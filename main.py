from __future__ import annotations

import os
import json
import requests
import gzip
import re
import base64
import time
import logging
import markdown2
from datetime import datetime
from dataclasses import dataclass, asdict
from heapq import heappush, heappushpop
from typing import Optional, Final, TypeAlias, List, Iterator, Tuple
from flask import (
    Flask,
    Response,
    request,
    render_template_string,
    make_response,
)
from anthropic import Anthropic
from urllib.parse import unquote, urlparse
from werkzeug.middleware.proxy_fix import ProxyFix
from functools import wraps
from typing_extensions import TypeGuard
from google.cloud import storage

# Type definitions
HTMLResponse: TypeAlias = Tuple[str, int]
URL: TypeAlias = str

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    jina_api_key: str
    claude_api_key: str
    bucket_name: str
    jina_timeout: Final[int] = 30
    claude_timeout: Final[int] = 120


@dataclass(frozen=True)
class CachedResult:
    title: str
    summary: str  # Markdown format


@dataclass(frozen=True)
class SummaryEntry:
    url: str
    timestamp: datetime

    def __lt__(self, other: SummaryEntry) -> bool:
        # For heap: older timestamps float to top for removal
        return self.timestamp < other.timestamp


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
            color: #333;
        }
        h1 { font-size: 1.5rem; margin-bottom: 1rem; }
        .source { color: #666; margin-bottom: 1rem; }
        .summary { 
            background: #f5f5f5; 
            padding: 1rem; 
            border-radius: 8px;
        }
        .summary h1 { font-size: 1.4rem; }
        .summary h2 { font-size: 1.2rem; }
        .summary h3 { font-size: 1.1rem; }
        .summary ul, .summary ol { 
            padding-left: 1.5rem;
            margin: 1rem 0;
        }
        .summary p { margin: 1rem 0; }
        .summary code {
            background: #e0e0e0;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: monospace;
        }
        .summary pre {
            background: #e0e0e0;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }
        .error { 
            background: #fff3f3; 
            color: #d63031; 
            padding: 1rem; 
            border-radius: 8px; 
            margin: 1rem 0; 
        }
    </style>
</head>
<body>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% else %}
        <h1>{{ title }}</h1>
        <div class="source">Source: <a href="{{ url }}">{{ url }}</a></div>
        <div class="summary">{{ summary|safe }}</div>
    {% endif %}
</body>
</html>
"""

LIST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recent Summaries</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        td, th {
            padding: 8px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .date {
            white-space: nowrap;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>Recent Summaries</h1>
    <table>
        <tr>
            <th>Date</th>
            <th>Article</th>
        </tr>
        {% for summary in summaries %}
        <tr>
            <td class="date">{{ summary.timestamp }}</td>
            <td><a href="/{{ summary.url }}">{{ summary.title }}</a></td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

SUMMARY_PROMPT = """
<content>{content}</content>

Please transform this content into an in-depth technical narrative that combines technical depth with the personality and insights from the original conversation. Write for a knowledgeable audience that wants both rich technical detail and the human story behind the innovations.

When describing technical innovations, combine thorough technical explanation with the speaker's perspective on why these choices matter. Maintain the original voice and personality while diving deep into the most compelling technical aspects. Begin directly with the content.

Please think through this summary task step by step in your internal reasoning, then provide the final summary in a structured format.

Your response must be wrapped in summary tags like this:
<summary>
[Your technical summary here with these characteristics:]
Deliver:
- Detailed explanations of novel technical approaches and why they matter
- The speaker's insights and reasoning behind technical choices
- Specific examples and implementation details
- Real-world context and practical implications

Structure requirements:
- 10-15 minute reading time
- Clear Markdown formatting
- Flowing narrative with minimal bullet points
- Natural blend of technical depth and personal insights

Writing style:
- Detailed technical explanations that reveal underlying complexity
- Preserve interesting quotes and personal observations
- Explain what makes techniques "novel" or interesting
- Let the speaker's excitement and expertise shine through
- Balance technical depth with clear, engaging exposition
- Connected ideas rather than isolated points
</summary>
"""


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

storage_client = storage.Client()
config = Config(
    jina_api_key=os.environ["JINA_API_KEY"],
    claude_api_key=os.environ["CLAUDE_API_KEY"],
    bucket_name=os.environ["BUCKET_NAME"],
)


def gzip_response(f):
    @wraps(f)
    def decorated_function(*args, **kwargs) -> Response:
        response = f(*args, **kwargs)
        if not isinstance(response, (str, bytes)):
            return response

        if "gzip" not in request.headers.get("Accept-Encoding", "").lower():
            return response

        response = make_response(gzip.compress(response.encode("utf-8")))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
        response.headers["Content-Length"] = len(response.data)
        return response

    return decorated_function


def get_blob_name(url: URL) -> str:
    """Generate blob name from URL using base64"""
    return f"{base64.urlsafe_b64encode(url.encode()).rstrip(b'=').decode()}.gz"


def url_from_blob_name(blob_name: str) -> URL:
    """Extract URL from blob name"""
    encoded = blob_name[:-3]  # remove .gz
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding).decode()


def is_valid_https_url(url: str) -> TypeGuard[URL]:
    """Validate that a string is a proper HTTPS URL"""
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and bool(parsed.netloc)
    except Exception:
        return False


def get_and_validate_url(path: str) -> Optional[URL]:
    """Extract and validate target URL from request path"""
    # https:// gets mapped to https:/ (single slash)
    https_prefix = "/https:/"
    if not path.startswith(https_prefix):
        logger.warning(f"Invalid URL format (must start with {https_prefix}): {path}")
        return None

    url = unquote("https://" + path.removeprefix(https_prefix))
    if not is_valid_https_url(url):
        logger.warning(f"URL validation failed: {url}")
        return None

    logger.info(f"URL validated: {url}")
    return url


def list_blobs_by_page(bucket: storage.Bucket) -> Iterator[storage.Blob]:
    """Iterator over all blobs, paginated to control memory usage"""
    page_token = None
    page_count = 0
    while True:
        blob_list = bucket.list_blobs(
            max_results=1000,
            page_token=page_token,
            fields="items(name,timeCreated),nextPageToken",
        )

        for blob in blob_list:
            yield blob

        page_token = blob_list.next_page_token
        page_count += 1
        logger.debug(f"Processed page {page_count} of blob listing")

        if not page_token:
            break


def get_recent_summaries(max_entries: int = 1000) -> List[Tuple[str, str]]:
    """Get the most recent summaries, properly paginated"""
    bucket = storage_client.bucket(config.bucket_name)
    recent_heap: List[SummaryEntry] = []
    total_processed = 0

    try:
        for blob in list_blobs_by_page(bucket):
            if not blob.name.endswith(".gz"):
                continue

            total_processed += 1
            entry = SummaryEntry(
                url=url_from_blob_name(blob.name), timestamp=blob.time_created
            )

            if len(recent_heap) < max_entries:
                heappush(recent_heap, entry)
            else:
                heappushpop(recent_heap, entry)

        # Sort newest first
        sorted_entries = sorted(recent_heap, key=lambda x: x.timestamp, reverse=True)
        logger.info(
            f"Processed {total_processed} entries, returning {len(sorted_entries)} most recent"
        )

        return [
            (entry.url, entry.timestamp.strftime("%Y-%m-%d %H:%M UTC"))
            for entry in sorted_entries
        ]

    except Exception as e:
        logger.error(f"Error listing summaries: {e}", exc_info=True)
        return []


def store_result(url: URL, title: str, summary: str) -> None:
    """Store a compressed result in Cloud Storage"""
    bucket = storage_client.bucket(config.bucket_name)
    blob_name = get_blob_name(url)
    blob = bucket.blob(blob_name)

    result = CachedResult(title=title, summary=summary)

    # Compress and store
    compressed = gzip.compress(json.dumps(asdict(result)).encode("utf-8"))
    blob.upload_from_string(compressed)
    logger.info(f"Stored result for {url} in {blob_name}")


def get_cached_result(url: URL) -> Optional[CachedResult]:
    """Try to get a cached result from Cloud Storage"""
    try:
        bucket = storage_client.bucket(config.bucket_name)
        blob_name = get_blob_name(url)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logger.debug(f"Cache miss for {url}")
            return None

        # Decompress and parse
        compressed_data = blob.download_as_bytes()
        data = json.loads(gzip.decompress(compressed_data))
        logger.info(f"Cache hit for {url}")
        return CachedResult(**data)
    except Exception as e:
        logger.error(f"Error retrieving cached result: {e}", exc_info=True)
        return None


def scrape_with_jina(url: URL) -> str:
    """Scrape content using Jina"""
    headers = {"Authorization": f"Bearer {config.jina_api_key}"}
    response = requests.get(
        f"https://r.jina.ai/{url}", headers=headers, timeout=config.jina_timeout
    )
    response.raise_for_status()
    return response.text


def summarize_with_claude(content: str) -> str:
    """Summarize content using Claude"""
    client = Anthropic(api_key=config.claude_api_key)

    message = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=8192,
        temperature=0.3,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(content=content)}],
        timeout=config.claude_timeout,
    )

    response = message.content[0].text

    # Try to extract summary section
    try:
        summary = (
            re.search(r"<summary>(.*?)</summary>", response, re.DOTALL).group(1).strip()
        )
        return summary
    except (AttributeError, IndexError):
        return f"[Failed to extract summary tags]\n\n{response}"


def process_request(url: URL) -> Tuple[str, str]:
    """Process a single summarization request with caching"""
    cached = get_cached_result(url)
    if cached:
        html_summary = markdown2.markdown(cached.summary)
        return cached.title, html_summary

    logger.info(f"Processing new request for {url}")
    start_time = time.time()

    content = scrape_with_jina(url)
    scrape_time = time.time()
    logger.info(f"Jina scraping completed in {scrape_time - start_time:.2f}s")

    title_match = re.search(r"Title: (.*?)\n", content)
    title = title_match.group(1) if title_match else "Untitled Article"

    markdown_summary = summarize_with_claude(content)
    summarize_time = time.time()
    logger.info(
        f"Claude summarization completed in {summarize_time - scrape_time:.2f}s"
    )

    # Store the markdown version
    store_result(url, title, markdown_summary)

    # Convert to HTML for display
    html_summary = markdown2.markdown(markdown_summary)

    return title, html_summary


@app.route("/<path:url>")
@gzip_response
def summarize(url: str) -> HTMLResponse:
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.info(f"Starting request {request_id} for URL path: {request.path}")
    start_time = time.time()

    target_url = get_and_validate_url(request.path)
    if not target_url:
        return (
            render_template_string(
                HTML_TEMPLATE, error="Invalid URL. Must be HTTPS.", title="Error"
            ),
            400,
        )

    try:
        title, summary = process_request(target_url)

        total_duration = time.time() - start_time
        logger.info(
            f"Request {request_id}: Complete success in {total_duration:.2f}s "
            f"for {target_url}"
        )

        return (
            render_template_string(
                HTML_TEMPLATE, title=title, url=target_url, summary=summary, error=None
            ),
            200,
        )

    except requests.Timeout:
        duration = time.time() - start_time
        logger.error(
            f"Request {request_id}: Timeout after {duration:.2f}s for {target_url}",
            exc_info=True,
        )
        return (
            render_template_string(
                HTML_TEMPLATE,
                error="Request timed out. Please try again.",
                title="Error",
            ),
            504,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Request {request_id}: Failed after {duration:.2f}s for {target_url}. "
            f"Error: {str(e)}",
            exc_info=True,
        )
        return (
            render_template_string(
                HTML_TEMPLATE,
                error=f"Error processing request: {str(e)}",
                title="Error",
            ),
            500,
        )


@app.route("/recent")
@gzip_response
def recent_summaries() -> HTMLResponse:
    """Show recent summaries"""
    logger.info("Fetching recent summaries")
    start_time = time.time()
    recent = get_recent_summaries(max_entries=1000)
    logger.info(f"Retrieved {len(recent)} summaries in {time.time() - start_time:.2f}s")

    return (
        render_template_string(
            LIST_TEMPLATE,
            summaries=[
                {
                    "url": url,
                    "timestamp": timestamp,
                    "title": url.removeprefix("https://").removesuffix("/"),
                }
                for url, timestamp in recent
            ],
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
