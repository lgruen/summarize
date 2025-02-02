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
from anthropic.types import TextBlock
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
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
</head>
<body class="bg-gray-50">
    <div class="max-w-4xl mx-auto p-4 sm:p-6 lg:p-8">
        {% if error %}
            <div class="bg-red-50 border-l-4 border-red-400 p-4 mb-6 rounded">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                    <div class="ml-3">
                        <p class="text-sm text-red-700">{{ error }}</p>
                    </div>
                </div>
            </div>
        {% else %}
            <h1 class="text-2xl font-bold text-gray-900 mb-4">{{ title }}</h1>
            <div class="text-gray-600 mb-6">
                Source: <a href="{{ url }}" class="text-indigo-600 hover:text-indigo-900 transition-colors">{{ url }}</a>
            </div>
            <div class="bg-white rounded-lg shadow-sm p-6">
                <div class="prose prose-slate max-w-none prose-headings:font-semibold prose-a:text-indigo-600">
                    {{ summary|safe }}
                </div>
            </div>
        {% endif %}
    </div>
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
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <script>
        async function deleteSummary(url, row) {
            try {
                const response = await fetch('/delete/' + url, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    row.remove();
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }
    </script>
</head>
<body class="bg-gray-50">
    <div class="max-w-4xl mx-auto p-4 sm:p-6 lg:p-8">
        <h1 class="text-2xl font-bold text-gray-900 mb-6">Recent Summaries</h1>
        <div class="bg-white rounded-lg shadow overflow-hidden">
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Article</th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"></th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    {% for summary in summaries %}
                    <tr class="hover:bg-gray-50 transition-colors" id="row-{{ loop.index }}">
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{ summary.timestamp }}</td>
                        <td class="px-6 py-4 text-sm">
                            <a href="/{{ summary.url }}" class="text-indigo-600 hover:text-indigo-900">{{ summary.title }}</a>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <button onclick="deleteSummary('{{ summary.url }}', document.getElementById('row-{{ loop.index }}'))" 
                                    class="text-red-600 hover:text-red-900 transition-colors">
                                Delete
                            </button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
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

MARKDOWN_EXTRAS = ["break-on-newline", "cuddled-lists", "markdown-in-html", "tables"]

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
        content: Response = f(*args, **kwargs)
        if not isinstance(content, str):
            return content

        if "gzip" not in request.headers.get("Accept-Encoding", "").lower():
            return Response(content)

        response = make_response(gzip.compress(content.encode("utf-8")))
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
            if (
                blob.name is None
                or not blob.name.endswith(".gz")
                or blob.time_created is None
            ):
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

    assert isinstance(message.content[0], TextBlock)
    response = message.content[0].text

    # Try to extract summary section
    match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return f"[Failed to extract summary tags]\n\n{response}"


def process_request(url: URL) -> Tuple[str, str]:
    """Process a single summarization request with caching"""
    cached = get_cached_result(url)
    if cached:
        html_summary = markdown2.markdown(cached.summary, extras=MARKDOWN_EXTRAS)
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
    html_summary = markdown2.markdown(markdown_summary, extras=MARKDOWN_EXTRAS)

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


@app.route("/raw/<path:path>")
def raw_content(path: str) -> Response:
    """Debug endpoint to show raw cached content"""
    # Add leading slash to match expected format
    target_url = get_and_validate_url("/" + path)
    if not target_url:
        return Response("Invalid URL", 400)

    cached = get_cached_result(target_url)
    if not cached:
        return Response("Not found in cache", 404)

    # Return both title and summary for completeness
    return Response(
        f"Title: {cached.title}\n\nSummary:\n{cached.summary}", mimetype="text/plain"
    )


@app.route("/delete/<path:path>", methods=["DELETE"])
def delete_summary(path: str) -> Response:
    """Delete a cached summary"""
    target_url = get_and_validate_url("/" + path)
    if not target_url:
        return Response("Invalid URL", 400)

    try:
        bucket = storage_client.bucket(config.bucket_name)
        blob_name = get_blob_name(target_url)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return Response("Not found", 404)

        blob.delete()
        return Response("Deleted", 200)
    except Exception as e:
        logger.error(f"Error deleting summary: {e}", exc_info=True)
        return Response("Error deleting summary", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
