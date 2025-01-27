# Self-hosted serverless LLM-based summarization service

A personal web service that creates AI-generated summaries of online articles using Claude 3 and Jina AI.

## Usage

1. Prepend any HTTPS URL with `https://summarize.gruenschloss.org/`:
   ```
   https://summarize.gruenschloss.org/https://example.com/article
   ```

2. View recently summarized articles:
   ```
   https://summarize.gruenschloss.org/recent
   ```

## How It Works

1. Content extraction via Jina AI 
2. Summarization using Claude 3
3. Results cached in Google Cloud Storage
4. Protected by oauth2-proxy (Google auth)

## Implementation Notes

- Built with Python/Flask on Cloud Run
- Uses oauth2-proxy sidecar for auth
- Authenticated users defined in `authenticated_emails.txt` in GCS bucket
- Caches use gzip compression
- Summaries in Markdown format
- `service.yaml` contains full deployment config -- update secrets accordingly
