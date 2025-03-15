# Self-hosted serverless LLM-based summarization service

A personal web service that creates AI-generated summaries of online articles using Claude 3 and client-side HTML-to-Markdown conversion.

## Usage

Two ways to use the service:

1. Using the bookmarklet (recommended):

   - Create a new bookmark with this code as the URL:

     ```javascript
     javascript: (function () {
       var s = document.createElement("script");
       s.src = "https://unpkg.com/turndown/dist/turndown.js";
       s.onload = function () {
         var t = new TurndownService(),
           m = t.turndown(document.body),
           f = document.createElement("form");
         f.method = "POST";
         f.enctype = "application/x-www-form-urlencoded";
         f.action = "YOUR_BACKEND_URL/summarize";

         [
           ["title", document.title],
           ["content", m],
           ["url", window.location.href],
         ].forEach(([k, v]) => {
           var i = document.createElement("input");
           i.type = "hidden";
           i.name = k;
           i.value = v;
           f.appendChild(i);
         });

         document.body.appendChild(f);
         f.submit();
       };
       document.body.appendChild(s);
     })();
     ```

   - Replace `YOUR_BACKEND_URL` with your actual service URL
   - Click the bookmark on any page you want to summarize

2. View recently summarized articles:

   ```txt
   https://YOUR_BACKEND_URL/recent
   ```

## How It Works

1. Content extraction via client-side Turndown.js (HTML to Markdown)
2. Summarization using Claude 3
3. Results cached in Google Cloud Storage
4. Protected by oauth2-proxy (Google auth)

## Implementation Notes

- Built with Python/Flask on Cloud Run
- Uses oauth2-proxy sidecar for auth
- Authenticated users defined in `authenticated_emails.txt` in GCS bucket
- Caches use gzip compression
- Summaries in Markdown format
- Deployed via GitHub Actions CI/CD pipeline

## Deployment

This application uses GitHub Actions for continuous deployment:

1. Secrets are stored securely in GitHub repository secrets
2. When code is pushed to the main branch, a workflow automatically:
   - Builds the Docker container
   - Deploys to Google Cloud Run
   - Configures service settings

To set up Workload Identity Federation for GitHub Actions:
- Follow guidance at: https://github.com/google-github-actions/auth#setting-up-workload-identity-federation
- Add the `WORKLOAD_IDENTITY_PROVIDER` and `SERVICE_ACCOUNT` secrets to GitHub

## Key Features

- Direct HTML-to-Markdown conversion in the browser
- No complex content extraction dependencies
- Clean, structured summaries from Claude 3
- Persistent caching of results
- Simple deployment and maintenance
