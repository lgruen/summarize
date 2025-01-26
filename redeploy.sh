#!/bin/bash

set -ex

gcloud builds submit --project leo-summarize --tag asia-southeast1-docker.pkg.dev/leo-summarize/images/summarize:latest
gcloud --quiet run services delete --project leo-summarize --region asia-southeast1 summarize
gcloud run services replace service.yaml --project leo-summarize --region asia-southeast1
gcloud --project=leo-summarize run services add-iam-policy-binding --region=asia-southeast1 summarize --member="allUsers" --role="roles/run.invoker"
