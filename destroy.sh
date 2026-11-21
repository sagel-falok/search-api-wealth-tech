#!/usr/bin/env bash
set -euo pipefail

# Move to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env if present
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

KEY_FILE="${GCP_KEY_FILE:-gcp_key.json}"
SERVICE_NAME="${GCP_SERVICE_NAME:-search-api}"
REGION="${GCP_REGION:-europe-west2}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: 'gcloud' CLI is not installed. Install Google Cloud SDK first." >&2
  exit 1
fi

if [[ ! -f "$KEY_FILE" ]]; then
  echo "Error: GCP service account key file '$KEY_FILE' not found in project root." >&2
  exit 1
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['project_id'])" "$KEY_FILE")}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: Could not determine GCP project_id from '$KEY_FILE' or GCP_PROJECT_ID." >&2
  exit 1
fi

echo "==> Authenticating with GCP using '$KEY_FILE' (Project: $PROJECT_ID)..."
gcloud auth activate-service-account --key-file="$KEY_FILE" --quiet
gcloud config set project "$PROJECT_ID" --quiet

echo "==> Deleting Cloud Run service '$SERVICE_NAME' in region '$REGION'..."
if gcloud run services describe "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" >/dev/null 2>&1; then
  gcloud run services delete "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --quiet
  echo "Cloud Run service '$SERVICE_NAME' deleted."
else
  echo "Cloud Run service '$SERVICE_NAME' does not exist in region '$REGION' (already deleted)."
fi

IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}"
echo "==> Cleaning up Artifact Registry container images at '${IMAGE_PATH}' (if present)..."
if gcloud artifacts docker images list "$IMAGE_PATH" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts docker images delete "$IMAGE_PATH" \
    --delete-tags \
    --project "$PROJECT_ID" \
    --quiet || true
  echo "Container images for '$SERVICE_NAME' deleted."
fi

echo "==> Cloud Run cleanup complete!"
