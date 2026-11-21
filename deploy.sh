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
MEMORY="${GCP_MEMORY:-1Gi}"
CPU="${GCP_CPU:-1}"
MIN_INSTANCES="${GCP_MIN_INSTANCES:-1}"
MAX_INSTANCES="${GCP_MAX_INSTANCES:-3}"

# 1. Verify prerequisites
if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: 'gcloud' CLI is not installed. Install Google Cloud SDK first." >&2
  exit 1
fi

if [[ ! -f "$KEY_FILE" ]]; then
  echo "Error: GCP service account key file '$KEY_FILE' not found in project root." >&2
  exit 1
fi

RAW_DB_URL="${NEON_DATABASE_URL:-${DATABASE_URL:-}}"
if [[ -z "$RAW_DB_URL" ]]; then
  echo "Error: NEON_DATABASE_URL or DATABASE_URL must be set in .env or environment." >&2
  exit 1
fi

if [[ "$RAW_DB_URL" == *"localhost"* || "$RAW_DB_URL" == *"127.0.0.1"* || "$RAW_DB_URL" == *"@db:"* ]]; then
  echo "Error: DATABASE_URL points to a local database ($RAW_DB_URL)." >&2
  echo "Please set NEON_DATABASE_URL (or DATABASE_URL) in .env to your Neon connection string." >&2
  exit 1
fi

# Normalize Neon URL for SQLAlchemy + asyncpg (ensure postgresql+asyncpg://, ssl=require, and strip channel_binding)
DEPLOY_DB_URL="$(python3 - "$RAW_DB_URL" <<'PY'
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

url = sys.argv[1].strip()
if url.startswith("postgres://"):
    url = "postgresql+asyncpg://" + url[len("postgres://"):]
elif url.startswith("postgresql://"):
    url = "postgresql+asyncpg://" + url[len("postgresql://"):]

parts = urlsplit(url)
query_params = []
for k, v in parse_qsl(parts.query, keep_blank_values=True):
    if k == "channel_binding":
        continue
    if k == "sslmode":
        query_params.append(("ssl", v))
    else:
        query_params.append((k, v))

print(urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query_params), parts.fragment)))
PY
)"

# Extract project_id from gcp_key.json if GCP_PROJECT_ID is not explicitly set
PROJECT_ID="${GCP_PROJECT_ID:-$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['project_id'])" "$KEY_FILE")}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Error: Could not determine GCP project_id from '$KEY_FILE' or GCP_PROJECT_ID." >&2
  exit 1
fi

echo "==> Step 1/3: Seeding data to Neon database..."
if command -v uv >/dev/null 2>&1; then
  DATABASE_URL="$DEPLOY_DB_URL" uv run --no-dev python seed.py
else
  DATABASE_URL="$DEPLOY_DB_URL" python3 seed.py
fi

echo "==> Step 2/3: Authenticating with GCP using '$KEY_FILE' (Project: $PROJECT_ID)..."
gcloud auth activate-service-account --key-file="$KEY_FILE" --quiet
gcloud config set project "$PROJECT_ID" --quiet

echo "==> Step 3/3: Deploying '$SERVICE_NAME' to Cloud Run (region: $REGION, min-instances: $MIN_INSTANCES)..."
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8000 \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --set-env-vars "^||^DATABASE_URL=${DEPLOY_DB_URL}||OPENAI_API_KEY=${PROD_OPENAI_API_KEY:-}||OPENAI_MODEL=${OPENAI_MODEL:-gpt-6-luna}" \
  --quiet

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"

echo ""
echo "Deployment complete!"
echo "  Service URL : ${SERVICE_URL}"
echo "  Swagger Docs: ${SERVICE_URL}/docs"
echo "  MCP Endpoint: ${SERVICE_URL}/mcp"
