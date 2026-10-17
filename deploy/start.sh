#!/usr/bin/env bash
# Start the API on localhost and the Streamlit UI on $PORT (default 7860), in one container.
# The UI only talks to the API, so the API is not exposed publicly.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-7860}"
uv run --no-sync uvicorn oncora.serve.app:app --host 127.0.0.1 --port 8000 &
api=$!

# Loading 1,360 target models and the lookup tables takes a few seconds; fail if the API dies.
for _ in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  kill -0 "$api" 2>/dev/null || { echo "API failed to start" >&2; exit 1; }
  sleep 1
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "API not healthy after 180 s" >&2; exit 1; }

export ONCORA_API_URL=http://127.0.0.1:8000
exec uv run --no-sync streamlit run app/main.py \
  --server.port "$PORT" --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
