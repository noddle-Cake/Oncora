#!/usr/bin/env bash
# Download the trained models + serving tables from a GitHub Release, verify, and unpack them,
# so a fresh clone can run the API/UI without the ~87 GB of raw data or a rebuild.
#
#   scripts/fetch_artifacts.sh                 # latest artifacts-* release
#   scripts/fetch_artifacts.sh artifacts-v1    # a specific release tag
#
# The repo is private, so this uses the GitHub CLI's login (`gh auth login`), or GH_TOKEN in CI.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${ONCORA_REPO:-noddle-Cake/Oncora}"
TAG="${1:-}"
if [ -z "$TAG" ]; then
  TAG=$(gh release list -R "$REPO" --json tagName -q '[.[] | select(.tagName | startswith("artifacts-"))][0].tagName')
fi
[ -n "$TAG" ] || { echo "No artifacts-* release found in $REPO" >&2; exit 1; }

mkdir -p dist
echo "==> downloading $TAG from $REPO"
gh release download "$TAG" -R "$REPO" -p 'oncora-artifacts-*.tar.gz*' -D dist --clobber

archive=$(ls dist/oncora-artifacts-*.tar.gz | head -1)
echo "==> checking $(basename "$archive")"
(cd dist && shasum -a 256 -c "$(basename "$archive").sha256")

echo "==> unpacking into $(pwd)"
tar -xzf "$archive"
uv run python -m oncora.serve.artifacts verify
echo "Done. Start the API: uv run uvicorn oncora.serve.app:app   UI: uv run streamlit run app/main.py"
