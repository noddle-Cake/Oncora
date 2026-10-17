#!/usr/bin/env bash
# Stage a self-contained Hugging Face Space folder: committed code (git archive of HEAD) + the
# models and serving tables of an extracted artifact bundle + the Space card and Dockerfile.
#
#   deploy/stage_space.sh <out_dir>
#   uvx --from huggingface_hub hf upload <user>/oncora <out_dir> . --repo-type space
#
# Needs the bundle extracted in this checkout (MANIFEST.json + models/ + data/serving/), i.e.
# after `scripts/fetch_artifacts.sh` or `python -m oncora.serve.artifacts package` + extract.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:?usage: deploy/stage_space.sh <out_dir>}"

[ -f MANIFEST.json ] || { echo "No MANIFEST.json here: fetch or extract an artifact bundle first" >&2; exit 1; }
uv run --no-sync python -m oncora.serve.artifacts verify

rm -rf "$out" && mkdir -p "$out"
git archive HEAD | tar -x -C "$out"
cp MANIFEST.json "$out/"
# Only the files the manifest lists (exactly the published bundle), dereferencing any symlinks.
python3 - "$out" <<'PY'
import json, shutil, sys
from pathlib import Path
out = Path(sys.argv[1])
for rel in json.load(open("MANIFEST.json"))["files"]:
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rel, dst)
PY
cp deploy/huggingface/README.md "$out/README.md"            # the Space card must be the root README
cp deploy/huggingface/Dockerfile.bundled "$out/Dockerfile"
echo "Staged $(du -sh "$out" | cut -f1) in $out"
