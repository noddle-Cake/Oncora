# Deploying Oncora to a public link (Hugging Face Space)

One container runs the FastAPI service on `127.0.0.1:8000` and the Streamlit UI on port 7860
(`deploy/start.sh`). The build clones this repo and downloads the `artifacts-*` release bundle,
so no data or models are uploaded to Hugging Face by hand.

**Resources:** ~2.5 GB RAM, 1–2 vCPUs, no GPU; ~0.1–0.4 s per prediction. The free
"CPU basic" Space hardware is enough.

## One-time setup

1. **GitHub token (read-only).** On GitHub, create a fine-grained personal access token limited to
   `noddle-Cake/Oncora` with **Contents: Read-only**. The build needs it because the repo and the
   release are private.
2. **Hugging Face account and token.** Create a write token at huggingface.co/settings/tokens, then
   log in locally:
   ```bash
   uvx --from huggingface_hub hf auth login
   ```
3. **Create the Space and its secret**, then upload the two Space files:
   ```bash
   SPACE=<your-hf-username>/oncora
   hf() { uvx --from huggingface_hub hf "$@"; }
   hf repos create "$SPACE" --repo-type space --space-sdk docker --private   # drop --private to go public
   GITHUB_TOKEN=<token from step 1> hf spaces secrets add "$SPACE" -s GITHUB_TOKEN
   hf upload "$SPACE" deploy/huggingface . --repo-type space
   hf spaces wait "$SPACE"                                                   # until it is running
   ```
   The Space settings page can switch a private Space to public later.
   Hugging Face builds the image (a few minutes) and serves it at
   `https://huggingface.co/spaces/<your-hf-username>/oncora`.

## Alternative: self-contained upload (no GitHub token)

Upload the code and the artifact bundle into the (private) Space, so its build never touches
GitHub. This is how the live Space is deployed:

```bash
./scripts/fetch_artifacts.sh                 # or extract a locally packaged bundle
deploy/stage_space.sh /tmp/oncora-space      # code (git archive of HEAD) + bundle + card + Dockerfile.bundled
hf repos create "$SPACE" --repo-type space --space-sdk docker --private --exist-ok
hf upload "$SPACE" /tmp/oncora-space . --repo-type space --delete .gitignore
hf spaces wait "$SPACE"
```

## Options

- **Claude explanations** are off by default (`ONCORA_LLM_ENABLED=0`): a public link could
  otherwise spend the API budget. The deterministic template explanation works without a key. To
  enable it, add the secret `ANTHROPIC_API_KEY` and the variable `ONCORA_LLM_ENABLED=1`.
- **New models:** publish a new `artifacts-vN` release (see the README quick start), then set the
  Space build argument / Dockerfile `ARTIFACTS_TAG` and restart the Space (factory rebuild).
- **Free Spaces sleep** after a period without visitors. The first visit after that waits for a
  restart. Wake it up shortly before a demo, or use paid hardware for always-on.

## Test the container locally

```bash
GITHUB_TOKEN=$(gh auth token) docker build --secret id=GITHUB_TOKEN,env=GITHUB_TOKEN -t oncora deploy/huggingface
docker run --rm -p 7860:7860 oncora      # open http://localhost:7860
```
