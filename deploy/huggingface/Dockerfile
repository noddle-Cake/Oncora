# syntax=docker/dockerfile:1.7
# Oncora as a Hugging Face Docker Space: FastAPI (internal :8000) + Streamlit UI (public :7860).
#
# The Space repo holds only this Dockerfile and README.md. The build clones the (private) GitHub
# repo and downloads the release bundle (models + serving tables, ~380 MB), both with a read-only
# GitHub token passed as the build secret GITHUB_TOKEN. The token is never written to an image layer.
#
# Local test:
#   docker build --secret id=GITHUB_TOKEN,env=GITHUB_TOKEN -t oncora deploy/huggingface
#   docker run --rm -p 7860:7860 oncora          # then open http://localhost:7860
FROM python:3.11-slim-bookworm

# libgomp1: LightGBM's OpenMP runtime. libxrender1/libxext6: RDKit molecule drawing.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl libgomp1 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# Hugging Face runs Docker Spaces as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    ONCORA_LLM_ENABLED=0
WORKDIR /home/user/app

ARG ONCORA_REPO=noddle-Cake/Oncora
ARG ONCORA_REF=main
ARG ARTIFACTS_TAG=artifacts-v1

# The token goes in an HTTP header for this one command, so it is not stored in .git/config.
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=true \
    AUTH=$(printf 'x-access-token:%s' "$(cat /run/secrets/GITHUB_TOKEN)" | base64 -w0) \
 && git -c http.extraHeader="Authorization: Basic ${AUTH}" \
        clone --depth 1 --branch "${ONCORA_REF}" "https://github.com/${ONCORA_REPO}.git" .

RUN uv sync --frozen --no-dev

RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=true \
    GITHUB_TOKEN="$(cat /run/secrets/GITHUB_TOKEN)" \
    uv run --no-sync python -m oncora.serve.artifacts fetch --repo "${ONCORA_REPO}" --tag "${ARTIFACTS_TAG}"

EXPOSE 7860
CMD ["bash", "deploy/start.sh"]
