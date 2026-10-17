"""FastAPI inference service (spec 08, Req 1, 3, 4; FR-19, FR-23, AC-9).

    uv run uvicorn oncora.serve.app:app --port 8000

Models load once at startup. A missing artifact fails startup with the path it expected; that
beats serving half a system. Run IDs come from ONCORA_TARGET_RUN / ONCORA_SENS_RUN, or default to
the latest.
"""

import os
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from oncora.config import DISCLAIMER
from oncora.serve.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ValidationErrorResponse,
)


def _default_pipeline():
    from oncora.serve.pipeline import OncoraPipeline

    return OncoraPipeline(os.environ.get("ONCORA_TARGET_RUN"), os.environ.get("ONCORA_SENS_RUN"))


def create_app(pipeline_factory: Callable = _default_pipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="Oncora", version="0.1.0", lifespan=lifespan,
                  description=f"Compound → targets, sensitive cell lines, cancer types. {DISCLAIMER}")

    @app.get("/health", response_model=HealthResponse)
    def health():
        p = app.state.pipeline
        return HealthResponse(model_versions=p.model_versions, disease_map=p.disease_map is not None)

    @app.post("/predict", response_model=PredictResponse,
              responses={422: {"model": ValidationErrorResponse}})
    def predict(req: PredictRequest):
        out = app.state.pipeline.predict(req.smiles, explain_with_llm=req.explain_with_llm)
        if not out["valid"]:
            return JSONResponse(status_code=422, content=ValidationErrorResponse(**out).model_dump())
        return PredictResponse(**out)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest):
        # Template renderer only: batch requests must not multiply LLM spend (NFR-10).
        results = []
        for s in req.smiles:
            out = app.state.pipeline.predict(s, explain_with_llm=False)
            results.append(PredictResponse(**out) if out["valid"] else ValidationErrorResponse(**out))
        return BatchResponse(results=results)

    return app


app = create_app()
