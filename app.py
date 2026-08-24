from __future__ import annotations

import hmac
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from features import PROMPT_COL, build_handcrafted_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/model_bundle.joblib"))
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "").strip()
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "50000"))

BUNDLE: dict[str, Any] | None = None
ENCODER: SentenceTransformer | None = None


def _origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*")
    return [x.strip() for x in raw.split(",") if x.strip()]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not MODEL_API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, MODEL_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_runtime() -> None:
    global BUNDLE, ENCODER
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model bundle not found at {MODEL_PATH}. Run export_model.py first and "
            "place model_bundle.joblib at the configured MODEL_PATH."
        )
    BUNDLE = joblib.load(MODEL_PATH)
    if BUNDLE.get("prompt_col") != PROMPT_COL:
        raise RuntimeError("Model bundle prompt column does not match service feature code.")
    ENCODER = SentenceTransformer(BUNDLE["embedding_model_name"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime()
    yield


app = FastAPI(
    title="Prompt Interaction-Risk API",
    version="1.0.0",
    description=(
        "Stateless scoring API for the frozen prompt-efficiency model. "
        "It does not persist prompt text."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class ScoreRequest(BaseModel):
    prompt: str = Field(min_length=1)
    participant_id: str | None = None


class ScoreResponse(BaseModel):
    risk: float
    risk_definition: str
    model_version: str
    model_name: str
    mode: str
    latency_ms: float
    probabilities: dict[str, float]
    risk_class2: float | None = None
    risk_1plus2: float | None = None


def _score(prompt: str) -> dict[str, Any]:
    if BUNDLE is None or ENCODER is None:
        raise RuntimeError("Model runtime is not initialized")

    frame = pd.DataFrame({PROMPT_COL: [prompt]})
    feat = build_handcrafted_features(frame)
    emb = ENCODER.encode([prompt], normalize_embeddings=False, show_progress_bar=False)

    embedding_cols = BUNDLE["embedding_cols"]
    if emb.shape[1] != len(embedding_cols):
        raise RuntimeError(
            f"Embedding dimension mismatch: service produced {emb.shape[1]}, "
            f"bundle expects {len(embedding_cols)}"
        )
    for j, col in enumerate(embedding_cols):
        feat[col] = emb[:, j]

    feature_cols = BUNDLE["numeric_cols"] + BUNDLE["categorical_cols"]
    probs = np.asarray(BUNDLE["pipeline"].predict_proba(feat[feature_cols]))[0]

    mode = BUNDLE["mode"]
    if mode.startswith("multiclass"):
        if len(probs) != 5:
            raise RuntimeError(f"Expected 5 probabilities, got {len(probs)}")
        probability_map = {str(i + 1): float(probs[i]) for i in range(5)}
        risk_class2 = float(probs[1])
        risk_1plus2 = float(probs[0] + probs[1])
        if mode == "multiclass_class2":
            risk = risk_class2
        elif mode == "multiclass_1plus2":
            risk = risk_1plus2
        else:
            risk = float(np.sum(probs[BUNDLE["risk_probability_indices"]]))
    else:
        if len(probs) != 2:
            raise RuntimeError(f"Expected 2 probabilities, got {len(probs)}")
        probability_map = {
            "scores_3_5": float(probs[0]),
            "scores_1_2": float(probs[1]),
        }
        risk = float(probs[1])
        risk_class2 = None
        risk_1plus2 = risk

    return {
        "risk": risk,
        "probabilities": probability_map,
        "risk_class2": risk_class2,
        "risk_1plus2": risk_1plus2,
    }


@app.get("/health")
def health():
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_name": BUNDLE["model_name"],
        "model_version": BUNDLE["model_version"],
        "mode": BUNDLE["mode"],
        "risk_definition": BUNDLE["risk_definition"],
        "training_n": BUNDLE.get("training_n"),
        "dataset_fingerprint": BUNDLE.get("dataset_fingerprint"),
    }


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score(req: ScoreRequest):
    prompt = req.prompt
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be blank")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Prompt exceeds MAX_PROMPT_CHARS={MAX_PROMPT_CHARS}",
        )

    started = time.perf_counter()
    try:
        result = _score(prompt)
    except Exception as exc:
        # Do not echo prompt content in errors or logs.
        raise HTTPException(status_code=500, detail=f"Scoring failed: {type(exc).__name__}") from exc
    latency_ms = (time.perf_counter() - started) * 1000

    assert BUNDLE is not None
    return ScoreResponse(
        risk=float(result["risk"]),
        risk_definition=BUNDLE["risk_definition"],
        model_version=BUNDLE["model_version"],
        model_name=BUNDLE["model_name"],
        mode=BUNDLE["mode"],
        latency_ms=round(latency_ms, 2),
        probabilities=result["probabilities"],
        risk_class2=result["risk_class2"],
        risk_1plus2=result["risk_1plus2"],
    )
