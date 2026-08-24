from __future__ import annotations

import hmac
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import PROMPT_COL, build_handcrafted_features


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "model" / "model_bundle.joblib"),
    )
)

MODEL_API_KEY = os.getenv("MODEL_API_KEY", "").strip()

MAX_PROMPT_CHARS = int(
    os.getenv("MAX_PROMPT_CHARS", "50000")
)


def get_allowed_origins() -> list[str]:
    """
    Read allowed origins from the environment.

    Examples:
        ALLOWED_ORIGINS=*
        ALLOWED_ORIGINS=https://my-app.lovable.app
        ALLOWED_ORIGINS=https://a.com,https://b.com
    """
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()

    if raw == "*":
        return ["*"]

    return [
        origin.strip()
        for origin in raw.split(",")
        if origin.strip()
    ]


# ============================================================
# Runtime globals
#
# IMPORTANT:
# We deliberately do NOT import sentence-transformers, torch,
# sklearn, or xgboost here.
#
# This allows Render to start Uvicorn and bind to $PORT before
# loading the memory-heavy ML stack.
# ============================================================

BUNDLE: dict[str, Any] | None = None
ENCODER: Any = None

LOAD_LOCK = threading.Lock()


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Prompt Interaction-Risk API",
    version="1.0.0",
    description=(
        "Stateless scoring API for the frozen prompt-efficiency model. "
        "The model uses only the initial prompt."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
    ],
)


# ============================================================
# Schemas
# ============================================================

class ScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
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
    participant_id: str | None = None


# ============================================================
# Authentication
# ============================================================

def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """
    Require an API key if MODEL_API_KEY is configured.

    If MODEL_API_KEY is empty, authentication is disabled.
    This can be useful for local testing but should not be used
    for the live experiment.
    """

    if not MODEL_API_KEY:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
        )

    if not hmac.compare_digest(
        x_api_key,
        MODEL_API_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


# ============================================================
# Lazy model loading
# ============================================================

def load_runtime() -> None:
    """
    Load the exported model and MiniLM encoder.

    This function is deliberately called only by /warmup or
    /score. It is NOT called when FastAPI starts.

    This is important on Render because importing PyTorch and
    SentenceTransformers before Uvicorn binds to the port can
    cause Render to report 'No open ports detected'.
    """

    global BUNDLE, ENCODER

    # Already loaded
    if BUNDLE is not None and ENCODER is not None:
        return

    with LOAD_LOCK:

        # A different request may have loaded everything while
        # this request was waiting for the lock.
        if BUNDLE is not None and ENCODER is not None:
            return

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model bundle not found at: {MODEL_PATH}"
            )

        print(
            f"[MODEL] Loading bundle from {MODEL_PATH}",
            flush=True,
        )

        # Import joblib lazily as well. Loading the pipeline may
        # cause sklearn/xgboost to be imported.
        import joblib

        bundle = joblib.load(MODEL_PATH)

        if not isinstance(bundle, dict):
            raise RuntimeError(
                "The model bundle is not a dictionary."
            )

        required_keys = [
            "pipeline",
            "model_name",
            "model_version",
            "mode",
            "risk_definition",
            "numeric_cols",
            "categorical_cols",
            "embedding_cols",
            "embedding_model_name",
            "prompt_col",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in bundle
        ]

        if missing_keys:
            raise RuntimeError(
                "Model bundle is missing required keys: "
                + ", ".join(missing_keys)
            )

        if bundle["prompt_col"] != PROMPT_COL:
            raise RuntimeError(
                "Prompt-column mismatch between model bundle "
                "and features.py. "
                f"Bundle: {bundle['prompt_col']}; "
                f"service: {PROMPT_COL}"
            )

        embedding_model_name = bundle[
            "embedding_model_name"
        ]

        print(
            "[MODEL] Loading sentence encoder: "
            f"{embedding_model_name}",
            flush=True,
        )

        # CRITICAL:
        # Do not move this import to the top of app.py.
        # sentence-transformers imports PyTorch and uses a lot
        # of memory.
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            embedding_model_name,
            device="cpu",
        )

        # Only set globals after everything loaded correctly.
        BUNDLE = bundle
        ENCODER = encoder

        print(
            "[MODEL] Runtime loaded successfully.",
            flush=True,
        )


# ============================================================
# Feature construction
# ============================================================

def build_model_frame(
    prompt: str,
) -> pd.DataFrame:
    """
    Reproduce the exact feature representation expected by the
    exported model:

        handcrafted features
        +
        384 MiniLM dimensions
    """

    if BUNDLE is None or ENCODER is None:
        raise RuntimeError(
            "Model runtime has not been loaded."
        )

    # --------------------------------------------------------
    # Handcrafted features
    # --------------------------------------------------------

    frame = pd.DataFrame(
        {
            PROMPT_COL: [prompt]
        }
    )

    feat = build_handcrafted_features(frame)

    # --------------------------------------------------------
    # MiniLM embedding
    # --------------------------------------------------------

    embedding = ENCODER.encode(
        [prompt],
        normalize_embeddings=bool(
            BUNDLE.get(
                "embedding_normalize",
                False,
            )
        ),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    embedding_cols = list(
        BUNDLE["embedding_cols"]
    )

    if embedding.ndim != 2:
        raise RuntimeError(
            "Unexpected embedding dimensionality: "
            f"{embedding.shape}"
        )

    if embedding.shape[0] != 1:
        raise RuntimeError(
            "Expected one embedding row, received "
            f"{embedding.shape[0]}"
        )

    if embedding.shape[1] != len(
        embedding_cols
    ):
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"encoder produced {embedding.shape[1]}, "
            f"but model expects {len(embedding_cols)}"
        )

    # Add all embedding columns at once rather than one by one.
    # This avoids pandas DataFrame fragmentation warnings.
    embedding_frame = pd.DataFrame(
        embedding,
        columns=embedding_cols,
        index=feat.index,
    )

    feat = pd.concat(
        [
            feat,
            embedding_frame,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Exact model columns
    # --------------------------------------------------------

    feature_cols = (
        list(BUNDLE["numeric_cols"])
        + list(BUNDLE["categorical_cols"])
    )

    missing_features = [
        col
        for col in feature_cols
        if col not in feat.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Missing model features: "
            + ", ".join(
                missing_features[:20]
            )
        )

    return feat[feature_cols]


# ============================================================
# Prediction
# ============================================================

def score_prompt(
    prompt: str,
) -> dict[str, Any]:
    """
    Calculate interaction-risk probabilities for one prompt.
    """

    load_runtime()

    if BUNDLE is None:
        raise RuntimeError(
            "Model bundle failed to load."
        )

    model_frame = build_model_frame(
        prompt
    )

    pipeline = BUNDLE["pipeline"]

    probabilities = np.asarray(
        pipeline.predict_proba(
            model_frame
        )
    )[0]

    mode = BUNDLE["mode"]

    risk_class2: float | None = None
    risk_1plus2: float | None = None

    # --------------------------------------------------------
    # Five-class model
    # --------------------------------------------------------

    if mode.startswith("multiclass"):

        if len(probabilities) != 5:
            raise RuntimeError(
                "Expected five probabilities from "
                f"multiclass model, got {len(probabilities)}"
            )

        probability_map = {
            str(i + 1): float(
                probabilities[i]
            )
            for i in range(5)
        }

        risk_class2 = float(
            probabilities[1]
        )

        risk_1plus2 = float(
            probabilities[0]
            + probabilities[1]
        )

        if mode == "multiclass_class2":
            risk = risk_class2

        elif mode == "multiclass_1plus2":
            risk = risk_1plus2

        else:
            risk_indices = BUNDLE.get(
                "risk_probability_indices",
                [],
            )

            if not risk_indices:
                raise RuntimeError(
                    "No risk probability indices "
                    "defined in model bundle."
                )

            risk = float(
                np.sum(
                    probabilities[
                        risk_indices
                    ]
                )
            )

    # --------------------------------------------------------
    # Binary 1+2 versus 3-5 model
    # --------------------------------------------------------

    elif mode == "binary_1plus2":

        if len(probabilities) != 2:
            raise RuntimeError(
                "Expected two probabilities from "
                f"binary model, got {len(probabilities)}"
            )

        probability_map = {
            "scores_3_5": float(
                probabilities[0]
            ),
            "scores_1_2": float(
                probabilities[1]
            ),
        }

        risk = float(
            probabilities[1]
        )

        risk_1plus2 = risk

    else:
        raise RuntimeError(
            f"Unsupported model mode: {mode}"
        )

    return {
        "risk": float(risk),
        "probabilities": probability_map,
        "risk_class2": risk_class2,
        "risk_1plus2": risk_1plus2,
    }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    """
    Lightweight route. Does not load ML libraries.
    """

    return {
        "service": "Prompt Interaction-Risk API",
        "status": "running",
        "model_loaded": (
            BUNDLE is not None
            and ENCODER is not None
        ),
        "health_endpoint": "/health",
        "score_endpoint": "/score",
        "warmup_endpoint": "/warmup",
        "docs_endpoint": "/docs",
    }


@app.get("/health")
def health():
    """
    VERY lightweight health check.

    Do not call load_runtime() here.

    Render should be able to detect the HTTP port without
    loading PyTorch/MiniLM/XGBoost.
    """

    return {
        "status": "ok",
        "model_loaded": (
            BUNDLE is not None
            and ENCODER is not None
        ),
        "model_file_exists": MODEL_PATH.exists(),
    }


@app.post(
    "/warmup",
    dependencies=[
        Depends(
            require_api_key
        )
    ],
)
def warmup():
    """
    Explicitly load the model after the HTTP server has started.

    Call this once before launching the experiment.
    """

    started = time.perf_counter()

    try:
        load_runtime()

    except Exception as exc:
        print(
            "[WARMUP ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Model warmup failed: "
                f"{type(exc).__name__}"
            ),
        ) from exc

    elapsed = (
        time.perf_counter()
        - started
    )

    assert BUNDLE is not None

    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": BUNDLE[
            "model_name"
        ],
        "model_version": BUNDLE[
            "model_version"
        ],
        "mode": BUNDLE[
            "mode"
        ],
        "risk_definition": BUNDLE[
            "risk_definition"
        ],
        "load_time_seconds": round(
            elapsed,
            2,
        ),
    }


@app.post(
    "/score",
    response_model=ScoreResponse,
    dependencies=[
        Depends(
            require_api_key
        )
    ],
)
def score(
    req: ScoreRequest,
):
    """
    Score one untouched initial prompt.
    """

    prompt = req.prompt

    if not prompt.strip():
        raise HTTPException(
            status_code=422,
            detail="Prompt cannot be blank.",
        )

    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Prompt exceeds "
                f"MAX_PROMPT_CHARS="
                f"{MAX_PROMPT_CHARS}"
            ),
        )

    started = time.perf_counter()

    try:
        result = score_prompt(
            prompt
        )

    except Exception as exc:

        # Do not log the participant's actual prompt.
        print(
            "[SCORING ERROR] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Scoring failed: "
                f"{type(exc).__name__}"
            ),
        ) from exc

    latency_ms = (
        time.perf_counter()
        - started
    ) * 1000

    assert BUNDLE is not None

    return ScoreResponse(
        risk=float(
            result["risk"]
        ),
        risk_definition=BUNDLE[
            "risk_definition"
        ],
        model_version=BUNDLE[
            "model_version"
        ],
        model_name=BUNDLE[
            "model_name"
        ],
        mode=BUNDLE[
            "mode"
        ],
        latency_ms=round(
            latency_ms,
            2,
        ),
        probabilities=result[
            "probabilities"
        ],
        risk_class2=result[
            "risk_class2"
        ],
        risk_1plus2=result[
            "risk_1plus2"
        ],
        participant_id=req.participant_id,
    )