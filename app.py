import os
import time
import threading
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

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

MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")

# Optional:
# ALLOWED_ORIGINS=https://your-lovable-app.lovable.app,https://yourdomain.com
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")

if allowed_origins_raw.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [
        x.strip()
        for x in allowed_origins_raw.split(",")
        if x.strip()
    ]


# ============================================================
# Global runtime objects
# ============================================================

BUNDLE: Optional[dict[str, Any]] = None
ENCODER: Optional[SentenceTransformer] = None

LOAD_LOCK = threading.Lock()


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Prompt Interaction-Risk API",
    version="1.0.0",
    description=(
        "Stateless scoring API for the frozen prompt-efficiency model. "
        "The API predicts interaction-inefficiency risk from the initial prompt."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# Request / response schemas
# ============================================================

class ScoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    participant_id: Optional[str] = None


# ============================================================
# Authentication
# ============================================================

def verify_api_key(x_api_key: Optional[str]) -> None:
    """
    Require X-API-Key when MODEL_API_KEY is configured.
    """

    if not MODEL_API_KEY:
        return

    if not x_api_key or x_api_key != MODEL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


# ============================================================
# Model loading
# ============================================================

def load_runtime() -> None:
    """
    Lazily load the frozen model bundle and MiniLM encoder.

    Nothing is loaded during FastAPI startup. This is intentional:
    Render can bind to its HTTP port immediately instead of waiting
    for sentence-transformers / PyTorch initialization.
    """

    global BUNDLE, ENCODER

    if BUNDLE is not None and ENCODER is not None:
        return

    with LOAD_LOCK:

        # Another request may have finished loading while this one waited.
        if BUNDLE is not None and ENCODER is not None:
            return

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model bundle not found at: {MODEL_PATH}"
            )

        print(f"Loading model bundle from: {MODEL_PATH}", flush=True)

        bundle = joblib.load(MODEL_PATH)

        if not isinstance(bundle, dict):
            raise RuntimeError(
                "model_bundle.joblib does not contain the expected dictionary."
            )

        if "model" not in bundle:
            raise RuntimeError(
                "Model bundle is missing key 'model'. "
                f"Available keys: {list(bundle.keys())}"
            )

        embedding_model_name = bundle.get(
            "embedding_model_name",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        print(
            f"Loading sentence encoder: {embedding_model_name}",
            flush=True,
        )

        encoder = SentenceTransformer(embedding_model_name)

        # Assign only after everything loaded successfully.
        BUNDLE = bundle
        ENCODER = encoder

        print(
            "Model runtime loaded successfully.",
            flush=True,
        )


# ============================================================
# Feature construction
# ============================================================

def make_model_input(prompt: str) -> pd.DataFrame:
    """
    Reproduce the inference features used by the frozen model:
      1. handcrafted initial-prompt features
      2. MiniLM sentence embedding
      3. exact feature ordering stored in the exported bundle
    """

    if BUNDLE is None or ENCODER is None:
        raise RuntimeError("Runtime has not been loaded.")

    # --------------------------------------------------------
    # Handcrafted features
    # --------------------------------------------------------

    raw_df = pd.DataFrame(
        {
            PROMPT_COL: [prompt]
        }
    )

    handcrafted = build_handcrafted_features(raw_df)

    if not isinstance(handcrafted, pd.DataFrame):
        raise RuntimeError(
            "build_handcrafted_features() did not return a DataFrame."
        )

    handcrafted = handcrafted.reset_index(drop=True)

    # --------------------------------------------------------
    # MiniLM embedding
    # --------------------------------------------------------

    embedding = ENCODER.encode(
        [prompt],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    # --------------------------------------------------------
    # Determine the exact model columns
    # --------------------------------------------------------

    feature_columns = BUNDLE.get("feature_columns")

    if feature_columns is None:
        feature_columns = BUNDLE.get("model_feature_columns")

    if feature_columns is None:
        model = BUNDLE["model"]

        if hasattr(model, "feature_names_in_"):
            feature_columns = list(model.feature_names_in_)
        else:
            raise RuntimeError(
                "Could not determine model feature ordering. "
                "The bundle must contain 'feature_columns' or "
                "'model_feature_columns'."
            )

    feature_columns = list(feature_columns)

    # Explicit exported column lists are preferred.
    handcrafted_columns = BUNDLE.get("handcrafted_columns")
    embedding_columns = BUNDLE.get("embedding_columns")

    if handcrafted_columns is not None:
        handcrafted_columns = list(handcrafted_columns)

    if embedding_columns is not None:
        embedding_columns = list(embedding_columns)

    # --------------------------------------------------------
    # Infer handcrafted columns if necessary
    # --------------------------------------------------------

    if handcrafted_columns is None:
        handcrafted_columns = [
            col
            for col in feature_columns
            if col in handcrafted.columns
        ]

    # --------------------------------------------------------
    # Infer embedding columns if necessary
    # --------------------------------------------------------

    if embedding_columns is None:
        embedding_columns = [
            col
            for col in feature_columns
            if col not in handcrafted_columns
        ]

    if len(embedding_columns) != embedding.shape[1]:
        raise RuntimeError(
            "Embedding dimensionality does not match the exported "
            "feature structure. "
            f"Encoder returned {embedding.shape[1]} dimensions, "
            f"but {len(embedding_columns)} embedding columns were expected."
        )

    # --------------------------------------------------------
    # Construct exact single-row model matrix
    # --------------------------------------------------------

    X = pd.DataFrame(
        np.zeros((1, len(feature_columns)), dtype=np.float32),
        columns=feature_columns,
    )

    # Handcrafted values
    for col in handcrafted_columns:
        if col not in handcrafted.columns:
            raise RuntimeError(
                f"Expected handcrafted feature '{col}' was not generated."
            )

        value = handcrafted.loc[0, col]

        try:
            X.loc[0, col] = float(value)
        except (TypeError, ValueError):
            X.loc[0, col] = 0.0

    # Embedding values
    X.loc[0, embedding_columns] = embedding[0]

    # Replace any problematic numeric values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Guarantee original training order
    X = X[feature_columns]

    return X


# ============================================================
# Scoring
# ============================================================

def _score(prompt: str) -> dict[str, Any]:

    start_time = time.perf_counter()

    # Lazy loading happens here, NOT during web-server startup.
    load_runtime()

    assert BUNDLE is not None

    model = BUNDLE["model"]

    X = make_model_input(prompt)

    probs = model.predict_proba(X)[0]

    classes = list(model.classes_)

    probability_map: dict[str, float] = {}

    for cls, prob in zip(classes, probs):
        try:
            cls_string = str(int(cls))
        except (TypeError, ValueError):
            cls_string = str(cls)

        probability_map[cls_string] = float(prob)

    mode = BUNDLE.get(
        "mode",
        "multiclass_class2",
    )

    model_version = BUNDLE.get(
        "model_version",
        "unknown",
    )

    model_name = BUNDLE.get(
        "model_name",
        model.__class__.__name__,
    )

    # --------------------------------------------------------
    # Different supported outcome definitions
    # --------------------------------------------------------

    risk_class2 = None
    risk_1plus2 = None

    if "2" in probability_map:
        risk_class2 = probability_map["2"]

    if "1" in probability_map and "2" in probability_map:
        risk_1plus2 = (
            probability_map["1"]
            + probability_map["2"]
        )

    if mode == "multiclass_class2":

        if risk_class2 is None:
            raise RuntimeError(
                "Class 2 probability is unavailable."
            )

        risk = risk_class2

        risk_definition = (
            "P(strict_efficiency_score = 2)"
        )

    elif mode == "multiclass_1plus2":

        if risk_1plus2 is None:
            raise RuntimeError(
                "Class 1+2 probability is unavailable."
            )

        risk = risk_1plus2

        risk_definition = (
            "P(strict_efficiency_score in {1,2})"
        )

    elif mode == "binary_1plus2":

        # For a binary model, positive class should be coded 1.
        if "1" not in probability_map:
            raise RuntimeError(
                "Positive class probability unavailable for "
                "binary_1plus2 mode."
            )

        risk = probability_map["1"]

        risk_1plus2 = risk

        risk_definition = (
            "P(strict_efficiency_score in {1,2})"
        )

    else:
        raise RuntimeError(
            f"Unsupported model mode: {mode}"
        )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    return {
        "risk": float(risk),
        "risk_definition": risk_definition,
        "model_version": model_version,
        "model_name": model_name,
        "mode": mode,
        "latency_ms": round(latency_ms, 2),
        "probabilities": probability_map,
        "risk_class2": (
            float(risk_class2)
            if risk_class2 is not None
            else None
        ),
        "risk_1plus2": (
            float(risk_1plus2)
            if risk_1plus2 is not None
            else None
        ),
    }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Prompt Interaction-Risk API",
        "status": "running",
        "model_loaded": (
            BUNDLE is not None
            and ENCODER is not None
        ),
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    """
    Lightweight health check.

    IMPORTANT:
    This endpoint deliberately does NOT load MiniLM or XGBoost.
    Render can therefore detect the open web port immediately.
    """

    return {
        "status": "ok",
        "model_loaded": (
            BUNDLE is not None
            and ENCODER is not None
        ),
        "model_file_exists": MODEL_PATH.exists(),
    }


@app.post("/warmup")
def warmup(
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key",
    )
):
    """
    Manually load the model after deployment.

    Useful for warming up Render before participants enter
    the experiment.
    """

    verify_api_key(x_api_key)

    start = time.perf_counter()

    load_runtime()

    elapsed = time.perf_counter() - start

    return {
        "status": "ready",
        "model_loaded": True,
        "load_time_seconds": round(elapsed, 2),
        "model_version": (
            BUNDLE.get("model_version", "unknown")
            if BUNDLE
            else "unknown"
        ),
    }


@app.post("/score")
def score(
    request: ScoreRequest,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key",
    ),
):
    """
    Score one untouched initial participant prompt.
    """

    verify_api_key(x_api_key)

    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=422,
            detail="Prompt cannot be empty.",
        )

    try:
        result = _score(prompt)

        # participant_id is deliberately returned only for
        # request matching; it is not used by the model.
        if request.participant_id is not None:
            result["participant_id"] = request.participant_id

        return result

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Scoring error: {type(exc).__name__}: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {str(exc)}",
        )