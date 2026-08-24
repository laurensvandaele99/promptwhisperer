from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from features import PROMPT_COL, build_handcrafted_features

TARGET_COL = "strict_efficiency_score"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42

HANDCRAFTED_NUMERIC = [
    "prompt_word_count","prompt_char_count","prompt_sentence_count",
    "prompt_avg_word_length","prompt_readability_score","prompt_question_rate",
    "prompt_exclamation_rate","prompt_caps_ratio","prompt_has_url","prompt_has_code",
    "prompt_has_numbers","prompt_constraint_count","prompt_polite_flag","prompt_urgent_flag",
    "prompt_hedge_rate","prompt_vague_pronoun_rate","prompt_negation_rate",
    "prompt_subordinate_clause_rate","prompt_first_turn_length","prompt_vader_compound",
    "prompt_vader_positive","prompt_vader_neutral","prompt_char_shannon_entropy",
    "prompt_code_density","prompt_creative_density","prompt_factual_density",
    "prompt_emotional_density","prompt_task_oriented_density","prompt_line_count",
    "prompt_bullet_count","prompt_quoted_string_count","prompt_comma_count",
    "prompt_colon_count","prompt_semicolon_count","prompt_parenthesis_count",
    "prompt_slash_count","prompt_newline_count","prompt_command_verb_count",
    "prompt_output_format_count","prompt_constraint_marker_count",
    "prompt_multi_task_marker_count","prompt_explicit_quantity_count",
    "prompt_question_word_count","prompt_proper_noun_proxy_count",
    "prompt_typo_gibberish_proxy","prompt_short_prompt_flag",
    "prompt_very_long_prompt_flag","prompt_no_clear_task_flag",
    "prompt_quiz_multiple_choice_flag","prompt_translation_flag",
    "prompt_summarization_flag","prompt_roleplay_flag","prompt_coding_task_flag",
    "prompt_writing_task_flag",
]
CATEGORICAL = ["prompt_type_flag"]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = [
        (
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            numeric_cols,
        )
    ]
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", make_ohe()),
                ]),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for conv, prompt in zip(
        frame["conversation_hash"].fillna("").astype(str),
        frame[PROMPT_COL].fillna("").astype(str),
    ):
        h.update(conv.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
        h.update(prompt.encode("utf-8", errors="ignore"))
        h.update(b"\x1e")
    return h.hexdigest()


def find_data(project_dir: Path, filename: str) -> Path:
    matches = sorted(project_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {project_dir}")
    if len(matches) > 1:
        print(f"Found {len(matches)} copies; using {matches[0]}")
    return matches[0]


def load_clean_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    header_like = (
        raw["conversation_hash"].astype(str).str.lower().eq("conversation_hash")
        | raw.get("model", pd.Series("", index=raw.index)).astype(str).str.lower().eq("model")
        | raw.get("timestamp", pd.Series("", index=raw.index)).astype(str).str.lower().eq("timestamp")
    )
    df = raw.loc[~header_like].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].isin([1, 2, 3, 4, 5])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df[PROMPT_COL] = df[PROMPT_COL].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_or_create_embeddings(project_dir: Path, frame: pd.DataFrame, fingerprint: str) -> np.ndarray:
    candidates = sorted(
        p for p in project_dir.rglob(f"minilm_embeddings_{fingerprint[:12]}.npy")
        if p.is_file()
    )
    if candidates:
        embeddings = np.load(candidates[0])
        print(f"Loaded cached embeddings: {candidates[0]} {embeddings.shape}")
    else:
        print("No matching embedding cache found. Computing MiniLM embeddings...")
        encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = encoder.encode(
            frame[PROMPT_COL].tolist(),
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
    if embeddings.shape[0] != len(frame):
        raise ValueError(f"Embedding row mismatch: {embeddings.shape[0]} vs {len(frame)}")
    if embeddings.shape[1] != 384:
        raise ValueError(f"Expected 384 MiniLM dimensions, got {embeddings.shape[1]}")
    return np.asarray(embeddings, dtype=np.float32)


def multiclass_model() -> XGBClassifier:
    # Exact XGBoost specification used in the v4 notebook.
    return XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=550,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def binary_model() -> XGBClassifier:
    # Same tree specification, adapted only to the binary 1-2 versus 3-5 target.
    # Use this only after you have evaluated that target in the analysis notebook.
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=550,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def runtime_versions() -> dict:
    import joblib as _joblib
    import sentence_transformers
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": _joblib.__version__,
        "sentence-transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "nltk": nltk.__version__,
    }


def write_model_requirements(path: Path, versions: dict) -> None:
    mapping = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
        "joblib": "joblib",
        "sentence-transformers": "sentence-transformers",
        "nltk": "nltk",
    }
    lines = ["# Auto-generated by export_model.py. Keep these versions for model serving."]
    for key, pkg in mapping.items():
        value = versions.get(key)
        if value:
            lines.append(f"{pkg}=={value}")
    # torch versions sometimes include a local suffix (+cpu / +cuXXX) that is not
    # accepted by every deployment index, so leave torch to sentence-transformers.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train and export the frozen XGBoost prompt-risk model used by the API."
    )
    parser.add_argument("--project-dir", required=True, help="Root of the chatGPT_study project")
    parser.add_argument("--data-filename", default="strict_efficiency_50k_with_topics.csv")
    parser.add_argument(
        "--mode",
        choices=["multiclass_class2", "multiclass_1plus2", "binary_1plus2"],
        default="multiclass_class2",
        help=(
            "multiclass_class2 reproduces the current v4 deployment score P(score=2). "
            "multiclass_1plus2 uses P(score=1)+P(score=2) from that same five-class model. "
            "binary_1plus2 retrains directly on scores 1-2 vs 3-5 and should only be used "
            "after that target is formally evaluated."
        ),
    )
    parser.add_argument("--output", default="model/model_bundle.joblib")
    parser.add_argument("--version", default=None, help="Optional human-readable model version")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    data_path = find_data(project_dir, args.data_filename)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_path}")
    df = load_clean_data(data_path)
    print(f"Clean N: {len(df):,}")
    print(df[TARGET_COL].value_counts().sort_index())

    fp = dataset_fingerprint(df)
    df_feat = build_handcrafted_features(df)
    embeddings = load_or_create_embeddings(project_dir, df_feat, fp)
    embed_cols = [f"prompt_embed_{j}" for j in range(embeddings.shape[1])]
    for j, col in enumerate(embed_cols):
        df_feat[col] = embeddings[:, j]

    numeric_cols = HANDCRAFTED_NUMERIC + embed_cols
    feature_cols = numeric_cols + CATEGORICAL
    X = df_feat[feature_cols]

    if args.mode.startswith("multiclass"):
        y = df_feat[TARGET_COL].to_numpy() - 1
        pipe = Pipeline([
            ("pre", make_preprocessor(numeric_cols, CATEGORICAL)),
            ("model", multiclass_model()),
        ])
        sw = compute_sample_weight(class_weight="balanced", y=y)
        pipe.fit(X, y, model__sample_weight=sw)
        if args.mode == "multiclass_class2":
            risk_definition = "P(strict_efficiency_score = 2)"
            risk_indices = [1]
        else:
            risk_definition = "P(strict_efficiency_score in {1,2}) from five-class model"
            risk_indices = [0, 1]
        class_labels = [1, 2, 3, 4, 5]
    else:
        y = (df_feat[TARGET_COL].to_numpy() <= 2).astype(int)
        pipe = Pipeline([
            ("pre", make_preprocessor(numeric_cols, CATEGORICAL)),
            ("model", binary_model()),
        ])
        sw = compute_sample_weight(class_weight="balanced", y=y)
        pipe.fit(X, y, model__sample_weight=sw)
        risk_definition = "P(strict_efficiency_score in {1,2}) from directly trained binary model"
        risk_indices = [1]
        class_labels = ["scores_3_5", "scores_1_2"]

    versions = runtime_versions()
    version = args.version or f"xgb-{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    bundle = {
        "pipeline": pipe,
        "model_name": "XGBoost",
        "model_version": version,
        "mode": args.mode,
        "risk_definition": risk_definition,
        "risk_probability_indices": risk_indices,
        "class_labels": class_labels,
        "numeric_cols": numeric_cols,
        "categorical_cols": CATEGORICAL,
        "handcrafted_numeric": HANDCRAFTED_NUMERIC,
        "embedding_cols": embed_cols,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_normalize": False,
        "prompt_col": PROMPT_COL,
        "target_col": TARGET_COL,
        "dataset_fingerprint": fp,
        "training_n": int(len(df_feat)),
        "training_class_counts": {
            str(k): int(v) for k, v in df_feat[TARGET_COL].value_counts().sort_index().items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": versions,
        "source": "prompt_efficiency_final_analysis_v4_ready.ipynb",
    }

    joblib.dump(bundle, output_path, compress=3)
    print(f"Saved model bundle: {output_path}")

    manifest = {k: v for k, v in bundle.items() if k != "pipeline"}
    manifest_path = output_path.with_name("model_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")

    req_path = output_path.with_name("requirements_model.txt")
    write_model_requirements(req_path, versions)
    print(f"Saved model-serving version pins: {req_path}")

    # Minimal sanity check on the exported pipeline.
    test_prompts = [
        "Hi",
        "What laptop should I buy?",
        "Recommend a Windows laptop under 1200 euros for Python, commuting and long battery life.",
    ]
    enc = SentenceTransformer(EMBEDDING_MODEL_NAME)
    test_df = pd.DataFrame({PROMPT_COL: test_prompts})
    test_feat = build_handcrafted_features(test_df)
    test_emb = enc.encode(test_prompts, normalize_embeddings=False, show_progress_bar=False)
    for j, col in enumerate(embed_cols):
        test_feat[col] = test_emb[:, j]
    probs = np.asarray(pipe.predict_proba(test_feat[feature_cols]))
    print("\nSanity-check probabilities:")
    for prompt, p in zip(test_prompts, probs):
        if args.mode.startswith("multiclass"):
            risk = float(np.sum(p[risk_indices]))
            print(prompt, "risk=", round(risk, 4), "p1-p5=", np.round(p, 4).tolist())
        else:
            print(prompt, "risk=", round(float(p[1]), 4), "p=", np.round(p, 4).tolist())


if __name__ == "__main__":
    main()
