"""
Shared helpers for the three-axis idea-scoring framework.

Provides the building blocks used across the insights notebooks:
    - get_model / generate_specter_embeddings : SPECTER text embeddings
    - compute_entropy                         : Surprise axis (GPT-2 cross-entropy)
    - compute_divergence_scores               : Divergence axis (FAISS kNN)
    - compute_stylometric_features            : stylometric features for the Value model
    - get_catboost_proba                      : Value axis (citation-impact probability)
    - train_and_evaluate                      : train + threshold-tune + save a CatBoost model

All file locations are configured in `paths` below; paths are relative to the
repository root, so launch Jupyter from the repo root.
"""

# ===== Imports =====
import json
import pickle
import re

import numpy as np
import pandas as pd
import faiss
import textstat
import joblib
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


# ===== Config =====
paths = {
    'SPECTER': r"data/specter_embeddings.csv",
    'Entropy_Model': r"data/gpt2_medium_scidomain/checkpoint-1636",
    'GraphSage': r"kg/project_graphsage_embeddings.csv",
    'target': r"data/targets.csv",
    "Dataset": r"data/combined_acl_arxiv_with_citations_final.csv",
    "split": r"train_val_test_split.json",
    "Internal_Entropy": r"data/entropy_scores_2.csv",
    "Combined_Scores": r"data/combined_scores.csv",
}

DEVICE = torch.device("cpu")
MAX_LEN = 256
_MODEL_CACHE = {}  # name -> (tokenizer, model), populated lazily by get_model


# ===== Model loading =====
def get_model(model_name: str):
    """
    Lazy-load and cache models + tokenizers.

    Supported:
    - "specter"
    - "gpt2-medium"
    """

    if model_name in _MODEL_CACHE:
        print(f"Using cached model for {model_name}")
        return _MODEL_CACHE[model_name]

    # -----------------------
    # SPECTER
    # -----------------------
    if model_name == "specter":
        hf_id = "allenai/specter"

        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(DEVICE)
        model.eval()

    # -----------------------
    # GPT-2
    # -----------------------
    elif model_name == "gpt2-medium":
        tokenizer = GPT2TokenizerFast.from_pretrained(paths["Entropy_Model"])
        model = GPT2LMHeadModel.from_pretrained(paths["Entropy_Model"]).to(DEVICE)
        model.eval()

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    print(f"Loaded model: {model_name}")
    # Cache it
    _MODEL_CACHE[model_name] = (tokenizer, model)

    return tokenizer, model


# ===== Embeddings =====
def generate_specter_embeddings(
    df: pd.DataFrame,
    id_col: str,
    text_col: str = None,
    batch_size: int = 64,
    max_length: int = 512,
) -> pd.DataFrame:
    """
    Generate SPECTER embeddings.

    Parameters
    ----------
    df : pd.DataFrame
    id_col : str
        Column name for unique ID
    text_col : str, optional
        Single text column (if already combined)
    batch_size : int
    max_length : int

    Returns
    -------
    pd.DataFrame
        DataFrame with id_col + embedding columns
    """

    if text_col is None:
        raise ValueError(
            "Provide text_col"
        )

    tokenizer, model = get_model("specter")

    # Prepare inputs
    texts = df[text_col].astype(str).tolist()

    embeddings_list = []
    n = len(df)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(DEVICE)

            outputs = model(**inputs)

            # CLS token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings_list.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings_list)

    # Create output DataFrame
    embedding_cols = [
        f"specter_embedding_{i}" for i in range(embeddings.shape[1])
    ]

    df_out = pd.concat(
        [
            df[[id_col]].reset_index(drop=True),
            pd.DataFrame(embeddings, columns=embedding_cols)
        ],
        axis=1
    )

    return df_out


# ===== Surprise axis (GPT-2 cross-entropy) =====
def compute_entropy(text):
    tokenizer, model = get_model("gpt2-medium")
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN
        )

        input_ids = inputs["input_ids"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            labels=input_ids
        )

        loss = outputs.loss  # mean cross-entropy
        return loss.item()


# ===== Divergence axis (FAISS kNN over SPECTER) =====
def compute_divergence_scores(
    query_df,
    k=20,
    dataset_path=paths["SPECTER"],
    internal_dataset=False
):
    """
    Compute divergence scores using FAISS kNN over SPECTER embeddings.

    Parameters
    ----------
    query_df : pd.DataFrame
        DataFrame containing embeddings (columns: specter_embedding_0 ... specter_embedding_755)
        Can be internal dataset OR external dataset (e.g., LiveIdeaBench)

    k : int
        Number of nearest neighbors

    dataset_path : str
        Path to full corpus embeddings CSV

    internal_dataset : bool
        If True → query_df is part of corpus (self-match exists)
        If False → query_df is external (no self-match)

    Returns
    -------
    divergence_scores : np.ndarray
        Array of divergence scores (length = len(query_df))
    """

    # -------------------------------
    # Load full corpus embeddings
    # -------------------------------
    full_df = pd.read_csv(dataset_path)

    emb_cols = [col for col in full_df.columns if col.startswith("specter_embedding_")]
    X_corpus = full_df[emb_cols].values.astype('float32')

    # -------------------------------
    # Normalize corpus embeddings
    # -------------------------------
    norms = np.linalg.norm(X_corpus, axis=1, keepdims=True)
    X_corpus = X_corpus / np.clip(norms, 1e-12, None)

    d = X_corpus.shape[1]

    # -------------------------------
    # Build FAISS index
    # -------------------------------
    index = faiss.IndexFlatIP(d)  # Inner product on normalized vectors
    index.add(X_corpus)

    print(f"FAISS index built with {index.ntotal} vectors of dimension {d}")

    # -------------------------------
    # Prepare query embeddings
    # -------------------------------
    X_query = query_df[emb_cols].values.astype('float32')

    norms = np.linalg.norm(X_query, axis=1, keepdims=True)
    X_query = X_query / np.clip(norms, 1e-12, None)

    # -------------------------------
    # kNN search
    # -------------------------------
    if internal_dataset:
        search_k = k + 1  # account for self-match
    else:
        search_k = k

    similarities, indices = index.search(X_query, search_k)

    # -------------------------------
    # Remove self-match if needed
    # -------------------------------
    if internal_dataset:
        similarities = similarities[:, 1:]
        indices = indices[:, 1:]

    # -------------------------------
    # Convert similarity → distance
    # (since using inner product on normalized vectors)
    # distance = 1 - similarity
    # -------------------------------
    distances = 1.0 - similarities

    # -------------------------------
    # Compute divergence (mean distance)
    # -------------------------------
    divergence_scores = np.mean(distances, axis=1)

    return divergence_scores


# ===== Stylometric features =====
def compute_stylometric_features(text):

    if pd.isna(text):
        text = ""

    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?\n]+', text)
    word_count = len(words)
    sentence_count = len([s for s in sentences if len(s.strip()) > 0])

    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0

    complex_words = [w for w in words if len(w) >= 7]
    complex_word_ratio = len(complex_words) / word_count if word_count > 0 else 0

    # Readability metrics
    # flesch_reading_ease = textstat.flesch_reading_ease(text) if word_count > 0 else 0
    # flesch_kincaid_grade = textstat.flesch_kincaid_grade(text) if word_count > 0 else 0

    return pd.Series({
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "complex_word_ratio": complex_word_ratio,
        # "flesch_reading_ease": flesch_reading_ease,
        # "flesch_kincaid_grade": flesch_kincaid_grade
    })


# ===== Value axis (CatBoost citation-impact probability) =====
def get_catboost_proba(df: pd.DataFrame, model_path=r"val_pred/results/specter_final_final_model.pkl"):
    """
    Load CatBoost model and return predicted probabilities for input DataFrame.

    Args:
        df (pd.DataFrame): Input features (same format as training)
        model_path (str): Path to saved CatBoost model (.pkl)

    Returns:
        np.ndarray: Probability of positive class (shape: [n_samples])
    """

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict probabilities
    proba = model.predict_proba(df[model.feature_names_])
    # Return probability of positive class (class 1)
    return proba[:, 1]


# ===== Train / evaluate a Value model =====
def train_and_evaluate(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_name,
    save_dir="results"
):

    print(f"\n===== Training {model_name} =====")

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        early_stopping_rounds=100,
        verbose=200
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    # --- Probabilities ---
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # --- PR-AUC ---
    val_pr_auc = average_precision_score(y_val, val_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)

    # --- Threshold tuning (VALIDATION ONLY) ---
    thresholds = np.linspace(0.05, 0.95, 200)

    best_thr = 0.5
    best_f1 = 0

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    print(f"\nBest threshold (val): {best_thr}")
    print(f"Best validation F1: {best_f1}")

    # --- Final metrics using best threshold ---
    val_preds = (val_probs >= best_thr).astype(int)
    test_preds = (test_probs >= best_thr).astype(int)

    val_metrics = {
        "PR_AUC": float(val_pr_auc),
        "F1": float(f1_score(y_val, val_preds)),
        "Recall": float(recall_score(y_val, val_preds)),
        "Precision": float(precision_score(y_val, val_preds)),
    }

    test_metrics = {
        "PR_AUC": float(test_pr_auc),
        "F1": float(f1_score(y_test, test_preds)),
        "Recall": float(recall_score(y_test, test_preds)),
        "Precision": float(precision_score(y_test, test_preds)),
    }

    results = {
        "model_name": model_name,
        "best_iteration": model.get_best_iteration(),
        "best_threshold": float(best_thr),
        "validation": val_metrics,
        "test": test_metrics
    }

    # --- Save model ---
    joblib.dump(model, f"{save_dir}/{model_name}_model.pkl")

    # --- Save results ---
    with open(f"{save_dir}/{model_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n--- Validation ---")
    print(val_metrics)
    print("\n--- Test ---")
    print(test_metrics)

    return model, results
