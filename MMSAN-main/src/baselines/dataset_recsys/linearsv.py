"""
Dataset Recommendation via Text Classification (TF-IDF + Linear SVM)
======================================================================
Reproduces the best-performing approach from:
Färber & Leisinger, "Recommending Datasets for Scientific Problem
Descriptions", CIKM 2021.

Given a publication's text (title + abstract), the model ranks all
known datasets by relevance, using a One-vs-Rest Linear SVM trained
on TF-IDF features. The SVM's decision_function output (signed
distance to the hyperplane) is used as a continuous relevance score,
which allows producing a RANKING instead of just a binary label.

Expected input files
---------------------
publications.csv : columns [id, title, abstract]
dataset.csv      : columns [id, title, abstract]
train.csv        : columns [source, target]  (source=publication id, target=dataset id)
val.csv          : columns [source, target]  (optional, for evaluation)
test.csv         : columns [source, target]  (optional, for evaluation)

A publication can appear multiple times in train/val/test (multilabel):
one row per (publication, dataset) pair it references.

Usage
-----
python recommend_datasets.py \
    --publications publications.csv \
    --datasets dataset.csv \
    --train train.csv \
    --val val.csv \
    --test test.csv \
    --out_dir outputs \
    --top_k 10
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_text_table(path: str, split: int = 0) -> pd.DataFrame:
    """Load a CSV with columns [id, title, abstract] and build a `text` field."""
    df = pd.read_csv(path, dtype={"id": str})
    for col in ("title", "description"):
        if col not in df.columns:
            df[col] = ""
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    if split > 0:
        n_hide = round(len(df) * split / 100)
        rows_to_hide = df.sample(n=n_hide, random_state=None).index

        df.loc[rows_to_hide, "title"] = ""
        df.loc[rows_to_hide, "description"] = ""

    df["text"] = (df["title"] + ". " + df["description"]).str.strip()
    df = df.drop_duplicates(subset="id").set_index("id", drop=False)
    return df


def load_pairs(path: str) -> pd.DataFrame:
    """Load a CSV with columns [source, target] (publication_id, dataset_id)."""
    df = pd.read_csv(path, dtype={"source": str, "target": str})
    assert {"source", "target"}.issubset(df.columns), (
        f"{path} must have columns 'source' and 'target'"
    )
    return df


def group_labels(pairs: pd.DataFrame) -> pd.Series:
    """Collapse (source, target) rows into source -> [target1, target2, ...]."""
    return pairs.groupby("source")["target"].apply(list)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class DatasetRecommender:
    """TF-IDF + One-vs-Rest Linear SVM multilabel dataset recommender."""

    def __init__(self, max_features: int = 50000, ngram_range=(1, 2), C: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
        )
        self.mlb = MultiLabelBinarizer()
        self.base_svm = LinearSVC(C=C, class_weight="balanced")
        self.clf = OneVsRestClassifier(self.base_svm, n_jobs=1)
        self.classes_ = None

    def fit(self, texts: list[str], labels: list[list[str]]):
        X = self.vectorizer.fit_transform(texts)
        Y = self.mlb.fit_transform(labels)
        self.classes_ = self.mlb.classes_
        self.clf.fit(X, Y)
        return self

    def score_matrix(self, texts: list[str]) -> np.ndarray:
        """Return a (n_texts, n_datasets) matrix of relevance scores."""
        X = self.vectorizer.transform(texts)
        scores = self.clf.decision_function(X)
        if issparse(scores):
            scores = scores.toarray()
        # decision_function on a single-class edge case returns 1D; normalize shape
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        return scores

    def rank(self, texts: list[str], ids: list[str], top_k: int = 10) -> pd.DataFrame:
        """Produce a long-format ranking table: publication_id, rank, dataset_id, score."""
        scores = self.score_matrix(texts)
        order = np.argsort(-scores, axis=1)[:, :top_k]
        rows = []
        for i, pub_id in enumerate(ids):
            for rank, class_idx in enumerate(order[i], start=1):
                rows.append(
                    {
                        "publication_id": pub_id,
                        "rank": rank,
                        "dataset_id": self.classes_[class_idx],
                        "score": float(scores[i, class_idx]),
                    }
                )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_classification(y_true_bin, y_pred_bin) -> dict:
    """Precision/Recall/F1, macro and weighted -- mirrors Table 2 of the paper."""
    from sklearn.metrics import precision_recall_fscore_support

    macro = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
    return {
        "macro_precision": macro[0], "macro_recall": macro[1], "macro_f1": macro[2],
        "weighted_precision": weighted[0], "weighted_recall": weighted[1], "weighted_f1": weighted[2],
    }


def ndcg_at_k(ranked_ids: list, truth: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ k, with binary relevance
    (1 if the dataset at that rank is a true citation, 0 otherwise).
    """
    dcg = 0.0
    for i, ds_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if ds_id in truth else 0.0
        if rel:
            dcg += rel / np.log2(i + 1)

    # Ideal DCG: all relevant items (up to k) placed at the top of the ranking
    n_ideal = min(len(truth), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_ideal + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(ranking_df: pd.DataFrame, true_labels: pd.Series, k_values=(1, 5, 10)) -> dict:
    """
    Ranking metrics: Precision@k, Recall@k, MRR, MAP.
    `ranking_df` must be sorted by rank within each publication_id (it is, by construction).
    `true_labels`: Series indexed by publication_id -> list of true dataset ids.
    """
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    reciprocal_ranks = []
    average_precisions = []

    grouped = ranking_df.groupby("publication_id")
    for pub_id, group in grouped:
        if pub_id not in true_labels.index:
            continue
        truth = set(true_labels.loc[pub_id])
        if not truth:
            continue
        ranked_ids = group.sort_values("rank")["dataset_id"].tolist()

        for k in k_values:
            top_k = ranked_ids[:k]
            hits = len(set(top_k) & truth)
            metrics[f"precision@{k}"].append(hits / k)
            metrics[f"recall@{k}"].append(hits / len(truth))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_ids, truth, k))
        # MRR: reciprocal rank of first correct hit
        rr = 0.0
        for rank, ds_id in enumerate(ranked_ids[:k], start=1):
            if ds_id in truth:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # MAP: average precision over the full ranked list
        hits, precisions = 0, []
        for rank, ds_id in enumerate(ranked_ids, start=1):
            if ds_id in truth:
                hits += 1
                precisions.append(hits / rank)
        ap = float(np.mean(precisions)) if precisions else 0.0
        average_precisions.append(ap)

    result = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in metrics.items()}
    result["MRR"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    result["MAP"] = float(np.mean(average_precisions)) if average_precisions else 0.0
    return result


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publications",default="datasets/pubmed/publications.csv", help="Path to publications.csv (id,title,abstract)")
    parser.add_argument("--datasets", default="datasets/pubmed/datasets.csv", help="Path to dataset.csv (id,title,abstract)")
    parser.add_argument("--train",default="datasets/pubmed/pubdataedges_train_kcore_2.csv", help="Path to train.csv (source,target)")
    parser.add_argument("--val",default="datasets/pubmed/pubdataedges_validation_trans_kcore_2.csv", help="Path to val.csv (source,target), optional")
    parser.add_argument("--test",default="datasets/pubmed/pubdataedges_test_trans_kcore_2.csv", help="Path to test.csv (source,target), optional")
    parser.add_argument("--out_dir", default="baselines/outputs", help="Directory to write ranking + metrics")
    parser.add_argument("--top_k", type=int, default=10, help="How many datasets to output per publication")
    parser.add_argument("--max_features", type=int, default=50000, help="TF-IDF vocabulary size cap")
    parser.add_argument("--C", type=float, default=1.0, help="Linear SVM regularization strength")
    parser.add_argument("--split", type=float, default=0.0, help="split %")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    pubs = load_text_table(args.publications)
    datasets = load_text_table(args.datasets,args.split)  # noqa: F841 (kept for future metadata use)
    train_pairs = load_pairs(args.train)

    print(f"  {len(pubs)} publications, {len(datasets)} datasets, {len(train_pairs)} train (pub, dataset) pairs")

    # Build training text + multilabel targets
    train_labels = group_labels(train_pairs)  # pub_id -> [dataset_id, ...]
    train_pub_ids = [pid for pid in train_labels.index if pid in pubs.index]
    missing = len(train_labels) - len(train_pub_ids)
    if missing:
        print(f"  Warning: {missing} training publications not found in publications.csv, skipped.")

    train_texts = [pubs.loc[pid, "text"] for pid in train_pub_ids]
    train_targets = [train_labels.loc[pid] for pid in train_pub_ids]

    print("Training TF-IDF + One-vs-Rest Linear SVM...")
    model = DatasetRecommender(max_features=args.max_features, C=args.C)
    model.fit(train_texts, train_targets)
    print(f"  Vocabulary size: {len(model.vectorizer.vocabulary_)}")
    print(f"  Number of dataset classes: {len(model.classes_)}")

    # Rank on train (sanity check) plus any provided eval splits
    splits = {"train": (train_pub_ids, train_labels)}
    if args.val:
        val_pairs = load_pairs(args.val)
        val_labels = group_labels(val_pairs)
        val_pub_ids = [pid for pid in val_labels.index if pid in pubs.index]
        splits["val"] = (val_pub_ids, val_labels)
    if args.test:
        test_pairs = load_pairs(args.test)
        test_labels = group_labels(test_pairs)
        test_pub_ids = [pid for pid in test_labels.index if pid in pubs.index]
        splits["test"] = (test_pub_ids, test_labels)

    all_metrics = {}
    for split_name, (pub_ids, labels) in splits.items():
        if not pub_ids:
            continue
        texts = [pubs.loc[pid, "text"] for pid in pub_ids]
        print(f"Ranking datasets for '{split_name}' split ({len(pub_ids)} publications)...")
        ranking_df = model.rank(texts, pub_ids, top_k=args.top_k)



        metrics = evaluate_ranking(ranking_df, labels, k_values=(1, 5, 10))
        all_metrics[split_name] = metrics
        print(f"  {split_name} metrics: {json.dumps(metrics, indent=2)}")


    print(f"\nAll metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()