"""
Bi-Encoder Dataset Recommender (DataFinder-style neural retrieval)
====================================================================
Implements the best-performing method from:
Viswanathan, Gao, Wu, Liu & Neubig, "DataFinder: Scientific Dataset
Recommendation from Natural Language Descriptions", ACL 2023.

A bi-encoder retriever: publications (queries) and datasets (documents)
are encoded independently into a shared embedding space with a
fine-tuned transformer (default: SciBERT). Relevance is the inner
product between the query's [CLS] embedding and each dataset's [CLS]
embedding:

    sim(q, d) = cls(Encoder(q))^T cls(Encoder(d))

The encoder is fine-tuned with a contrastive loss over (query, positive
dataset) pairs, using in-batch negatives plus BM25-mined hard negatives
-- mirroring the paper's Tevatron-based training recipe (Karpukhin et
al., 2020 DPR-style training).

Why this instead of the TF-IDF + Linear SVM classifier?
----------------------------------------------------------
The classifier (Färber & Leisinger, 2021 -- see recommend_datasets.py)
treats each dataset as an opaque class label: it can only recommend
datasets seen during training and never looks at dataset text at all.
The bi-encoder does genuine text-to-text semantic matching, so it:
  - generalizes to datasets added to dataset.csv AFTER training, with
    zero retraining (just re-encode the new row);
  - actually uses the dataset description/title as a model input;
  - is the strongest method in the DataFinder benchmarks (full-sentence
    queries: P@5 16.0 / MRR 42.6, vs. 5.5 / 15.5 for TF-IDF kNN).

Expected input files (same schema as recommend_datasets.py)
----------------------------------------------------------------
publications.csv : columns [id, title, description]
dataset.csv      : columns [id, title, description]
train.csv        : columns [source, target]  (source=pub id, target=dataset id)
val.csv          : columns [source, target]  (optional)
test.csv         : columns [source, target]  (optional)

Usage
-----
pip install sentence-transformers rank-bm25 torch

python bi_encoder_recommender.py \
    --publications publications.csv \
    --datasets dataset.csv \
    --train train.csv --val val.csv --test test.csv \
    --out_dir outputs \
    --base_model allenai/scibert_scivocab_uncased \
    --epochs 3 --batch_size 32 --top_k 10 --hard_negatives 3

Note: downloading the base model requires internet access to
huggingface.co the first time it's run. If SciBERT isn't available in
your environment, a good lightweight fallback is
--base_model sentence-transformers/all-MiniLM-L6-v2 (not scientific-
domain pretrained, but requires no special access and trains fast).
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer




# --------------------------------------------------------------------------- #
# Data loading (same schema/logic as recommend_datasets.py)
# --------------------------------------------------------------------------- #

def load_text_table(path: str, split: int = 0) -> pd.DataFrame:
    """Load a CSV with columns [id, title, description] and build a `text` field."""
    df = pd.read_csv(path, dtype={"id": str})
    for col in ("title", "description"):
        if col not in df.columns:
            df[col] = ""
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    if split > 0:
        print('split!!')
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
# Hard-negative mining with BM25
# --------------------------------------------------------------------------- #

def build_bm25_index(dataset_texts: list[str]):
    """BM25 index over dataset texts, used to mine hard negatives for training."""
    from rank_bm25 import BM25Okapi

    tokenized = [t.lower().split() for t in dataset_texts]
    return BM25Okapi(tokenized)


def mine_hard_negatives(
    bm25, dataset_ids: list[str], query_text: str, exclude_ids: set, k: int
) -> list[str]:
    """
    Return up to k dataset IDs that are lexically similar to the query
    (high BM25 score) but are NOT true positives for it. These are the
    "confusable" datasets a weak retriever would mistakenly rank highly --
    training against them sharpens the embedding space the most.
    """
    scores = bm25.get_scores(query_text.lower().split())
    order = np.argsort(-scores)
    negs = []
    for idx in order:
        did = dataset_ids[idx]
        if did in exclude_ids:
            continue
        negs.append(did)
        if len(negs) >= k:
            break
    return negs


# --------------------------------------------------------------------------- #
# Training example construction
# --------------------------------------------------------------------------- #

def build_training_examples(
    train_pub_ids: list[str],
    train_labels: pd.Series,
    pubs: pd.DataFrame,
    datasets: pd.DataFrame,
    bm25,
    dataset_ids: list[str],
    n_hard_negatives: int,
    seed: int = 42,
):
    """
    Build one InputExample per (publication, positive dataset) pair. Each
    example's `texts` list is [query, positive, hard_neg_1, ..., hard_neg_k].
    MultipleNegativesRankingLoss treats texts[0] as anchor, texts[1] as the
    positive, and everything else (texts[2:], plus every OTHER example's
    positive in the same batch) as negatives.
    """
    from sentence_transformers import InputExample

    rng = random.Random(seed)
    examples = []
    skipped_missing_dataset = 0

    for pid in train_pub_ids:
        pos_ids = train_labels.loc[pid]
        query_text = pubs.loc[pid, "text"]
        if not query_text:
            continue
        pos_id_set = set(pos_ids)

        for pos_id in pos_ids:
            if pos_id not in datasets.index:
                skipped_missing_dataset += 1
                continue
            texts = [query_text, datasets.loc[pos_id, "text"]]

            if n_hard_negatives > 0 and bm25 is not None:
                hard_negs = mine_hard_negatives(
                    bm25, dataset_ids, query_text, pos_id_set, n_hard_negatives
                )
                texts += [datasets.loc[nid, "text"] for nid in hard_negs if nid in datasets.index]

            examples.append(InputExample(texts=texts))

    rng.shuffle(examples)
    if skipped_missing_dataset:
        print(f"  Warning: skipped {skipped_missing_dataset} training pairs whose dataset id "
              f"was not found in dataset.csv")
    return examples


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #

def build_model(base_model_name: str, max_seq_length: int = 256):
    """
    A single shared transformer encodes both queries and datasets (a
    "Siamese" bi-encoder). Pooling uses the [CLS] token, matching the
    paper's sim(q,d) = cls(BERT(q))^T cls(BERT(d)) formulation exactly
    (as opposed to sentence-transformers' usual mean-pooling default).
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models

    word_embedding_model = models.Transformer(base_model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def rank_datasets(
    model,
    pub_ids: list[str],
    pub_texts: list[str],
    dataset_ids: list[str],
    dataset_texts: list[str],
    top_k: int,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Encode all datasets once, encode queries, rank by inner product."""
    dataset_emb = model.encode(
        dataset_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    query_emb = model.encode(
        pub_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    scores = query_emb @ dataset_emb.T  # inner product, matches the paper's sim(q, d)

    k = min(top_k, len(dataset_ids))
    order = np.argsort(-scores, axis=1)[:, :k]

    rows = []
    for i, pub_id in enumerate(pub_ids):
        for rank, idx in enumerate(order[i], start=1):
            rows.append(
                {
                    "publication_id": pub_id,
                    "rank": rank,
                    "dataset_id": dataset_ids[idx],
                    "score": float(scores[i, idx]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Evaluation (identical metrics to recommend_datasets.py, for direct comparison)
# --------------------------------------------------------------------------- #

def ndcg_at_k(ranked_ids: list, truth: set, k: int) -> float:
    dcg = 0.0
    for i, ds_id in enumerate(ranked_ids[:k], start=1):
        if ds_id in truth:
            dcg += 1.0 / np.log2(i + 1)
    n_ideal = min(len(truth), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(ranking_df: pd.DataFrame, true_labels: pd.Series, k_values=(1, 5, 10)) -> dict:
    """Precision@k, Recall@k, NDCG@k, MRR, MAP."""
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    reciprocal_ranks, average_precisions = [], []

    for pub_id, group in ranking_df.groupby("publication_id"):
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

        rr = 0.0
        for rank, ds_id in enumerate(ranked_ids, start=1):
            if ds_id in truth:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        hits, precisions = 0, []
        for rank, ds_id in enumerate(ranked_ids, start=1):
            if ds_id in truth:
                hits += 1
                precisions.append(hits / rank)
        average_precisions.append(float(np.mean(precisions)) if precisions else 0.0)

    result = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in metrics.items()}
    result["MRR"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    result["MAP"] = float(np.mean(average_precisions)) if average_precisions else 0.0
    return result


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main(dataset: str = 'mes',split: int = 0):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publications",default="datasets/pubmed/publications.csv", help="Path to publications.csv (id,title,abstract)")
    parser.add_argument("--datasets", default="datasets/pubmed/datasets.csv", help="Path to dataset.csv (id,title,abstract)")
    parser.add_argument("--train",default="datasets/pubmed/pubdataedges_train_kcore_2.csv", help="Path to train.csv (source,target)")
    parser.add_argument("--val",default="datasets/pubmed/pubdataedges_validation_trans_kcore_2.csv", help="Path to val.csv (source,target), optional")
    parser.add_argument("--test",default="datasets/pubmed/pubdataedges_test_trans_kcore_2.csv", help="Path to test.csv (source,target), optional")
    parser.add_argument("--out_dir", default="baselines/outputs_bi_encoder")
    parser.add_argument("--base_model", default="allenai/scibert_scivocab_uncased",
                         help="HuggingFace model name. Needs internet access on first run.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hard_negatives", type=int, default=0,
                         help="Number of BM25-mined hard negatives per training example. 0 disables.")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--save_model", default=None, help="Optional path to save the finetuned encoder.")
    args = parser.parse_args()
    args.split = split
    args.datasets = f"datasets/{dataset}/datasets.csv"
    args.publications = f"datasets/{dataset}/publications.csv"
    args.train = f"datasets/{dataset}/pubdataedges_train_kcore_2.csv"
    args.val = f"datasets/{dataset}/pubdataedges_validation_trans_kcore_2.csv"
    args.test = f"datasets/{dataset}/pubdataedges_test_trans_kcore_2.csv"
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print("Loading data...")
    pubs = load_text_table(args.publications)
    datasets = load_text_table(args.datasets)
    train_pairs = load_pairs(args.train)
    print(f"  {len(pubs)} publications, {len(datasets)} datasets, {len(train_pairs)} train (pub, dataset) pairs")

    train_labels = group_labels(train_pairs)
    train_pub_ids = [pid for pid in train_labels.index if pid in pubs.index]
    missing = len(train_labels) - len(train_pub_ids)
    if missing:
        print(f"  Warning: {missing} training publications not found in publications.csv, skipped.")

    dataset_ids = datasets["id"].tolist()
    dataset_texts = datasets["text"].tolist()

    bm25 = None
    if args.hard_negatives > 0:
        print("Building BM25 index for hard-negative mining...")
        bm25 = build_bm25_index(dataset_texts)

    print("Building training examples...")
    train_examples = build_training_examples(
        train_pub_ids, train_labels, pubs, datasets, bm25, dataset_ids, args.hard_negatives, args.seed
    )
    print(f"  {len(train_examples)} training examples "
          f"(1 positive + up to {args.hard_negatives} hard negatives each, plus in-batch negatives)")

    print(f"Loading base model '{args.base_model}'...")
    model = build_model(args.base_model, args.max_seq_length)

    from torch.utils.data import DataLoader
    from sentence_transformers import losses

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)  # in-batch + hard-negative contrastive loss
    warmup_steps = max(1, int(len(train_dataloader) * args.epochs * 0.1))

    print(f"Fine-tuning bi-encoder for {args.epochs} epoch(s)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    if args.save_model:
        model.save(args.save_model)
        print(f"Model saved to {args.save_model}")

    # Cold-start diagnostic: how many eval-set datasets were never seen as a
    # positive during training? A classifier-based approach could never rank
    # these correctly; the bi-encoder can, since it reads dataset text.
    train_seen_datasets = {d for ids in train_labels.reindex(train_pub_ids) for d in ids}

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
        pub_texts = [pubs.loc[pid, "text"] for pid in pub_ids]

        if split_name != "train":
            unseen = {d for ids in labels.reindex(pub_ids) for d in ids} - train_seen_datasets
            if unseen:
                print(f"  [{split_name}] {len(unseen)} true target dataset(s) never appeared in "
                      f"training -- a classifier-based approach could never recommend these.")

        print(f"Ranking datasets for '{split_name}' split ({len(pub_ids)} publications)...")
        ranking_df = rank_datasets(model, pub_ids, pub_texts, dataset_ids, dataset_texts, args.top_k, args.batch_size)

        out_path = os.path.join(args.out_dir, f"ranking_{split_name}.csv")
        ranking_df.to_csv(out_path, index=False)
        print(f"  Saved ranking to {out_path}")

        k_values = tuple(sorted({k for k in (1, 5, 10) if k <= args.top_k}))
        metrics = evaluate_ranking(ranking_df, labels, k_values=k_values)
        all_metrics[split_name] = metrics
        print(f"  {split_name} metrics: {json.dumps(metrics, indent=2)}")

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")


if __name__ == "__main__":
    for dataset in ['mes','pubmed']:
        for split in [0,25,50,75]:
            print('DATASET',dataset)
            print('SPLIT',split)
            main(dataset,split)