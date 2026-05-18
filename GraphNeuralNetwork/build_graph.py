
import os
import argparse

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from tqdm import tqdm
from dotenv import load_dotenv
import sys

# Add project root to system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import compute_agreement

# Load environment variables from the .env file
load_dotenv()

def get_args():
    parser = argparse.ArgumentParser(description="Build similarity graph from article embeddings")

    # Database (Defaults pulled from .env)
    parser.add_argument("--db_host", type=str, default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db_port", type=int, default=int(os.getenv("DB_PORT", 5432)))
    parser.add_argument("--db_name", type=str, default=os.getenv("DB_NAME"))
    parser.add_argument("--db_user", type=str, default=os.getenv("DB_USER"))
    parser.add_argument("--db_password", type=str, default=os.getenv("DB_PASSWORD"))

    # File Paths
    parser.add_argument("--babe_csv", type=str, default=os.getenv("BABE_CSV", "../data/final_labels_MBIC.csv"))
    parser.add_argument("--sg1_csv", type=str, default=os.getenv("SG1_CSV", "../data/raw_labels_SG1.csv"))
    parser.add_argument("--sg2_csv", type=str, default=os.getenv("SG2_CSV", "../data/raw_labels_SG2.csv"))

    # Column mappings (Kept as hardcoded defaults, but you can add these to .env too if you want)
    # Column mappings (Now pulled from .env)
    parser.add_argument("--babe_id_col", type=str, default=os.getenv("BABE_ID_COL", "article_id"))
    parser.add_argument("--babe_label_col", type=str, default=os.getenv("BABE_LABEL_COL", "label_bias"),
                        help="Column with Biased/Non-biased labels")
    parser.add_argument("--sg_id_col", type=str, default=os.getenv("SG_ID_COL", "article_id"))
    parser.add_argument("--sg_label_col", type=str, default=os.getenv("SG_LABEL_COL", "label_bias"))

    # Hyperparameters (Defaults pulled from .env)
    parser.add_argument("--top_k", type=int, default=int(os.getenv("TOP_K", 10)), help="K nearest neighbors per node")
    parser.add_argument("--sim_threshold", type=float, default=float(os.getenv("SIM_THRESHOLD", 0.75)), help="Min cosine similarity to add edge")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", 1000)), help="Chunk size for similarity computation")

    # Agreement Thresholds
    parser.add_argument("--high_agreement", type=float, default=float(os.getenv("HIGH_AGREEMENT", 0.80)), help="Fraction of annotators that must agree for train_mask")
    parser.add_argument("--med_agreement", type=float, default=float(os.getenv("MED_AGREEMENT", 0.60)), help="Fraction of annotators that must agree for val_mask")

    # Output
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_FILE", "graph.pt"))

    return parser.parse_args()

def _parse_vector(v) -> list:
    """
    pgvector returns VECTOR columns as a string like "[0.1,0.2,...]".
    psycopg2 does NOT automatically cast them to lists, so we parse manually.
    If it's already a list (future psycopg3 behaviour), pass through unchanged.
    """
    if isinstance(v, (list, np.ndarray)):
        return v
    # Strip surrounding brackets and split on commas
    return [float(x) for x in str(v).strip("[]").split(",")]


def load_embeddings(conn_params: dict):
    """
    Returns:
        article_ids : list of str, length N
        embeddings  : np.ndarray [N, 768]
        emotions    : np.ndarray [N, 13]

    Uses a named server-side cursor (itersize=2000) so rows are streamed
    from Postgres in batches rather than loaded all at once with fetchall().
    """
    conn = psycopg2.connect(**conn_params)
    # Named cursor → server-side: fetches `itersize` rows at a time
    cur = conn.cursor(name="emb_stream", cursor_factory=psycopg2.extras.DictCursor)
    cur.itersize = 2000

    print("[build_graph] Loading embeddings from PostgreSQL...")
    cur.execute("SELECT article_id, embedding, emotion_scores FROM article_embeddings ORDER BY article_id")

    article_ids = []
    embeddings_list = []
    emotions_list = []
    for row in cur:
        article_ids.append(row["article_id"])
        # pgvector returns embedding as a string "[0.1,0.2,...]" — parse it
        embeddings_list.append(_parse_vector(row["embedding"]))
        emotions_list.append(row["emotion_scores"])

    cur.close()
    conn.close()

    embeddings = np.array(embeddings_list, dtype=np.float32)
    emotions   = np.array(emotions_list,   dtype=np.float32)

    print(f"[build_graph] Loaded {len(article_ids):,} articles")
    print(f"[build_graph] Embedding shape : {embeddings.shape}")
    print(f"[build_graph] Emotion shape   : {emotions.shape}")

    return article_ids, embeddings, emotions





def build_label_tensors(
    article_ids: list,
    babe_path: str,
    sg1_path: str,
    sg2_path: str,
    babe_id_col: str,
    babe_label_col: str,
    sg_id_col: str,
    sg_label_col: str,
    high_agreement: float,
    med_agreement: float,
):
    """

    :param article_ids: a list of article ids
    :param babe_path: the path to the babe dataset file
    :param sg1_path:  the path to the first annonators file
    :param sg2_path: the path to the second annonators file
    :param babe_id_col: the article id column
    :param babe_label_col: the collumn we try to predict from babe
    :param sg_id_col: the article id column from sg data set
    :param sg_label_col: the collumn we try to predict from sg
    :param high_agreement: parameter that helps us build a train dataset for the graph of only articles with high agreement
    :param med_agreement: parameter that helps us build a validation dataset for the graph of only articles with medium agreement

    Methodology:
        The function iterates through every row from babe and based on the agreement score between annotators it adds it to either train dataset or validation  dataset

    Returns:
            y a list of labels strings in general ,
            train_mask the mask of datasets used for training,
            val_mask the mask of datasets used for validation,
            weights  how confident is the prediction for each article

    """
    N = len(article_ids)
    # create dict between embeddings ids and indexes
    id_to_idx = {aid:i for i,aid in enumerate(article_ids)}


    babe = pd.read_csv(babe_path, sep=';', on_bad_lines='skip')
    babe[babe_id_col] = babe[babe_id_col].astype(str)

    # compute aggreement
    agreement_df = compute_agreement(sg1_path, sg2_path, sg_id_col, sg_label_col)
    agreement_df["article_id"] = agreement_df["article_id"].astype(str)
    # zip article_id, agreement as a dictionary
    agreement_map = dict(zip(agreement_df["article_id"], agreement_df["agreement"]))


    # Data loading
    y = torch.full((N,), -1, dtype=torch.long)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)  # new: medium-low confidence
    low_conf_mask = torch.zeros(N, dtype=torch.bool)  # new: ambiguous, track only
    weights = torch.zeros(N, dtype=torch.float)

    label_map = {"Biased": 1, "Non-biased": 0}
    labeled_count = 0
    unlabeled_count = 0

    for _,row in babe.iterrows():
        aid = str(row[babe_id_col])
        label = row[babe_label_col]

        if aid not in id_to_idx:
            unlabeled_count += 1
            continue
        if label not in label_map:
            continue

        # label the article
        idx = id_to_idx[aid]
        y[idx] = label_map[label]
        labeled_count += 1

        agr = agreement_map.get(aid,0.5)

        weights[idx] = agr

        if agr >= high_agreement:
            train_mask[idx] = True
        elif agr >= med_agreement:
            val_mask[idx] = True
        elif agr >= 0.40:
            test_mask[idx] = True  # has a label but low confidence
        else:
            low_conf_mask[idx] = True

    print(f"[build_graph] BABE articles matched in graph : {labeled_count:,}")
    print(f"[build_graph] BABE articles not in graph     : {unlabeled_count:,}")
    print(f"[build_graph] train_mask  (>= {high_agreement}) : {train_mask.sum().item():,}")
    print(f"[build_graph] val_mask    (>= {med_agreement})  : {val_mask.sum().item():,}")
    print(f"[build_graph] test_mask   (>= 0.40)             : {test_mask.sum().item():,}")
    print(f"[build_graph] low_conf    (<  0.40)             : {low_conf_mask.sum().item():,}")
    print(f"[build_graph] Unlabeled nodes (no BABE label)   : {(y == -1).sum().item():,}")
    return y, train_mask, val_mask, test_mask, low_conf_mask, weights


def build_edges(article_ids, embeddings: np.ndarray, emotions: np.ndarray, top_k: int, sim_threshold: float, chunk_size=1000):
    """
        Builds a bidirectional k-Nearest Neighbors (k-NN) graph based on cosine similarity.

        The function normalizes the input vectors (L2 norm) and computes cosine similarity
        in chunks to optimize memory usage (RAM/VRAM). It creates edges only to the `top_k`
        nearest neighbors, provided the similarity exceeds the `sim_threshold`. The returned
        graph is undirected (edges are bidirectional) and deduplicated, formatted specifically
        for PyTorch Geometric.

        Important: edge construction is label-agnostic. Labels and train/val/test masks are
        created after the graph topology and must not influence neighbor selection.

        Args:
            embeddings (np.ndarray): A 2D numpy array containing the vectors (shape: [num_nodes, embedding_dim]).
            top_k (int): The maximum number of neighbors connected to a single node.
            sim_threshold (float): The minimum cosine similarity threshold to validate an edge (e.g., 0.7).
            chunk_size (int, optional): The batch size processed in a single iteration. This is highly
                useful for preventing Out-Of-Memory (OOM) errors on large datasets. Defaults to 1000.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple consisting of two tensors:
                - edge_index (torch.Tensor): Tensor of shape [2, num_edges] (dtype=torch.long).
                  Contains the source and destination node indices.
                - edge_attr (torch.Tensor): Tensor of shape [num_edges, 1] (dtype=torch.float).
                  Contains the edge weights (the exact cosine similarity value).
    """

    # make it a unit vector
    # how much we take into account embeddings and emotion scores
    ALPHA = 0.8
    BETA = 0.2
    normed_emb = normalize(embeddings, norm='l2')
    normed_emo = normalize(emotions, norm='l2')

    N = len(normed_emb)
    rows_list = []
    cols_list = []
    vals_list = []


    print(f"[build_graph] Building KNN graph (top_k={top_k}, threshold={sim_threshold})...")
    # One loop over chunks — `chunk_start` is the outer variable, never shadowed.
    for chunk_start in tqdm(range(0, N, chunk_size)):
        chunk_end = min(N, chunk_start + chunk_size)

        sim_emb = cosine_similarity(normed_emb[chunk_start:chunk_end], normed_emb)
        sim_emo = cosine_similarity(normed_emo[chunk_start:chunk_end], normed_emo)
        sims = ALPHA * sim_emb + BETA * sim_emo  # [chunk, N]

        for local_i, row_sims in enumerate(sims):
            global_i = chunk_start + local_i
            row_sims = row_sims.copy()
            row_sims[global_i] = -1.0  # exclude self-loops

            top_indices = np.argsort(row_sims)[::-1][:top_k]
            for j in top_indices:
                if row_sims[j] >= sim_threshold:
                    rows_list.append(global_i)
                    cols_list.append(j)
                    vals_list.append(float(row_sims[j]))

    # --- Bidirectional dedup (OUTSIDE the loop — runs once after all chunks) ---
    rows_arr = np.array(rows_list + cols_list, dtype=np.int64)
    cols_arr = np.array(cols_list + rows_list, dtype=np.int64)
    vals_arr = np.array(vals_list + vals_list, dtype=np.float32)

    edge_set = {}
    for r, c, v in zip(rows_arr, cols_arr, vals_arr):
        key = (min(r, c), max(r, c))
        if key not in edge_set:
            edge_set[key] = v

    final_rows, final_cols, final_vals = [], [], []
    for (r, c), v in edge_set.items():
        final_rows.extend([r, c])
        final_cols.extend([c, r])
        final_vals.extend([v, v])

    edge_index = torch.tensor([final_rows, final_cols], dtype=torch.long)
    edge_attr  = torch.tensor(final_vals, dtype=torch.float).unsqueeze(1)

    print(f"[build_graph] Total edges (bidirectional): {edge_index.shape[1]:,}")
    return edge_index, edge_attr

def main():
    args =  get_args()

    conn_params = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.db_name,
        "user": args.db_user,
        "password": args.db_password,
    }

    # i think we should use only the embeddings not the emotions
    article_ids,embeddings,emotions = load_embeddings(conn_params)

    edge_index, edge_attr = build_edges(
        article_ids,
        embeddings,
        emotions,
        top_k=args.top_k,
        sim_threshold=args.sim_threshold,
        chunk_size=args.chunk_size,
    )

    y, train_mask, val_mask, test_mask, low_conf_mask, weights = build_label_tensors(
        article_ids,
        babe_path=args.babe_csv,
        sg1_path=args.sg1_csv,
        sg2_path=args.sg2_csv,
        babe_id_col=args.babe_id_col,
        babe_label_col=args.babe_label_col,
        sg_id_col=args.sg_id_col,
        sg_label_col=args.sg_label_col,
        high_agreement=args.high_agreement,
        med_agreement=args.med_agreement,
    )

    data = Data(
        x=torch.tensor(embeddings, dtype=torch.float),
        emotions=torch.tensor(emotions, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        low_conf_mask=low_conf_mask,
    )
    # Store article IDs as a plain list (not a tensor) for later lookup
    data.article_ids = article_ids
    data.label_weights = weights

    print(f"\n[build_graph] Graph summary:")
    print(f"  Nodes          : {data.num_nodes:,}")
    print(f"  Edges          : {data.num_edges:,}")
    print(f"  Node feature dim: {data.num_node_features}")
    print(f"  Labeled (train): {train_mask.sum().item():,}")
    print(f"  Labeled (val)  : {val_mask.sum().item():,}")

    torch.save(data, args.output)
    print(f"\n[build_graph] Saved graph to: {args.output}")


if __name__ == "__main__":
    main()
