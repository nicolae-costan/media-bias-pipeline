
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


def get_args():
    parser = argparse.ArgumentParser(description="Build similarity graph from article embeddings")

    # Database
    parser.add_argument("--db_host",     type=str, default="localhost")
    parser.add_argument("--db_port",     type=int, default=5432)
    parser.add_argument("--db_name",     type=str, required=True)
    parser.add_argument("--db_user",     type=str, required=True)
    parser.add_argument("--db_password", type=str, required=True)

    parser.add_argument("--babe_csv",  type=str, required=True, help="final_labels_mbic.csv")
    parser.add_argument("--sg1_csv",   type=str, required=True, help="sg1.csv for agreement scores")
    parser.add_argument("--sg2_csv",   type=str, required=True, help="sg2.csv for agreement scores")

    parser.add_argument("--babe_id_col", type=str, default="article_id")
    parser.add_argument("--babe_label_col", type=str, default="label_bias",
                        help="Column with Biased/Non-biased labels")
    parser.add_argument("--sg_id_col", type=str, default="article_id")
    parser.add_argument("--sg_label_col", type=str, default="label_bias")

    parser.add_argument("--top_k", type=int, default=10, help="K nearest neighbors per node")
    parser.add_argument("--sim_threshold", type=float, default=0.75, help="Min cosine similarity to add edge")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for similarity computation")

    parser.add_argument("--high_agreement", type=float, default=0.80,
                        help="Fraction of annotators that must agree for train_mask")
    parser.add_argument("--med_agreement", type=float, default=0.60,
                        help="Fraction of annotators that must agree for val_mask")

    parser.add_argument("--output", type=str, default="graph.pt")


def compute_agreement(sg1_path:str, sg2_path:str,id_col,label_col):

    sg1 = pd.read_csv(sg1_path)
    sg2 = pd.read_csv(sg2_path)

    all_anotations = pd.concat([sg1, sg2],ignore_index=True)

    records = []

    for article_id,group in all_anotations.groupby(id_col):

        counts = group[label_col].value_counts()
        majority_label = counts.index[0]
        agreement = counts.iloc[0] / len(group)

        records.append({

            "article_id": article_id,
            "majority_label": majority_label,
            "agreement": agreement,
            })


    return pd.DataFrame(records)





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


    babe = pd.read_csv(babe_path)
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

        agr = agreement_map.get(aid,0)

        weights[idx] = agr


        if agr >= high_agreement:
            train_mask[idx] = True
        elif agr >= med_agreement:
            val_mask[idx] = True

    # print overall accuracy
    print(f"[build_graph] BABE articles matched in graph : {labeled_count:,}")
    print(f"[build_graph] BABE articles not in graph     : {unlabeled_count:,}")
    print(f"[build_graph] train_mask (high agreement)    : {train_mask.sum().item():,}")
    print(f"[build_graph] val_mask   (med  agreement)    : {val_mask.sum().item():,}")
    print(f"[build_graph] Unlabeled nodes                : {(y == -1).sum().item():,}")

    return y, train_mask, val_mask, weights


def build_edges(embeddings: np.ndarray,top_k:int,sim_threshold:float,chunk_size = 1000):
    """
        Builds a bidirectional k-Nearest Neighbors (k-NN) graph based on cosine similarity.

        The function normalizes the input vectors (L2 norm) and computes cosine similarity
        in chunks to optimize memory usage (RAM/VRAM). It creates edges only to the `top_k`
        nearest neighbors, provided the similarity exceeds the `sim_threshold`. The returned
        graph is undirected (edges are bidirectional) and deduplicated, formatted specifically
        for PyTorch Geometric.

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
    normed = normalize(embeddings,norm='l2')
    N = len(normed)
    rows_list = []
    cols_list = []
    vals_list = []

    print(f"[build_graph] Building KNN graph (top_k={top_k}, threshold={sim_threshold})...")
    for start in tqdm(range(0,N,chunk_size)):
        end = min(N,start+chunk_size)
        chunk = normed[start:end]

        sims = cosine_similarity(chunk,normed)

        # get every row
        for local_i, row_sims in enumerate(sims):
            global_i = start + local_i

            # Exclude self-similarity
            row_sims[global_i] = -1.0

            # Get top_k indices above threshold by reversing the srot
            top_indices = np.argsort(row_sims)[::-1][:top_k]

            for j in top_indices:
                if row_sims[j] >= sim_threshold:
                    rows_list.append(global_i)
                    cols_list.append(j)
                    vals_list.append(float(row_sims[j]))

    # make arrays
    rows_arr = np.array(rows_list + cols_list, dtype=np.int64)
    cols_arr = np.array(cols_list + rows_list, dtype=np.int64)
    vals_arr = np.array(vals_list + vals_list, dtype=np.float32)

    # Deduplicate having an array as [0,1] and the other one as [1,0]
    edge_set = {}
    for r, c, v in zip(rows_arr, cols_arr, vals_arr):
        key = (min(r, c), max(r, c))
        if key not in edge_set:
            edge_set[key] = v

    final_rows = []
    final_cols = []
    final_vals = []
    for (r, c), v in edge_set.items():
        final_rows.extend([r, c])
        final_cols.extend([c, r])
        final_vals.extend([v, v])

    edge_index = torch.tensor([final_rows, final_cols], dtype=torch.long)
    edge_attr = torch.tensor(final_vals, dtype=torch.float).unsqueeze(1)

    print(f"[build_graph] Total edges (bidirectional): {edge_index.shape[1]:,}")
    return edge_index, edge_attr
