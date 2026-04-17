
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
