
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
