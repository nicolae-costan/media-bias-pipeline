
import os
import argparse

import numpy as np
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from tqdm import tqdm
from dotenv import load_dotenv
import sys

# Add project root to system path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import compute_agreement
from EmotionModels.dataloader import RedditDataset
COLUMNS = RedditDataset.EMOTION_COLUMNS

# Load environment variables from the .env file
load_dotenv()

def get_args():
    parser = argparse.ArgumentParser(description="Build similarity graph from article embeddings")

    # Database (Defaults pulled from .env)
    parser.add_argument("--db_host", type=str, default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db_port", type=int, default=int(os.getenv("DB_PORT", 5433)))
    parser.add_argument("--db_name", type=str, default=os.getenv("DB_NAME"))
    parser.add_argument("--db_user", type=str, default=os.getenv("DB_USER"))
    parser.add_argument("--db_password", type=str, default=os.getenv("DB_PASSWORD"))

    # File Paths
    parser.add_argument("--babe_csv", type=str, default=os.getenv("BABE_CSV", "../data/consensus_labels_sg1_sg2.csv"))
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
    parser.add_argument("--top_k", type=int, default=int(os.getenv("TOP_K", 15)),
                        help="Guaranteed number of nearest neighbors per node (no threshold pruning)")
    parser.add_argument("--sim_threshold", type=float, default=float(os.getenv("SIM_THRESHOLD", 0.5)),
                        help="Soft floor — used only to count weak edges for diagnostics; does NOT prune edges")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", 1000)), help="Chunk size for similarity computation")

    # Agreement Thresholds (used only when --split_mode=agreement)
    parser.add_argument("--high_agreement", type=float, default=float(os.getenv("HIGH_AGREEMENT", 0.80)), help="Fraction of annotators that must agree for train_mask")
    parser.add_argument("--med_agreement", type=float, default=float(os.getenv("MED_AGREEMENT", 0.60)), help="Fraction of annotators that must agree for val_mask")

    # Split strategy: how train/val/test are carved out of the labeled articles.
    #   "random_stratified" — random 70/15/15 stratified by label. Agreement is
    #     used only as a per-sample LOSS weight (label_weights), not to pick
    #     which split an article lands in. Recommended: the test set then
    #     represents the same label-quality distribution as train/val.
    #   "agreement" — original behaviour: train = high-agreement, val = mid,
    #     test = low. Useful if you actually want a "hardest-articles" test.
    parser.add_argument(
        "--split_mode", type=str,
        choices=["random_stratified", "agreement"],
        default=os.getenv("SPLIT_MODE", "random_stratified"),
        help="How to assign labeled articles to train/val/test"
    )
    parser.add_argument("--train_frac", type=float, default=float(os.getenv("TRAIN_FRAC", 0.70)),
                        help="Fraction of labeled articles assigned to train (random_stratified only)")
    parser.add_argument("--val_frac",   type=float, default=float(os.getenv("VAL_FRAC",   0.15)),
                        help="Fraction of labeled articles assigned to val (random_stratified only); test is the remainder")
    parser.add_argument("--split_seed", type=int,   default=int(os.getenv("SPLIT_SEED", 42)),
                        help="RNG seed for random_stratified split — set for reproducible builds")

    # Output
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_FILE", "graph.pt"))

    return parser.parse_args()

def load_embeddings(conn_params: dict):
    """
    Returns:
        article_ids : list of str, length N
        embeddings  : np.ndarray [N, 768]
        emotions    : np.ndarray [N, 13]

    Performance notes:
        - Registers pgvector's psycopg2 adapter so VECTOR columns arrive as
          numpy arrays directly from the driver (binary mode), skipping the
          per-row text parse that used to dominate this function.
        - Uses a plain client-side cursor (no named server-side cursor and
          no DictCursor). On localhost the whole table fits comfortably in
          RAM, and avoiding per-batch round-trips + per-row dict construction
          is dramatically faster.
        - Pre-allocates the output numpy arrays from a COUNT(*) so we don't
          rebuild a giant list-of-lists and then copy it at the end.
    """
    conn = psycopg2.connect(**conn_params)
    # Tell psycopg2 how to decode VECTOR columns → numpy arrays (binary).
    register_vector(conn)

    print("[build_graph] Loading embeddings from PostgreSQL...")

    with conn.cursor() as count_cur:
        count_cur.execute("SELECT COUNT(*) FROM article_embeddings")
        n = count_cur.fetchone()[0]

    if n == 0:
        conn.close()
        raise RuntimeError(
            "article_embeddings table is empty — run article_embeddings.py first."
        )

    # Pre-allocate output buffers. Embedding dim is fixed at 768 (VECTOR(768))
    # and emotion_scores is a len(COLUMNS)-dim FLOAT4[] per the schema.
    article_ids = [None] * n
    embeddings = np.empty((n, 768), dtype=np.float32)
    emotions = np.empty((n, len(COLUMNS)), dtype=np.float32)

    with conn.cursor() as cur:
        # article_id is the primary key, so ORDER BY uses the existing B-tree
        # index — free sort, deterministic ordering for downstream id_to_idx.
        cur.execute(
            "SELECT article_id, embedding, emotion_scores "
            "FROM article_embeddings ORDER BY article_id"
        )
        for i, (aid, emb, emo) in enumerate(cur):
            article_ids[i] = aid
            embeddings[i] = emb  # numpy array thanks to register_vector
            emotions[i] = emo

    conn.close()

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
    split_mode: str = "random_stratified",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    split_seed: int = 42,
):
    """
    Returns (y, train_mask, val_mask, test_mask, low_conf_mask, weights).

    Labels and per-article agreement weights are always computed the same way.
    The split_mode parameter controls how labeled articles are routed into
    train / val / test:

    - "random_stratified" (recommended): random 70/15/15 of labeled articles,
       stratified by class. Annotator agreement is preserved as `weights` and
       used by the loss as a per-sample multiplier. This gives a meaningful
       test number — the test articles have the same label-quality distribution
       as train.

    - "agreement" (legacy): train = articles with agreement >= high_agreement,
       val = articles with agreement >= med_agreement, test = articles in
       [0.40, med_agreement). The test set is by construction the worst-
       agreement articles, which caps achievable accuracy near the human
       agreement ceiling (~0.60–0.65 on this dataset).

    Articles with agreement < 0.40 always go to low_conf_mask (genuine noise,
    excluded from any split). Unlabeled articles stay y == -1 / no mask.
    """
    N = len(article_ids)
    # create dict between embeddings ids and indexes
    id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    # Auto-detect delimiter: the project mixes ';'-separated files
    # (raw_labels_SG1/SG2.csv, final_labels_MBIC.csv) with ','-separated ones
    # (consensus_labels_sg1_sg2.csv). engine='python' is required for sniffing.
    babe = pd.read_csv(babe_path, sep=None, engine='python', on_bad_lines='skip')
    if babe_id_col not in babe.columns:
        raise KeyError(
            f"Column '{babe_id_col}' not found in {babe_path}. "
            f"Available columns: {list(babe.columns)}"
        )
    babe[babe_id_col] = babe[babe_id_col].astype(str)

    # Compute agreement from SG1+SG2 raw annotator files
    agreement_df = compute_agreement(sg1_path, sg2_path, sg_id_col, sg_label_col)
    agreement_df["article_id"] = agreement_df["article_id"].astype(str)
    agreement_map = dict(zip(agreement_df["article_id"], agreement_df["agreement"]))

    # ---- Pass 1: label everything that maps cleanly, regardless of split mode
    y = torch.full((N,), -1, dtype=torch.long)
    weights = torch.zeros(N, dtype=torch.float)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    low_conf_mask = torch.zeros(N, dtype=torch.bool)

    label_map = {"Biased": 1, "Non-biased": 0}
    labeled_count = 0
    unlabeled_count = 0

    for _, row in babe.iterrows():
        aid = str(row[babe_id_col])
        label = row[babe_label_col]

        if aid not in id_to_idx:
            unlabeled_count += 1
            continue
        if label not in label_map:
            continue

        idx = id_to_idx[aid]
        y[idx] = label_map[label]
        weights[idx] = agreement_map.get(aid, 0.5)
        labeled_count += 1

    # ---- Pass 2: assign train/val/test masks per the chosen split mode

    if split_mode == "agreement":
        for i in range(N):
            if y[i] == -1:
                continue
            agr = float(weights[i])
            if agr >= high_agreement:
                train_mask[i] = True
            elif agr >= med_agreement:
                val_mask[i] = True
            elif agr >= 0.40:
                test_mask[i] = True
            else:
                low_conf_mask[i] = True

    elif split_mode == "random_stratified":
        if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0
                and train_frac + val_frac < 1.0):
            raise ValueError(
                f"Invalid split fractions: train_frac={train_frac}, val_frac={val_frac}. "
                "Need 0 < train_frac, val_frac and train_frac + val_frac < 1."
            )
        test_frac = 1.0 - train_frac - val_frac

        # Pull noise floor (< 0.40 agreement) into low_conf_mask, excluded from splits.
        labeled = (y != -1)
        eligible = labeled & (weights >= 0.40)
        low_conf_mask = labeled & (weights < 0.40)

        eligible_idx = np.where(eligible.numpy())[0]
        y_eligible = y.numpy()[eligible_idx]

        # First split: train vs (val + test)
        train_idx, temp_idx, _, y_temp = train_test_split(
            eligible_idx, y_eligible,
            test_size=(val_frac + test_frac),
            stratify=y_eligible,
            random_state=split_seed,
        )
        # Second split: val vs test (proportional inside the leftover bucket)
        test_share_of_temp = test_frac / (val_frac + test_frac)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_share_of_temp,
            stratify=y_temp,
            random_state=split_seed,
        )

        train_mask[torch.as_tensor(train_idx, dtype=torch.long)] = True
        val_mask[torch.as_tensor(val_idx, dtype=torch.long)] = True
        test_mask[torch.as_tensor(test_idx, dtype=torch.long)] = True

    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")

    # ---- Logging
    print(f"[build_graph] BABE articles matched in graph : {labeled_count:,}")
    print(f"[build_graph] BABE articles not in graph     : {unlabeled_count:,}")
    print(f"[build_graph] Split mode                     : {split_mode}")
    if split_mode == "random_stratified":
        print(f"[build_graph]   fractions — train: {train_frac:.2f} | "
              f"val: {val_frac:.2f} | test: {1.0 - train_frac - val_frac:.2f} "
              f"(seed={split_seed})")
    print(f"[build_graph] train_mask                     : {train_mask.sum().item():,}")
    print(f"[build_graph] val_mask                       : {val_mask.sum().item():,}")
    print(f"[build_graph] test_mask                      : {test_mask.sum().item():,}")
    print(f"[build_graph] low_conf  (excluded, agr<0.40) : {low_conf_mask.sum().item():,}")
    print(f"[build_graph] Unlabeled nodes (no BABE label): {(y == -1).sum().item():,}")
    # Per-class breakdown so we can confirm stratification worked.
    for name, m in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        ym = y[m]
        if ym.numel() == 0:
            continue
        n_pos = int((ym == 1).sum())
        n_neg = int((ym == 0).sum())
        print(f"[build_graph]   {name:>5s} class balance — "
              f"Non-biased: {n_neg:>4d} | Biased: {n_pos:>4d}")

    return y, train_mask, val_mask, test_mask, low_conf_mask, weights


def build_edges(article_ids, embeddings: np.ndarray, emotions: np.ndarray, top_k: int, sim_threshold: float, chunk_size=1000):
    """
        Builds a bidirectional k-Nearest Neighbors (k-NN) graph based on cosine similarity.

        Every node is guaranteed to connect to its `top_k` nearest neighbors, regardless
        of their absolute similarity score. This is intentional: a hard similarity cutoff
        (the previous behaviour) silently isolated nodes whose neighbourhoods lived in
        slightly lower-similarity regions of the embedding space, leaving the GNN with
        nothing to aggregate over for those nodes. `sim_threshold` is retained as a SOFT
        FLOOR for diagnostics — we log how many of the kept edges fall below it so callers
        can spot when neighborhoods are unusually noisy, but it does NOT prune edges.

        Important: edge construction is label-agnostic. Labels and train/val/test masks are
        created after the graph topology and must not influence neighbor selection.

        Args:
            embeddings (np.ndarray): A 2D numpy array containing the vectors (shape: [num_nodes, embedding_dim]).
            top_k (int): Number of neighbors each node will be connected to (guaranteed).
            sim_threshold (float): Soft floor used only for the diagnostic count of "weak"
                edges — does NOT prune edges. Set to 0.0 to silence the diagnostic.
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
    # Pre-size the directed edge buffers: exactly top_k per node before dedup.
    k = min(top_k, N - 1)
    total = N * k
    rows_arr_dir = np.empty(total, dtype=np.int64)
    cols_arr_dir = np.empty(total, dtype=np.int64)
    vals_arr_dir = np.empty(total, dtype=np.float32)
    write_ptr = 0
    weak_edges = 0  # count of kept edges whose similarity falls below sim_threshold

    print(f"[build_graph] Building KNN graph (guaranteed top_k={k}, soft floor={sim_threshold})...")
    # One loop over chunks — `chunk_start` is the outer variable, never shadowed.
    for chunk_start in tqdm(range(0, N, chunk_size)):
        chunk_end = min(N, chunk_start + chunk_size)

        sim_emb = cosine_similarity(normed_emb[chunk_start:chunk_end], normed_emb)
        sim_emo = cosine_similarity(normed_emo[chunk_start:chunk_end], normed_emo)
        sims = ALPHA * sim_emb + BETA * sim_emo  # [chunk, N]

        for local_i, row_sims in enumerate(sims):
            global_i = chunk_start + local_i
            row_sims = row_sims.copy()
            row_sims[global_i] = -np.inf  # exclude self-loops

            # argpartition is O(N) — much cheaper than a full O(N log N) sort.
            # The selected k indices are unordered; that's fine, we don't rank within.
            top_indices = np.argpartition(row_sims, -k)[-k:]
            rows_arr_dir[write_ptr:write_ptr + k] = global_i
            cols_arr_dir[write_ptr:write_ptr + k] = top_indices
            vals_arr_dir[write_ptr:write_ptr + k] = row_sims[top_indices]
            weak_edges += int((row_sims[top_indices] < sim_threshold).sum())
            write_ptr += k

    print(f"[build_graph] Directed edges built: {write_ptr:,} (= N * top_k)")
    print(f"[build_graph] Edges below soft floor ({sim_threshold}): {weak_edges:,}")

    # Carry the directed edges into the dedup phase below.
    rows_list = rows_arr_dir.tolist()
    cols_list = cols_arr_dir.tolist()
    vals_list = vals_arr_dir.tolist()

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

    # Diagnostics: per-node degree histogram so we can spot under-connected nodes.
    degree = np.bincount(edge_index[0].numpy(), minlength=N)
    print(f"[build_graph] Total edges (bidirectional): {edge_index.shape[1]:,}")
    print(
        f"[build_graph] Degree stats — min: {int(degree.min())} | "
        f"median: {int(np.median(degree))} | mean: {degree.mean():.1f} | "
        f"max: {int(degree.max())} | isolated nodes: {int((degree == 0).sum())}"
    )
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
        split_mode=args.split_mode,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
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
