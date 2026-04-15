"""
Drug Repurposing with Graph Neural Networks
Step 1: Data Loading & Knowledge Graph Construction
====================================================
Uses Hetionet — a biomedical knowledge graph with:
  - 47,031 nodes (drugs, diseases, genes, pathways, etc.)
  - 2,250,197 edges across 24 relationship types

Run this first to download, parse, and build your graph.
"""

import json
import os
import urllib.request
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HETIONET_NODES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-nodes.tsv"
HETIONET_EDGES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-edges.sif.gz"

# Edge types we care about most for drug repurposing
FOCUS_EDGE_TYPES = [
    ("Compound", "treats",           "Disease"),   # ground truth labels!
    ("Compound", "binds",            "Gene"),
    ("Compound", "upregulates",      "Gene"),
    ("Compound", "downregulates",    "Gene"),
    ("Disease",  "associates",       "Gene"),
    ("Disease",  "upregulates",      "Gene"),
    ("Disease",  "downregulates",    "Gene"),
    ("Gene",     "participates",     "Pathway"),
    ("Disease",  "localizes",        "Anatomy"),
    ("Compound", "causes",           "Side Effect"),
]


# ── Download helpers ────────────────────────────────────────────────────────
def download_hetionet():
    """Download Hetionet node and edge files if not already present."""
    nodes_path = os.path.join(DATA_DIR, "nodes.tsv")
    edges_path = os.path.join(DATA_DIR, "edges.tsv.gz")

    if not os.path.exists(nodes_path):
        print("Downloading Hetionet nodes...")
        urllib.request.urlretrieve(HETIONET_NODES_URL, nodes_path)
        print(f"  Saved to {nodes_path}")
    else:
        print("  Nodes already downloaded.")

    if not os.path.exists(edges_path):
        print("Downloading Hetionet edges (~50MB)...")
        urllib.request.urlretrieve(HETIONET_EDGES_URL, edges_path)
        print(f"  Saved to {edges_path}")
    else:
        print("  Edges already downloaded.")

    return nodes_path, edges_path


# ── Parse nodes ────────────────────────────────────────────────────────────
def load_nodes(nodes_path):
    """
    Returns:
        nodes_df : DataFrame with columns [id, name, kind]
        node2idx : dict mapping (kind, id) -> int index within that kind
        kind_dfs : dict mapping kind -> subset DataFrame
    """
    nodes_df = pd.read_csv(nodes_path, sep="\t")
    print(f"\nTotal nodes: {len(nodes_df)}")
    print(nodes_df["kind"].value_counts().to_string())

    # Build per-kind integer indices
    kind_dfs = {}
    node2idx  = {}
    for kind, grp in nodes_df.groupby("kind"):
        grp = grp.reset_index(drop=True)
        kind_dfs[kind] = grp
        for local_idx, row in grp.iterrows():
            node2idx[(kind, row["id"])] = local_idx

    return nodes_df, node2idx, kind_dfs


# ── Parse edges ────────────────────────────────────────────────────────────
def load_edges(edges_path, node2idx):
    """
    Returns a dict:
        edge_dict[(src_kind, rel, dst_kind)] = (src_indices, dst_indices)
    """
    print("\nLoading edges...")
    edges_df = pd.read_csv(edges_path, sep="\t", compression="gzip",
                           names=["source", "metaedge", "target"])
    print(f"Total edges: {len(edges_df)}")

    # Metaedge format:  "CbG" = Compound-binds-Gene
    # We use a lookup from abbreviation to (src_kind, rel, dst_kind)
    # Build it from FOCUS_EDGE_TYPES names
    edge_dict = defaultdict(lambda: ([], []))

    # Parse metaedge abbreviations from node data
    # Hetionet abbreviation map (common ones)
    ABBREV = {
        "CtD":  ("Compound",  "treats",           "Disease"),
        "CbG":  ("Compound",  "binds",             "Gene"),
        "CuG":  ("Compound",  "upregulates",       "Gene"),
        "CdG":  ("Compound",  "downregulates",     "Gene"),
        "DaG":  ("Disease",   "associates",        "Gene"),
        "DuG":  ("Disease",   "upregulates",       "Gene"),
        "DdG":  ("Disease",   "downregulates",     "Gene"),
        "GpPW": ("Gene",      "participates",      "Pathway"),
        "DlA":  ("Disease",   "localizes",         "Anatomy"),
        "CcSE": ("Compound",  "causes",            "Side Effect"),
        "GiG":  ("Gene",      "interacts",         "Gene"),
        "GrG":  ("Gene",      "regulates",         "Gene"),
    }

    skipped = 0
    for _, row in edges_df.iterrows():
        abbrev = row["metaedge"]
        if abbrev not in ABBREV:
            skipped += 1
            continue

        src_kind, rel, dst_kind = ABBREV[abbrev]
        src_key = (src_kind, row["source"])
        dst_key = (dst_kind, row["target"])

        if src_key not in node2idx or dst_key not in node2idx:
            skipped += 1
            continue

        edge_type = (src_kind, rel, dst_kind)
        edge_dict[edge_type][0].append(node2idx[src_key])
        edge_dict[edge_type][1].append(node2idx[dst_key])

    print(f"  Skipped {skipped} edges (unknown type or missing nodes)")
    for et, (srcs, _) in edge_dict.items():
        print(f"  {et}: {len(srcs)} edges")

    return dict(edge_dict)


# ── Build PyG HeteroData ────────────────────────────────────────────────────
def build_hetero_graph(kind_dfs, edge_dict):
    """
    Constructs a PyTorch Geometric HeteroData object.
    Node features: one-hot encoding of node type + random 64-dim embedding
    (replace with molecular fingerprints in step 2 for real features).
    """
    data = HeteroData()

    print("\nBuilding HeteroData graph...")

    FEAT_DIM = 64  # Will be replaced by real features in step 2

    for kind, df in kind_dfs.items():
        n = len(df)
        # Placeholder features — Step 2 replaces these with real embeddings
        data[kind].x = torch.randn(n, FEAT_DIM)
        data[kind].node_id = torch.arange(n)
        data[kind].num_nodes = n
        print(f"  {kind}: {n} nodes, feature shape {data[kind].x.shape}")

    for (src_kind, rel, dst_kind), (srcs, dsts) in edge_dict.items():
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        data[src_kind, rel, dst_kind].edge_index = edge_index
        print(f"  ({src_kind}, {rel}, {dst_kind}): {edge_index.shape[1]} edges")

    return data


# ── Train/val/test split on "treats" edges ─────────────────────────────────
def split_treat_edges(edge_dict, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split the gold-standard (Compound, treats, Disease) edges into
    train / val / test sets. Returns index arrays.
    """
    key = ("Compound", "treats", "Disease")
    srcs, dsts = edge_dict[key]
    n = len(srcs)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    print(f"\nTreats-edge split — train: {len(train_idx)}, "
          f"val: {len(val_idx)}, test: {len(test_idx)}")

    return {
        "train": (np.array(srcs)[train_idx], np.array(dsts)[train_idx]),
        "val":   (np.array(srcs)[val_idx],   np.array(dsts)[val_idx]),
        "test":  (np.array(srcs)[test_idx],  np.array(dsts)[test_idx]),
    }


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nodes_path, edges_path = download_hetionet()
    nodes_df, node2idx, kind_dfs = load_nodes(nodes_path)
    edge_dict = load_edges(edges_path, node2idx)
    data = build_hetero_graph(kind_dfs, edge_dict)
    splits = split_treat_edges(edge_dict)

    # Save for next steps
    torch.save({
        "hetero_data": data,
        "splits":      splits,
        "kind_dfs":    kind_dfs,
        "edge_dict":   edge_dict,
        "node2idx":    node2idx,
    }, os.path.join(DATA_DIR, "graph_data.pt"))

    print("\n✓ graph_data.pt saved. Run 2_features.py next.")
    print(f"\nGraph summary:\n{data}")
