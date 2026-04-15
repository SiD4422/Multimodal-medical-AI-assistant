"""
Drug Repurposing with Graph Neural Networks
Step 3: R-GCN Model + Link Predictor
======================================
Architecture:
  1. Linear input projections  — map each node type to shared hidden dim
  2. Relational GCN layers     — aggregate over heterogeneous edge types
  3. Link predictor head       — score (drug, disease) pairs

Why R-GCN?
  Standard GCNs treat all edges the same. R-GCN uses separate weight
  matrices per relation type, letting the model learn that
  "Compound-binds-Gene" is different from "Disease-associates-Gene".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv, Linear
from torch_geometric.data import HeteroData


# ── Input projection: align all node types to same hidden dim ─────────────
class NodeProjector(nn.Module):
    """
    Each node type has a different feature dimension.
    This projects them all into a common hidden_dim.
    """
    def __init__(self, in_dims: dict, hidden_dim: int):
        """
        Args:
            in_dims    : {node_type: feature_dim}
            hidden_dim : target dimension
        """
        super().__init__()
        self.projectors = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim, bias=False)
            for ntype, fdim in in_dims.items()
        })

    def forward(self, x_dict: dict) -> dict:
        return {
            ntype: F.relu(self.projectors[ntype](x))
            for ntype, x in x_dict.items()
            if ntype in self.projectors
        }


# ── Heterogeneous GCN block ────────────────────────────────────────────────
class HeteroGCNLayer(nn.Module):
    """
    One layer of message passing across all edge types simultaneously.
    Uses SAGEConv per relation (mean aggregation — fast and effective).
    """
    def __init__(self, hidden_dim: int, metadata):
        super().__init__()
        node_types, edge_types = metadata
        self.conv = HeteroConv(
            {
                et: SAGEConv(hidden_dim, hidden_dim, normalize=True)
                for et in edge_types
            },
            aggr="sum"
        )
        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(hidden_dim) for nt in node_types
        })

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        return {
            ntype: F.relu(self.norms[ntype](h))
            for ntype, h in out.items()
            if ntype in self.norms
        }


# ── Full GNN encoder ───────────────────────────────────────────────────────
class DrugRepurposingGNN(nn.Module):
    """
    Multi-layer heterogeneous GNN that produces embeddings for all nodes.

    Args:
        in_dims    : dict of {node_type: input_feature_dim}
        hidden_dim : hidden/embedding dimension (default 256)
        num_layers : number of GCN layers (default 3)
        dropout    : dropout rate (default 0.3)
        metadata   : (node_types, edge_types) from HeteroData
    """
    def __init__(self, in_dims, hidden_dim=256, num_layers=3,
                 dropout=0.3, metadata=None):
        super().__init__()
        self.dropout = dropout

        # 1. Project each node type to hidden_dim
        self.projector = NodeProjector(in_dims, hidden_dim)

        # 2. Stack GCN layers
        self.layers = nn.ModuleList([
            HeteroGCNLayer(hidden_dim, metadata)
            for _ in range(num_layers)
        ])

        # 3. Final linear to produce embeddings
        self.out_proj = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, hidden_dim)
            for nt in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        # Project inputs
        h = self.projector(x_dict)

        # GCN layers with skip connections
        for layer in self.layers:
            h_new = layer(h, edge_index_dict)
            # Residual connection (same dim)
            h = {nt: h_new[nt] + h[nt] if nt in h_new else h[nt]
                 for nt in h}
            h = {nt: F.dropout(v, p=self.dropout, training=self.training)
                 for nt, v in h.items()}

        # Final projection
        h = {nt: self.out_proj[nt](v)
             for nt, v in h.items() if nt in self.out_proj}

        return h


# ── Link predictor ─────────────────────────────────────────────────────────
class LinkPredictor(nn.Module):
    """
    Scores (drug, disease) pairs.
    Takes embeddings of both nodes, concatenates, passes through MLP.

    Score ≈ probability that drug treats disease.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, drug_emb, disease_emb):
        """
        Args:
            drug_emb    : (N, hidden_dim)
            disease_emb : (N, hidden_dim)
        Returns:
            scores      : (N,) raw logits
        """
        pair = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.mlp(pair).squeeze(-1)


# ── Full model wrapper ────────────────────────────────────────────────────
class DrugRepurposingModel(nn.Module):
    def __init__(self, in_dims, hidden_dim=256, num_layers=3,
                 dropout=0.3, metadata=None):
        super().__init__()
        self.encoder   = DrugRepurposingGNN(in_dims, hidden_dim,
                                            num_layers, dropout, metadata)
        self.predictor = LinkPredictor(hidden_dim)

    def forward(self, data: HeteroData,
                drug_idx: torch.Tensor,
                disease_idx: torch.Tensor):
        """
        Args:
            data        : HeteroData graph
            drug_idx    : (N,) indices of Compound nodes
            disease_idx : (N,) indices of Disease nodes
        Returns:
            logits      : (N,) link scores
        """
        embeddings = self.encoder(data.x_dict, data.edge_index_dict)
        drug_emb    = embeddings["Compound"][drug_idx]
        disease_emb = embeddings["Disease"][disease_idx]
        return self.predictor(drug_emb, disease_emb)

    def get_all_embeddings(self, data: HeteroData):
        """Returns full embedding dict for all node types."""
        return self.encoder(data.x_dict, data.edge_index_dict)


# ── Negative sampling ─────────────────────────────────────────────────────
def negative_sample(pos_drugs, pos_diseases, n_diseases, neg_ratio=1):
    """
    For each positive (drug, disease) pair, sample neg_ratio random
    negative disease nodes that the drug does NOT treat.

    Returns (neg_drugs, neg_diseases).
    """
    n = len(pos_drugs) * neg_ratio
    neg_diseases = torch.randint(0, n_diseases, (n,))
    neg_drugs    = pos_drugs.repeat(neg_ratio)
    return neg_drugs, neg_diseases


# ── Model factory ─────────────────────────────────────────────────────────
def build_model(hetero_data: HeteroData, hidden_dim=256,
                num_layers=3, dropout=0.3):
    """Convenience function — builds the full model from a HeteroData object."""
    in_dims = {nt: hetero_data[nt].x.shape[1]
               for nt in hetero_data.node_types}
    metadata = (hetero_data.node_types, hetero_data.edge_types)

    model = DrugRepurposingModel(
        in_dims    = in_dims,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        dropout    = dropout,
        metadata   = metadata,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel built — {total_params:,} trainable parameters")
    print(f"  Hidden dim : {hidden_dim}")
    print(f"  GCN layers : {num_layers}")
    return model


if __name__ == "__main__":
    import os
    data_path = os.path.join("data", "graph_data.pt")
    print(f"Loading graph from {data_path}...")
    checkpoint  = torch.load(data_path)
    hetero_data = checkpoint["hetero_data"]

    model = build_model(hetero_data)
    print(model)
    print("\n✓ Model defined. Run 4_train.py next.")
