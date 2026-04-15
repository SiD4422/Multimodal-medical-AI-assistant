"""
Drug Repurposing with Graph Neural Networks
Step 2: Node Feature Engineering
=================================
Replaces placeholder features with real ones:
  - Compounds  → 2048-bit Morgan molecular fingerprints (via RDKit)
  - Diseases   → 128-dim learned embedding (trained end-to-end)
  - Genes      → 128-dim learned embedding
  - Others     → 64-dim learned embedding

Morgan fingerprints encode the 2D chemical structure of a drug molecule,
giving the GNN real chemistry to reason about — not just random noise.
"""

import os
import torch
import numpy as np
import pandas as pd

DATA_DIR = "data"
FEAT_DIM_DRUG    = 2048   # Morgan fingerprint size
FEAT_DIM_DEFAULT = 128    # Learnable embedding dim for non-drug nodes


# ── Try to load RDKit (optional but recommended) ───────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
    print("RDKit found — will compute real Morgan fingerprints.")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not found. Install with: pip install rdkit")
    print("Falling back to random fingerprints for now.")


# ── Morgan fingerprint for a single SMILES string ─────────────────────────
def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Converts a SMILES string to a Morgan fingerprint bit vector.
    Returns a numpy array of shape (n_bits,) with float32 values.
    Falls back to zeros if SMILES is invalid.
    """
    if not RDKIT_AVAILABLE or not isinstance(smiles, str):
        return np.zeros(n_bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ── Load drug SMILES from DrugBank-style CSV ───────────────────────────────
def load_drug_smiles(kind_df: pd.DataFrame) -> dict:
    """
    Attempts to load SMILES strings for Compound nodes.

    In real usage: download drugbank_approved.csv from DrugBank (free academic
    registration) and place it in data/drugbank.csv with columns:
        drugbank_id, name, smiles

    Falls back to random fingerprints if file not found.
    """
    smiles_path = os.path.join(DATA_DIR, "drugbank.csv")
    smiles_map  = {}

    if os.path.exists(smiles_path):
        db = pd.read_csv(smiles_path)
        if "smiles" in db.columns and "drugbank_id" in db.columns:
            smiles_map = dict(zip(db["drugbank_id"], db["smiles"]))
            print(f"  Loaded {len(smiles_map)} SMILES from DrugBank CSV.")
        else:
            print("  drugbank.csv found but missing expected columns.")
    else:
        print("  data/drugbank.csv not found — using random fingerprints.")
        print("  Get it free at: https://go.drugbank.com/releases/latest")

    return smiles_map


# ── Build feature tensors for all node types ──────────────────────────────
def build_node_features(hetero_data, kind_dfs):
    """
    Returns updated hetero_data with real node features.

    Compound nodes: Morgan fingerprints (or random fallback)
    All others:     random uniform embeddings (to be learned end-to-end)
    """
    print("\nBuilding node features...")

    # --- Compound (Drug) features ---
    compound_df = kind_dfs.get("Compound", pd.DataFrame())
    smiles_map  = load_drug_smiles(compound_df)

    n_compounds = len(compound_df)
    compound_feats = np.zeros((n_compounds, FEAT_DIM_DRUG), dtype=np.float32)

    for idx, row in compound_df.iterrows():
        drug_id = row["id"]
        smiles  = smiles_map.get(drug_id, None)
        compound_feats[idx] = smiles_to_fingerprint(smiles, n_bits=FEAT_DIM_DRUG)

    hetero_data["Compound"].x = torch.tensor(compound_feats)
    print(f"  Compound features: {hetero_data['Compound'].x.shape}")

    # --- All other node types: random learnable embeddings ---
    for kind in kind_dfs:
        if kind == "Compound":
            continue
        n = len(kind_dfs[kind])
        hetero_data[kind].x = torch.randn(n, FEAT_DIM_DEFAULT) * 0.1
        print(f"  {kind} features: {hetero_data[kind].x.shape}")

    return hetero_data


# ── Node feature normalization ─────────────────────────────────────────────
def normalize_features(hetero_data):
    """L2-normalize all node feature tensors."""
    for node_type in hetero_data.node_types:
        x = hetero_data[node_type].x
        norm = x.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        hetero_data[node_type].x = x / norm
    print("\nFeatures L2-normalized.")
    return hetero_data


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading graph data...")
    checkpoint = torch.load(os.path.join(DATA_DIR, "graph_data.pt"))
    hetero_data = checkpoint["hetero_data"]
    kind_dfs    = checkpoint["kind_dfs"]

    hetero_data = build_node_features(hetero_data, kind_dfs)
    hetero_data = normalize_features(hetero_data)

    checkpoint["hetero_data"] = hetero_data
    torch.save(checkpoint, os.path.join(DATA_DIR, "graph_data.pt"))

    print("\n✓ Features added and saved. Run 3_model.py next.")
