# Drug Repurposing with Graph Neural Networks

> Predicting new uses for existing drugs using a Relational GCN trained on the Hetionet biomedical knowledge graph.

---

## What this project does

Models drugs, diseases, genes, and proteins as nodes in a **biomedical knowledge graph** (Hetionet). A heterogeneous GNN learns to predict which approved drugs could treat diseases they weren't originally designed for — essentially a recommendation system backed by real biology.

---

## Architecture

```
Hetionet (47K nodes, 2.25M edges)
  └─ Node features (Morgan fingerprints for drugs, learned embeddings for others)
       └─ R-GCN encoder (3 heterogeneous GCN layers with residual connections)
            └─ Link predictor MLP
                 └─ Ranked drug list per disease + biological explanation paths
```

---

## Quick start

### 1. Install dependencies

```bash
pip install torch torch-geometric
pip install -r requirements.txt
```

> Note: Install PyTorch first, then torch-geometric. See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 2. Run in order

```bash
python 1_data_setup.py     # Download Hetionet, build graph
python 2_features.py       # Add molecular fingerprints
python 3_model.py          # Verify model architecture
python 4_train.py          # Train the GNN (200 epochs)
python 5_explain.py        # Generate explanation paths
streamlit run 6_app.py     # Launch interactive web app
uvicorn 7_api:app --reload # (Optional) REST API
```

---

## Datasets

| Dataset | How to get it |
|---------|--------------|
| **Hetionet** | Auto-downloaded by `1_data_setup.py` |
| **DrugBank** | Free academic registration at https://go.drugbank.com — download `drugbank_approved.csv` to `data/` |
| **DisGeNET** | https://www.disgenet.org — for validation |
| **ClinicalTrials.gov** | https://clinicaltrials.gov — manual cross-check |

---

## Project structure

```
drug_repurposing_gnn/
├── 1_data_setup.py     # Download + parse Hetionet → HeteroData
├── 2_features.py       # Morgan fingerprints + node embeddings
├── 3_model.py          # R-GCN encoder + link predictor
├── 4_train.py          # Training loop + evaluation (AUC, Hits@K, MRR)
├── 5_explain.py        # GNNExplainer + biological path analysis
├── 6_app.py            # Streamlit interactive demo
├── 7_api.py            # FastAPI REST API
├── requirements.txt
├── data/               # Auto-created, stores downloaded data
└── checkpoints/        # Saved model + training history
```

---

## Key metrics to target

| Metric | Baseline (GCN) | Target (R-GCN) |
|--------|----------------|----------------|
| ROC-AUC | ~0.85 | ~0.92+ |
| Hits@10 | ~0.30 | ~0.50+ |
| Hits@50 | ~0.55 | ~0.72+ |
| MRR | ~0.15 | ~0.28+ |

---

## What makes this stand out

1. **Heterogeneous graph reasoning** — not flat tabular ML
2. **Real molecular features** — Morgan fingerprints from RDKit
3. **Biological interpretability** — shared gene paths explain every prediction
4. **Full deployment** — Streamlit app + FastAPI, ready for Hugging Face Spaces
5. **Validation hook** — cross-check predictions against ClinicalTrials.gov

---

## Deployment on Hugging Face Spaces

1. Create a new Space (Streamlit SDK)
2. Push this repo
3. Add `requirements.txt`
4. Done — Spaces auto-runs `streamlit run 6_app.py`

---

## References

- Hetionet: https://het.io
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- Himmelstein et al. (2017) — "Systematic integration of biomedical knowledge" *eLife*
- Schlichtkrull et al. (2018) — "Modeling Relational Data with Graph Convolutional Networks"
