"""
Drug Repurposing with Graph Neural Networks
Step 4: Training Loop + Evaluation
=====================================
Trains the R-GCN link predictor and evaluates using:
  - ROC-AUC   : area under ROC curve
  - Hits@10   : fraction of true drugs in top-10 ranked candidates
  - Hits@50   : fraction of true drugs in top-50 ranked candidates
  - MRR       : mean reciprocal rank

Saves best checkpoint and training curves.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from model import build_model, negative_sample

# ── Config ────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = "data"
CKPT_DIR   = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS      = 200
LR          = 3e-4
HIDDEN_DIM  = 256
NUM_LAYERS  = 3
DROPOUT     = 0.3
NEG_RATIO   = 3        # negative samples per positive
PATIENCE    = 20       # early stopping patience


# ── Data loaders ──────────────────────────────────────────────────────────
def load_data():
    ckpt = torch.load(os.path.join(DATA_DIR, "graph_data.pt"),
                      map_location=DEVICE)
    data    = ckpt["hetero_data"].to(DEVICE)
    splits  = ckpt["splits"]
    return data, splits


def to_tensors(arr_tuple, device):
    return (torch.tensor(arr_tuple[0], dtype=torch.long, device=device),
            torch.tensor(arr_tuple[1], dtype=torch.long, device=device))


# ── Loss: binary cross entropy with positive weighting ───────────────────
def compute_loss(model, data, pos_drugs, pos_diseases, n_diseases):
    # Positive scores
    pos_scores = model(data, pos_drugs, pos_diseases)

    # Negative samples
    neg_drugs, neg_diseases = negative_sample(
        pos_drugs, pos_diseases, n_diseases, neg_ratio=NEG_RATIO)
    neg_scores = model(data, neg_drugs, neg_diseases)

    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    return F.binary_cross_entropy_with_logits(scores, labels)


# ── Evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, data, pos_drugs, pos_diseases, n_diseases, split_name=""):
    model.eval()

    pos_scores = torch.sigmoid(model(data, pos_drugs, pos_diseases)).cpu().numpy()

    # Sample negatives (fixed seed for reproducibility)
    rng = np.random.default_rng(0)
    neg_d_idx = torch.tensor(
        rng.integers(0, n_diseases, len(pos_drugs) * NEG_RATIO),
        dtype=torch.long, device=DEVICE)
    neg_drug_idx = pos_drugs.repeat(NEG_RATIO)
    neg_scores = torch.sigmoid(
        model(data, neg_drug_idx, neg_d_idx)).cpu().numpy()

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([
        np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    auc = roc_auc_score(all_labels, all_scores)
    ap  = average_precision_score(all_labels, all_scores)

    # Hits@K — for each positive pair, rank among all diseases
    hits10 = hits50 = mrr = 0
    sample_size = min(200, len(pos_drugs))   # sample for speed
    sample_idx  = rng.choice(len(pos_drugs), sample_size, replace=False)

    all_disease_idx = torch.arange(n_diseases, device=DEVICE)

    for i in sample_idx:
        d_idx = pos_drugs[i].unsqueeze(0).expand(n_diseases)
        all_scores_i = torch.sigmoid(
            model(data, d_idx, all_disease_idx)).cpu().numpy()
        true_score = all_scores_i[pos_diseases[i].item()]
        rank = (all_scores_i >= true_score).sum()
        hits10 += int(rank <= 10)
        hits50 += int(rank <= 50)
        mrr    += 1.0 / rank

    hits10 /= sample_size
    hits50 /= sample_size
    mrr    /= sample_size

    print(f"  [{split_name}] AUC={auc:.4f}  AP={ap:.4f}  "
          f"Hits@10={hits10:.3f}  Hits@50={hits50:.3f}  MRR={mrr:.4f}")

    return {"auc": auc, "ap": ap, "hits10": hits10,
            "hits50": hits50, "mrr": mrr}


# ── Training loop ─────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")
    data, splits = load_data()
    n_diseases   = data["Disease"].num_nodes

    train_drugs, train_dis = to_tensors(splits["train"], DEVICE)
    val_drugs,   val_dis   = to_tensors(splits["val"],   DEVICE)
    test_drugs,  test_dis  = to_tensors(splits["test"],  DEVICE)

    model = build_model(data, hidden_dim=HIDDEN_DIM,
                        num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val_auc = 0.0
    no_improve   = 0
    history      = []

    print(f"\nTraining for up to {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0   = time.time()
        loss = compute_loss(model, data, train_drugs, train_dis, n_diseases)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} "
                  f"| time={time.time()-t0:.1f}s")
            metrics = evaluate(model, data, val_drugs, val_dis,
                               n_diseases, "val")
            history.append({"epoch": epoch, "loss": loss.item(), **metrics})

            if metrics["auc"] > best_val_auc:
                best_val_auc = metrics["auc"]
                no_improve   = 0
                torch.save({
                    "epoch":   epoch,
                    "model":   model.state_dict(),
                    "metrics": metrics,
                }, os.path.join(CKPT_DIR, "best_model.pt"))
                print(f"  ✓ New best model saved (val AUC={best_val_auc:.4f})")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

    # ── Final test evaluation ──────────────────────────────────────────────
    print("\n── Test evaluation ──")
    best = torch.load(os.path.join(CKPT_DIR, "best_model.pt"),
                      map_location=DEVICE)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, data, test_drugs, test_dis,
                            n_diseases, "test")

    with open(os.path.join(CKPT_DIR, "history.json"), "w") as f:
        json.dump({"history": history, "test": test_metrics}, f, indent=2)

    print(f"\n✓ Done. Best val AUC: {best_val_auc:.4f}")
    print(f"  Test metrics saved to checkpoints/history.json")
    print("  Run 5_explain.py next for GNNExplainer.")

    return model, data, test_metrics


if __name__ == "__main__":
    train()
