"""
Multimodal Medical AI — Module 1: Chest
=========================================
Covers:
  - Chest X-ray: DenseNet-121, 14-class multi-label classification
  - ECG: 1D-CNN Transformer, 5-class arrhythmia classification
  - Late fusion: combines both embeddings for joint chest diagnosis

Datasets:
  - NIH ChestX-ray14  →  data/chest_xray/
  - PTB-XL ECG        →  data/ptbxl/
"""

import os
import ast
import wfdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from config import DEVICE, CHEST_CFG as CFG
from utils  import (get_train_transforms, get_val_transforms,
                    ImageDataset, normalise_ecg, compute_multilabel_auc,
                    GradCAM)


# ════════════════════════════════════════════════════════════════════════════
# PART A — CHEST X-RAY
# ════════════════════════════════════════════════════════════════════════════

# ── Dataset ────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 dataset.

    Expected layout:
        data/chest_xray/
            images/          ← all .png images
            Data_Entry_2017.csv
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        from utils import load_image_rgb
        img = load_image_rgb(img_path)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[CFG.xray_labels].values.astype(float),
                             dtype=torch.float32)
        return img, label


def load_xray_dataframes(data_dir: str):
    """
    Parses Data_Entry_2017.csv into train/val/test DataFrames
    with one-hot columns for each of the 14 disease labels.
    """
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df       = pd.read_csv(csv_path)

    for label in CFG.xray_labels:
        df[label] = df["Finding Labels"].apply(
            lambda x: 1.0 if label in x else 0.0)

    # Official train/test split file (download separately)
    split_file = os.path.join(data_dir, "train_val_list.txt")
    if os.path.exists(split_file):
        with open(split_file) as f:
            train_files = set(f.read().splitlines())
        train_df = df[df["Image Index"].isin(train_files)]
        test_df  = df[~df["Image Index"].isin(train_files)]
    else:
        train_df, test_df = train_test_split(df, test_size=0.15,
                                             random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1,
                                         random_state=42)
    print(f"X-ray split — train:{len(train_df)} val:{len(val_df)} "
          f"test:{len(test_df)}")
    return train_df, val_df, test_df


# ── Model ──────────────────────────────────────────────────────────────────

class XrayEncoder(nn.Module):
    """
    DenseNet-121 backbone producing a 512-dim embedding.
    Pretrained on ImageNet, fine-tuned on chest X-rays.
    """
    def __init__(self, embed_dim: int = 512, dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        base             = models.densenet121(pretrained=pretrained)
        self.features    = base.features
        self.pool        = nn.AdaptiveAvgPool2d(1)
        feat_dim         = base.classifier.in_features   # 1024
        self.projector   = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
        )
        self.classifier  = nn.Linear(embed_dim, CFG.xray_num_classes)

    def forward(self, x, return_embed=False):
        feat  = self.features(x)
        feat  = F.relu(feat, inplace=True)
        feat  = self.pool(feat).flatten(1)
        embed = self.projector(feat)
        if return_embed:
            return embed
        return self.classifier(embed)


# ── Training ───────────────────────────────────────────────────────────────

def train_xray(data_dir: str = CFG.xray_data_dir):
    img_dir  = os.path.join(data_dir, "images")
    train_df, val_df, test_df = load_xray_dataframes(data_dir)

    t_ds = ChestXrayDataset(train_df, img_dir,
                            get_train_transforms(CFG.xray_img_size))
    v_ds = ChestXrayDataset(val_df,   img_dir,
                            get_val_transforms(CFG.xray_img_size))

    t_dl = DataLoader(t_ds, batch_size=CFG.batch_size,
                      shuffle=True,  num_workers=4)
    v_dl = DataLoader(v_ds, batch_size=CFG.batch_size,
                      shuffle=False, num_workers=4)

    model     = XrayEncoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=CFG.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in t_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 5 == 0:
            auc = evaluate_xray(model, v_dl)
            print(f"Epoch {epoch:3d} | loss={total_loss/len(t_dl):.4f} "
                  f"| val mean-AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(),
                           "checkpoints/xray_best.pt")
                print("  ✓ Saved best X-ray model")

    return model


@torch.no_grad()
def evaluate_xray(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        preds = torch.sigmoid(model(imgs)).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_multilabel_auc(y_true, y_pred, CFG.xray_labels)
    return metrics["mean_auc"]


# ════════════════════════════════════════════════════════════════════════════
# PART B — ECG (PTB-XL)
# ════════════════════════════════════════════════════════════════════════════

# ── Dataset ────────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PTB-XL 12-lead ECG dataset.

    Expected layout:
        data/ptbxl/
            ptbxl_database.csv
            records100/          ← 100Hz ECG records
            scp_statements.csv
    """
    def __init__(self, df: pd.DataFrame, data_dir: str,
                 sampling_rate: int = 100):
        self.df           = df.reset_index(drop=True)
        self.data_dir     = data_dir
        self.sampling_rate= sampling_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        filename = row["filename_lr"] if self.sampling_rate == 100 \
                   else row["filename_hr"]
        path     = os.path.join(self.data_dir, filename)
        signal, _= wfdb.rdsamp(path)            # (1000, 12)
        signal   = normalise_ecg(signal.T)       # (12, 1000)
        signal   = torch.tensor(signal, dtype=torch.float32)
        label    = torch.tensor(int(row["label"]), dtype=torch.long)
        return signal, label


def load_ptbxl_dataframes(data_dir: str):
    """
    Parses PTB-XL CSV and assigns superclass labels:
    NORM=0, MI=1, STTC=2, CD=3, HYP=4
    """
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    df       = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_path = os.path.join(data_dir, "scp_statements.csv")
    scp      = pd.read_csv(scp_path, index_col=0)
    scp      = scp[scp["diagnostic"] == 1]

    superclass_map = {"NORM":0,"MI":1,"STTC":2,"CD":3,"HYP":4}

    def get_label(codes):
        for code in codes:
            if code in scp.index:
                sc = scp.loc[code, "diagnostic_class"]
                if sc in superclass_map:
                    return superclass_map[sc]
        return -1

    df["label"] = df["scp_codes"].apply(
        lambda x: get_label(list(x.keys())))
    df = df[df["label"] >= 0]

    # PTB-XL official split: fold 10 = test, fold 9 = val
    test_df  = df[df["strat_fold"] == 10]
    val_df   = df[df["strat_fold"] == 9]
    train_df = df[~df["strat_fold"].isin([9, 10])]
    print(f"ECG split — train:{len(train_df)} val:{len(val_df)} "
          f"test:{len(test_df)}")
    return train_df, val_df, test_df


# ── Model ──────────────────────────────────────────────────────────────────

class ECGEncoder(nn.Module):
    """
    1D-CNN + Transformer encoder for 12-lead ECG.
    Input: (B, 12, 1000)
    Output: 256-dim embedding
    """
    def __init__(self, n_leads: int = 12, seq_len: int = 1000,
                 embed_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # 1D-CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_leads, 64,  kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(64,      128, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128,     256, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(256,     256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256,     embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),   # → (B, embed_dim, 32)
        )
        # Transformer over time steps
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8,
            dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, CFG.ecg_num_classes)
        self.embed_dim  = embed_dim

    def forward(self, x, return_embed=False):
        feat  = self.cnn(x)                      # (B, D, 32)
        feat  = feat.permute(0, 2, 1)            # (B, 32, D)
        feat  = self.transformer(feat)           # (B, 32, D)
        embed = feat.mean(dim=1)                 # (B, D)
        if return_embed:
            return embed
        return self.classifier(embed)


# ── Training ───────────────────────────────────────────────────────────────

def train_ecg(data_dir: str = CFG.ecg_data_dir):
    train_df, val_df, _ = load_ptbxl_dataframes(data_dir)

    t_ds = PTBXLDataset(train_df, data_dir)
    v_ds = PTBXLDataset(val_df,   data_dir)
    t_dl = DataLoader(t_ds, batch_size=CFG.batch_size,
                      shuffle=True,  num_workers=4)
    v_dl = DataLoader(v_ds, batch_size=CFG.batch_size,
                      shuffle=False, num_workers=4)

    model     = ECGEncoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        for sigs, labels in t_dl:
            sigs, labels = sigs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(sigs), labels)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            acc = evaluate_ecg(model, v_dl)
            print(f"ECG Epoch {epoch:3d} | val acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "checkpoints/ecg_best.pt")

    return model


@torch.no_grad()
def evaluate_ecg(model, loader):
    model.eval()
    correct = total = 0
    for sigs, labels in loader:
        sigs   = sigs.to(DEVICE)
        preds  = model(sigs).argmax(dim=1).cpu()
        correct += (preds == labels).sum().item()
        total   += len(labels)
    return correct / total


# ════════════════════════════════════════════════════════════════════════════
# PART C — CHEST FUSION (X-ray + ECG)
# ════════════════════════════════════════════════════════════════════════════

class ChestFusionModel(nn.Module):
    """
    Fuses X-ray embedding (512-dim) + ECG embedding (256-dim)
    with cross-attention → joint 14-class + 5-class heads.
    """
    def __init__(self):
        super().__init__()
        self.xray_enc = XrayEncoder(embed_dim=512)
        self.ecg_enc  = ECGEncoder(embed_dim=256)

        # Project ECG to same dim as X-ray
        self.ecg_proj = nn.Linear(256, 512)

        # Cross-modal attention: X-ray queries ECG context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True)

        fusion_dim = 512 + 512
        self.xray_head = nn.Linear(fusion_dim, CFG.xray_num_classes)
        self.ecg_head  = nn.Linear(fusion_dim, CFG.ecg_num_classes)
        self.urgency   = nn.Linear(fusion_dim, 3)   # low/med/high

    def forward(self, xray_img, ecg_signal):
        x_emb = self.xray_enc(xray_img, return_embed=True)   # (B, 512)
        e_emb = self.ecg_enc (ecg_signal, return_embed=True) # (B, 256)
        e_emb = self.ecg_proj(e_emb)                          # (B, 512)

        # Cross attention: x-ray attends to ECG
        x_seq    = x_emb.unsqueeze(1)     # (B, 1, 512)
        e_seq    = e_emb.unsqueeze(1)     # (B, 1, 512)
        attn_out, _ = self.cross_attn(x_seq, e_seq, e_seq)
        attn_out = attn_out.squeeze(1)    # (B, 512)

        fused = torch.cat([x_emb, attn_out], dim=-1)  # (B, 1024)

        return {
            "xray_logits":    self.xray_head(fused),
            "ecg_logits":     self.ecg_head(fused),
            "urgency_logits": self.urgency(fused),
        }


if __name__ == "__main__":
    print("Module 1 — Chest")
    print("  train_xray()  → trains X-ray DenseNet-121")
    print("  train_ecg()   → trains ECG 1D-CNN Transformer")
    print("  ChestFusionModel → fused inference")
    print("\nRun train_xray() and train_ecg() first, then fuse.")
