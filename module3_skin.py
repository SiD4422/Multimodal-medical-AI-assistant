"""
Multimodal Medical AI — Module 3: Skin
=========================================
Covers:
  - Skin lesion classification: EfficientNet-B4, 7 classes
  - Patient metadata fusion: age, sex, anatomical site
  - Test-Time Augmentation (TTA) for robust inference
  - Class activation mapping for lesion localisation

Dataset:
  - HAM10000 / ISIC 2020  →  data/isic2020/
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

import timm

from config import DEVICE, SKIN_CFG as CFG
from utils  import (get_train_transforms, get_val_transforms,
                    get_tta_transforms, load_image_rgb,
                    ImageMetaDataset, make_weighted_sampler, GradCAM)


# ── Metadata preprocessing ─────────────────────────────────────────────────

SITE_COLS = [
    "head/neck", "upper extremity", "lower extremity",
    "torso", "palms/soles", "oral/genital", "unknown",
]

def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes patient metadata into numeric features:
      - age_approx  → normalised 0–1
      - sex         → 0/1 binary
      - anatom_site → 7-dim one-hot
    """
    df = df.copy()

    # Age normalisation
    df["age_norm"] = df["age_approx"].fillna(df["age_approx"].median()) / 90.0

    # Sex encoding
    df["sex_enc"] = (df["sex"].fillna("unknown")
                              .str.lower()
                              .map({"male": 1.0, "female": 0.0})
                              .fillna(0.5))

    # Anatomical site one-hot
    site_col = "anatom_site_general_challenge"
    df[site_col] = df[site_col].fillna("unknown").str.lower()
    for site in SITE_COLS:
        df[f"site_{site.replace('/','-')}"] = (
            df[site_col] == site).astype(float)

    return df


META_COLS = (["age_norm", "sex_enc"] +
             [f"site_{s.replace('/','-')}" for s in SITE_COLS])


# ── Dataset ────────────────────────────────────────────────────────────────

class ISICDataset(ImageMetaDataset):
    """
    ISIC 2020 / HAM10000 skin lesion dataset.

    Expected layout:
        data/isic2020/
            train/            ← JPEG images
            train.csv         ← image_name, diagnosis, age_approx, sex,
                                 anatom_site_general_challenge
    """
    pass   # inherits everything from ImageMetaDataset


def load_isic_dataframes(data_dir: str):
    csv_path = os.path.join(data_dir, "train.csv")
    df       = pd.read_csv(csv_path)

    # HAM10000 uses 'dx'; ISIC 2020 uses 'diagnosis'
    label_col = "diagnosis" if "diagnosis" in df.columns else "dx"
    df["diagnosis_str"] = df[label_col].str.lower()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["diagnosis_str"])
    label_map   = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label map: {label_map}")

    df["image_path"] = df["image_name"].apply(
        lambda x: os.path.join(data_dir, "train",
                               x if x.endswith(".jpg") else x + ".jpg"))

    df = preprocess_metadata(df)

    # Stratified 5-fold — use fold 0 as val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
            skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    train_df = df[df["fold"] != 0].reset_index(drop=True)
    val_df   = df[df["fold"] == 0].reset_index(drop=True)
    print(f"Skin split — train:{len(train_df)} val:{len(val_df)}")
    return train_df, val_df, label_map


# ── Model ──────────────────────────────────────────────────────────────────

class SkinEncoder(nn.Module):
    """
    EfficientNet-B4 image encoder + MLP metadata encoder.
    Late fusion: concat image_embed + meta_embed → classifier.

    Image path  → EfficientNet-B4 → 512-dim
    Metadata    → 2-layer MLP     → 128-dim
    Concat      → 640-dim         → 7-class head
    """
    def __init__(self, num_classes: int = 7,
                 meta_dim: int = 9,       # len(META_COLS)
                 img_embed: int = 512,
                 meta_embed: int = 128,
                 dropout: float = 0.4):
        super().__init__()

        # Image branch
        self.backbone  = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=0)
        feat_dim       = self.backbone.num_features   # 1792
        self.img_proj  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, img_embed),
            nn.ReLU(),
        )

        # Metadata branch
        self.meta_mlp  = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, meta_embed),
            nn.ReLU(),
        )

        # Fusion classifier
        fused_dim      = img_embed + meta_embed
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        self.img_embed_dim = img_embed

    def forward(self, img, meta, return_embed=False):
        img_feat  = self.backbone(img)
        img_embed = self.img_proj(img_feat)
        meta_embed= self.meta_mlp(meta)
        fused     = torch.cat([img_embed, meta_embed], dim=-1)
        if return_embed:
            return fused
        return self.classifier(fused)

    def encode_image_only(self, img):
        """For inference when metadata is unavailable."""
        feat  = self.backbone(img)
        embed = self.img_proj(feat)
        dummy = torch.zeros(img.shape[0], 128, device=img.device)
        fused = torch.cat([embed, dummy], dim=-1)
        return self.classifier(fused)


# ── Training ───────────────────────────────────────────────────────────────

def train_skin(data_dir: str = CFG.data_dir):
    train_df, val_df, label_map = load_isic_dataframes(data_dir)
    img_size = CFG.img_size

    t_ds = ISICDataset(train_df, META_COLS,
                       get_train_transforms(img_size))
    v_ds = ISICDataset(val_df,   META_COLS,
                       get_val_transforms(img_size))

    # Weighted sampler for imbalanced classes
    sampler = make_weighted_sampler(train_df["label"].tolist())
    t_dl = DataLoader(t_ds, batch_size=CFG.batch_size,
                      sampler=sampler,  num_workers=4)
    v_dl = DataLoader(v_ds, batch_size=CFG.batch_size,
                      shuffle=False, num_workers=4)

    model     = SkinEncoder(
        num_classes=CFG.num_classes,
        meta_dim=len(META_COLS)).to(DEVICE)

    # Label smoothing helps with dermatology (inter-rater disagreement)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10)

    best_auc  = 0.0
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, meta, labels in t_dl:
            imgs, meta, labels = (imgs.to(DEVICE), meta.to(DEVICE),
                                   labels.to(DEVICE))
            optimizer.zero_grad()
            logits = model(imgs, meta)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 5 == 0:
            auc = evaluate_skin(model, v_dl)
            print(f"Skin Epoch {epoch:3d} | loss={total_loss/len(t_dl):.4f}"
                  f" | val macro-AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(),
                           "checkpoints/skin_best.pt")
                print("  ✓ Saved best skin model")

    return model


@torch.no_grad()
def evaluate_skin(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    for imgs, meta, labels in loader:
        imgs, meta = imgs.to(DEVICE), meta.to(DEVICE)
        probs = F.softmax(model(imgs, meta), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
    y_pred = np.concatenate(all_probs)
    y_true = np.array(all_labels)
    try:
        auc = roc_auc_score(y_true, y_pred,
                            multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    return auc


# ── TTA inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict_skin_tta(model, img_pil: Image.Image,
                     meta_tensor: torch.Tensor,
                     n_augments: int = 5) -> dict:
    """
    TTA skin lesion prediction.
    img_pil      : PIL Image
    meta_tensor  : (1, meta_dim) tensor
    """
    model.eval()
    transforms_list = get_tta_transforms(CFG.img_size, n_augments)
    probs_list = []

    for t in transforms_list:
        img_t = t(img_pil).unsqueeze(0).to(DEVICE)
        meta  = meta_tensor.to(DEVICE)
        probs = F.softmax(model(img_t, meta), dim=1).cpu().numpy()[0]
        probs_list.append(probs)

    avg_probs  = np.mean(probs_list, axis=0)
    pred_class = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_class])

    return {
        "class":      pred_class,
        "label":      CFG.labels[pred_class],
        "confidence": confidence,
        "probs":      {lbl: float(p)
                       for lbl, p in zip(CFG.labels, avg_probs)},
        "is_malignant": CFG.labels[pred_class] in [
            "Melanoma", "Basal cell carcinoma",
            "Actinic keratosis", "Squamous cell carcinoma"],
    }


if __name__ == "__main__":
    print("Module 3 — Skin")
    print("  train_skin()         → trains EfficientNet-B4 + meta fusion")
    print("  predict_skin_tta()   → TTA inference")
    print("  SkinEncoder          → full model class")
