"""
Multimodal Medical AI — Shared Config
======================================
Central config for all 3 diagnostic modules.
Edit paths and hyperparameters here before running.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ── Directory layout ───────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
LOG_DIR    = os.path.join(ROOT, "logs")

for d in [DATA_DIR, CKPT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Module 1 — Chest (X-ray + ECG) ────────────────────────────────────────
@dataclass
class ChestConfig:
    # X-ray
    xray_data_dir:   str   = os.path.join(DATA_DIR, "chest_xray")
    xray_img_size:   int   = 224
    xray_backbone:   str   = "densenet121"       # or "efficientnet_b3"
    xray_num_classes:int   = 14
    xray_labels: List[str] = field(default_factory=lambda: [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ])

    # ECG
    ecg_data_dir:    str   = os.path.join(DATA_DIR, "ptbxl")
    ecg_seq_len:     int   = 1000              # 10s @ 100Hz
    ecg_leads:       int   = 12
    ecg_num_classes: int   = 5                 # NORM, MI, STTC, CD, HYP
    ecg_labels: List[str]  = field(default_factory=lambda: [
        "NORM", "MI", "STTC", "CD", "HYP"
    ])

    # Training
    batch_size:      int   = 32
    epochs:          int   = 50
    lr:              float = 1e-4
    dropout:         float = 0.3
    fusion_dim:      int   = 512


# ── Module 2 — Eye (Retina + Glaucoma) ────────────────────────────────────
@dataclass
class EyeConfig:
    # Diabetic retinopathy
    dr_data_dir:     str   = os.path.join(DATA_DIR, "aptos")
    dr_img_size:     int   = 380
    dr_backbone:     str   = "efficientnet_b4"
    dr_num_classes:  int   = 5                 # 0=No DR … 4=Proliferative
    dr_labels: List[str]   = field(default_factory=lambda: [
        "No DR", "Mild", "Moderate", "Severe", "Proliferative DR"
    ])

    # Glaucoma
    glaucoma_data_dir: str = os.path.join(DATA_DIR, "rim_one")
    glaucoma_img_size: int = 224
    glaucoma_backbone: str = "efficientnet_b2"

    # Training
    batch_size:      int   = 16
    epochs:          int   = 40
    lr:              float = 3e-5
    dropout:         float = 0.4
    img_size:        int   = 380


# ── Module 3 — Skin (Lesion + metadata) ───────────────────────────────────
@dataclass
class SkinConfig:
    data_dir:        str   = os.path.join(DATA_DIR, "isic2020")
    img_size:        int   = 256
    backbone:        str   = "efficientnet_b4"
    num_classes:     int   = 7
    labels: List[str]      = field(default_factory=lambda: [
        "Melanoma", "Melanocytic nevus", "Basal cell carcinoma",
        "Actinic keratosis", "Benign keratosis",
        "Dermatofibroma", "Vascular lesion",
    ])
    meta_features:   int   = 6      # age, sex, location (one-hot)

    # Training
    batch_size:      int   = 32
    epochs:          int   = 40
    lr:              float = 3e-5
    dropout:         float = 0.4
    use_tta:         bool  = True   # test-time augmentation


# ── LLM report generator ───────────────────────────────────────────────────
@dataclass
class ReportConfig:
    # Set ANTHROPIC_API_KEY env variable, or paste key here (not recommended)
    api_key:         str   = os.environ.get("ANTHROPIC_API_KEY", "")
    model:           str   = "claude-opus-4-20250514"
    max_tokens:      int   = 600


# ── Singleton instances ────────────────────────────────────────────────────
CHEST_CFG  = ChestConfig()
EYE_CFG    = EyeConfig()
SKIN_CFG   = SkinConfig()
REPORT_CFG = ReportConfig()
