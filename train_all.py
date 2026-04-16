"""
Multimodal Medical AI — Master Training Script
================================================
Trains all 3 modules in sequence.
Run individual modules or all at once.

Usage:
    python train_all.py --module all        # train everything
    python train_all.py --module chest
    python train_all.py --module eye
    python train_all.py --module skin
"""

import argparse
from config import CHEST_CFG, EYE_CFG, SKIN_CFG


def train_chest():
    print("\n" + "="*50)
    print("MODULE 1 — CHEST (X-ray + ECG)")
    print("="*50)
    from module1_chest import train_xray, train_ecg
    print("\n[1/2] Training X-ray DenseNet-121...")
    train_xray(CHEST_CFG.xray_data_dir)
    print("\n[2/2] Training ECG 1D-CNN Transformer...")
    train_ecg(CHEST_CFG.ecg_data_dir)
    print("\n✓ Chest module complete.")


def train_eye():
    print("\n" + "="*50)
    print("MODULE 2 — EYE (DR + Glaucoma)")
    print("="*50)
    from module2_eye import train_dr, train_glaucoma
    print("\n[1/2] Training DR EfficientNet-B4...")
    train_dr(EYE_CFG.dr_data_dir)
    print("\n[2/2] Training Glaucoma EfficientNet-B2...")
    train_glaucoma(EYE_CFG.glaucoma_data_dir)
    print("\n✓ Eye module complete.")


def train_skin():
    print("\n" + "="*50)
    print("MODULE 3 — SKIN (Lesion + Metadata)")
    print("="*50)
    from module3_skin import train_skin as _train_skin
    print("\nTraining Skin EfficientNet-B4 + metadata fusion...")
    _train_skin(SKIN_CFG.data_dir)
    print("\n✓ Skin module complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="all",
                        choices=["all","chest","eye","skin"])
    args = parser.parse_args()

    if args.module in ("all", "chest"):
        train_chest()
    if args.module in ("all", "eye"):
        train_eye()
    if args.module in ("all", "skin"):
        train_skin()

    print("\n\n✓ All requested modules trained.")
    print("Run:  streamlit run app.py  to launch the demo.")
