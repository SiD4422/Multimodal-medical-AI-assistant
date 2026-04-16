# Multimodal Medical AI — Diagnostic Screening System

> Three deep learning modules covering Chest, Eye, and Skin diagnosis — fused with an LLM clinical report generator.

---

## What this project does

A unified AI diagnostic system that accepts medical images across 3 domains and produces structured clinical reports:

| Module | Input | Model | Task |
|--------|-------|-------|------|
| Chest  | X-ray image + ECG signal | DenseNet-121 + 1D-CNN Transformer | 14 pathologies + 5 rhythm classes |
| Eye    | Retinal fundus photo | EfficientNet-B4 + B2 | DR grading (0–4) + glaucoma risk |
| Skin   | Lesion photo + age/sex/site | EfficientNet-B4 + MLP fusion | 7-class lesion + malignancy flag |

---

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Get data (free)

| Dataset | How |
|---------|-----|
| NIH ChestX-ray14 | [nih.gov](https://nihcc.app.box.com/v/ChestXray-NIHCC) → `data/chest_xray/` |
| PTB-XL ECG | [physionet.org/ptb-xl](https://physionet.org/content/ptb-xl/) → `data/ptbxl/` |
| APTOS 2019 | [kaggle.com/c/aptos2019](https://www.kaggle.com/c/aptos2019-blindness-detection) → `data/aptos/` |
| RIM-ONE DL | [rimone.retinaanalysis.org](http://rimone.retinaanalysis.org) → `data/rim_one/` |
| ISIC 2020 | [kaggle.com/c/siim-isic](https://www.kaggle.com/c/siim-isic-melanoma-classification) → `data/isic2020/` |

### 3. Train
```bash
python train_all.py --module all     # all 3 modules
python train_all.py --module chest   # just chest
python train_all.py --module eye     # just eye
python train_all.py --module skin    # just skin
```

### 4. Launch app
```bash
streamlit run app.py
```

### 5. Enable LLM reports (optional)
```bash
export ANTHROPIC_API_KEY=your_key_here
```

---

## Project structure

```
multimodal_medical_ai/
├── config.py             # Central config for all modules
├── utils.py              # Shared transforms, GradCAM, metrics
├── module1_chest.py      # X-ray DenseNet + ECG 1D-CNN + fusion
├── module2_eye.py        # DR EfficientNet-B4 + Glaucoma B2 + fusion
├── module3_skin.py       # Skin EfficientNet-B4 + metadata MLP
├── report_generator.py   # Claude API → structured clinical reports
├── train_all.py          # Master training script
├── app.py                # Streamlit app (all 3 modules)
├── requirements.txt
├── data/                 # Downloaded datasets (gitignored)
└── checkpoints/          # Saved model weights
```

---

## Target metrics

| Module | Metric | Target |
|--------|--------|--------|
| X-ray | Mean AUC (14 labels) | ≥ 0.85 |
| ECG | 5-class accuracy | ≥ 0.88 |
| DR grading | Quadratic Weighted Kappa | ≥ 0.85 |
| Glaucoma | ROC-AUC | ≥ 0.92 |
| Skin | Macro AUC (7-class) | ≥ 0.90 |

---

## Key technical highlights

1. **Ben Graham preprocessing** on fundus images (removes vignette, boosts contrast)
2. **Two-phase DR training**: regression first (ordinal-aware), then classification
3. **Metadata fusion** for skin: image + patient demographics = better malignancy detection
4. **Weighted sampling** for imbalanced skin lesion classes
5. **Grad-CAM overlays** on every image prediction
6. **Missing modality handling** — each module works independently
7. **LLM report generation** — predictions → structured clinical language via Claude API

---

## Deployment on Hugging Face Spaces

```
1. Create new Space → SDK: Streamlit
2. Push this repo
3. Add ANTHROPIC_API_KEY as a Space secret
4. Add requirements.txt
5. Space auto-runs: streamlit run app.py
```

---

## References

- Wang et al. (2017) — ChestX-ray8, NIH
- Wagner et al. (2020) — PTB-XL ECG dataset
- APTOS 2019 Blindness Detection, Kaggle
- Tschandl et al. (2018) — HAM10000 dermatoscopy dataset
- Tan & Le (2019) — EfficientNet
- Huang et al. (2017) — DenseNet (CheXNet)
