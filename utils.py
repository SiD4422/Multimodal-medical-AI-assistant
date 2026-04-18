"""
Multimodal Medical AI — Shared Data Utilities
===============================================
Common transforms, augmentation pipelines, and
base dataset helpers used across all 3 modules.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── Image transforms ───────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(img_size: int = 224, n_augments: int = 5):
    """
    Test-Time Augmentation: returns a list of transform pipelines.
    Run the same image through all of them and average predictions.
    """
    base = get_val_transforms(img_size)
    aug  = get_train_transforms(img_size)
    return [base] + [aug] * (n_augments - 1)


# ── Grayscale → RGB converter (for X-ray / fundus images) ─────────────────

def load_image_rgb(path: str) -> Image.Image:
    """Load any image and ensure it's RGB (handles grayscale X-rays)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ── Normalise ECG signals ──────────────────────────────────────────────────

def normalise_ecg(signal: np.ndarray) -> np.ndarray:
    """
    Per-lead z-score normalisation.
    signal shape: (n_leads, seq_len)
    """
    mean = signal.mean(axis=1, keepdims=True)
    std  = signal.std(axis=1, keepdims=True) + 1e-8
    return (signal - mean) / std


# ── Generic image dataset ──────────────────────────────────────────────────

class ImageDataset(Dataset):
    """
    Generic dataset for image-label pairs.
    Expects a DataFrame with columns: [image_path, label]
    label can be int (single-class) or list/array (multi-label).
    """
    def __init__(self, df: pd.DataFrame, transform=None,
                 multilabel: bool = False, num_classes: int = None):
        self.df          = df.reset_index(drop=True)
        self.transform   = transform
        self.multilabel  = multilabel
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = load_image_rgb(row["image_path"])
        if self.transform:
            img = self.transform(img)

        if self.multilabel:
            label = torch.tensor(row["label"], dtype=torch.float32)
        else:
            label = torch.tensor(int(row["label"]), dtype=torch.long)

        return img, label


# ── Image + metadata dataset (for skin module) ────────────────────────────

class ImageMetaDataset(Dataset):
    """
    Dataset that returns (image_tensor, meta_tensor, label).
    meta_cols: list of column names to include as numeric features.
    """
    def __init__(self, df: pd.DataFrame, meta_cols: list,
                 transform=None):
        self.df        = df.reset_index(drop=True)
        self.meta_cols = meta_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = load_image_rgb(row["image_path"])
        if self.transform:
            img = self.transform(img)

        meta  = torch.tensor(
            row[self.meta_cols].values.astype(np.float32),
            dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return img, meta, label


# ── Weighted sampler for class imbalance ──────────────────────────────────

def make_weighted_sampler(labels: list):
    """
    Returns a WeightedRandomSampler that oversamples minority classes.
    Essential for imbalanced medical datasets (e.g. melanoma << benign).
    """
    from torch.utils.data import WeightedRandomSampler
    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts
    sample_wts   = weights[labels]
    return WeightedRandomSampler(
        torch.tensor(sample_wts, dtype=torch.float),
        num_samples=len(sample_wts),
        replacement=True,
    )


# ── Grad-CAM utility ───────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works with any CNN that has a named target layer.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(img_tensor, class_idx)
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, img_tensor: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).requires_grad_(True)
        logits     = self.model(img_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Metric helpers ─────────────────────────────────────────────────────────

def compute_multilabel_auc(y_true: np.ndarray, y_pred: np.ndarray,
                            labels: list) -> dict:
    """ROC-AUC per label for multi-label classification."""
    from sklearn.metrics import roc_auc_score
    results = {}
    for i, lbl in enumerate(labels):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = float("nan")
        results[lbl] = round(auc, 4)
    results["mean_auc"] = round(
        np.nanmean(list(results.values())), 4)
    return results
