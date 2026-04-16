"""
Multimodal Medical AI — Streamlit App
=======================================
Unified diagnostic interface for all 3 modules:
  - Chest: X-ray + ECG analysis
  - Eye:   Retinal DR grading + glaucoma screening
  - Skin:  Lesion classification + malignancy risk

Run with:
    streamlit run app.py
"""

import os
import io
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config  import DEVICE, CHEST_CFG, EYE_CFG, SKIN_CFG
from utils   import get_val_transforms, GradCAM

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Medical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.report-box {
    background: #f8f9fa;
    border-left: 4px solid #0d6efd;
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
}
.warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: .75rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model loaders (cached) ─────────────────────────────────────────────────

@st.cache_resource
def load_chest_model():
    from module1_chest import XrayEncoder, ECGEncoder
    xray = XrayEncoder().to(DEVICE)
    ecg  = ECGEncoder().to(DEVICE)
    if os.path.exists("checkpoints/xray_best.pt"):
        xray.load_state_dict(
            torch.load("checkpoints/xray_best.pt", map_location=DEVICE))
    if os.path.exists("checkpoints/ecg_best.pt"):
        ecg.load_state_dict(
            torch.load("checkpoints/ecg_best.pt",  map_location=DEVICE))
    xray.eval(); ecg.eval()
    return xray, ecg


@st.cache_resource
def load_eye_model():
    from module2_eye import DREncoder, GlaucomaEncoder
    dr  = DREncoder().to(DEVICE)
    glc = GlaucomaEncoder().to(DEVICE)
    if os.path.exists("checkpoints/dr_best.pt"):
        dr.load_state_dict(
            torch.load("checkpoints/dr_best.pt",       map_location=DEVICE))
    if os.path.exists("checkpoints/glaucoma_best.pt"):
        glc.load_state_dict(
            torch.load("checkpoints/glaucoma_best.pt", map_location=DEVICE))
    dr.eval(); glc.eval()
    return dr, glc


@st.cache_resource
def load_skin_model():
    from module3_skin import SkinEncoder, META_COLS
    model = SkinEncoder(
        num_classes=SKIN_CFG.num_classes,
        meta_dim=len(META_COLS)).to(DEVICE)
    if os.path.exists("checkpoints/skin_best.pt"):
        model.load_state_dict(
            torch.load("checkpoints/skin_best.pt", map_location=DEVICE))
    model.eval()
    return model


# ── Grad-CAM overlay ───────────────────────────────────────────────────────

def make_gradcam_overlay(model, img_tensor: torch.Tensor,
                          orig_img: Image.Image,
                          target_layer) -> Image.Image:
    cam  = GradCAM(model, target_layer)
    heat = cam(img_tensor)

    # Resize heatmap to original image size
    from PIL import Image as PILImage
    heat_img = PILImage.fromarray(
        (cm.jet(heat)[:, :, :3] * 255).astype(np.uint8)
    ).resize(orig_img.size)

    orig_arr = np.array(orig_img.convert("RGB")).astype(float)
    heat_arr = np.array(heat_img).astype(float)
    overlay  = (0.55 * orig_arr + 0.45 * heat_arr).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(overlay)


# ── Probability bar chart ──────────────────────────────────────────────────

def prob_chart(probs: dict, title: str = ""):
    df = pd.DataFrame({
        "Class": list(probs.keys()),
        "Probability": [v * 100 for v in probs.values()],
    }).sort_values("Probability", ascending=True)

    fig = px.bar(df, x="Probability", y="Class",
                 orientation="h", title=title,
                 color="Probability",
                 color_continuous_scale="teal",
                 range_x=[0, 100])
    fig.update_layout(height=300, showlegend=False,
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MODULE PAGES
# ════════════════════════════════════════════════════════════════════════════

def page_chest():
    st.header("Chest Diagnosis — X-ray + ECG")
    st.caption("Upload a chest X-ray and/or ECG signal to detect 14 pathologies "
               "and 5 cardiac rhythm classes.")

    xray_model, ecg_model = load_chest_model()
    transform = get_val_transforms(CHEST_CFG.xray_img_size)

    col1, col2 = st.columns(2)
    xray_preds = ecg_preds = None

    with col1:
        st.subheader("Chest X-ray")
        xray_file = st.file_uploader("Upload X-ray (.jpg / .png)",
                                      type=["jpg","jpeg","png"],
                                      key="xray_upload")
        if xray_file:
            img = Image.open(xray_file).convert("RGB")
            st.image(img, caption="Uploaded X-ray", use_column_width=True)
            img_t = transform(img)

            with torch.no_grad():
                probs = torch.sigmoid(
                    xray_model(img_t.unsqueeze(0).to(DEVICE))
                ).cpu().numpy()[0]

            xray_preds = dict(zip(CHEST_CFG.xray_labels, probs.tolist()))

            # Show Grad-CAM for top finding
            top_idx = int(np.argmax(probs))
            try:
                overlay = make_gradcam_overlay(
                    xray_model, img_t, img,
                    xray_model.features[-1])
                st.image(overlay,
                         caption=f"Grad-CAM: {CHEST_CFG.xray_labels[top_idx]}",
                         use_column_width=True)
            except Exception:
                pass

            st.plotly_chart(prob_chart(xray_preds, "X-ray findings"),
                            use_container_width=True)

    with col2:
        st.subheader("ECG Signal")
        st.info("Upload a PTB-XL format .npy file (shape: 12×1000) "
                "or use the demo signal below.")

        use_demo = st.checkbox("Use demo ECG signal", value=True)
        if use_demo:
            demo_ecg = np.random.randn(1, 12, 1000).astype(np.float32)
            ecg_t    = torch.tensor(demo_ecg)
        else:
            ecg_file = st.file_uploader("Upload ECG (.npy)", type=["npy"],
                                         key="ecg_upload")
            if ecg_file:
                ecg_np = np.load(ecg_file)
                ecg_t  = torch.tensor(ecg_np).float()
                if ecg_t.dim() == 2:
                    ecg_t = ecg_t.unsqueeze(0)
            else:
                ecg_t = None

        if ecg_t is not None:
            with torch.no_grad():
                probs = F.softmax(
                    ecg_model(ecg_t.to(DEVICE)), dim=1
                ).cpu().numpy()[0]

            pred_idx    = int(np.argmax(probs))
            ecg_preds   = {
                "label":      CHEST_CFG.ecg_labels[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probs":      dict(zip(CHEST_CFG.ecg_labels, probs.tolist())),
            }
            st.metric("Rhythm class",
                      ecg_preds["label"],
                      f"{ecg_preds['confidence']:.1%} confidence")
            st.plotly_chart(
                prob_chart(ecg_preds["probs"], "ECG rhythm classification"),
                use_container_width=True)

    # ── Report ────────────────────────────────────────────────────────────
    if (xray_preds or ecg_preds) and st.button("Generate clinical report"):
        with st.spinner("Generating report with Claude AI..."):
            from report_generator import chest_report
            report = chest_report(
                xray_preds=xray_preds or {},
                ecg_preds=ecg_preds   or {},
            )
        st.subheader("Clinical Report")
        st.markdown(f'<div class="report-box">{report}</div>',
                    unsafe_allow_html=True)


def page_eye():
    st.header("Eye Screening — Diabetic Retinopathy + Glaucoma")
    st.caption("Upload a retinal fundus photograph for DR grading and glaucoma risk.")

    dr_model, glc_model = load_eye_model()
    transform = get_val_transforms(EYE_CFG.dr_img_size)

    col1, col2 = st.columns([1, 1])

    with col1:
        fundus_file = st.file_uploader(
            "Upload fundus image (.jpg / .png)",
            type=["jpg","jpeg","png"], key="fundus")

        if fundus_file:
            img = Image.open(fundus_file).convert("RGB")
            st.image(img, caption="Fundus image", use_column_width=True)

    with col2:
        if fundus_file:
            img_t = transform(img)
            inp   = img_t.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                dr_probs  = F.softmax(dr_model(inp),  dim=1).cpu().numpy()[0]
                glc_score = torch.sigmoid(
                    glc_model(inp)).cpu().item()

            dr_idx   = int(np.argmax(dr_probs))
            dr_preds = {
                "grade":      dr_idx,
                "label":      EYE_CFG.dr_labels[dr_idx],
                "confidence": float(dr_probs[dr_idx]),
                "probs":      dict(zip(EYE_CFG.dr_labels, dr_probs.tolist())),
            }

            # DR grade indicator
            grade_color = ["green","yellow","orange","red","darkred"][dr_idx]
            st.metric("DR Grade", f"{dr_idx} — {EYE_CFG.dr_labels[dr_idx]}",
                      f"{dr_preds['confidence']:.1%} confidence")
            st.progress(dr_idx / 4)

            # Glaucoma
            glc_risk = "High" if glc_score > 0.5 else "Low"
            st.metric("Glaucoma Risk",
                      f"{glc_risk} ({glc_score:.1%})")

            st.plotly_chart(
                prob_chart(dr_preds["probs"], "DR grade probabilities"),
                use_container_width=True)

            # Patient context
            st.subheader("Patient context (optional)")
            age  = st.number_input("Age", 0, 120, 50)
            hba1c= st.text_input("HbA1c (if known)", "")

            if st.button("Generate eye report"):
                with st.spinner("Generating..."):
                    from report_generator import eye_report
                    ctx = {"age": age}
                    if hba1c:
                        ctx["hba1c"] = hba1c
                    report = eye_report(dr_preds, glc_score, ctx)
                st.markdown(f'<div class="report-box">{report}</div>',
                            unsafe_allow_html=True)


def page_skin():
    st.header("Skin Lesion Analysis — Malignancy Screening")
    st.caption("Upload a dermoscopy or clinical skin image for lesion classification.")

    from module3_skin import META_COLS, SITE_COLS
    model = load_skin_model()
    transform = get_val_transforms(SKIN_CFG.img_size)

    col1, col2 = st.columns([1, 1])

    with col1:
        skin_file = st.file_uploader(
            "Upload skin lesion image (.jpg / .png)",
            type=["jpg","jpeg","png"], key="skin")

        if skin_file:
            img = Image.open(skin_file).convert("RGB")
            st.image(img, caption="Lesion image", use_column_width=True)

        st.subheader("Patient metadata")
        age  = st.slider("Patient age", 0, 100, 45)
        sex  = st.selectbox("Sex", ["Male", "Female", "Unknown"])
        site = st.selectbox("Lesion location", SITE_COLS)

    with col2:
        if skin_file:
            img_t = transform(img)

            # Build metadata tensor
            age_norm = age / 90.0
            sex_enc  = {"Male": 1.0, "Female": 0.0, "Unknown": 0.5}[sex]
            site_vec = [1.0 if s == site else 0.0 for s in SITE_COLS]
            meta_np  = np.array([age_norm, sex_enc] + site_vec,
                                 dtype=np.float32)
            meta_t   = torch.tensor(meta_np).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = F.softmax(
                    model(img_t.unsqueeze(0).to(DEVICE), meta_t),
                    dim=1).cpu().numpy()[0]

            pred_idx    = int(np.argmax(probs))
            pred_label  = SKIN_CFG.labels[pred_idx]
            confidence  = float(probs[pred_idx])
            is_malignant= pred_label in [
                "Melanoma", "Basal cell carcinoma",
                "Actinic keratosis"]

            if is_malignant:
                st.error(f"Potential malignancy detected: {pred_label}")
            else:
                st.success(f"Primary finding: {pred_label}")

            st.metric("Confidence", f"{confidence:.1%}")
            st.metric("Malignancy concern",
                      "HIGH" if is_malignant else "LOW")

            preds_dict = dict(zip(SKIN_CFG.labels, probs.tolist()))
            st.plotly_chart(
                prob_chart(preds_dict, "Lesion classification"),
                use_container_width=True)

            if st.button("Generate dermatology report"):
                with st.spinner("Generating..."):
                    from report_generator import skin_report
                    skin_preds = {
                        "class":       pred_idx,
                        "label":       pred_label,
                        "confidence":  confidence,
                        "is_malignant":is_malignant,
                        "probs":       preds_dict,
                    }
                    report = skin_report(
                        skin_preds,
                        {"age": age, "sex": sex, "site": site})
                st.markdown(f'<div class="report-box">{report}</div>',
                            unsafe_allow_html=True)

            st.markdown(
                '<div class="warning-box">AI screening only. '
                'Not a substitute for clinical examination or histopathology.</div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🏥 Multimodal Medical AI")
    st.markdown("*AI-powered diagnostic screening across Chest, Eye, and Skin*")

    module = st.sidebar.radio(
        "Select diagnostic module",
        ["Chest (X-ray + ECG)", "Eye (Retina)", "Skin (Lesion)"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**")
    st.sidebar.caption(
        "3 deep learning modules trained on public medical datasets. "
        "Reports generated by Claude AI. "
        "For research and educational use only."
    )
    st.sidebar.markdown("**Datasets**")
    st.sidebar.caption(
        "NIH ChestX-ray14 · PTB-XL · APTOS 2019 · RIM-ONE · ISIC 2020"
    )

    if module == "Chest (X-ray + ECG)":
        page_chest()
    elif module == "Eye (Retina)":
        page_eye()
    else:
        page_skin()


if __name__ == "__main__":
    main()
