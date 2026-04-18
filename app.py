"""
Multimodal Medical AI — Professional UI (Redesigned)
=====================================================
Clinical dark-navy aesthetic with:
  - Playfair Display headers, DM Sans body, DM Mono data
  - Animated confidence bars per finding
  - Severity badges (normal / mild / moderate / severe)
  - Grad-CAM overlay side-by-side with original image
  - Styled clinical report output
  - Consistent card layout across all 3 modules

Run with:
    streamlit run app.py
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go

from config import DEVICE, CHEST_CFG, EYE_CFG, SKIN_CFG
from utils  import get_val_transforms, GradCAM

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600&display=swap');

:root {
  --navy:       #0a1628;
  --navy-mid:   #0f2040;
  --navy-light: #1a3358;
  --accent:     #00d4aa;
  --accent-dim: #00a882;
  --red:        #ff4d6d;
  --amber:      #ffb703;
  --green:      #06d6a0;
  --blue:       #3a86ff;
  --text:       #e8edf5;
  --muted:      #8a9bb5;
  --border:     rgba(255,255,255,0.07);
  --card:       rgba(15,32,64,0.6);
}

html, body, [class*="css"] { font-family:'DM Sans',sans-serif !important; color:var(--text) !important; }
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }

.stApp {
  background: var(--navy) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,212,170,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(58,134,255,0.05) 0%, transparent 50%) !important;
}

[data-testid="stSidebar"] {
  background: var(--navy-mid) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color:var(--text) !important; }
[data-testid="stSidebar"] .stRadio label { font-size:.875rem !important; padding:5px 0 !important; }

.ph { padding:2rem 0 1.5rem; border-bottom:1px solid var(--border); margin-bottom:2rem; }
.ph h1 { font-family:'Playfair Display',serif !important; font-size:2.1rem !important; font-weight:600 !important; color:var(--text) !important; margin:0 0 5px !important; letter-spacing:-.02em; }
.ph p  { color:var(--muted) !important; font-size:.88rem !important; margin:0 !important; }
.aline { width:36px; height:3px; background:var(--accent); border-radius:2px; margin-bottom:10px; }

.slbl { font-size:.68rem; font-weight:600; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:0 0 .75rem; }
.sdiv { height:1px; background:var(--border); margin:1.25rem 0; }

.stile { display:inline-flex; flex-direction:column; align-items:center; background:rgba(255,255,255,.04); border:1px solid var(--border); border-radius:10px; padding:12px 18px; min-width:90px; }
.stile-val { font-family:'DM Mono',monospace; font-size:1.5rem; font-weight:600; color:var(--accent); line-height:1.1; }
.stile-lbl { font-size:.68rem; color:var(--muted); margin-top:3px; text-align:center; }

.sbadge { display:inline-flex; align-items:center; gap:6px; padding:5px 14px; border-radius:999px; font-size:.78rem; font-weight:600; letter-spacing:.03em; margin-bottom:.5rem; }
.sb-dot { width:7px; height:7px; border-radius:50%; background:currentColor; }
.sev-normal   { background:rgba(6,214,160,.12);  color:#06d6a0; border:1px solid rgba(6,214,160,.25); }
.sev-mild     { background:rgba(255,183,3,.12);   color:#ffb703; border:1px solid rgba(255,183,3,.25); }
.sev-moderate { background:rgba(255,140,0,.12);   color:#ff8c00; border:1px solid rgba(255,140,0,.25); }
.sev-severe   { background:rgba(255,77,109,.12);  color:#ff4d6d; border:1px solid rgba(255,77,109,.25); }

.cbar-wrap { margin:5px 0 12px; }
.cbar-top  { display:flex; justify-content:space-between; font-size:.7rem; color:var(--muted); margin-bottom:4px; }
.cbar-trk  { height:5px; border-radius:3px; background:rgba(255,255,255,.08); overflow:hidden; }
.cbar-fill { height:100%; border-radius:3px; }

.ibox { background:rgba(58,134,255,.07); border:1px solid rgba(58,134,255,.18); border-radius:8px; padding:10px 14px; font-size:.82rem; color:#7eb8ff; }
.wbox { background:rgba(255,183,3,.07); border:1px solid rgba(255,183,3,.18); border-radius:8px; padding:10px 14px; font-size:.8rem; color:#ffb703; display:flex; gap:8px; align-items:flex-start; margin-top:.75rem; }
.rbox-wrap { background:rgba(0,0,0,.3); border:1px solid rgba(0,212,170,.18); border-radius:12px; overflow:hidden; margin-top:1rem; }
.rbox-head { background:rgba(0,212,170,.09); padding:9px 16px; display:flex; align-items:center; gap:9px; border-bottom:1px solid rgba(0,212,170,.18); }
.rbox-dot  { width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 7px var(--accent); }
.rbox-lbl  { font-size:.68rem; font-weight:600; color:var(--accent); letter-spacing:.08em; text-transform:uppercase; }
.rbox-body { padding:1.1rem 1.4rem; font-family:'DM Mono',monospace; font-size:.8rem; color:var(--text); white-space:pre-wrap; line-height:1.75; }

.sidebar-brand { padding:1.5rem 1rem 1rem; border-bottom:1px solid var(--border); margin-bottom:1.25rem; }
.sb-name { font-family:'Playfair Display',serif; font-size:1.25rem; font-weight:600; color:var(--text); }
.sb-sub  { font-size:.68rem; color:var(--accent); text-transform:uppercase; letter-spacing:.1em; margin-top:1px; }

[data-testid="stFileUploader"] { background:rgba(255,255,255,.03) !important; border:1.5px dashed rgba(0,212,170,.28) !important; border-radius:10px !important; }
[data-testid="stFileUploader"]:hover { border-color:var(--accent) !important; background:rgba(0,212,170,.05) !important; }
[data-testid="stFileUploader"] * { color:var(--muted) !important; }

.stButton > button { background:var(--accent) !important; color:var(--navy) !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-size:.85rem !important; padding:10px 22px !important; font-family:'DM Sans',sans-serif !important; transition:all .2s !important; }
.stButton > button:hover { background:var(--accent-dim) !important; transform:translateY(-1px) !important; box-shadow:0 4px 14px rgba(0,212,170,.28) !important; }

.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input { background:rgba(255,255,255,.05) !important; border:1px solid var(--border) !important; border-radius:8px !important; color:var(--text) !important; font-family:'DM Sans',sans-serif !important; }
.stCheckbox label { font-size:.85rem !important; color:var(--text) !important; }

[data-testid="stMetric"] { background:rgba(255,255,255,.03); border:1px solid var(--border); border-radius:10px; padding:12px 16px !important; }
[data-testid="stMetricLabel"] > div { font-size:.68rem !important; color:var(--muted) !important; text-transform:uppercase; letter-spacing:.06em; }
[data-testid="stMetricValue"] { font-family:'DM Mono',monospace !important; font-size:1.35rem !important; color:var(--text) !important; }
[data-testid="stMetricDelta"] { font-size:.78rem !important; color:var(--accent) !important; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,.1); border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────
def badge(label, sev):
    return (f'<span class="sbadge sev-{sev}">'
            f'<span class="sb-dot"></span>{label}</span>')

def cbar(label, pct, color="#00d4aa"):
    return (f'<div class="cbar-wrap">'
            f'<div class="cbar-top"><span>{label}</span><span>{pct:.1f}%</span></div>'
            f'<div class="cbar-trk"><div class="cbar-fill" '
            f'style="width:{pct}%;background:{color}"></div></div></div>')

def tiles_row(*pairs):
    inner = "".join(
        f'<div class="stile"><div class="stile-val">{v}</div>'
        f'<div class="stile-lbl">{l}</div></div>'
        for v, l in pairs)
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:.5rem 0 1rem;">{inner}</div>'

def report_html(text):
    return (f'<div class="rbox-wrap"><div class="rbox-head">'
            f'<div class="rbox-dot"></div>'
            f'<div class="rbox-lbl">AI Clinical Report</div></div>'
            f'<div class="rbox-body">{text}</div></div>')

def prob_chart(probs, title=""):
    labels = list(probs.keys())
    values = [v*100 for v in probs.values()]
    mx     = max(values)
    colors = ["#00d4aa" if v == mx else "#1e3a5f" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#8a9bb5", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8a9bb5", size=11),
        xaxis=dict(range=[0,115], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(showgrid=False, color="#e8edf5",
                   tickfont=dict(size=12, color="#e8edf5")),
        margin=dict(l=0,r=50,t=8,b=0),
        height=max(160, len(labels)*40),
        bargap=0.38,
    )
    return fig


# ── Model loaders ─────────────────────────────────────────────────────────
@st.cache_resource
def load_chest():
    from module1_chest import XrayEncoder, ECGEncoder
    xray = XrayEncoder().to(DEVICE); ecg = ECGEncoder().to(DEVICE)
    for ckpt, m in [("checkpoints/xray_best.pt",xray),("checkpoints/ecg_best.pt",ecg)]:
        if os.path.exists(ckpt): m.load_state_dict(torch.load(ckpt,map_location=DEVICE))
    return xray.eval(), ecg.eval()

@st.cache_resource
def load_eye():
    from module2_eye import DREncoder, GlaucomaEncoder
    dr = DREncoder().to(DEVICE); glc = GlaucomaEncoder().to(DEVICE)
    for ckpt, m in [("checkpoints/dr_best.pt",dr),("checkpoints/glaucoma_best.pt",glc)]:
        if os.path.exists(ckpt): m.load_state_dict(torch.load(ckpt,map_location=DEVICE))
    return dr.eval(), glc.eval()

@st.cache_resource
def load_skin():
    from module3_skin import SkinEncoder, META_COLS
    m = SkinEncoder(num_classes=SKIN_CFG.num_classes, meta_dim=len(META_COLS)).to(DEVICE)
    if os.path.exists("checkpoints/skin_best.pt"):
        m.load_state_dict(torch.load("checkpoints/skin_best.pt",map_location=DEVICE))
    return m.eval()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">'
                '<div class="sb-name">MediScan AI</div>'
                '<div class="sb-sub">Diagnostic Intelligence</div>'
                '</div>', unsafe_allow_html=True)

    st.markdown('<p class="slbl">Diagnostic Modules</p>', unsafe_allow_html=True)
    module = st.radio("", ["Chest — X-ray + ECG",
                           "Eye — Retina + Glaucoma",
                           "Skin — Lesion Analysis"],
                      label_visibility="collapsed")

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown('<p class="slbl">Training Datasets</p>', unsafe_allow_html=True)
    for ds in ["NIH ChestX-ray14","PTB-XL ECG","APTOS 2019",
               "RIM-ONE DL","ISIC 2020"]:
        st.markdown(f'<p style="font-size:.8rem;color:var(--muted);margin:3px 0">{ds}</p>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wbox"><span>⚠</span>'
                '<span>Research & educational use only. '
                'Not a substitute for clinical diagnosis.</span></div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — CHEST
# ════════════════════════════════════════════════════════════════════════════
def page_chest():
    st.markdown('<div class="ph"><div class="aline"></div>'
                '<h1>Chest Diagnosis</h1>'
                '<p>14-finding X-ray pathology detection · 5-class ECG cardiac rhythm classification</p>'
                '</div>', unsafe_allow_html=True)

    xray_model, ecg_model = load_chest()
    tf = get_val_transforms(CHEST_CFG.xray_img_size)
    xray_preds = ecg_preds = None

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="slbl">Chest X-ray</p>', unsafe_allow_html=True)
        xray_file = st.file_uploader("Drop X-ray image", type=["jpg","jpeg","png"],
                                      key="xray", label_visibility="collapsed")
        if xray_file:
            img   = Image.open(xray_file).convert("RGB")
            img_t = tf(img)
            with torch.no_grad():
                probs = torch.sigmoid(
                    xray_model(img_t.unsqueeze(0).to(DEVICE))
                ).cpu().numpy()[0]
            xray_preds = dict(zip(CHEST_CFG.xray_labels, probs.tolist()))
            top_idx    = int(np.argmax(probs))
            top_label  = CHEST_CFG.xray_labels[top_idx]
            top_conf   = float(probs[top_idx])

            # Original + Grad-CAM
            c1, c2 = st.columns(2)
            with c1: st.image(img, caption="Original", use_column_width=True)
            with c2:
                try:
                    import matplotlib.cm as cm
                    gc   = GradCAM(xray_model, xray_model.features[-1])
                    heat = gc(img_t)
                    hi   = Image.fromarray(
                        (cm.inferno(heat)[:,:,:3]*255).astype(np.uint8)
                    ).resize(img.size)
                    ov   = np.clip(.6*np.array(img.convert("RGB")).astype(float)
                                   +.4*np.array(hi).astype(float), 0, 255).astype(np.uint8)
                    st.image(Image.fromarray(ov),
                             caption=f"Grad-CAM · {top_label}", use_column_width=True)
                except Exception:
                    st.image(img, caption="Grad-CAM (unavailable)", use_column_width=True)

            sev = "severe" if top_conf>.7 else "moderate" if top_conf>.4 else "mild"
            st.markdown(badge(f"Primary finding: {top_label}", sev), unsafe_allow_html=True)
            st.markdown(cbar("Detection confidence", top_conf*100), unsafe_allow_html=True)

            top5 = dict(sorted(xray_preds.items(), key=lambda x:-x[1])[:5])
            st.plotly_chart(prob_chart(top5, "Top 5 findings"),
                            use_container_width=True)
        else:
            st.markdown('<div class="ibox">Upload a chest X-ray to detect 14 pathologies '
                        'with Grad-CAM heatmap localisation.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="slbl">ECG Rhythm</p>', unsafe_allow_html=True)
        use_demo = st.checkbox("Use demo ECG signal", value=True)
        ecg_t    = None

        if use_demo:
            ecg_t = torch.randn(1, 12, 1000)
        else:
            ecg_file = st.file_uploader("Upload ECG .npy (12×1000)", type=["npy"],
                                         key="ecg", label_visibility="collapsed")
            if ecg_file:
                arr   = np.load(ecg_file)
                ecg_t = torch.tensor(arr).float()
                if ecg_t.dim() == 2: ecg_t = ecg_t.unsqueeze(0)

        if ecg_t is not None:
            with torch.no_grad():
                probs = F.softmax(
                    ecg_model(ecg_t.to(DEVICE)), dim=1
                ).cpu().numpy()[0]
            pi    = int(np.argmax(probs))
            label = CHEST_CFG.ecg_labels[pi]
            conf  = float(probs[pi])
            ecg_preds = {"label":label, "confidence":conf,
                         "probs":dict(zip(CHEST_CFG.ecg_labels, probs.tolist()))}

            full_map = {"NORM":("Normal Sinus Rhythm","normal"),
                        "MI":  ("Myocardial Infarction","severe"),
                        "STTC":("ST/T-wave Change","moderate"),
                        "CD":  ("Conduction Defect","moderate"),
                        "HYP": ("Hypertrophy","mild")}
            fl, sev  = full_map.get(label, (label,"mild"))
            st.markdown(badge(fl, sev), unsafe_allow_html=True)
            st.markdown(tiles_row((label,"Rhythm class"),(f"{conf:.0%}","Confidence")),
                        unsafe_allow_html=True)

            st.markdown('<p class="slbl" style="margin-top:.5rem;">Class probabilities</p>',
                        unsafe_allow_html=True)
            cm_map = {"NORM":"#06d6a0","MI":"#ff4d6d",
                      "STTC":"#ffb703","CD":"#ff8c00","HYP":"#3a86ff"}
            for lbl, p in ecg_preds["probs"].items():
                st.markdown(cbar(lbl, p*100, cm_map.get(lbl,"#00d4aa")),
                            unsafe_allow_html=True)

    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
    if xray_preds or ecg_preds:
        if st.button("Generate Clinical Report"):
            with st.spinner("Generating AI report..."):
                try:
                    from report_generator import chest_report
                    rpt = chest_report(xray_preds or {}, ecg_preds or {})
                except Exception as e:
                    rpt = str(e)
            st.markdown(report_html(rpt), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — EYE
# ════════════════════════════════════════════════════════════════════════════
def page_eye():
    st.markdown('<div class="ph"><div class="aline"></div>'
                '<h1>Eye Screening</h1>'
                '<p>Diabetic retinopathy grading (Grade 0–4) · Glaucoma risk assessment</p>'
                '</div>', unsafe_allow_html=True)

    dr_model, glc_model = load_eye()
    tf = get_val_transforms(EYE_CFG.dr_img_size)

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown('<p class="slbl">Retinal Fundus Image</p>', unsafe_allow_html=True)
        f = st.file_uploader("Upload fundus image", type=["jpg","jpeg","png"],
                              key="fundus", label_visibility="collapsed")
        if f:
            img = Image.open(f).convert("RGB")
            st.image(img, use_column_width=True)

    with col2:
        if f:
            inp = tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dr_probs  = F.softmax(dr_model(inp), dim=1).cpu().numpy()[0]
                glc_score = torch.sigmoid(glc_model(inp)).cpu().item()

            di   = int(np.argmax(dr_probs))
            dlbl = EYE_CFG.dr_labels[di]
            dcon = float(dr_probs[di])
            sm   = ["normal","mild","moderate","severe","severe"]

            st.markdown(badge(f"DR Grade {di} — {dlbl}", sm[di]),
                        unsafe_allow_html=True)

            # Grade progress bar
            seg_cols = ["#06d6a0","#a8dadc","#ffb703","#ff8c00","#ff4d6d"]
            segs = "".join(
                f'<div style="flex:1;height:7px;border-radius:4px;'
                f'background:{"#1e3a5f" if i>di else seg_cols[di]}"></div>'
                for i in range(5))
            st.markdown(
                f'<div style="display:flex;gap:4px;margin:6px 0 14px">{segs}</div>',
                unsafe_allow_html=True)

            glc_risk  = "High" if glc_score>.5 else "Low"
            glc_color = "#ff4d6d" if glc_score>.5 else "#06d6a0"
            st.markdown(tiles_row(
                (str(di), "DR Grade"),
                (f"{dcon:.0%}", "Confidence"),
                (glc_risk, "Glaucoma risk")),
                unsafe_allow_html=True)

            st.markdown('<p class="slbl">Grade probabilities</p>',
                        unsafe_allow_html=True)
            for lbl, p in zip(EYE_CFG.dr_labels, dr_probs):
                st.markdown(cbar(lbl, p*100), unsafe_allow_html=True)

            st.markdown(cbar("Glaucoma risk score", glc_score*100, glc_color),
                        unsafe_allow_html=True)

            st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
            st.markdown('<p class="slbl">Patient Context</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: age   = st.number_input("Age", 0, 120, 50)
            with c2: hba1c = st.text_input("HbA1c", placeholder="e.g. 8.5%")

            if st.button("Generate Eye Report"):
                with st.spinner("Generating..."):
                    try:
                        from report_generator import eye_report
                        dp  = {"grade":di,"label":dlbl,"confidence":dcon,
                               "probs":dict(zip(EYE_CFG.dr_labels, dr_probs.tolist()))}
                        ctx = {"age":age}
                        if hba1c: ctx["hba1c"] = hba1c
                        rpt = eye_report(dp, glc_score, ctx)
                    except Exception as e: rpt = str(e)
                st.markdown(report_html(rpt), unsafe_allow_html=True)
        else:
            st.markdown('<div class="ibox" style="margin-top:3rem;">'
                        'Upload a retinal fundus photograph to begin DR grading '
                        'and glaucoma risk screening.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — SKIN
# ════════════════════════════════════════════════════════════════════════════
def page_skin():
    st.markdown('<div class="ph"><div class="aline"></div>'
                '<h1>Skin Lesion Analysis</h1>'
                '<p>7-class lesion classification · Malignancy risk scoring · Metadata fusion</p>'
                '</div>', unsafe_allow_html=True)

    from module3_skin import META_COLS, SITE_COLS
    model = load_skin()
    tf    = get_val_transforms(SKIN_CFG.img_size)

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown('<p class="slbl">Lesion Image</p>', unsafe_allow_html=True)
        f = st.file_uploader("Upload skin lesion image",
                              type=["jpg","jpeg","png"], key="skin",
                              label_visibility="collapsed")
        if f:
            img = Image.open(f).convert("RGB")
            st.image(img, use_column_width=True)

        st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)
        st.markdown('<p class="slbl">Patient Metadata</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: age = st.slider("Age", 0, 100, 45)
        with c2: sex = st.selectbox("Sex", ["Male","Female","Unknown"])
        site = st.selectbox("Anatomical site", SITE_COLS)

    with col2:
        if f:
            img_t  = tf(img)
            meta_t = torch.tensor(
                np.array([age/90., {"Male":1.,"Female":0.,"Unknown":.5}[sex]]
                         + [1. if s==site else 0. for s in SITE_COLS],
                         dtype=np.float32)
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = F.softmax(
                    model(img_t.unsqueeze(0).to(DEVICE), meta_t),
                    dim=1).cpu().numpy()[0]

            pi  = int(np.argmax(probs))
            plb = SKIN_CFG.labels[pi]
            pcf = float(probs[pi])
            mal = plb in ["Melanoma","Basal cell carcinoma","Actinic keratosis"]

            sev = "severe" if mal else "normal"
            st.markdown(badge(plb, sev), unsafe_allow_html=True)
            st.markdown(tiles_row(
                (f"{pcf:.0%}","Confidence"),
                ("HIGH" if mal else "LOW","Malignancy risk")),
                unsafe_allow_html=True)

            if mal:
                st.markdown('<div class="wbox"><span>⚠</span>'
                            '<span>Potential malignancy detected. '
                            'Urgent dermatology referral recommended. '
                            'Biopsy may be warranted.</span></div>',
                            unsafe_allow_html=True)

            st.markdown('<p class="slbl" style="margin-top:.75rem;">'
                        'Classification probabilities</p>',
                        unsafe_allow_html=True)
            MALIGNANT = {"Melanoma","Basal cell carcinoma","Actinic keratosis"}
            for lbl, p in sorted(zip(SKIN_CFG.labels, probs), key=lambda x:-x[1]):
                col = "#ff4d6d" if lbl in MALIGNANT else "#00d4aa"
                st.markdown(cbar(lbl, p*100, col), unsafe_allow_html=True)

            if st.button("Generate Dermatology Report"):
                with st.spinner("Generating..."):
                    try:
                        from report_generator import skin_report
                        sp  = {"class":pi,"label":plb,"confidence":pcf,
                               "is_malignant":mal,
                               "probs":dict(zip(SKIN_CFG.labels, probs.tolist()))}
                        rpt = skin_report(sp, {"age":age,"sex":sex,"site":site})
                    except Exception as e: rpt = str(e)
                st.markdown(report_html(rpt), unsafe_allow_html=True)
        else:
            st.markdown('<div class="ibox" style="margin-top:3rem;">'
                        'Upload a dermoscopy or clinical image, enter patient details, '
                        'and run classification.</div>', unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────
if "Chest" in module:   page_chest()
elif "Eye" in module:   page_eye()
else:                   page_skin()
