"""
MediScan AI — FastAPI Backend
==============================
Replaces `streamlit run app.py` with a plain HTTP REST server.
The frontend (index.html) talks to this via fetch().

Run with:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

Then open index.html in your browser.
"""

import os, json, datetime, io, base64
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests as req_lib

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from config import DEVICE, CHEST_CFG, EYE_CFG, SKIN_CFG
from utils  import get_val_transforms, GradCAM

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MediScan AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html and static files from the same directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html not found</h1>"

@app.get("/manifest.json")
def manifest():
    return FileResponse(os.path.join(BASE_DIR, "manifest.json"))

@app.get("/sw.js")
def sw():
    return FileResponse(os.path.join(BASE_DIR, "sw.js"))

# ── Model loaders (cached in module-level globals) ────────────────────────────
_chest_models = None
_eye_models   = None
_skin_model   = None

def get_chest():
    global _chest_models
    if _chest_models is None:
        from module1_chest import XrayEncoder, ECGEncoder
        xray = XrayEncoder().to(DEVICE)
        ecg  = ECGEncoder().to(DEVICE)
        for ck, m in [("checkpoints/xray_best.pt", xray),
                      ("checkpoints/ecg_best.pt",  ecg)]:
            path = os.path.join(BASE_DIR, ck)
            if os.path.exists(path):
                m.load_state_dict(torch.load(path, map_location=DEVICE))
        _chest_models = (xray.eval(), ecg.eval())
    return _chest_models

def get_eye():
    global _eye_models
    if _eye_models is None:
        from module2_eye import DREncoder, GlaucomaEncoder
        dr  = DREncoder().to(DEVICE)
        glc = GlaucomaEncoder().to(DEVICE)
        for ck, m in [("checkpoints/dr_best.pt",      dr),
                      ("checkpoints/glaucoma_best.pt", glc)]:
            path = os.path.join(BASE_DIR, ck)
            if os.path.exists(path):
                m.load_state_dict(torch.load(path, map_location=DEVICE))
        _eye_models = (dr.eval(), glc.eval())
    return _eye_models

def get_skin():
    global _skin_model
    if _skin_model is None:
        from module3_skin import SkinEncoder, META_COLS
        m = SkinEncoder(
            num_classes=SKIN_CFG.num_classes,
            meta_dim=len(META_COLS)
        ).to(DEVICE)
        path = os.path.join(BASE_DIR, "checkpoints/skin_best.pt")
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=DEVICE))
        _skin_model = m.eval()
    return _skin_model

# ── Claude helpers ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")

if API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)

import time

def call_gemini(system, messages, max_tokens=500, max_retries=3):
    if not API_KEY:
        return "⚠ Set GEMINI_API_KEY environment variable to enable AI features."
    
    import google.generativeai as genai
    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        system_instruction=system
    )
    
    gemini_messages = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        gemini_messages.append({"role": role, "parts": [m["content"]]})
        
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                # Wait before retrying: 12 seconds on first retry, 24 on second
                time.sleep(12 * (attempt + 1))
                continue
            return f"AI error: {e}"


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


# ── CHEST endpoints ───────────────────────────────────────────────────────────
@app.post("/chest/xray")
async def chest_xray(file: UploadFile = File(...)):
    """Upload an X-ray image → returns per-label probabilities + Grad-CAM overlay."""
    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    tf   = get_val_transforms(CHEST_CFG.xray_img_size)
    img_t = tf(img)

    xray_model, _ = get_chest()
    with torch.no_grad():
        probs = torch.sigmoid(
            xray_model(img_t.unsqueeze(0).to(DEVICE))
        ).cpu().numpy()[0]

    preds = dict(zip(CHEST_CFG.xray_labels, probs.tolist()))
    top_idx   = int(np.argmax(probs))
    top_label = CHEST_CFG.xray_labels[top_idx]
    top_conf  = float(probs[top_idx])

    # Grad-CAM overlay → base64 for JSON transport
    gradcam_b64 = None
    try:
        import matplotlib.cm as cm
        gc   = GradCAM(xray_model, xray_model.features[-1])
        heat = gc(img_t)
        hi   = Image.fromarray(
            (cm.jet(heat)[:, :, :3] * 255).astype(np.uint8)
        ).resize(img.size)
        ov   = np.clip(
            0.55 * np.array(img.convert("RGB")).astype(float) +
            0.45 * np.array(hi).astype(float), 0, 255
        ).astype(np.uint8)
        buf  = io.BytesIO()
        Image.fromarray(ov).save(buf, format="JPEG", quality=85)
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass

    # Original image as base64
    buf2 = io.BytesIO()
    img.thumbnail((400, 400))
    img.save(buf2, format="JPEG", quality=85)
    orig_b64 = base64.b64encode(buf2.getvalue()).decode()

    return {
        "predictions": preds,
        "top_label":   top_label,
        "top_conf":    top_conf,
        "gradcam_b64": gradcam_b64,
        "orig_b64":    orig_b64,
    }


@app.post("/chest/ecg")
async def chest_ecg(file: Optional[UploadFile] = File(None), demo: bool = False):
    """Upload a .npy ECG file OR use demo random signal."""
    _, ecg_model = get_chest()
    if demo or file is None:
        ecg_t = torch.randn(1, 12, 1000)
    else:
        data  = await file.read()
        arr   = np.load(io.BytesIO(data))
        ecg_t = torch.tensor(arr).float()
        if ecg_t.dim() == 2:
            ecg_t = ecg_t.unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(ecg_model(ecg_t.to(DEVICE)), dim=1).cpu().numpy()[0]

    pi    = int(np.argmax(probs))
    label = CHEST_CFG.ecg_labels[pi]
    conf  = float(probs[pi])
    return {
        "label":  label,
        "confidence": conf,
        "probs": dict(zip(CHEST_CFG.ecg_labels, probs.tolist())),
    }


# ── EYE endpoints ─────────────────────────────────────────────────────────────
@app.post("/eye/fundus")
async def eye_fundus(file: UploadFile = File(...)):
    """Upload a fundus photograph → DR grade + glaucoma risk."""
    data  = await file.read()
    img   = Image.open(io.BytesIO(data)).convert("RGB")
    tf    = get_val_transforms(EYE_CFG.dr_img_size)
    inp   = tf(img).unsqueeze(0).to(DEVICE)

    dr_model, glc_model = get_eye()
    with torch.no_grad():
        dr_probs  = F.softmax(dr_model(inp), dim=1).cpu().numpy()[0]
        glc_score = torch.sigmoid(glc_model(inp)).cpu().item()

    di   = int(np.argmax(dr_probs))
    dlbl = EYE_CFG.dr_labels[di]
    dcon = float(dr_probs[di])

    # Grad-CAM overlay → base64
    gradcam_b64 = None
    try:
        import matplotlib.cm as cm
        gc = GradCAM(dr_model, dr_model.backbone.conv_head)
        heat = gc(inp.squeeze(0))
        hi = Image.fromarray((cm.jet(heat)[:, :, :3] * 255).astype(np.uint8)).resize(img.size)
        ov = np.clip(0.55 * np.array(img.convert("RGB")).astype(float) + 0.45 * np.array(hi).astype(float), 0, 255).astype(np.uint8)
        buf_gc = io.BytesIO()
        Image.fromarray(ov).save(buf_gc, format="JPEG", quality=85)
        gradcam_b64 = base64.b64encode(buf_gc.getvalue()).decode()
    except Exception as e:
        print("Eye GradCAM error:", e)
        pass

    # Original image as base64
    buf = io.BytesIO()
    img.thumbnail((400, 400))
    img.save(buf, format="JPEG", quality=85)
    orig_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "dr": {
            "grade":      di,
            "label":      dlbl,
            "confidence": dcon,
            "probs":      dict(zip(EYE_CFG.dr_labels, dr_probs.tolist())),
        },
        "glaucoma": {"score": glc_score},
        "orig_b64": orig_b64,
        "gradcam_b64": gradcam_b64,
    }


# ── SKIN endpoints ────────────────────────────────────────────────────────────
class SkinMeta(BaseModel):
    age:  float = 45.0
    sex:  str   = "Male"
    site: str   = "back"

@app.post("/skin/classify")
async def skin_classify(
    file: UploadFile = File(...),
    age:  float = Form(45.0),
    sex:  str   = Form("Male"),
    site: str   = Form("back"),
):
    """Upload skin image + metadata → 7-class lesion classification."""
    from module3_skin import META_COLS, SITE_COLS
    data  = await file.read()
    img   = Image.open(io.BytesIO(data)).convert("RGB")
    tf    = get_val_transforms(SKIN_CFG.img_size)
    img_t = tf(img)

    model = get_skin()
    meta_arr = np.array(
        [age / 90.0, {"Male": 1.0, "Female": 0.0, "Unknown": 0.5}[sex]] +
        [1.0 if s == site else 0.0 for s in SITE_COLS],
        dtype=np.float32
    )
    meta_t = torch.tensor(meta_arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(
            model(img_t.unsqueeze(0).to(DEVICE), meta_t), dim=1
        ).cpu().numpy()[0]

    pi  = int(np.argmax(probs))
    plb = SKIN_CFG.labels[pi]
    pcf = float(probs[pi])
    mal = plb in {"Melanoma", "Basal cell carcinoma", "Actinic keratosis"}

    class SkinModelWrapper(nn.Module):
        def __init__(self, m, mt):
            super().__init__()
            self.m = m
            self.mt = mt
        def forward(self, i):
            return self.m(i, self.mt)

    gradcam_b64 = None
    try:
        import matplotlib.cm as cm
        wrapper = SkinModelWrapper(model, meta_t)
        gc = GradCAM(wrapper, model.backbone.conv_head)
        heat = gc(img_t)
        hi = Image.fromarray((cm.jet(heat)[:, :, :3] * 255).astype(np.uint8)).resize(img.size)
        ov = np.clip(0.55 * np.array(img.convert("RGB")).astype(float) + 0.45 * np.array(hi).astype(float), 0, 255).astype(np.uint8)
        buf_gc = io.BytesIO()
        Image.fromarray(ov).save(buf_gc, format="JPEG", quality=85)
        gradcam_b64 = base64.b64encode(buf_gc.getvalue()).decode()
    except Exception as e:
        print("Skin GradCAM error:", e)
        pass

    buf = io.BytesIO()
    img.thumbnail((400, 400))
    img.save(buf, format="JPEG", quality=85)
    orig_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "class":        pi,
        "label":        plb,
        "confidence":   pcf,
        "is_malignant": mal,
        "probs":        dict(zip(SKIN_CFG.labels, probs.tolist())),
        "orig_b64":     orig_b64,
        "gradcam_b64":  gradcam_b64,
    }


# ── AI endpoints ──────────────────────────────────────────────────────────────
class TriageReq(BaseModel):
    module:   str
    findings: dict

@app.post("/ai/triage")
def ai_triage(req: TriageReq):
    sys = ('Assess urgency from AI screening findings. '
           'Respond ONLY valid JSON: '
           '{"level":"low"|"medium"|"high","icon":"✅"|"⚠️"|"🚨","label":"Routine"|"Monitor"|"Urgent","reason":"1 sentence"}')
    raw = call_gemini(sys, [{"role": "user", "content":
        f"Module:{req.module}\n{json.dumps(req.findings)}"}], 120)
    try:
        t = json.loads(raw.strip().strip("```json").strip("```").strip())
    except Exception:
        t = {"level": "medium", "icon": "⚠️", "label": "Review needed",
             "reason": "Manual review recommended."}
    return t


class ReportReq(BaseModel):
    module:   str
    findings: dict
    ctx:      dict = {}

@app.post("/ai/report")
def ai_report(req: ReportReq):
    sys_map = {
        "chest": "Senior radiologist. Write structured report: FINDINGS, IMPRESSION, RECOMMENDATION. Max 220 words. Clinical terminology.",
        "eye":   "Ophthalmologist. Write structured report: FINDINGS, IMPRESSION, RECOMMENDATION. Max 200 words. ETDRS grading.",
        "skin":  "Dermatologist. Write structured report: FINDINGS, IMPRESSION, RECOMMENDATION. Flag malignancy clearly. Max 200 words.",
    }
    sys = sys_map.get(req.module, "Write medical report.")
    text = call_gemini(sys, [{"role": "user", "content":
        f"Findings:\n{json.dumps(req.findings)}\nContext:\n{json.dumps(req.ctx)}"}], 600)
    return {"report": text, "timestamp": datetime.datetime.now().strftime("%d %b %Y · %H:%M")}


class QuestionsReq(BaseModel):
    module:   str
    findings: dict

@app.post("/ai/questions")
def ai_questions(req: QuestionsReq):
    sys = 'Generate exactly 3 patient follow-up questions. Return ONLY a JSON array of 3 strings.'
    raw = call_gemini(sys, [{"role": "user", "content":
        f"{req.module}: {json.dumps(req.findings)}"}], 180)
    try:
        qs = json.loads(raw.strip().strip("```json").strip("```").strip())
        qs = qs[:3] if isinstance(qs, list) else []
    except Exception:
        qs = ["What do these results mean?", "What should I do next?", "How accurate is this AI?"]
    return {"questions": qs}


class ChatReq(BaseModel):
    module:   str
    findings: dict
    history:  list
    message:  str

@app.post("/ai/chat")
def ai_chat(req: ChatReq):
    sys = (f"You are MediScan AI assistant. Patient got {req.module} screening: "
           f"{json.dumps(req.findings)}. Answer questions clearly. Remind to consult doctor. Max 120 words.")
    msgs = req.history + [{"role": "user", "content": req.message}]
    reply = call_gemini(sys, msgs, 200)
    return {"reply": reply}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
