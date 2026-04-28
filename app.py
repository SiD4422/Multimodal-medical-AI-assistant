"""
MediScan AI — Nuvica-Inspired Full Product UI
===============================================
Design: Sky-blue gradient, bold Syne headers, white glass cards,
        floating stat badges, organ module cards, top navbar always visible.

Run:  streamlit run app.py
"""

import os, datetime, json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
import requests

from config import DEVICE, CHEST_CFG, EYE_CFG, SKIN_CFG
from utils  import get_val_transforms, GradCAM

# ── page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="MediScan AI", page_icon="🩺",
                   layout="wide", initial_sidebar_state="collapsed")

# ── session defaults ──────────────────────────────────────────────────────
for k, v in [("active_module","chest"),
             ("chest_report",""),("eye_report",""),("skin_report",""),
             ("chest_preds",{}),("eye_preds",{}),("skin_preds",{}),
             ("chest_ctx",{}),("eye_ctx",{}),("skin_ctx",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --sky:#ddeeff; --sky2:#c8e4f8; --sky3:#b0d4f0;
  --blue:#1a6fc4; --bd:#0f4f8f; --bl:#e8f2fb; --bxl:#f0f8ff;
  --teal:#2dd4bf; --teal-bg:#e0f7f5;
  --green:#10b981; --green-bg:#ecfdf5;
  --amber:#f59e0b; --amber-bg:#fffbeb;
  --red:#ef4444; --red-bg:#fef2f2;
  --purple:#8b5cf6;
  --text:#0a1628; --t2:#2d4a6e; --t3:#6a8caa; --t4:#b0cce0;
  --white:#fff; --border:rgba(26,111,196,0.12);
  --sh:0 2px 20px rgba(26,111,196,0.10);
  --sh-lg:0 8px 40px rgba(26,111,196,0.15);
}

*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif !important; color:var(--text) !important; -webkit-font-smoothing:antialiased;}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}

/* ── Background: sky gradient like Nuvica ── */
.stApp{
  background:linear-gradient(160deg,#ddeeff 0%,#c8e4f8 30%,#b8daf5 60%,#caf0ec 100%) !important;
  background-attachment:fixed !important;
  min-height:100vh;
}

/* ── Hide sidebar completely — we use top nav ── */
[data-testid="stSidebar"]{display:none !important;}
[data-testid="collapsedControl"]{display:none !important;}

/* ══════════════════════════════════════
   TOP NAVBAR
══════════════════════════════════════ */
.nav{
  background:rgba(255,255,255,0.92);
  backdrop-filter:blur(16px);
  border-bottom:1px solid rgba(26,111,196,0.12);
  padding:0 32px;
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:0;
  box-shadow:0 1px 12px rgba(26,111,196,0.08);
  position:sticky; top:0; z-index:999;
}
.nav-brand{display:flex;align-items:center;gap:10px;padding:14px 0;}
.nav-logo{
  width:34px;height:34px;border-radius:9px;
  background:linear-gradient(135deg,#1a6fc4,#2dd4bf);
  display:flex;align-items:center;justify-content:center;
  font-size:17px;flex-shrink:0;
  box-shadow:0 2px 10px rgba(26,111,196,0.35);
}
.nav-name{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:var(--blue);}
.nav-sub{font-size:0.6rem;color:var(--t3);text-transform:uppercase;letter-spacing:.1em;}
.nav-links{display:flex;gap:4px;align-items:center;}
.nav-link{
  padding:7px 16px;font-size:0.84rem;font-weight:600;
  color:var(--t2);border-radius:8px;cursor:pointer;
  border:none;background:transparent;
  font-family:'Plus Jakarta Sans',sans-serif;
  transition:all .15s;white-space:nowrap;
}
.nav-link:hover{background:var(--bl);color:var(--blue);}
.nav-link.active{background:var(--blue);color:white;box-shadow:0 2px 10px rgba(26,111,196,.3);}
.nav-right{display:flex;align-items:center;gap:10px;}
.live-badge{
  display:inline-flex;align-items:center;gap:5px;
  background:var(--green-bg);border:1px solid rgba(16,185,129,.25);
  color:#059669;font-size:.68rem;font-weight:700;
  padding:4px 12px;border-radius:999px;
}
.ld{width:6px;height:6px;border-radius:50%;background:#10b981;animation:blink 2s infinite;display:inline-block !important;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ══════════════════════════════════════
   HERO SECTION  (Nuvica style)
══════════════════════════════════════ */
.hero{
  background:rgba(255,255,255,0.75);
  backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.9);
  border-radius:24px;
  padding:40px 44px 36px;
  margin:28px 0 24px;
  position:relative; overflow:hidden;
  box-shadow:var(--sh-lg);
}
.hero::before{
  content:'';position:absolute;top:-100px;right:-80px;
  width:320px;height:320px;border-radius:50%;
  background:radial-gradient(circle,rgba(45,212,191,0.14) 0%,transparent 70%);
  pointer-events:none;
}
.hero::after{
  content:'';position:absolute;bottom:-80px;left:20%;
  width:240px;height:240px;border-radius:50%;
  background:radial-gradient(circle,rgba(26,111,196,0.09) 0%,transparent 70%);
  pointer-events:none;
}
.hero-tag{
  display:inline-flex;align-items:center;gap:6px;
  background:var(--bl);color:var(--blue);
  font-size:.7rem;font-weight:700;padding:4px 14px;
  border-radius:999px;border:1px solid rgba(26,111,196,.2);
  margin-bottom:14px;letter-spacing:.04em;
}
.hero-title{
  font-family:'Syne',sans-serif;
  font-size:2.8rem;font-weight:800;
  color:var(--text);letter-spacing:-.03em;
  line-height:1.08;margin:0 0 12px;
}
.hero-title em{color:var(--blue);font-style:normal;}
.hero-desc{color:var(--t2);font-size:.92rem;line-height:1.7;margin:0 0 22px;max-width:580px;}
.hero-row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;}

/* Floating stat cards — like Nuvica's "490 Awards" "22 Years" badges */
.fcard{
  display:inline-flex;align-items:center;gap:9px;
  background:white;border-radius:14px;
  padding:10px 16px;
  box-shadow:0 4px 20px rgba(26,111,196,.15);
  border:1px solid rgba(26,111,196,.08);
}
.fcard-icon{
  width:32px;height:32px;border-radius:9px;
  display:flex;align-items:center;justify-content:center;
  font-size:15px;flex-shrink:0;
}
.fcard-val{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;color:var(--text);line-height:1.1;}
.fcard-lbl{font-size:.62rem;color:var(--t3);text-transform:uppercase;letter-spacing:.06em;}
.explore-btn{
  display:inline-flex;align-items:center;gap:8px;
  background:var(--blue);color:white;border:none;
  border-radius:10px;padding:12px 24px;
  font-size:.88rem;font-weight:700;cursor:pointer;
  font-family:'Plus Jakarta Sans',sans-serif;
  box-shadow:0 4px 16px rgba(26,111,196,.35);
  transition:all .2s;text-decoration:none;
}
.explore-btn:hover{background:var(--bd);transform:translateY(-1px);box-shadow:0 6px 22px rgba(26,111,196,.4);}

/* ══════════════════════════════════════
   ORGAN MODULE CARDS  (like Nuvica brain/liver/kidney row)
══════════════════════════════════════ */
.module-row{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:0 0 28px;}
.module-card{
  background:white;border:1.5px solid var(--border);
  border-radius:18px;padding:22px;
  position:relative;overflow:hidden;
  cursor:pointer;transition:all .22s;
  box-shadow:0 2px 14px rgba(26,111,196,.07);
}
.module-card:hover{transform:translateY(-3px);box-shadow:0 8px 28px rgba(26,111,196,.14);}
.module-card.active{border-color:var(--blue);background:rgba(26,111,196,.03);}
.module-card-icon{font-size:2.4rem;margin-bottom:10px;display:block;}
.module-card-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:var(--text);margin:0 0 4px;}
.module-card-sub{font-size:.75rem;color:var(--t3);line-height:1.45;}
.module-card-arrow{
  position:absolute;top:16px;right:16px;
  width:28px;height:28px;border-radius:50%;
  background:var(--bl);display:flex;align-items:center;
  justify-content:center;font-size:12px;color:var(--blue);
  font-weight:700;transition:all .2s;
}
.module-card:hover .module-card-arrow{background:var(--blue);color:white;}
.module-card.active .module-card-arrow{background:var(--blue);color:white;}
.module-tag{
  display:inline-flex;align-items:center;gap:4px;
  background:var(--bl);color:var(--blue);
  font-size:.65rem;font-weight:700;padding:3px 10px;
  border-radius:999px;margin-top:8px;letter-spacing:.04em;
}

/* ══════════════════════════════════════
   STAT ROW  (like Nuvica $250M / 20M+ row)
══════════════════════════════════════ */
.stat-row{
  background:rgba(255,255,255,0.8);
  backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,.9);
  border-radius:18px;padding:24px 32px;
  margin:0 0 28px;
  display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:16px;
  box-shadow:0 2px 16px rgba(26,111,196,.08);
}
.stat-item{text-align:center;}
.stat-val{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--blue);}
.stat-lbl{font-size:.68rem;color:var(--t3);text-transform:uppercase;letter-spacing:.07em;margin-top:2px;}

/* ══════════════════════════════════════
   CONTENT CARDS
══════════════════════════════════════ */
.wcard{
  background:white;border:1px solid var(--border);
  border-radius:16px;padding:20px 22px;
  margin-bottom:14px;
  box-shadow:0 2px 12px rgba(26,111,196,.06);
  transition:box-shadow .2s;
}
.wcard:hover{box-shadow:0 4px 22px rgba(26,111,196,.11);}
.card-title{
  font-size:.67rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--t3);margin:0 0 14px;
  display:flex;align-items:center;gap:6px;
}
.ctdot{width:6px;height:6px;border-radius:50%;background:var(--blue);}

/* ══════════════════════════════════════
   BADGES & BARS
══════════════════════════════════════ */
.sbadge{display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:600;margin-bottom:10px;}
.sb-dot{width:6px;height:6px;border-radius:50%;background:currentColor;}
.sev-normal  {background:var(--green-bg);color:#059669;border:1px solid rgba(16,185,129,.2);}
.sev-mild    {background:var(--amber-bg);color:#d97706;border:1px solid rgba(245,158,11,.2);}
.sev-moderate{background:#fff7ed;color:#ea580c;border:1px solid rgba(249,115,22,.2);}
.sev-severe  {background:var(--red-bg);color:#dc2626;border:1px solid rgba(239,68,68,.2);}

.cbar{margin:4px 0 10px;}
.cbar-top{display:flex;justify-content:space-between;font-size:.73rem;color:var(--t2);margin-bottom:4px;font-weight:500;}
.cbar-trk{height:7px;border-radius:4px;background:rgba(26,111,196,.09);overflow:hidden;}
.cbar-fill{height:100%;border-radius:4px;}

.stile-row{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 14px;}
.stile{flex:1;min-width:76px;background:var(--bxl);border:1px solid rgba(26,111,196,.15);border-radius:10px;padding:10px 12px;text-align:center;}
.stile-val{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:var(--blue);line-height:1.1;}
.stile-lbl{font-size:.62rem;color:var(--t3);margin-top:2px;text-transform:uppercase;letter-spacing:.05em;}

.grade-track{display:flex;gap:4px;margin:8px 0 14px;}
.grade-seg{flex:1;height:8px;border-radius:4px;}

/* ══════════════════════════════════════
   TRIAGE CARD
══════════════════════════════════════ */
.triage-card{border-radius:14px;padding:16px 18px;margin-bottom:14px;border:2px solid;}
.triage-card.low{background:var(--green-bg);border-color:rgba(16,185,129,.28);}
.triage-card.medium{background:var(--amber-bg);border-color:rgba(245,158,11,.28);}
.triage-card.high{background:var(--red-bg);border-color:rgba(239,68,68,.28);}
.triage-header{display:flex;align-items:center;gap:9px;margin-bottom:5px;}
.triage-icon{font-size:1.4rem;}
.triage-label{font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;}
.triage-card.low .triage-label{color:#059669;}
.triage-card.medium .triage-label{color:#d97706;}
.triage-card.high .triage-label{color:#dc2626;}
.triage-desc{font-size:.82rem;line-height:1.55;}
.triage-card.low .triage-desc{color:#065f46;}
.triage-card.medium .triage-desc{color:#78350f;}
.triage-card.high .triage-desc{color:#7f1d1d;}

/* ══════════════════════════════════════
   AI CHAT
══════════════════════════════════════ */
.chat-wrap{background:white;border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--sh);margin-top:14px;}
.chat-head{background:linear-gradient(135deg,var(--blue),var(--bd));padding:12px 18px;display:flex;align-items:center;gap:9px;}
.chat-dot{width:8px;height:8px;border-radius:50%;background:var(--teal);box-shadow:0 0 6px var(--teal);animation:blink 2s infinite;}
.chat-title{font-size:.77rem;font-weight:700;color:white;letter-spacing:.05em;text-transform:uppercase;}
.chat-sub{font-size:.66rem;color:rgba(255,255,255,.6);}
.chat-msgs{padding:14px 16px;max-height:300px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;}
.msg-ai{background:var(--bxl);border:1px solid rgba(26,111,196,.13);border-radius:12px 12px 12px 2px;padding:10px 14px;font-size:.83rem;line-height:1.6;color:var(--text);max-width:85%;align-self:flex-start;}
.msg-user{background:var(--blue);border-radius:12px 12px 2px 12px;padding:10px 14px;font-size:.83rem;line-height:1.6;color:white;max-width:85%;align-self:flex-end;}
.msg-sender{font-size:.63rem;font-weight:700;letter-spacing:.06em;margin-bottom:3px;}
.msg-ai .msg-sender{color:var(--blue);}
.msg-user .msg-sender{color:rgba(255,255,255,.7);}
.quick-btns{padding:0 16px 12px;display:flex;flex-wrap:wrap;gap:6px;}
.qbtn{background:var(--bl);color:var(--blue);border:1px solid rgba(26,111,196,.18);border-radius:999px;font-size:.74rem;font-weight:500;padding:4px 12px;cursor:pointer;transition:all .15s;font-family:inherit;}
.qbtn:hover{background:var(--blue);color:white;}

/* ══════════════════════════════════════
   REPORT BOX
══════════════════════════════════════ */
.rbox{background:white;border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--sh);margin-top:14px;}
.rbox-head{background:linear-gradient(135deg,var(--blue),#1558a0);padding:12px 18px;display:flex;align-items:center;justify-content:space-between;}
.rbox-head-l{display:flex;align-items:center;gap:9px;}
.rbox-dot{width:7px;height:7px;border-radius:50%;background:var(--teal);box-shadow:0 0 6px var(--teal);}
.rbox-lbl{font-size:.72rem;font-weight:700;color:white;letter-spacing:.08em;text-transform:uppercase;}
.rbox-date{font-size:.68rem;color:rgba(255,255,255,.6);}
.rbox-body{padding:18px 20px;font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--text);white-space:pre-wrap;line-height:1.85;background:white;}

/* ══════════════════════════════════════
   MISC
══════════════════════════════════════ */
.ibox{background:var(--bxl);border:1px solid rgba(26,111,196,.18);border-radius:10px;padding:10px 14px;font-size:.84rem;color:var(--blue);}
.wbox{background:var(--amber-bg);border:1px solid rgba(245,158,11,.22);border-radius:10px;padding:10px 14px;font-size:.82rem;color:#92400e;display:flex;gap:8px;margin-top:8px;}
.sdiv{height:1px;background:var(--border);margin:16px 0;}
.slbl{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--t3);margin:0 0 10px;}
.warn-box{background:#fffbeb;border:1px solid rgba(245,158,11,.28);border-radius:10px;padding:10px 14px;font-size:.8rem;color:#92400e;margin-top:6px;}

/* ── Buttons ── */
.stButton > button{
  background:linear-gradient(135deg,var(--blue),var(--bd)) !important;
  color:white !important;border:none !important;border-radius:10px !important;
  font-weight:600 !important;font-size:.87rem !important;padding:10px 22px !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
  box-shadow:0 3px 12px rgba(26,111,196,.28) !important;
  transition:all .2s !important;
}
.stButton > button:hover{transform:translateY(-1px) !important;box-shadow:0 6px 20px rgba(26,111,196,.36) !important;}

/* Download button green */
[data-testid="stDownloadButton"] > button{
  background:linear-gradient(135deg,#10b981,#059669) !important;
  box-shadow:0 3px 12px rgba(16,185,129,.3) !important;
}
[data-testid="stDownloadButton"] > button:hover{box-shadow:0 6px 20px rgba(16,185,129,.4) !important;}

/* Nav buttons — flat style */
div[data-testid="stHorizontalBlock"] .stButton > button{
  background:rgba(255,255,255,.8) !important;
  color:var(--t2) !important;border:1px solid var(--border) !important;
  box-shadow:none !important;border-radius:9px !important;
  font-size:.83rem !important;padding:8px 10px !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover{
  background:var(--bl) !important;color:var(--blue) !important;
  transform:none !important;box-shadow:none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"]{background:var(--bxl) !important;border:2px dashed rgba(26,111,196,.25) !important;border-radius:12px !important;}
[data-testid="stFileUploader"]:hover{border-color:var(--blue) !important;}
[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] p,[data-testid="stFileUploader"] small{color:var(--t2) !important;}

/* ── Inputs ── */
.stSelectbox>div>div,.stTextInput>div>div>input,.stNumberInput>div>div>input,.stTextArea>div>div>textarea{
  background:white !important;border:1px solid var(--border) !important;
  border-radius:9px !important;color:var(--text) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important;
}
.stSelectbox label,.stTextInput label,.stNumberInput label,.stSlider label,.stCheckbox label{
  color:var(--t2) !important;font-size:.82rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"]{background:white;border:1px solid var(--border);border-radius:12px;padding:12px 16px !important;box-shadow:0 1px 8px rgba(26,111,196,.06);}
[data-testid="stMetricLabel"]>div{font-size:.66rem !important;color:var(--t3) !important;text-transform:uppercase;letter-spacing:.06em;}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif !important;font-size:1.3rem !important;font-weight:700 !important;color:var(--text) !important;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:rgba(26,111,196,.2);border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def badge(label, sev):
    return f'<span class="sbadge sev-{sev}"><span class="sb-dot"></span>{label}</span>'

def cbar(label, pct, color="var(--blue)"):
    pct = min(float(pct), 100)
    return (f'<div class="cbar"><div class="cbar-top"><span>{label}</span>'
            f'<span>{pct:.1f}%</span></div>'
            f'<div class="cbar-trk"><div class="cbar-fill" '
            f'style="width:{pct}%;background:{color}"></div></div></div>')

def tiles(*pairs):
    inner = "".join(f'<div class="stile"><div class="stile-val">{v}</div>'
                    f'<div class="stile-lbl">{l}</div></div>' for v,l in pairs)
    return f'<div class="stile-row">{inner}</div>'

def fcard(icon, val, lbl, color="#1a6fc4"):
    return (f'<div class="fcard">'
            f'<div class="fcard-icon" style="background:{color}18">{icon}</div>'
            f'<div><div class="fcard-val" style="color:{color}">{val}</div>'
            f'<div class="fcard-lbl">{lbl}</div></div></div>')

def rbox(text):
    ts = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
    return (f'<div class="rbox"><div class="rbox-head">'
            f'<div class="rbox-head-l"><div class="rbox-dot"></div>'
            f'<div class="rbox-lbl">AI Clinical Report</div></div>'
            f'<span class="rbox-date">{ts}</span></div>'
            f'<div class="rbox-body">{text}</div></div>')

def prob_chart(probs):
    labels = list(probs.keys()); values = [v*100 for v in probs.values()]
    mx = max(values) if values else 1
    colors = ["#1a6fc4" if v==mx else "#c8dff5" for v in values]
    fig = go.Figure(go.Bar(x=values,y=labels,orientation="h",
        marker_color=colors,marker_line_width=0,
        text=[f"{v:.1f}%" for v in values],textposition="outside",
        textfont=dict(color="#3d5a80",size=11,family="Plus Jakarta Sans")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans",color="#3d5a80",size=11),
        xaxis=dict(range=[0,115],showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,color="#0a1628",tickfont=dict(size=12,color="#0a1628")),
        margin=dict(l=0,r=50,t=6,b=0),height=max(150,len(labels)*40),bargap=0.38)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  AI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY","")

def call_claude(system, messages, max_tokens=500):
    if not API_KEY:
        return "⚠ Set ANTHROPIC_API_KEY to enable AI features."
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json","x-api-key":API_KEY,"anthropic-version":"2023-06-01"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":max_tokens,"system":system,"messages":messages},
            timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception as e:
        return f"AI error: {e}"

def ai_triage(module, findings):
    key = f"{module}_triage_cache"
    if key in st.session_state: return st.session_state[key]
    sys = ('Assess urgency from AI screening findings. '
           'Respond ONLY valid JSON: {"level":"low"|"medium"|"high","icon":"✅"|"⚠️"|"🚨","label":"Routine"|"Monitor"|"Urgent","reason":"1 sentence"}')
    raw = call_claude(sys,[{"role":"user","content":f"Module:{module}\n{json.dumps(findings)}"}],120)
    try:
        t = json.loads(raw.strip().strip("```json").strip("```").strip())
    except:
        t = {"level":"medium","icon":"⚠️","label":"Review needed","reason":"Manual review recommended."}
    st.session_state[key] = t
    return t

def ai_report(module, findings, ctx=None):
    sys = {"chest":"Senior radiologist. Write structured report: FINDINGS, IMPRESSION, RECOMMENDATION. Max 220 words. Clinical terminology.",
           "eye":"Ophthalmologist. Write structured report: FINDINGS, IMPRESSION, RECOMMENDATION. Max 200 words. ETDRS grading.",
           "skin":"Dermatologist. Write structured report: FINDINGS, IMPRESSION, RECOMMENDATION. Flag malignancy clearly. Max 200 words."}.get(module,"Write medical report.")
    return call_claude(sys,[{"role":"user","content":f"Findings:\n{json.dumps(findings)}\nContext:\n{json.dumps(ctx or {})}"}],600)

def ai_questions(module, findings):
    key = f"{module}_qs"
    if key in st.session_state: return st.session_state[key]
    sys = 'Generate exactly 3 patient follow-up questions. Return ONLY a JSON array of 3 strings.'
    raw = call_claude(sys,[{"role":"user","content":f"{module}: {json.dumps(findings)}"}],180)
    try:
        qs = json.loads(raw.strip().strip("```json").strip("```").strip())
        qs = qs[:3] if isinstance(qs,list) else []
    except:
        qs = ["What do these results mean?","What should I do next?","How accurate is this AI?"]
    st.session_state[key] = qs
    return qs

def ai_chat_reply(module, findings, history, user_msg):
    sys = (f"You are MediScan AI assistant. Patient got {module} screening: {json.dumps(findings)}. "
           f"Answer questions clearly. Remind to consult doctor. Max 120 words.")
    msgs = history + [{"role":"user","content":user_msg}]
    return call_claude(sys, msgs, 200)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_chest():
    from module1_chest import XrayEncoder, ECGEncoder
    xray=XrayEncoder().to(DEVICE); ecg=ECGEncoder().to(DEVICE)
    for ck,m in [("checkpoints/xray_best.pt",xray),("checkpoints/ecg_best.pt",ecg)]:
        if os.path.exists(ck): m.load_state_dict(torch.load(ck,map_location=DEVICE))
    return xray.eval(), ecg.eval()

@st.cache_resource
def load_eye():
    from module2_eye import DREncoder, GlaucomaEncoder
    dr=DREncoder().to(DEVICE); glc=GlaucomaEncoder().to(DEVICE)
    for ck,m in [("checkpoints/dr_best.pt",dr),("checkpoints/glaucoma_best.pt",glc)]:
        if os.path.exists(ck): m.load_state_dict(torch.load(ck,map_location=DEVICE))
    return dr.eval(), glc.eval()

@st.cache_resource
def load_skin():
    from module3_skin import SkinEncoder, META_COLS
    m=SkinEncoder(num_classes=SKIN_CFG.num_classes,meta_dim=len(META_COLS)).to(DEVICE)
    if os.path.exists("checkpoints/skin_best.pt"):
        m.load_state_dict(torch.load("checkpoints/skin_best.pt",map_location=DEVICE))
    return m.eval()

# ─────────────────────────────────────────────────────────────────────────────
#  RENDER: TRIAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_triage(module, findings):
    t = ai_triage(module, findings)
    level = t.get("level","medium")
    st.markdown(
        f'<div class="triage-card {level}">'
        f'<div class="triage-header"><span class="triage-icon">{t.get("icon","⚠️")}</span>'
        f'<span class="triage-label">{t.get("label","Review")} Priority</span></div>'
        f'<div class="triage-desc">{t.get("reason","")}</div></div>',
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  RENDER: AI CHAT
# ─────────────────────────────────────────────────────────────────────────────
def render_chat(module, findings):
    if not findings: return
    key = f"{module}_chat"
    if key not in st.session_state:
        st.session_state[key] = [{"role":"assistant",
            "content":f"Hi! I've reviewed your {module} screening results. Ask me anything about what they mean."}]

    msgs_html = "".join(
        f'<div class="msg-ai"><div class="msg-sender">MEDISCAN AI</div>{m["content"]}</div>'
        if m["role"]=="assistant" else
        f'<div class="msg-user"><div class="msg-sender">YOU</div>{m["content"]}</div>'
        for m in st.session_state[key])

    qs = ai_questions(module, findings)
    qbtns = "".join(f'<button class="qbtn">{q}</button>' for q in qs)

    st.markdown(
        f'<div class="chat-wrap">'
        f'<div class="chat-head"><div class="chat-dot"></div>'
        f'<div><div class="chat-title">MediScan AI Assistant</div>'
        f'<div class="chat-sub">Ask about your results</div></div></div>'
        f'<div class="chat-msgs">{msgs_html}</div>'
        f'<div class="quick-btns">{qbtns}</div></div>',
        unsafe_allow_html=True)

    c1,c2 = st.columns([5,1])
    with c1:
        user_input = st.text_input("",placeholder="Type a question...",
                                    key=f"{key}_inp",label_visibility="collapsed")
    with c2:
        send = st.button("Send",key=f"{key}_send")

    sel = st.selectbox("Quick question:",[""] + qs,key=f"{key}_sel",label_visibility="collapsed")
    msg = user_input if (send and user_input) else (sel if sel else None)
    if msg:
        conv = [{"role":m["role"],"content":m["content"]} for m in st.session_state[key]]
        with st.spinner("AI thinking..."):
            reply = ai_chat_reply(module, findings, conv, msg)
        st.session_state[key] += [{"role":"user","content":msg},{"role":"assistant","content":reply}]
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  PDF DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def pdf_btn(module_key, preds, report_text, ctx):
    try:
        from pdf_report import generate_pdf_report
        pdf = generate_pdf_report(module_key, preds, report_text, ctx)
        fname = f"mediscan_{module_key}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button("⬇ Download PDF Report",data=pdf,file_name=fname,mime="application/pdf")
    except Exception as e:
        st.error(f"PDF error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  TOP NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
  <div class="nav-brand">
    <div class="nav-logo">🩺</div>
    <div>
      <div class="nav-name">MediScan AI</div>
      <div class="nav-sub">Diagnostic Intelligence</div>
    </div>
  </div>
  <div class="nav-right">
    <span class="live-badge"><span class="ld"></span>AI Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
fcards_html = " ".join([
    fcard("🫁","14 Findings","Chest X-ray","#1a6fc4"),
    fcard("💓","5 Classes","ECG Rhythm","#2dd4bf"),
    fcard("👁","Grade 0–4","DR Grading","#10b981"),
    fcard("🩹","7 Types","Skin Lesion","#f59e0b"),
    fcard("🤖","AI Powered","Chat + Triage","#8b5cf6"),
])

st.markdown(f"""
<div class="hero">
  <div class="hero-tag">🔬 Multimodal Medical AI</div>
  <div class="hero-title">Quick. Smart.<br><em>Medical AI.</em></div>
  <div class="hero-desc">
    AI-powered diagnostic screening across Chest, Eye, and Skin — powered by DenseNet, EfficientNet,
    and 1D-CNN Transformer models trained on 5+ public medical datasets.
    Generate structured clinical reports with one click.
  </div>
  <div class="hero-row">
    {fcards_html}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STAT ROW  (Nuvica-style numbers)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-item"><div class="stat-val">26+</div><div class="stat-lbl">Disease Classes</div></div>
  <div class="stat-item"><div class="stat-val">3</div><div class="stat-lbl">Diagnostic Modules</div></div>
  <div class="stat-item"><div class="stat-val">5+</div><div class="stat-lbl">Free Datasets</div></div>
  <div class="stat-item"><div class="stat-val">AI</div><div class="stat-lbl">Chat + Triage</div></div>
  <div class="stat-item"><div class="stat-val">PDF</div><div class="stat-lbl">Report Export</div></div>
  <div class="stat-item" style="font-size:.75rem;color:#6a8caa;text-align:right;">
    MediScan AI<br>Research & Educational Use Only
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE SELECTOR CARDS  (Nuvica organ cards)
# ─────────────────────────────────────────────────────────────────────────────
active = st.session_state["active_module"]

st.markdown(f"""
<div class="module-row">
  <div class="module-card {"active" if active=="chest" else ""}" id="mc-chest">
    <div class="module-card-arrow">↗</div>
    <span class="module-card-icon">🫁</span>
    <div class="module-card-title">Chest Diagnosis</div>
    <div class="module-card-sub">X-ray pathology detection + ECG rhythm classification</div>
    <span class="module-tag">DenseNet-121 · 1D-CNN</span>
  </div>
  <div class="module-card {"active" if active=="eye" else ""}" id="mc-eye">
    <div class="module-card-arrow">↗</div>
    <span class="module-card-icon">👁</span>
    <div class="module-card-title">Eye Screening</div>
    <div class="module-card-sub">Diabetic retinopathy grading + glaucoma risk</div>
    <span class="module-tag">EfficientNet-B4 · B2</span>
  </div>
  <div class="module-card {"active" if active=="skin" else ""}" id="mc-skin">
    <div class="module-card-arrow">↗</div>
    <span class="module-card-icon">🩹</span>
    <div class="module-card-title">Skin Analysis</div>
    <div class="module-card-sub">7-class lesion classification + malignancy scoring</div>
    <span class="module-tag">EfficientNet-B4 · Fusion</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Real Streamlit buttons that actually switch modules
c1,c2,c3 = st.columns(3)
with c1:
    if st.button("🫁 Open Chest Module",key="nav_c",use_container_width=True):
        st.session_state["active_module"]="chest"; st.rerun()
with c2:
    if st.button("👁 Open Eye Module",key="nav_e",use_container_width=True):
        st.session_state["active_module"]="eye"; st.rerun()
with c3:
    if st.button("🩹 Open Skin Module",key="nav_s",use_container_width=True):
        st.session_state["active_module"]="skin"; st.rerun()

st.markdown('<div class="sdiv"></div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1 — CHEST
# ─────────────────────────────────────────────────────────────────────────────
def page_chest():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
      <span style="font-size:1.8rem;">🫁</span>
      <div>
        <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#0a1628;">Chest Diagnosis</div>
        <div style="font-size:.82rem;color:#6a8caa;">X-ray pathology detection · ECG rhythm analysis</div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    xray_model,ecg_model = load_chest()
    tf = get_val_transforms(CHEST_CFG.xray_img_size)

    col1,col2 = st.columns(2,gap="large")

    with col1:
        st.markdown('<div class="wcard"><div class="card-title"><div class="ctdot"></div>Chest X-ray</div>',unsafe_allow_html=True)
        xf = st.file_uploader("Upload X-ray image",type=["jpg","jpeg","png"],key="xray",label_visibility="collapsed")
        st.markdown('</div>',unsafe_allow_html=True)

        if xf:
            img=Image.open(xf).convert("RGB"); img_t=tf(img)
            with torch.no_grad():
                probs=torch.sigmoid(xray_model(img_t.unsqueeze(0).to(DEVICE))).cpu().numpy()[0]
            xray_preds=dict(zip(CHEST_CFG.xray_labels,probs.tolist()))
            top_idx=int(np.argmax(probs)); top_label=CHEST_CFG.xray_labels[top_idx]; top_conf=float(probs[top_idx])
            st.session_state["chest_preds"]["xray"]=xray_preds

            st.markdown('<div class="wcard">',unsafe_allow_html=True)
            c1a,c1b=st.columns(2)
            with c1a: st.image(img,caption="Original",use_column_width=True)
            with c1b:
                try:
                    import matplotlib.cm as cm
                    gc=GradCAM(xray_model,xray_model.features[-1]); heat=gc(img_t)
                    hi=Image.fromarray((cm.jet(heat)[:,:,:3]*255).astype(np.uint8)).resize(img.size)
                    ov=np.clip(.55*np.array(img.convert("RGB")).astype(float)+.45*np.array(hi).astype(float),0,255).astype(np.uint8)
                    st.image(Image.fromarray(ov),caption=f"Grad-CAM · {top_label}",use_column_width=True)
                except: st.image(img,caption="Grad-CAM",use_column_width=True)
            sev="severe" if top_conf>.7 else "moderate" if top_conf>.4 else "mild"
            st.markdown(badge(f"Primary: {top_label}",sev),unsafe_allow_html=True)
            st.markdown(cbar("Confidence",top_conf*100),unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

            st.markdown('<div class="wcard"><div class="card-title"><div class="ctdot"></div>Top 5 Findings</div>',unsafe_allow_html=True)
            top5=dict(sorted(xray_preds.items(),key=lambda x:-x[1])[:5])
            st.plotly_chart(prob_chart(top5),use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="ibox">Upload a chest X-ray (.jpg/.png) to detect 14 pathologies with Grad-CAM localisation.</div>',unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="wcard"><div class="card-title"><div class="ctdot"></div>ECG Rhythm</div>',unsafe_allow_html=True)
        use_demo=st.checkbox("Use demo ECG signal",value=True)
        st.markdown('</div>',unsafe_allow_html=True)
        ecg_t=None
        if use_demo: ecg_t=torch.randn(1,12,1000)
        else:
            ef=st.file_uploader("Upload ECG .npy",type=["npy"],key="ecg",label_visibility="collapsed")
            if ef:
                arr=np.load(ef); ecg_t=torch.tensor(arr).float()
                if ecg_t.dim()==2: ecg_t=ecg_t.unsqueeze(0)

        if ecg_t is not None:
            with torch.no_grad():
                probs=F.softmax(ecg_model(ecg_t.to(DEVICE)),dim=1).cpu().numpy()[0]
            pi=int(np.argmax(probs)); label=CHEST_CFG.ecg_labels[pi]; conf=float(probs[pi])
            ecg_preds={"label":label,"confidence":conf,"probs":dict(zip(CHEST_CFG.ecg_labels,probs.tolist()))}
            st.session_state["chest_preds"]["ecg"]=ecg_preds
            full_map={"NORM":("Normal Sinus Rhythm","normal","#10b981"),
                      "MI":("Myocardial Infarction","severe","#ef4444"),
                      "STTC":("ST/T-wave Change","moderate","#f97316"),
                      "CD":("Conduction Defect","moderate","#f59e0b"),
                      "HYP":("Hypertrophy","mild","#6366f1")}
            fl,sev,col=full_map.get(label,(label,"mild","#1a6fc4"))
            st.markdown('<div class="wcard">',unsafe_allow_html=True)
            st.markdown(badge(fl,sev),unsafe_allow_html=True)
            st.markdown(tiles((label,"Rhythm class"),(f"{conf:.0%}","Confidence")),unsafe_allow_html=True)
            cm_map={"NORM":"#10b981","MI":"#ef4444","STTC":"#f97316","CD":"#f59e0b","HYP":"#6366f1"}
            for lbl,p in ecg_preds["probs"].items():
                st.markdown(cbar(lbl,p*100,cm_map.get(lbl,"#1a6fc4")),unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

        preds=st.session_state.get("chest_preds",{})
        if preds: render_triage("chest",preds)

    st.markdown('<div class="sdiv"></div>',unsafe_allow_html=True)
    preds=st.session_state.get("chest_preds",{})
    if preds:
        c1,c2,_ = st.columns([1,1,2])
        with c1:
            if st.button("Generate AI Report",key="ch_rpt"):
                with st.spinner("Claude writing report..."):
                    rt=ai_report("chest",preds)
                    st.session_state["chest_report"]=rt
        with c2:
            if st.session_state.get("chest_report"):
                pdf_btn("chest",preds,st.session_state["chest_report"],st.session_state.get("chest_ctx",{}))
        if st.session_state.get("chest_report"):
            st.markdown(rbox(st.session_state["chest_report"]),unsafe_allow_html=True)
        render_chat("chest",preds)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 — EYE
# ─────────────────────────────────────────────────────────────────────────────
def page_eye():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
      <span style="font-size:1.8rem;">👁</span>
      <div>
        <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#0a1628;">Eye Screening</div>
        <div style="font-size:.82rem;color:#6a8caa;">Diabetic retinopathy grading · Glaucoma risk</div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    dr_model,glc_model=load_eye(); tf=get_val_transforms(EYE_CFG.dr_img_size)
    col1,col2=st.columns([1,1],gap="large")

    with col1:
        st.markdown('<div class="wcard"><div class="card-title"><div class="ctdot"></div>Retinal Fundus Image</div>',unsafe_allow_html=True)
        f=st.file_uploader("Upload fundus photograph",type=["jpg","jpeg","png"],key="fundus",label_visibility="collapsed")
        st.markdown('</div>',unsafe_allow_html=True)
        if f:
            img=Image.open(f).convert("RGB")
            st.markdown('<div class="wcard">',unsafe_allow_html=True)
            st.image(img,use_column_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

    with col2:
        if f:
            inp=tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dr_probs=F.softmax(dr_model(inp),dim=1).cpu().numpy()[0]
                glc_score=torch.sigmoid(glc_model(inp)).cpu().item()
            di=int(np.argmax(dr_probs)); dlbl=EYE_CFG.dr_labels[di]; dcon=float(dr_probs[di])
            eye_preds={"dr":{"grade":di,"label":dlbl,"confidence":dcon,"probs":dict(zip(EYE_CFG.dr_labels,dr_probs.tolist()))},"glaucoma":{"score":glc_score}}
            st.session_state["eye_preds"]=eye_preds
            sm=["normal","mild","moderate","severe","severe"]
            st.markdown('<div class="wcard">',unsafe_allow_html=True)
            st.markdown(badge(f"DR Grade {di} — {dlbl}",sm[di]),unsafe_allow_html=True)
            seg_c=["#10b981","#a7f3d0","#fbbf24","#f97316","#ef4444"]
            segs="".join(f'<div class="grade-seg" style="background:{"#dbeeff" if i>di else seg_c[di]}"></div>' for i in range(5))
            st.markdown(f'<div class="grade-track">{segs}</div>',unsafe_allow_html=True)
            glc_col="#ef4444" if glc_score>.5 else "#10b981"
            st.markdown(tiles((str(di),"DR grade"),(f"{dcon:.0%}","Confidence"),("High" if glc_score>.5 else "Low","Glaucoma risk")),unsafe_allow_html=True)
            grd_c=["#10b981","#34d399","#fbbf24","#f97316","#ef4444"]
            for i,(lbl,p) in enumerate(zip(EYE_CFG.dr_labels,dr_probs)):
                st.markdown(cbar(lbl,p*100,grd_c[i]),unsafe_allow_html=True)
            st.markdown(cbar("Glaucoma risk",glc_score*100,glc_col),unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
            render_triage("eye",eye_preds)

            st.markdown('<div class="wcard"><div class="card-title"><div class="ctdot"></div>Patient Context</div>',unsafe_allow_html=True)
            ca,cb=st.columns(2)
            with ca: age=st.number_input("Age",0,120,50)
            with cb: hba1c=st.text_input("HbA1c",placeholder="e.g. 8.5%")
            ctx={"age":age}
            if hba1c: ctx["hba1c"]=hba1c
            st.session_state["eye_ctx"]=ctx
            c1b,c2b,_=st.columns([1,1,2])
            with c1b:
                if st.button("Generate AI Report",key="eye_rpt"):
                    with st.spinner("Claude writing..."):
                        rt=ai_report("eye",eye_preds,ctx)
                        st.session_state["eye_report"]=rt
            with c2b:
                if st.session_state.get("eye_report"):
                    pdf_btn("eye",eye_preds,st.session_state["eye_report"],ctx)
            st.markdown('</div>',unsafe_allow_html=True)
            if st.session_state.get("eye_report"):
                st.markdown(rbox(st.session_state["eye_report"]),unsafe_allow_html=True)
            render_chat("eye",eye_preds)
        else:
            st.markdown('<div class="ibox" style="margin-top:2rem;">Upload a retinal fundus photograph to begin DR grading and glaucoma screening.</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 — SKIN
# ─────────────────────────────────────────────────────────────────────────────
def page_skin():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
      <span style="font-size:1.8rem;">🩹</span>
      <div>
        <div style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#0a1628;">Skin Lesion Analysis</div>
        <div style="font-size:.82rem;color:#6a8caa;">7-class classification · Malignancy scoring · Metadata fusion</div>
      </div>
    </div>
    """,unsafe_allow_html=True)

    from module3_skin import META_COLS,SITE_COLS
    model=load_skin(); tf=get_val_transforms(SKIN_CFG.img_size)
    MALIGNANT={"Melanoma","Basal cell carcinoma","Actinic keratosis"}
    col1,col2=st.columns([1,1],gap="large")

    with col1:
        st.markdown('<div class="wcard"><div class="card-title"><div class="ctdot"></div>Lesion Image</div>',unsafe_allow_html=True)
        f=st.file_uploader("Upload skin image",type=["jpg","jpeg","png"],key="skin",label_visibility="collapsed")
        st.markdown('</div>',unsafe_allow_html=True)
        if f:
            img=Image.open(f).convert("RGB")
            st.markdown('<div class="wcard">',unsafe_allow_html=True)
            st.image(img,use_column_width=True)
            st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('<div class="wcard"><div class="card-title"><div class="ctdot"></div>Patient Metadata</div>',unsafe_allow_html=True)
        ca,cb=st.columns(2)
        with ca: age=st.slider("Age",0,100,45)
        with cb: sex=st.selectbox("Sex",["Male","Female","Unknown"])
        site=st.selectbox("Anatomical site",SITE_COLS)
        st.markdown('</div>',unsafe_allow_html=True)

    with col2:
        if f:
            img_t=tf(img)
            meta_t=torch.tensor(np.array([age/90.,{"Male":1.,"Female":0.,"Unknown":.5}[sex]]+[1. if s==site else 0. for s in SITE_COLS],dtype=np.float32)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs=F.softmax(model(img_t.unsqueeze(0).to(DEVICE),meta_t),dim=1).cpu().numpy()[0]
            pi=int(np.argmax(probs)); plb=SKIN_CFG.labels[pi]; pcf=float(probs[pi])
            mal=plb in MALIGNANT
            skin_preds={"class":pi,"label":plb,"confidence":pcf,"is_malignant":mal,"probs":dict(zip(SKIN_CFG.labels,probs.tolist()))}
            ctx={"age":age,"sex":sex,"site":site}
            st.session_state["skin_preds"]=skin_preds; st.session_state["skin_ctx"]=ctx
            st.markdown('<div class="wcard">',unsafe_allow_html=True)
            st.markdown(badge(plb,"severe" if mal else "normal"),unsafe_allow_html=True)
            st.markdown(tiles((f"{pcf:.0%}","Confidence"),("HIGH" if mal else "LOW","Malignancy")),unsafe_allow_html=True)
            if mal:
                st.markdown('<div class="wbox"><span>⚠</span><span>Potential malignancy. Urgent dermatology referral recommended.</span></div>',unsafe_allow_html=True)
            for lbl,p in sorted(zip(SKIN_CFG.labels,probs),key=lambda x:-x[1]):
                st.markdown(cbar(lbl,p*100,"#ef4444" if lbl in MALIGNANT else "#1a6fc4"),unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
            render_triage("skin",skin_preds)
            c1b,c2b,_=st.columns([1,1,2])
            with c1b:
                if st.button("Generate AI Report",key="sk_rpt"):
                    with st.spinner("Claude writing..."):
                        rt=ai_report("skin",skin_preds,ctx)
                        st.session_state["skin_report"]=rt
            with c2b:
                if st.session_state.get("skin_report"):
                    pdf_btn("skin",skin_preds,st.session_state["skin_report"],ctx)
            if st.session_state.get("skin_report"):
                st.markdown(rbox(st.session_state["skin_report"]),unsafe_allow_html=True)
            render_chat("skin",skin_preds)
        else:
            st.markdown('<div class="ibox" style="margin-top:2rem;">Upload a skin lesion image, fill patient metadata, then run classification.</div>',unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────────────────────
mod = st.session_state["active_module"]
if   mod=="chest": page_chest()
elif mod=="eye":   page_eye()
else:              page_skin()

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sdiv"></div>
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;padding:.5rem 0 1rem;">
  <div style="display:flex;gap:24px;flex-wrap:wrap;">
    <div style="text-align:center;"><div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#1a6fc4;">3</div><div style="font-size:.62rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.06em;">Modules</div></div>
    <div style="text-align:center;"><div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#10b981;">26+</div><div style="font-size:.62rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.06em;">Disease classes</div></div>
    <div style="text-align:center;"><div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#f59e0b;">5+</div><div style="font-size:.62rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.06em;">Datasets</div></div>
    <div style="text-align:center;"><div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#8b5cf6;">AI</div><div style="font-size:.62rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.06em;">Chat + Triage</div></div>
    <div style="text-align:center;"><div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#ef4444;">PDF</div><div style="font-size:.62rem;color:#6a8caa;text-transform:uppercase;letter-spacing:.06em;">Reports</div></div>
  </div>
  <div style="font-size:.72rem;color:#6a8caa;">MediScan AI · Research & Educational Use Only · Not for Clinical Diagnosis</div>
</div>
""",unsafe_allow_html=True)
