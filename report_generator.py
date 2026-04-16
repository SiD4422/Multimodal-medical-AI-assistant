"""
Multimodal Medical AI — LLM Clinical Report Generator
=======================================================
Takes raw model predictions from any module and generates
a structured, human-readable clinical report using the Claude API.

This is the feature that turns "Prediction: Grade 3 DR (0.87)"
into something a clinician can actually hand to a patient.

Usage:
    report = generate_report("eye", predictions, patient_context)
    print(report)
"""

import os
import json
import requests
from dataclasses import dataclass
from typing import Optional

from config import REPORT_CFG as CFG


# ── Report templates per module ────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "chest": """You are a clinical AI assistant generating structured radiology-style reports.
Given model predictions for chest X-ray and ECG analysis, produce a concise clinical report with:
1. FINDINGS: What the AI detected and confidence levels
2. IMPRESSION: 1-2 sentence clinical summary
3. RECOMMENDATION: Suggested next steps (tests, referrals, urgency)
4. DISCLAIMER: Standard AI-assisted diagnosis disclaimer

Be precise, use clinical terminology, and flag high-urgency findings clearly.
Keep the report under 250 words. Do not invent findings not supported by the predictions.""",

    "eye": """You are a clinical AI assistant generating structured ophthalmology reports.
Given model predictions for diabetic retinopathy grading and glaucoma screening, produce a report with:
1. FINDINGS: DR grade, glaucoma risk, confidence
2. IMPRESSION: Clinical significance of findings
3. RECOMMENDATION: Referral urgency (routine/urgent/emergency) and suggested follow-up
4. DISCLAIMER: Standard AI-assisted screening disclaimer

Use ETDRS grading terminology for DR. Keep under 200 words.""",

    "skin": """You are a clinical AI assistant generating structured dermatology screening reports.
Given model predictions for skin lesion classification, produce a report with:
1. FINDINGS: Most likely diagnosis, differential diagnoses, malignancy risk
2. IMPRESSION: Overall risk assessment
3. RECOMMENDATION: Urgency of dermatology referral (routine/urgent), whether biopsy may be warranted
4. DISCLAIMER: Standard AI screening disclaimer — not a replacement for biopsy/pathology

Keep under 200 words. Be appropriately cautious about malignant findings.""",
}


# ── Prediction formatters ──────────────────────────────────────────────────

def format_chest_predictions(preds: dict) -> str:
    lines = ["CHEST AI ANALYSIS RESULTS:\n"]

    if "xray" in preds:
        lines.append("X-RAY FINDINGS:")
        for label, prob in sorted(preds["xray"].items(),
                                   key=lambda x: -x[1]):
            if prob > 0.1:
                lines.append(f"  {label}: {prob:.1%}")

    if "ecg" in preds:
        lines.append("\nECG FINDINGS:")
        lines.append(f"  Rhythm class: {preds['ecg']['label']} "
                     f"(confidence: {preds['ecg']['confidence']:.1%})")
        for label, prob in preds['ecg'].get('probs', {}).items():
            lines.append(f"    {label}: {prob:.1%}")

    if "urgency" in preds:
        urgency_map = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        lines.append(f"\nPREDICTED URGENCY: "
                     f"{urgency_map.get(preds['urgency'], 'UNKNOWN')}")

    return "\n".join(lines)


def format_eye_predictions(preds: dict) -> str:
    lines = ["EYE AI SCREENING RESULTS:\n"]

    if "dr" in preds:
        dr = preds["dr"]
        lines.append(f"DIABETIC RETINOPATHY:")
        lines.append(f"  Grade: {dr['grade']} — {dr['label']}")
        lines.append(f"  Confidence: {dr['confidence']:.1%}")
        lines.append("  Grade distribution:")
        for lbl, p in dr["probs"].items():
            lines.append(f"    {lbl}: {p:.1%}")

    if "glaucoma" in preds:
        g = preds["glaucoma"]
        risk = "HIGH" if g["score"] > 0.5 else "LOW"
        lines.append(f"\nGLAUCOMA SCREENING:")
        lines.append(f"  Risk score: {g['score']:.1%} ({risk} risk)")

    return "\n".join(lines)


def format_skin_predictions(preds: dict) -> str:
    lines = ["SKIN LESION AI ANALYSIS:\n"]
    lines.append(f"PRIMARY DIAGNOSIS: {preds['label']} "
                 f"(confidence: {preds['confidence']:.1%})")
    lines.append(f"MALIGNANCY CONCERN: "
                 f"{'YES' if preds.get('is_malignant') else 'NO'}")
    lines.append("\nDIFFERENTIAL (all classes):")
    for lbl, p in sorted(preds["probs"].items(), key=lambda x: -x[1]):
        lines.append(f"  {lbl}: {p:.1%}")
    return "\n".join(lines)


FORMATTERS = {
    "chest": format_chest_predictions,
    "eye":   format_eye_predictions,
    "skin":  format_skin_predictions,
}


# ── Claude API call ────────────────────────────────────────────────────────

def call_claude(system: str, user_message: str) -> str:
    """
    Calls the Claude API to generate a clinical report.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = CFG.api_key
    if not api_key:
        return _fallback_report(user_message)

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model":      CFG.model,
        "max_tokens": CFG.max_tokens,
        "system":     system,
        "messages":   [{"role": "user", "content": user_message}],
    }

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        return f"[Report generation error: {e}]\n\n{_fallback_report(user_message)}"


def _fallback_report(prediction_text: str) -> str:
    """Simple template fallback if API key is not set."""
    return (
        "AUTOMATED SCREENING REPORT\n"
        "===========================\n\n"
        f"{prediction_text}\n\n"
        "DISCLAIMER: This report is generated by an AI screening system "
        "and is not a substitute for clinical evaluation. "
        "Please consult a qualified healthcare professional for diagnosis and treatment."
    )


# ── Main report generator ──────────────────────────────────────────────────

def generate_report(module: str,
                    predictions: dict,
                    patient_context: Optional[dict] = None) -> str:
    """
    Generate a structured clinical report from model predictions.

    Args:
        module          : "chest", "eye", or "skin"
        predictions     : raw output dict from the corresponding module
        patient_context : optional dict with age, sex, chief_complaint, etc.

    Returns:
        Formatted clinical report string.
    """
    if module not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown module '{module}'. "
                         f"Choose from: {list(SYSTEM_PROMPTS.keys())}")

    formatter      = FORMATTERS[module]
    prediction_str = formatter(predictions)

    user_msg_parts = [prediction_str]

    if patient_context:
        ctx_lines = ["\nPATIENT CONTEXT:"]
        for k, v in patient_context.items():
            ctx_lines.append(f"  {k.replace('_',' ').title()}: {v}")
        user_msg_parts.append("\n".join(ctx_lines))

    user_msg_parts.append(
        "\nPlease generate a structured clinical report based on the above.")

    user_message = "\n".join(user_msg_parts)
    system       = SYSTEM_PROMPTS[module]

    return call_claude(system, user_message)


# ── Convenience wrappers ───────────────────────────────────────────────────

def chest_report(xray_preds: dict, ecg_preds: dict,
                 urgency: int = None,
                 patient_context: dict = None) -> str:
    preds = {"xray": xray_preds, "ecg": ecg_preds}
    if urgency is not None:
        preds["urgency"] = urgency
    return generate_report("chest", preds, patient_context)


def eye_report(dr_preds: dict, glaucoma_score: float,
               patient_context: dict = None) -> str:
    preds = {
        "dr":       dr_preds,
        "glaucoma": {"score": glaucoma_score},
    }
    return generate_report("eye", preds, patient_context)


def skin_report(skin_preds: dict,
                patient_context: dict = None) -> str:
    return generate_report("skin", skin_preds, patient_context)


# ── Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: eye report
    sample_dr_preds = {
        "grade": 3, "label": "Severe",
        "confidence": 0.84,
        "probs": {
            "No DR": 0.02, "Mild": 0.05, "Moderate": 0.09,
            "Severe": 0.84, "Proliferative DR": 0.00,
        },
    }
    print(eye_report(
        dr_preds=sample_dr_preds,
        glaucoma_score=0.23,
        patient_context={
            "age": 58, "sex": "female",
            "chief_complaint": "blurry vision for 3 months",
            "hba1c": "9.2%",
        }
    ))
