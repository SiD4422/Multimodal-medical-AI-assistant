"""
MediScan AI — PDF Report Generator
=====================================
Generates a professional medical-style PDF report
inspired by clean clinical invoice / report layouts.

Usage:
    from pdf_report import generate_pdf_report
    pdf_bytes = generate_pdf_report("chest", predictions, patient_context)
    # returns bytes — pass to st.download_button
"""

import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Brand colours (matching app palette) ─────────────────────────────────
TEAL      = colors.HexColor("#2dd4bf")
BLUE      = colors.HexColor("#1a6fc4")
DARK      = colors.HexColor("#0f1f3d")
MID       = colors.HexColor("#3d5a80")
LIGHT     = colors.HexColor("#7a9cc0")
SKY       = colors.HexColor("#e8f4fd")
WHITE     = colors.white
RED       = colors.HexColor("#ef4444")
AMBER     = colors.HexColor("#f59e0b")
GREEN     = colors.HexColor("#10b981")
LIGHT_TEAL= colors.HexColor("#e0f7f5")
LIGHT_BLUE= colors.HexColor("#dbeeff")

W, H = A4   # 210 × 297 mm

# ── Paragraph styles ──────────────────────────────────────────────────────
def _styles():
    return {
        "title": ParagraphStyle("title",
            fontName="Helvetica-Bold", fontSize=22,
            textColor=DARK, leading=26, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle",
            fontName="Helvetica", fontSize=10,
            textColor=MID, leading=14, spaceAfter=0),
        "section": ParagraphStyle("section",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=TEAL, leading=11, spaceBefore=10, spaceAfter=4,
            letterSpacing=1.5),
        "label": ParagraphStyle("label",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=DARK, leading=13),
        "value": ParagraphStyle("value",
            fontName="Helvetica", fontSize=9,
            textColor=MID, leading=13),
        "body": ParagraphStyle("body",
            fontName="Helvetica", fontSize=9,
            textColor=MID, leading=14, spaceAfter=4),
        "finding_title": ParagraphStyle("finding_title",
            fontName="Helvetica-Bold", fontSize=9.5,
            textColor=DARK, leading=13),
        "finding_val": ParagraphStyle("finding_val",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=BLUE, leading=16),
        "disclaimer": ParagraphStyle("disclaimer",
            fontName="Helvetica-Oblique", fontSize=7.5,
            textColor=LIGHT, leading=11),
        "header_name": ParagraphStyle("header_name",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=WHITE, leading=14),
        "header_sub": ParagraphStyle("header_sub",
            fontName="Helvetica", fontSize=8,
            textColor=colors.HexColor("#b3d9f5"), leading=11),
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _divider(color=TEAL, thickness=0.8, space=6):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=space, spaceBefore=space)


def _prob_bar_table(label, pct, bar_color=BLUE, width=120*mm):
    """Renders a label + filled bar + percentage as a Table row."""
    filled = max(1, int(pct / 100 * 100))
    empty  = 100 - filled
    bar_data = [[""]*filled + [""]*empty]
    bar_style = TableStyle([
        ("BACKGROUND", (0,0), (filled-1, 0), bar_color),
        ("BACKGROUND", (filled,0), (-1, 0), colors.HexColor("#e0ecf8")),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ])
    bar_tbl = Table(bar_data, colWidths=[width/100]*100, rowHeights=[6])
    bar_tbl.setStyle(bar_style)
    return bar_tbl


def _finding_card(label, value, sub="", color=BLUE):
    """A small stat card — label / big value / sub."""
    data = [[
        Paragraph(label.upper(), ParagraphStyle("cl",
            fontName="Helvetica-Bold", fontSize=7, textColor=LIGHT,
            letterSpacing=1)),
        "",
    ],[
        Paragraph(str(value), ParagraphStyle("cv",
            fontName="Helvetica-Bold", fontSize=16, textColor=color, leading=18)),
        "",
    ],[
        Paragraph(sub, ParagraphStyle("cs",
            fontName="Helvetica", fontSize=7.5, textColor=MID, leading=10)),
        "",
    ]]
    tbl = Table(data, colWidths=[36*mm, 0], rowHeights=[10, 19, 10])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), LIGHT_BLUE),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("SPAN", (0,0),(1,0)),
        ("SPAN", (0,1),(1,1)),
        ("SPAN", (0,2),(1,2)),
    ]))
    return tbl


# ── Header builder ────────────────────────────────────────────────────────

def _build_header(story, module, report_id, patient_ctx, styles):
    """Top header: blue band with logo + report info."""
    date_str = datetime.datetime.now().strftime("%d %B %Y")
    time_str = datetime.datetime.now().strftime("%H:%M")

    module_labels = {
        "chest": "Chest Diagnosis Report",
        "eye":   "Ophthalmology Screening Report",
        "skin":  "Dermatology Screening Report",
    }
    module_sub = {
        "chest": "X-ray Pathology Detection · ECG Rhythm Analysis",
        "eye":   "Diabetic Retinopathy Grading · Glaucoma Risk Assessment",
        "skin":  "Skin Lesion Classification · Malignancy Risk Scoring",
    }

    header_data = [[
        Paragraph("+ MediScan AI", ParagraphStyle("logo",
            fontName="Helvetica-Bold", fontSize=16,
            textColor=WHITE, leading=19)),
        Paragraph(
            f"<b>{module_labels.get(module,'AI Diagnostic Report')}</b><br/>"
            f"<font color='#b3d9f5'>{module_sub.get(module,'')}</font>",
            ParagraphStyle("hmod", fontName="Helvetica-Bold",
                           fontSize=10, textColor=WHITE, leading=14)),
        Paragraph(
            f"DATE: {date_str}<br/>TIME: {time_str}<br/>REF: {report_id}",
            ParagraphStyle("hinfo", fontName="Helvetica",
                           fontSize=8, textColor=colors.HexColor("#b3d9f5"),
                           leading=12, alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data,
                       colWidths=[42*mm, 90*mm, 50*mm],
                       rowHeights=[20*mm])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BLUE),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8))


def _build_patient_section(story, patient_ctx, styles):
    """Patient / session info row."""
    ctx = patient_ctx or {}
    fields = [
        ("Patient Age",   str(ctx.get("age", "Not provided"))),
        ("Sex",           str(ctx.get("sex", "Not provided"))),
        ("HbA1c",         str(ctx.get("hba1c", "—"))),
        ("Chief Complaint",str(ctx.get("chief_complaint", "—"))),
        ("Anatomical Site",str(ctx.get("site", "—"))),
        ("AI Model",      "MediScan v1.0"),
    ]
    # Only show fields with real values
    fields = [(k,v) for k,v in fields if v not in ("—","Not provided","None","nan")]

    story.append(Paragraph("PATIENT & SESSION INFORMATION", styles["section"]))
    _divider()

    if fields:
        col_count = min(3, len(fields))
        chunk     = [fields[i:i+col_count] for i in range(0, len(fields), col_count)]
        for row_fields in chunk:
            row_data   = []
            col_widths = []
            for k, v in row_fields:
                cell = [Paragraph(k, styles["label"]),
                        Paragraph(v, styles["value"])]
                row_data.append(cell)
                col_widths.append(58*mm)
            tbl = Table([row_data], colWidths=col_widths, rowHeights=[16])
            tbl.setStyle(TableStyle([
                ("TOPPADDING",    (0,0),(-1,-1), 3),
                ("BOTTOMPADDING", (0,0),(-1,-1), 3),
                ("LEFTPADDING",   (0,0),(-1,-1), 0),
                ("RIGHTPADDING",  (0,0),(-1,-1), 8),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ]))
            story.append(tbl)
    story.append(Spacer(1, 6))


# ── Findings tables ───────────────────────────────────────────────────────

def _build_findings_table(story, title, rows, styles):
    """
    rows = list of (label, pct_float, color)
    Renders a table with label | bar | percentage.
    """
    story.append(Paragraph(title.upper(), styles["section"]))
    story.append(_divider())

    tbl_data = [
        [
            Paragraph("<b>FINDING</b>", ParagraphStyle("th",
                fontName="Helvetica-Bold", fontSize=7.5,
                textColor=WHITE, letterSpacing=0.8)),
            Paragraph("<b>PROBABILITY</b>", ParagraphStyle("th2",
                fontName="Helvetica-Bold", fontSize=7.5,
                textColor=WHITE, letterSpacing=0.8)),
            Paragraph("<b>SCORE</b>", ParagraphStyle("th3",
                fontName="Helvetica-Bold", fontSize=7.5,
                textColor=WHITE, letterSpacing=0.8,
                alignment=TA_RIGHT)),
        ]
    ]

    for label, pct, bar_color in rows:
        filled   = max(1, int(pct))
        empty    = 100 - filled
        bar_data = [[""] * filled + [""] * empty]
        bar_tbl  = Table(bar_data,
                         colWidths=[1.0*mm]*100,
                         rowHeights=[5])
        bar_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (filled-1,0), bar_color),
            ("BACKGROUND", (filled,0), (-1,0), colors.HexColor("#dbeeff")),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))

        tbl_data.append([
            Paragraph(label, ParagraphStyle("fl",
                fontName="Helvetica", fontSize=9, textColor=DARK)),
            bar_tbl,
            Paragraph(f"<b>{pct:.1f}%</b>", ParagraphStyle("fv",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=bar_color, alignment=TA_RIGHT)),
        ])

    tbl = Table(tbl_data, colWidths=[52*mm, 100*mm, 22*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  BLUE),
        ("BACKGROUND",    (0,1),(-1,-1), WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, SKY]),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("LINEBELOW",     (0,0),(-1,-1), 0.4, colors.HexColor("#d0e8f8")),
        ("BOX",           (0,0),(-1,-1), 0.8, colors.HexColor("#b8d9f2")),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def _build_ai_report_section(story, report_text, styles):
    """The LLM-generated clinical narrative."""
    story.append(Paragraph("AI CLINICAL ASSESSMENT", styles["section"]))
    story.append(_divider())

    # Parse report into sections (FINDINGS:, IMPRESSION:, etc.)
    lines = report_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        # Section headers in the report text
        if any(line.startswith(h) for h in
               ["FINDINGS","IMPRESSION","RECOMMENDATION","DISCLAIMER",
                "1.","2.","3.","4."]):
            story.append(Paragraph(line, ParagraphStyle("rhead",
                fontName="Helvetica-Bold", fontSize=9,
                textColor=BLUE, leading=13, spaceBefore=6)))
        else:
            story.append(Paragraph(line, styles["body"]))


def _build_footer(story, styles):
    story.append(Spacer(1, 10))
    story.append(_divider(color=colors.HexColor("#b8d9f2"), thickness=0.5))
    footer_data = [[
        Paragraph(
            "<b>MediScan AI</b> · Multimodal Diagnostic Intelligence Platform",
            ParagraphStyle("fl", fontName="Helvetica", fontSize=7.5,
                           textColor=LIGHT, leading=10)),
        Paragraph(
            "FOR RESEARCH AND EDUCATIONAL USE ONLY · NOT FOR CLINICAL DIAGNOSIS",
            ParagraphStyle("fr", fontName="Helvetica-Bold", fontSize=7,
                           textColor=AMBER, letterSpacing=0.5,
                           alignment=TA_RIGHT, leading=10)),
    ]]
    ftbl = Table(footer_data, colWidths=[100*mm, 82*mm])
    ftbl.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
    ]))
    story.append(ftbl)


# ── Main generator ────────────────────────────────────────────────────────

def generate_pdf_report(module: str,
                        predictions: dict,
                        report_text: str = "",
                        patient_ctx: dict = None) -> bytes:
    """
    Generates a professional PDF report.

    Args:
        module       : "chest" | "eye" | "skin"
        predictions  : raw predictions dict from the module
        report_text  : LLM-generated clinical narrative (optional)
        patient_ctx  : patient context dict (optional)

    Returns:
        bytes — PDF file contents, ready for st.download_button
    """
    buf      = io.BytesIO()
    doc      = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=14*mm, rightMargin=14*mm,
        topMargin=12*mm,  bottomMargin=12*mm,
    )
    styles   = _styles()
    story    = []
    report_id= f"MS-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    # ── Header ─────────────────────────────────────────────────────────
    _build_header(story, module, report_id, patient_ctx, styles)
    _build_patient_section(story, patient_ctx, styles)

    # ── Module-specific findings ────────────────────────────────────────
    if module == "chest":
        _build_chest_findings(story, predictions, styles)
    elif module == "eye":
        _build_eye_findings(story, predictions, styles)
    elif module == "skin":
        _build_skin_findings(story, predictions, styles)

    # ── AI narrative ────────────────────────────────────────────────────
    if report_text:
        story.append(KeepTogether([
            Paragraph("AI CLINICAL ASSESSMENT", styles["section"]),
            _divider(),
        ]))
        _build_ai_report_section(story, report_text, styles)

    # ── Disclaimer box ──────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    disc_data = [[
        Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by an AI screening system "
            "and has NOT been reviewed by a licensed medical professional. "
            "It is intended for research and educational purposes only and must not "
            "be used for clinical diagnosis, treatment decisions, or patient care. "
            "Consult a qualified healthcare provider for any medical concerns.",
            ParagraphStyle("disc", fontName="Helvetica", fontSize=7.5,
                           textColor=colors.HexColor("#854f0b"), leading=11))
    ]]
    disc_tbl = Table(disc_data, colWidths=[182*mm])
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#fffbeb")),
        ("BOX",           (0,0),(-1,-1), 0.8, AMBER),
        ("TOPPADDING",    (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 8),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
    ]))
    story.append(disc_tbl)

    _build_footer(story, styles)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Module-specific findings builders ─────────────────────────────────────

def _build_chest_findings(story, preds, styles):
    # ── Stat cards row ──────────────────────────────────────────────────
    xray = preds.get("xray", {})
    ecg  = preds.get("ecg",  {})

    if xray:
        top_label = max(xray, key=xray.get)
        top_conf  = xray[top_label]
        ecg_label = ecg.get("label", "—")
        ecg_conf  = ecg.get("confidence", 0)

        cards_data = [[
            _finding_card("Primary Finding",  top_label,
                          f"{top_conf*100:.1f}% confidence", BLUE),
            _finding_card("ECG Rhythm",       ecg_label,
                          f"{ecg_conf*100:.1f}% confidence",
                          GREEN if ecg_label=="NORM" else RED),
        ]]
        ctbl = Table(cards_data, colWidths=[90*mm, 90*mm])
        ctbl.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ]))
        story.append(ctbl)
        story.append(Spacer(1, 10))

    # ── X-ray findings table ─────────────────────────────────────────────
    if xray:
        rows = []
        color_map = lambda p: RED if p > 0.7 else AMBER if p > 0.4 else BLUE
        for label, prob in sorted(xray.items(), key=lambda x: -x[1])[:10]:
            rows.append((label, prob*100, color_map(prob)))
        _build_findings_table(story, "Chest X-ray Pathology Detection", rows, styles)

    # ── ECG table ────────────────────────────────────────────────────────
    if ecg and "probs" in ecg:
        ecg_colors = {"NORM":GREEN,"MI":RED,"STTC":AMBER,"CD":AMBER,"HYP":BLUE}
        rows = [(lbl, p*100, ecg_colors.get(lbl, BLUE))
                for lbl, p in sorted(ecg["probs"].items(), key=lambda x:-x[1])]
        _build_findings_table(story, "ECG Rhythm Classification", rows, styles)


def _build_eye_findings(story, preds, styles):
    dr  = preds.get("dr",       {})
    glc = preds.get("glaucoma", {})

    if dr:
        grade     = dr.get("grade", 0)
        dr_label  = dr.get("label", "—")
        dr_conf   = dr.get("confidence", 0)
        glc_score = glc.get("score", 0)
        glc_risk  = "High" if glc_score > 0.5 else "Low"
        grade_color = [GREEN, colors.HexColor("#34d399"),
                       AMBER, colors.HexColor("#f97316"), RED][min(grade,4)]

        cards_data = [[
            _finding_card("DR Grade",        f"{grade} — {dr_label}",
                          f"{dr_conf*100:.1f}% confidence", grade_color),
            _finding_card("Glaucoma Risk",   glc_risk,
                          f"Score: {glc_score*100:.1f}%",
                          RED if glc_score>.5 else GREEN),
        ]]
        ctbl = Table(cards_data, colWidths=[90*mm, 90*mm])
        ctbl.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ]))
        story.append(ctbl)
        story.append(Spacer(1, 10))

        grade_colors = [GREEN, colors.HexColor("#34d399"),
                        AMBER, colors.HexColor("#f97316"), RED]
        rows = [(lbl, p*100, grade_colors[i])
                for i,(lbl,p) in enumerate(dr.get("probs",{}).items())]
        _build_findings_table(story, "Diabetic Retinopathy — Grade Probabilities",
                               rows, styles)

        story.append(Paragraph("GLAUCOMA SCREENING", styles["section"]))
        story.append(_divider())
        g_color = RED if glc_score>.5 else GREEN
        story.append(Paragraph(
            f"Risk score: <b>{glc_score*100:.1f}%</b> — Risk level: <b>{glc_risk}</b>",
            ParagraphStyle("gs", fontName="Helvetica", fontSize=10,
                           textColor=g_color, leading=14)))
        story.append(Spacer(1, 8))


def _build_skin_findings(story, preds, styles):
    label   = preds.get("label", "—")
    conf    = preds.get("confidence", 0)
    mal     = preds.get("is_malignant", False)
    probs   = preds.get("probs", {})

    MALIGNANT = {"Melanoma","Basal cell carcinoma","Actinic keratosis"}
    cards_data = [[
        _finding_card("Primary Diagnosis", label,
                      f"{conf*100:.1f}% confidence",
                      RED if mal else GREEN),
        _finding_card("Malignancy Risk",   "HIGH" if mal else "LOW",
                      "Urgent referral" if mal else "Routine follow-up",
                      RED if mal else GREEN),
    ]]
    ctbl = Table(cards_data, colWidths=[90*mm, 90*mm])
    ctbl.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
    ]))
    story.append(ctbl)
    story.append(Spacer(1, 10))

    rows = [(lbl, p*100, RED if lbl in MALIGNANT else BLUE)
            for lbl, p in sorted(probs.items(), key=lambda x:-x[1])]
    _build_findings_table(story, "Lesion Classification Probabilities", rows, styles)
