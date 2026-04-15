"""
Drug Repurposing with Graph Neural Networks
Step 6: Streamlit Web App
==========================
Interactive demo where users can:
  1. Search for a disease by name
  2. Get ranked drug candidates from the GNN
  3. See the biological path explaining each prediction
  4. Compare against known treatments

Run with:
    streamlit run 6_app.py
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from model import build_model
from explain import (load_model_and_data, predict_drugs_for_disease,
                     explain_prediction, _idx_to_name)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Repurposing GNN",
    page_icon="💊",
    layout="wide",
)

# ── Load model (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_everything():
    return load_model_and_data()


# ── Helper: disease name search ───────────────────────────────────────────
def search_diseases(kind_dfs, query: str, top_n=20):
    df = kind_dfs.get("Disease", pd.DataFrame())
    if "name" not in df.columns:
        return []
    mask = df["name"].str.contains(query, case=False, na=False)
    return df[mask].head(top_n)[["name"]].reset_index().rename(
        columns={"index": "node_idx"})


# ── Main UI ───────────────────────────────────────────────────────────────
def main():
    # --- Header ---
    st.title("💊 Drug Repurposing with Graph Neural Networks")
    st.markdown(
        "A GNN trained on **Hetionet** — a biomedical knowledge graph "
        "with 47K+ nodes — predicts which existing approved drugs could "
        "treat diseases they weren't originally designed for."
    )

    with st.spinner("Loading model and graph data..."):
        model, data, kind_dfs = load_everything()

    st.success("Model ready!", icon="✅")

    # --- Sidebar ---
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top-K drug candidates", 5, 50, 10)
    show_paths = st.sidebar.checkbox("Show biological explanation paths", True)

    # --- Disease search ---
    st.header("1. Search for a disease")
    query = st.text_input("Type a disease name (e.g. 'diabetes', 'alzheimer')", "")

    if query:
        results = search_diseases(kind_dfs, query)
        if results.empty:
            st.warning("No diseases found. Try a different search term.")
            return

        selected = st.selectbox(
            "Select disease:",
            options=results["name"].tolist(),
        )
        disease_row = results[results["name"] == selected].iloc[0]
        disease_idx = int(disease_row["node_idx"])

        # --- Predict ---
        st.header(f"2. Top-{top_k} drug candidates for **{selected}**")

        with st.spinner("Running GNN predictions..."):
            top_drugs = predict_drugs_for_disease(
                model, data, disease_idx, top_k=top_k)

        # Build results table
        rows = []
        for drug_idx, score in top_drugs:
            drug_name = _idx_to_name(kind_dfs, "Compound", drug_idx)
            rows.append({
                "Drug":  drug_name,
                "Score": round(score, 4),
                "drug_idx": drug_idx,
            })
        results_df = pd.DataFrame(rows)

        # --- Bar chart ---
        fig = px.bar(
            results_df,
            x="Score", y="Drug",
            orientation="h",
            color="Score",
            color_continuous_scale="teal",
            title=f"Predicted treatment scores for {selected}",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

        # --- Detailed table ---
        st.dataframe(
            results_df[["Drug", "Score"]].style.background_gradient(
                subset=["Score"], cmap="YlGn"),
            use_container_width=True,
        )

        # --- Explanation paths ---
        if show_paths:
            st.header("3. Biological explanation paths")
            st.markdown(
                "For each predicted drug, the GNN identifies shared genes/pathways "
                "between the drug and the disease — the biological evidence behind "
                "each prediction."
            )

            selected_drug = st.selectbox(
                "Explore explanation for:",
                options=results_df["Drug"].tolist()
            )
            sel_row  = results_df[results_df["Drug"] == selected_drug].iloc[0]
            drug_idx = int(sel_row["drug_idx"])

            with st.spinner("Generating explanation..."):
                exp = explain_prediction(
                    model, data, drug_idx, disease_idx, kind_dfs)

            st.metric("Prediction score", f"{exp['score']:.4f}")
            st.info(exp["interpretation"])

            if exp["paths"]:
                st.subheader("Top biological paths")
                for i, p in enumerate(exp["paths"], 1):
                    st.markdown(f"**Path {i}:** `{p['path']}`")
                    cols = st.columns(3)
                    cols[0].metric("Shared gene", p["shared_gene"])
                    cols[1].metric("Drug relation", p["drug_rel"])
                    cols[2].metric("Disease relation", p["disease_rel"])
                    st.divider()
            else:
                st.warning("No direct gene-level paths found. "
                           "The GNN is reasoning from higher-order patterns.")

        # --- Cross-check with clinical trials hint ---
        st.header("4. Validate against known data")
        st.markdown(
            f"Cross-check your top predictions on "
            f"[ClinicalTrials.gov](https://clinicaltrials.gov/search?cond={selected.replace(' ', '+')}) "
            f"and [DrugBank](https://go.drugbank.com/) to see if any predicted "
            f"drugs are already in trials for **{selected}**."
        )

    # --- About ---
    with st.expander("About this project"):
        st.markdown("""
        **Architecture:** Relational GCN (R-GCN) with heterogeneous message passing

        **Dataset:** [Hetionet v1.0](https://het.io) — 47K nodes, 2.25M edges

        **Node types:** Compound, Disease, Gene, Pathway, Anatomy, Side Effect

        **Key edge type:** `Compound → treats → Disease` (ground truth labels)

        **Explainability:** Biological path analysis identifying shared gene neighbours

        **Stack:** PyTorch Geometric · Streamlit · Plotly · RDKit
        """)


if __name__ == "__main__":
    main()
