"""Streamlit demo — upload a leaf image, compare System A vs System B side by side.

Run:  streamlit run src/app/app.py

Used for the defense demo: show the same image producing a hallucinated baseline answer
and a corrected reflection-loop answer.
"""
from __future__ import annotations

import tempfile

import streamlit as st

from src import system_a_baseline
from src.system_b_reflection import graph as system_b

st.set_page_config(page_title="Crop Reflection RAG", layout="wide")
st.title("🌿 Crop Diagnostics — Baseline vs Reflection Loop")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
query = st.text_input("Your question", "What disease affects this plant?")

if uploaded and st.button("Diagnose"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.getvalue())
        image_path = tmp.name

    st.image(image_path, width=300)
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("System A — Single-pass RAG")
        with st.spinner("Diagnosing..."):
            a = system_a_baseline.diagnose(image_path, query)
        st.write(a["diagnosis"])
        st.caption(f"latency: {a['latency_s']:.2f}s")

    with col_b:
        st.subheader("System B — Reflection Loop")
        with st.spinner("Diagnosing + reflecting..."):
            b = system_b.diagnose(image_path, query)
        st.write(b["diagnosis"])
        st.caption(f"iterations: {b['iterations']} · latency: {b['latency_s']:.2f}s")
        if b.get("critic_feedback", {}).get("contradictions"):
            with st.expander("What the Critic caught"):
                st.json(b["critic_feedback"])
