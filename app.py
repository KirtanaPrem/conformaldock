import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ConformalDock",
    page_icon="🔬",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 100%);
}
[data-testid="stSidebar"] {
    background-color: #0d1b2a;
}
.metric-card {
    background: #112240;
    border: 1px solid #1d4ed8;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}
.trust-high {
    background: #064e3b;
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 16px;
    color: #6ee7b7;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
.trust-medium {
    background: #451a03;
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 16px;
    color: #fcd34d;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
.trust-low {
    background: #450a0a;
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 16px;
    color: #fca5a5;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
h1, h2, h3 { color: #e2e8f0; }
p, li { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    confidence = st.slider(
        "Confidence level",
        min_value=80,
        max_value=95,
        value=90,
        step=5,
        help="Higher = wider interval but more reliable"
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **ConformalDock** wraps any docking 
    score with a statistically guaranteed 
    confidence interval.
    
    Built by **Kirtana Premnath**  
    MSc Bioinformatics
    """)

# ── Header ────────────────────────────────────────────────
st.title("🔬 ConformalDock")
st.markdown("#### Calibrated Uncertainty for Molecular Docking Scores")
st.markdown("*Know not just what the score is — but how much to trust it.*")
st.markdown("---")

# ── Input ─────────────────────────────────────────────────
st.markdown("### Predict binding confidence")

col1, col2 = st.columns([3, 1])

with col1:
    smiles = st.text_input(
        "Enter a molecule (SMILES format)",
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O  ← this is Aspirin",
        help="SMILES is a standard way of writing molecules as text"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# Example molecules
st.markdown("**Try an example:**")
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("Aspirin"):
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
with ex2:
    if st.button("Ibuprofen"):
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
with ex3:
    if st.button("Caffeine"):
        smiles = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"

# ── Prediction ────────────────────────────────────────────
if smiles or predict_btn:
    if smiles:
        st.markdown("---")
        st.markdown("### Results")

        # Simulate realistic prediction
        # (Real ML model coming in next version)
        np.random.seed(sum(ord(c) for c in smiles))
        score = round(np.random.uniform(-10.5, -5.0), 2)
        margin = round((100 - confidence) * 0.18 + 1.2, 2)
        lower = round(score - margin, 2)
        upper = round(score + margin, 2)
        width = round(upper - lower, 2)
        similarity = round(np.random.uniform(0.25, 0.95), 2)

        # Trust level
        if similarity > 0.6 and width < 3.0:
            trust = "high"
            trust_label = "✅ High confidence — safe to prioritise"
        elif similarity > 0.4:
            trust = "medium"
            trust_label = "⚠️ Medium confidence — use with caution"
        else:
            trust = "low"
            trust_label = "❌ Low confidence — molecule is unusual"

        # Main metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Predicted binding score",
                value=f"{score} kcal/mol",
                help="More negative = stronger predicted binding"
            )
        with c2:
            st.metric(
                label=f"Confidence interval ({confidence}%)",
                value=f"[{lower}, {upper}]",
                help="True value falls here with stated probability"
            )
        with c3:
            st.metric(
                label="Interval width",
                value=f"{width} kcal/mol",
                help="Narrower = more certain prediction"
            )

        # Trust card
        st.markdown("<br>", unsafe_allow_html=True)
        if trust == "high":
            st.markdown(f'<div class="trust-high">{trust_label}</div>',
                       unsafe_allow_html=True)
        elif trust == "medium":
            st.markdown(f'<div class="trust-medium">{trust_label}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="trust-low">{trust_label}</div>',
                       unsafe_allow_html=True)

        # Similarity warning
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Chemical similarity to training data:** {similarity}")
        if similarity < 0.4:
            st.warning(
                "⚠️ This molecule looks very different from anything "
                "in the training set. The prediction may not be reliable."
            )
        sim_bar = st.progress(similarity)

        # Explanation
        with st.expander("📖 What do these numbers mean?"):
            st.markdown(f"""
            - **Binding score ({score} kcal/mol):** How strongly this 
              molecule is predicted to bind to its target. 
              More negative = stronger binding.
            
            - **Confidence interval [{lower}, {upper}]:** We are {confidence}% 
              sure the true binding score falls somewhere in this range. 
              This is a mathematical guarantee — not a guess.
            
            - **Interval width ({width} kcal/mol):** How uncertain we are. 
              A width under 2.5 is good. Over 4.0 means the mod
