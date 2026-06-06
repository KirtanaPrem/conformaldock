import streamlit as st
import numpy as np

st.set_page_config(
    page_title="ConformalDock",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0f1e; }
[data-testid="stSidebar"] { background-color: #0d1b2a; }
.trust-high {
    background: #064e3b; border: 1px solid #10b981;
    border-radius: 12px; padding: 16px; color: #6ee7b7;
    font-size: 18px; font-weight: bold; text-align: center;
}
.trust-medium {
    background: #451a03; border: 1px solid #f59e0b;
    border-radius: 12px; padding: 16px; color: #fcd34d;
    font-size: 18px; font-weight: bold; text-align: center;
}
.trust-low {
    background: #450a0a; border: 1px solid #ef4444;
    border-radius: 12px; padding: 16px; color: #fca5a5;
    font-size: 18px; font-weight: bold; text-align: center;
}
.mol-card {
    background: #112240; border: 1px solid #1d4ed8;
    border-radius: 12px; padding: 16px; margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Settings")
    confidence = st.slider("Confidence level", 80, 95, 90, 5)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Built by **Kirtana Premnath**\nMSc Bioinformatics")
    st.markdown("---")
    st.markdown("### What is SMILES?")
    st.markdown("""
    SMILES is a way of writing a molecule 
    as a line of text. For example:
    
    Aspirin = `CC(=O)Oc1ccccc1C(=O)O`
    
    Every atom and bond is encoded 
    in that string of letters.
    """)

st.title("🔬 ConformalDock")
st.markdown("#### Calibrated Uncertainty for Molecular Docking Scores")
st.markdown("*Know not just what the score is — but how much to trust it.*")
st.markdown("---")

st.markdown("### Step 1 — Enter your molecule")

smiles = st.text_input(
    "Paste a SMILES string here",
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
)

st.markdown("**Or click an example:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Aspirin", use_container_width=True):
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
with col2:
    if st.button("Ibuprofen", use_container_width=True):
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
with col3:
    if st.button("Caffeine", use_container_width=True):
        smiles = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"

if smiles:
    st.markdown("---")

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        import io
        from PIL import Image

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            st.error("Could not read that molecule. Please check the SMILES string and try again.")
        else:
            st.success("Molecule recognised successfully!")

            st.markdown("### Step 2 — Molecule properties")
            st.markdown("These are **real values** calculated directly from the molecule structure.")

            mw = round(Descriptors.MolWt(mol), 2)
            logp = round(Descriptors.MolLogP(mol), 2)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rings = rdMolDescriptors.CalcNumRings(mol)
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            tpsa = round(Descriptors.TPSA(mol), 2)
            atoms = mol.GetNumAtoms()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Molecular weight", str(mw) + " g/mol",
                    help="How heavy the molecule is")
            with c2:
                st.metric("LogP", str(logp),
                    help="How oily vs watery the molecule is. Between -2 and 5 is ideal for drugs.")
            with c3:
                st.metric("H-bond donors", str(hbd),
                    help="How many hydrogen bonds this molecule can donate")
            with c4:
                st.metric("H-bond acceptors", str(hba),
                    help="How many hydrogen bonds this molecule can accept")

            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.metric("Ring count", str(rings),
                    help="Number of ring structures in the molecule")
            with c6:
                st.metric("Rotatable bonds", str(rotbonds),
                    help="Flexibility of the molecule")
            with c7:
                st.metric("TPSA", str(tpsa),
                    help="Topological polar surface area — relates to how well drug crosses cell membranes")
            with c8:
                st.metric("Atom count", str(atoms),
                    help="Total number of heavy atoms")

            st.markdown("### Step 3 — Drug-likeness check")
            st.markdown("Lipinski's Rule of Five — the standard checklist for whether a molecule could be an oral drug:")

            rules = {
                "Molecular weight under 500": mw <= 500,
                "LogP under 5 (not too oily)": logp <= 5,
                "H-bond donors 5 or fewer": hbd <= 5,
                "H-bond acceptors 10 or fewer": hba <= 10,
            }

            passed = sum(rules.values())
            rc1, rc2 = st.columns(2)
            for i, (rule, ok) in enumerate(rules.items()):
                col = rc1 if i % 2 == 0 else rc2
                with col:
                    if ok:
                        st.success("✅ " + rule)
                    else:
                        st.error("❌ " + rule)

            if passed == 4:
                st.markdown('<div class="trust-high">✅ Passes all drug-likeness rules — good oral drug candidate</div>', unsafe_allow_html=True)
            elif passed >= 3:
