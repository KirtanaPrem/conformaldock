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
.trust-high { background: #064e3b; border: 1px solid #10b981; border-radius: 12px; padding: 16px; color: #6ee7b7; font-size: 18px; font-weight: bold; text-align: center; }
.trust-medium { background: #451a03; border: 1px solid #f59e0b; border-radius: 12px; padding: 16px; color: #fcd34d; font-size: 18px; font-weight: bold; text-align: center; }
.trust-low { background: #450a0a; border: 1px solid #ef4444; border-radius: 12px; padding: 16px; color: #fca5a5; font-size: 18px; font-weight: bold; text-align: center; }
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
    st.markdown("SMILES is a way of writing a molecule as text. Aspirin = `CC(=O)Oc1ccccc1C(=O)O`")

st.title("🔬 ConformalDock")
st.markdown("#### Calibrated Uncertainty for Molecular Docking Scores")
st.markdown("*Know not just what the score is — but how much to trust it.*")
st.markdown("---")

st.markdown("### Enter your molecule")

smiles = st.text_input("Paste a SMILES string here", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")

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

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            st.error("Could not read that molecule. Please check the SMILES string.")
        else:
            st.success("Molecule recognised!")

            st.markdown("### Real molecular properties")
            st.markdown("These numbers come directly from reading the molecule structure — not made up.")

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
                st.metric("Molecular weight", str(mw) + " g/mol")
            with c2:
                st.metric("LogP", str(logp))
            with c3:
                st.metric("H-bond donors", str(hbd))
            with c4:
                st.metric("H-bond acceptors", str(hba))

            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.metric("Rings", str(rings))
            with c6:
                st.metric("Rotatable bonds", str(rotbonds))
            with c7:
                st.metric("TPSA", str(tpsa))
            with c8:
                st.metric("Atoms", str(atoms))

            st.markdown("### Drug-likeness check (Lipinski rules)")

            rule1 = mw <= 500
            rule2 = logp <= 5
            rule3 = hbd <= 5
            rule4 = hba <= 10
            passed = sum([rule1, rule2, rule3, rule4])

            rc1, rc2 = st.columns(2)
            with rc1:
                if rule1:
                    st.success("Molecular weight under 500")
                else:
                    st.error("Molecular weight over 500")
                if rule2:
                    st.success("LogP under 5")
                else:
                    st.error("LogP over 5")
            with rc2:
                if rule3:
                    st.success("H-bond donors 5 or fewer")
                else:
                    st.error("H-bond donors over 5")
                if rule4:
                    st.success("H-bond acceptors 10 or fewer")
                else:
                    st.error("H-bond acceptors over 10")

            st.markdown("<br>", unsafe_allow_html=True)

            if passed == 4:
                st.markdown('<div class="trust-high">All 4 drug-likeness rules passed — good oral drug candidate</div>', unsafe_allow_html=True)
            elif passed == 3:
                st.markdown('<div class="trust-medium">3 out of 4 rules passed — borderline drug candidate</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="trust-low">Only ' + str(passed) + ' rules passed — unlikely oral drug</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Binding prediction")
            st.info("Real ML model coming next session. Placeholder numbers shown below.")

            np.random.seed(sum(ord(c) for c in smiles))
            score = round(np.random.uniform(-10.5, -5.0), 2)
            margin = round((100 - confidence) * 0.18 + 1.2, 2)
            lower = round(score - margin, 2)
            upper = round(score + margin, 2)
            width = round(upper - lower, 2)

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Predicted score", str(score) + " kcal/mol")
            with bc2:
                st.metric("Confidence interval (" + str(confidence) + "%)", "[" + str(lower) + ", " + str(upper) + "]")
            with bc3:
                st.metric("Interval width", str(width) + " kcal/mol")

    except Exception as e:
        st.error("Something went wrong. Try a different SMILES string.")
        st.write(str(e))

st.markdown("---")
st.markdown("<p style='text-align:center; color:#475569;'>ConformalDock · Built by Kirtana Premnath · MSc Bioinformatics · 2024</p>", unsafe_allow_html=True)
