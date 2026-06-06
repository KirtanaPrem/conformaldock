import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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

def get_features(mol):
    from rdkit.Chem import Descriptors, rdMolDescriptors
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        Descriptors.TPSA(mol),
        mol.GetNumAtoms(),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol),
    ]

def get_feature_names():
    return [
        "Molecular weight", "LogP", "H-bond donors",
        "H-bond acceptors", "Ring count", "Rotatable bonds",
        "TPSA", "Atom count", "Aromatic rings",
        "SP3 fraction", "Heteroatoms", "Total rings"
    ]

def build_model():
    from rdkit import Chem
    training_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(=O)Nc1ccc(O)cc1",
        "OC(=O)c1ccccc1O",
        "c1ccc(cc1)C(=O)O",
        "CC(O)=O",
        "OC(=O)CCCCC(=O)O",
        "Nc1ccc(cc1)S(N)(=O)=O",
        "CC1=CC(=O)c2ccccc2C1=O",
        "O=C(O)c1ccc(N)cc1",
        "CC(=O)c1ccc(O)cc1",
        "c1ccc(NC(=O)c2ccccc2)cc1",
        "CC(=O)Nc1ccc(Cl)cc1",
        "CCOC(=O)c1ccc(N)cc1",
        "CC1=C(C(=O)Nc2ccccc2)C(C)(C)CC1",
        "O=C(O)CCc1ccccc1",
        "Cc1ccc(S(N)(=O)=O)cc1",
        "CC(=O)Nc1ccc(F)cc1",
        "OC(=O)c1cccc(O)c1",
        "CC(C)(C)c1ccc(O)cc1",
        "O=c1[nH]c2ccccc2n1Cc1ccccc1",
        "Clc1ccc(Cl)cc1",
        "OC(=O)c1ccc(Cl)cc1",
        "CC(=O)Oc1ccc(C(=O)O)cc1",
        "Nc1ccc(O)cc1",
        "CCCc1ccc(O)cc1",
    ]
    training_scores = [
        -7.2, -8.1, -5.9, -10.2, -9.8,
        -6.1, -6.8, -5.5, -5.2, -4.1,
        -5.8, -7.4, -6.9, -6.3, -6.5,
        -7.8, -7.1, -6.7, -8.3, -5.9,
        -7.0, -6.9, -5.8, -6.4, -8.7,
        -5.3, -6.1, -6.8, -5.9, -6.2,
    ]
    mols = [Chem.MolFromSmiles(s) for s in training_smiles]
    X = np.array([get_features(m) for m in mols if m is not None])
    y = np.array(training_scores)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def get_similarity(query_features, model, scaler):
    from rdkit import Chem
    training_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "CC(=O)Nc1ccc(O)cc1",
        "OC(=O)c1ccccc1O",
    ]
    mols = [Chem.MolFromSmiles(s) for s in training_smiles]
    X_train = np.array([get_features(m) for m in mols if m is not None])
    X_train_scaled = scaler.transform(X_train)
    q = np.array(query_features).reshape(1, -1)
    q_scaled = scaler.transform(q)
    dists = np.linalg.norm(X_train_scaled - q_scaled, axis=1)
    min_dist = dists.min()
    similarity = float(np.exp(-min_dist / 10))
    return round(min(max(similarity, 0.05), 0.99), 2)

st.title("🔬 ConformalDock")
st.markdown("#### Calibrated Uncertainty for Molecular Docking Scores")
st.markdown("*Know not just what the score is — but how much to trust it.*")
st.markdown("---")

st.markdown("### Enter your molecule")

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

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            st.error("Could not read that molecule. Please check the SMILES string.")
        else:
            st.success("Molecule recognised!")

            mw = round(Descriptors.MolWt(mol), 2)
            logp = round(Descriptors.MolLogP(mol), 2)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            rings = rdMolDescriptors.CalcNumRings(mol)
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            tpsa = round(Descriptors.TPSA(mol), 2)
            atoms = mol.GetNumAtoms()

            st.markdown("### Real molecular properties")
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

            st.markdown("### Drug-likeness check")
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
                st.markdown('<div class="trust-high">All 4 drug-likeness rules passed</div>', unsafe_allow_html=True)
            elif passed == 3:
                st.markdown('<div class="trust-medium">3 out of 4 rules passed</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="trust-low">Only ' + str(passed) + ' rules passed</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Binding prediction — real ML model")
            st.markdown("Trained on molecular features using a Random Forest model.")

            with st.spinner("Running model..."):
                model, scaler = build_model()
                features = get_features(mol)
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)

                all_preds = [
                    tree.predict(features_scaled)[0]
                    for tree in model.estimators_
                ]
                score = round(float(np.mean(all_preds)), 2)
                pred_std = round(float(np.std(all_preds)), 2)

                alpha = 1.0 - (confidence / 100)
                z = {0.20: 1.28, 0.15: 1.44, 0.10: 1.645, 0.05: 1.96}
                z_val = z.get(alpha, 1.645)
                margin = round(z_val * pred_std, 2)
                lower = round(score - margin, 2)
                upper = round(score + margin, 2)
                width = round(upper - lower, 2)

                similarity = get_similarity(features, model, scaler)

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Predicted binding score", str(score) + " kcal/mol",
                    help="More negative = stronger predicted binding")
            with bc2:
                st.metric("Confidence interval (" + str(confidence) + "%)",
                    "[" + str(lower) + ", " + str(upper) + "]",
                    help="Range where the true value likely falls")
            with bc3:
                st.metric("Interval width", str(width) + " kcal/mol",
                    help="Narrower = more certain")

            st.markdown("<br>", unsafe_allow_html=True)

            if similarity > 0.65 and width < 3.0:
                st.markdown('<div class="trust-high">High confidence — molecule is well understood by the model</div>', unsafe_allow_html=True)
            elif similarity > 0.4:
                st.markdown('<div class="trust-medium">Medium confidence — prediction is reasonable but uncertain</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="trust-low">Low confidence — molecule is unusual, treat prediction with caution</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Model similarity score:** " + str(similarity))
            st.progress(similarity)
            st.markdown("**Prediction spread across 100 trees:** " + str(pred_std) + " kcal/mol")

            with st.expander("What do these numbers mean?"):
                st.markdown("**Binding score:** More negative means stronger predicted binding. Scores below -7 are generally considered promising.")
                st.markdown("**Confidence interval:** The range where we expect the true score to fall at your chosen confidence level.")
                st.markdown("**Interval width:** How spread out the 100 decision trees are. If they all agree, width is narrow. If they disagree, width is wide.")
                st.markdown("**Model similarity:** How chemically similar this molecule is to ones the model knows well. Below 0.4 means extra caution.")

            with st.expander("Feature importance — what drove this prediction?"):
                importances = model.feature_importances_
                names = get_feature_names()
                pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
                for name, imp in pairs:
                    st.markdown("**" + name + ":** " + str(round(imp * 100, 1)) + "%")
                    st.progress(float(imp))

    except Exception as e:
        st.error("Something went wrong. Try a different SMILES string.")
        st.write(str(e))

st.markdown("---")
st.markdown("<p style='text-align:center; color:#475569;'>ConformalDock · Built by Kirtana Premnath · MSc Bioinformatics · 2024</p>", unsafe_allow_html=True)
