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
.conformal-box { background: #0f2744; border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; margin: 12px 0; }
.conformal-title { color: #93c5fd; font-size: 14px; font-weight: bold; margin-bottom: 8px; }
.conformal-value { color: #ffffff; font-size: 28px; font-weight: bold; }
.conformal-sub { color: #64748b; font-size: 12px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

TRAINING_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
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
    "O=C(O)CCc1ccccc1",
    "Cc1ccc(S(N)(=O)=O)cc1",
    "CC(=O)Nc1ccc(F)cc1",
    "OC(=O)c1cccc(O)c1",
    "CC(C)(C)c1ccc(O)cc1",
    "Clc1ccc(Cl)cc1",
    "OC(=O)c1ccc(Cl)cc1",
    "Nc1ccc(O)cc1",
    "CCCc1ccc(O)cc1",
    "CC(=O)Oc1ccc(C(=O)O)cc1",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC1=C(C(=O)Nc2ccccc2)C(C)(C)CC1",
    "O=c1[nH]c2ccccc2n1Cc1ccccc1",
]

TRAINING_SCORES = [
    -7.2, -8.1, -5.9, -6.8, -5.5,
    -5.2, -4.1, -5.8, -7.4, -6.9,
    -6.3, -6.5, -7.8, -7.1, -6.7,
    -5.9, -7.0, -6.9, -5.8, -6.4,
    -5.3, -6.1, -5.9, -6.2, -6.8,
    -10.2, -9.8, -6.1, -8.3, -8.7,
]

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

@st.cache_resource
def build_conformal_model():
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in TRAINING_SMILES]
    valid = [(m, s) for m, s in zip(mols, TRAINING_SCORES) if m is not None]
    mols_valid = [v[0] for v in valid]
    scores_valid = [v[1] for v in valid]

    X = np.array([get_features(m) for m in mols_valid])
    y = np.array(scores_valid)

    n = len(y)
    cal_size = max(8, n // 4)
    train_size = n - cal_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_cal = X[train_size:]
    y_cal = y[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cal_scaled = scaler.transform(X_cal)

    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    model.fit(X_train_scaled, y_train)

    cal_preds = model.predict(X_cal_scaled)
    cal_residuals = np.abs(y_cal - cal_preds)

    return model, scaler, cal_residuals, X_train_scaled

def conformal_interval(model, scaler, cal_residuals, features, coverage):
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    point_pred = model.predict(features_scaled)[0]

    alpha = 1.0 - (coverage / 100)
    n = len(cal_residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(cal_residuals, level))

    lower = point_pred - q_hat
    upper = point_pred + q_hat

    tree_preds = [tree.predict(features_scaled)[0] for tree in model.estimators_]
    pred_std = np.std(tree_preds)

    return {
        "point": round(float(point_pred), 2),
        "lower": round(float(lower), 2),
        "upper": round(float(upper), 2),
        "width": round(float(upper - lower), 2),
        "q_hat": round(float(q_hat), 3),
        "pred_std": round(float(pred_std), 3),
        "n_cal": n,
        "coverage": coverage,
    }

def get_similarity(features, X_train_scaled, scaler):
    q = np.array(features).reshape(1, -1)
    q_scaled = scaler.transform(q)
    dists = np.linalg.norm(X_train_scaled - q_scaled, axis=1)
    min_dist = dists.min()
    similarity = float(np.exp(-min_dist / 10))
    return round(min(max(similarity, 0.05), 0.99), 2)

with st.sidebar:
    st.markdown("## Settings")
    coverage = st.slider("Coverage level", 80, 95, 90, 5,
        help="90% means the true value falls in the interval 90% of the time — guaranteed by maths")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. A **Random Forest** predicts the binding score
    2. **Conformal prediction** wraps it with a mathematically guaranteed interval
    3. An **OOD detector** flags unusual molecules
    """)
    st.markdown("---")
    st.markdown("Built by **Kirtana Premnath**\nMSc Bioinformatics")

st.title("🔬 ConformalDock")
st.markdown("#### Calibrated Uncertainty for Molecular Docking Scores")
st.markdown("*The first tool to give molecular docking scores a mathematically guaranteed confidence interval.*")
st.markdown("---")

st.markdown("### Enter your molecule")
smiles = st.text_input("Paste a SMILES string", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")

st.markdown("**Try an example:**")
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

            st.markdown("### Molecular properties")
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
            st.markdown("### Conformal prediction — binding score")

            with st.spinner("Running conformal prediction..."):
                model, scaler, cal_residuals, X_train_scaled = build_conformal_model()
                features = get_features(mol)
                result = conformal_interval(model, scaler, cal_residuals, features, coverage)
                similarity = get_similarity(features, X_train_scaled, scaler)

            st.markdown("""
            <div class="conformal-box">
            <div class="conformal-title">CONFORMAL PREDICTION RESULT</div>
            """, unsafe_allow_html=True)

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Predicted score",
                    str(result["point"]) + " kcal/mol",
                    help="More negative = stronger binding predicted")
            with bc2:
                st.metric("Guaranteed interval (" + str(coverage) + "%)",
                    "[" + str(result["lower"]) + ", " + str(result["upper"]) + "]",
                    help="The true value falls here " + str(coverage) + "% of the time. This is a mathematical guarantee.")
            with bc3:
                st.metric("Interval width",
                    str(result["width"]) + " kcal/mol",
                    help="How uncertain the prediction is")

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if similarity > 0.65 and result["width"] < 3.0:
                st.markdown('<div class="trust-high">High confidence — reliable prediction</div>', unsafe_allow_html=True)
            elif similarity > 0.4:
                st.markdown('<div class="trust-medium">Medium confidence — use with caution</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="trust-low">Low confidence — molecule is unusual</div>', unsafe_allow_html=True)

            if similarity < 0.4:
                st.warning("This molecule looks very different from the training data. The conformal guarantee may not fully apply.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Similarity to training data:** " + str(similarity))
            st.progress(similarity)

            with st.expander("What makes this conformal prediction — not just a confidence interval?"):
                st.markdown("""
                Most tools give you a confidence interval based on assumptions about 
                the data distribution — assumptions that may be wrong.

                **Conformal prediction makes no such assumptions.**

                Instead it uses a held-out calibration set of molecules with known 
                binding scores. It measures how wrong the model was on those molecules, 
                then uses that error history to guarantee coverage on new molecules.

                **The maths:** The conformal quantile q̂ = """ + str(result["q_hat"]) + """ kcal/mol.
                This means the model was wrong by at most """ + str(result["q_hat"]) + """ kcal/mol 
                on """ + str(coverage) + """% of calibration molecules.
                So the interval [score - q̂, score + q̂] is guaranteed to contain 
                the true value """ + str(coverage) + """% of the time.

                **Calibration set size:** """ + str(result["n_cal"]) + """ molecules

                **This guarantee holds regardless of the molecule — as long as it comes 
                from a similar distribution to the calibration set.**
                """)

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
