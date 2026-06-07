import streamlit as st
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
.paper-box { background: #0f1f0f; border: 1px solid #22c55e; border-radius: 12px; padding: 20px; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

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
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.NumValenceElectrons(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumHeterocycles(mol),
    ]

def get_feature_names():
    return [
        "Molecular weight", "LogP", "H-bond donors", "H-bond acceptors",
        "Ring count", "Rotatable bonds", "TPSA", "Atom count",
        "Aromatic rings", "SP3 fraction", "Heteroatoms", "Total rings",
        "Radical electrons", "Valence electrons", "Amide bonds", "Heterocycles"
    ]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chembl_data():
    from rdkit import Chem
    all_smiles = []
    all_scores = []

    targets = [
        "CHEMBL202",   # COX-2
        "CHEMBL204",   # COX-1
        "CHEMBL1827",  # EGFR
        "CHEMBL261",   # ACE
        "CHEMBL340",   # Thrombin
        "CHEMBL205",   # DHFR
        "CHEMBL1862",  # Aurora A
        "CHEMBL3784",  # JAK2
    ]

    for target_id in targets:
        try:
            url = (
                "https://www.ebi.ac.uk/chembl/api/data/activity"
                "?target_chembl_id=" + target_id +
                "&standard_type=IC50"
                "&standard_relation=%3D"
                "&assay_type=B"
                "&pchembl_value__isnull=false"
                "&limit=80"
                "&format=json"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            activities = data.get("activities", [])
            for act in activities:
                smiles = act.get("canonical_smiles")
                pchembl = act.get("pchembl_value")
                if not smiles or not pchembl:
                    continue
                try:
                    pval = float(pchembl)
                except:
                    continue
                score = -(pval * 1.364)
                if not (-15 < score < -2):
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                if mol.GetNumAtoms() > 80:
                    continue
                all_smiles.append(smiles)
                all_scores.append(score)
        except:
            continue

    return all_smiles, all_scores

@st.cache_resource(show_spinner=False)
def build_model_from_chembl():
    from rdkit import Chem

    smiles_list, scores_list = fetch_chembl_data()

    if len(smiles_list) < 30:
        return None, None, None, None, 0, 0

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [(m, sc) for m, sc in zip(mols, scores_list) if m is not None]

    X = np.array([get_features(m) for m, _ in valid])
    y = np.array([sc for _, sc in valid])

    n = len(y)
    cal_size = max(30, n // 5)
    train_size = n - cal_size

    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:train_size]
    cal_idx = indices[train_size:]

    X_train, X_cal = X[train_idx], X[cal_idx]
    y_train, y_cal = y[train_idx], y[cal_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        subsample=0.8,
    )
    model.fit(X_train_s, y_train)

    cal_preds = model.predict(X_cal_s)
    cal_residuals = np.abs(y_cal - cal_preds)

    from sklearn.metrics import mean_squared_error
    train_preds = model.predict(X_train_s)
    rmse = round(float(np.sqrt(mean_squared_error(y_train, train_preds))), 3)

    return model, scaler, cal_residuals, X_train_s, len(y_train), len(y_cal)

def conformal_predict(model, scaler, cal_residuals, features, coverage):
    fa = np.array(features).reshape(1, -1)
    fs = scaler.transform(fa)
    point = float(model.predict(fs)[0])
    alpha = 1.0 - coverage / 100
    n = len(cal_residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(cal_residuals, level))
    return {
        "point": round(point, 2),
        "lower": round(point - q_hat, 2),
        "upper": round(point + q_hat, 2),
        "width": round(2 * q_hat, 2),
        "q_hat": round(q_hat, 3),
        "n_cal": n,
    }

def get_similarity(features, X_train_s, scaler):
    q = scaler.transform(np.array(features).reshape(1, -1))
    dists = np.linalg.norm(X_train_s - q, axis=1)
    return round(min(max(float(np.exp(-dists.min() / 15)), 0.05), 0.99), 2)

with st.sidebar:
    st.markdown("## Settings")
    coverage = st.slider("Coverage level", 80, 95, 90, 5)
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. **ChEMBL data** — real experimental binding measurements
2. **Gradient Boosting** predicts binding score
3. **Conformal prediction** adds guaranteed interval
4. **OOD detector** flags unusual molecules
    """)
    st.markdown("---")
    st.markdown("Built by **Kirtana Premnath**\nMSc Bioinformatics")

tab1, tab2, tab3 = st.tabs(["Predict", "Benchmark", "About"])

with tab1:
    st.title("🔬 ConformalDock")
    st.markdown("#### Calibrated Uncertainty for Molecular Docking Scores")
    st.markdown("*Trained on real experimental binding data from ChEMBL. Guaranteed confidence intervals using conformal prediction.*")
    st.markdown("---")

    with st.spinner("Loading real ChEMBL experimental data and training model..."):
        result_model = build_model_from_chembl()
        model, scaler, cal_residuals, X_train_s, n_train, n_cal = result_model

    if model is None:
        st.error("Could not load ChEMBL data right now. Please refresh the page.")
        st.stop()

    st.success("Model trained on " + str(n_train) + " real experimental measurements from ChEMBL")

    smiles = st.text_input(
        "Paste any drug-like molecule (SMILES format)",
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
    )

    st.markdown("**Try an example:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Aspirin", use_container_width=True):
            smiles = "CC(=O)Oc1ccccc1C(=O)O"
    with col2:
        if st.button("Ibuprofen", use_container_width=True):
            smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    with col3:
        if st.button("Paracetamol", use_container_width=True):
            smiles = "CC(=O)Nc1ccc(O)cc1"
    with col4:
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
                st.markdown("### Binding prediction")
                st.markdown("Based on real experimental IC50 measurements from ChEMBL database.")

                features = get_features(mol)
                result = conformal_predict(model, scaler, cal_residuals, features, coverage)
                similarity = get_similarity(features, X_train_s, scaler)

                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    st.metric(
                        "Predicted binding score",
                        str(result["point"]) + " kcal/mol",
                        help="Derived from pChEMBL values. More negative = stronger binding."
                    )
                with bc2:
                    st.metric(
                        "Guaranteed interval (" + str(coverage) + "%)",
                        "[" + str(result["lower"]) + ", " + str(result["upper"]) + "]",
                        help="True value falls here " + str(coverage) + "% of the time."
                    )
                with bc3:
                    st.metric(
                        "Interval width",
                        str(result["width"]) + " kcal/mol",
                        help="Narrower = more certain"
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                if similarity > 0.6 and result["width"] < 5.0:
                    st.markdown('<div class="trust-high">High confidence — molecule is similar to training data</div>', unsafe_allow_html=True)
                elif similarity > 0.35:
                    st.markdown('<div class="trust-medium">Medium confidence — treat with some caution</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="trust-low">Low confidence — molecule is very different from training data</div>', unsafe_allow_html=True)

                if similarity < 0.35:
                    st.warning("This molecule is outside the training distribution. The conformal guarantee may not fully apply here.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Similarity to training data:** " + str(similarity))
                st.progress(similarity)

                with st.expander("What is conformal prediction?"):
                    st.markdown("""
Conformal prediction converts any ML model into one with a **mathematically guaranteed** interval.

Instead of assuming the data follows a specific distribution (which standard confidence intervals do),
conformal prediction uses a held-out calibration set to measure how wrong the model typically is.
It then sets the interval width so that the stated coverage is guaranteed.

**Conformal quantile used:** """ + str(result["q_hat"]) + """ kcal/mol
**Calibration set size:** """ + str(result["n_cal"]) + """ real molecules
                    """)

                with st.expander("Feature importance — what drove this prediction?"):
                    importances = model.feature_importances_
                    names = get_feature_names()
                    pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
                    for name, imp in pairs:
                        st.markdown("**" + name + ":** " + str(round(imp * 100, 1)) + "%")
                        st.progress(float(imp))

        except Exception as e:
            st.error("Something went wrong.")
            st.write(str(e))

with tab2:
    st.markdown("## Benchmark results")
    st.markdown("---")

    with st.spinner("Loading data..."):
        result_model2 = build_model_from_chembl()
        model2, scaler2, cal_res2, X_tr2, n_tr2, n_cal2 = result_model2

    if model2 is not None:
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Training molecules", str(n_tr2))
        with mc2:
            st.metric("Calibration molecules", str(n_cal2))
        with mc3:
            st.metric("Data source", "ChEMBL")
        with mc4:
            st.metric("Features per molecule", "16")

        st.markdown("### Method comparison")
        comparison = pd.DataFrame({
            "Method": [
                "AutoDock Vina (standard)",
                "Random Forest (no uncertainty)",
                "RF + heuristic interval",
                "ConformalDock (this work)",
            ],
            "Real experimental data": ["No", "Depends", "Depends", "Yes — ChEMBL"],
            "Uncertainty estimate": ["No", "No", "Heuristic", "Guaranteed"],
            "OOD detection": ["No", "No", "No", "Yes"],
            "Explainability": ["No", "Partial", "Partial", "Yes"],
            "Free": ["Yes", "Yes", "Yes", "Yes"],
        })
        st.dataframe(comparison, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="paper-box"><b>Publication note:</b> Full validation on CASF-2016 (285 complexes) in preparation. Target journal: Journal of Chemical Information and Modeling.</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("## About ConformalDock")
    st.markdown("---")
    st.markdown("""
### The problem
Every molecular docking tool gives a single score with no uncertainty.
Drug discovery teams make expensive synthesis decisions based on a number
that could be wildly wrong — with no way to know.

### The solution
ConformalDock wraps predictions with a **mathematically guaranteed**
confidence interval using conformal prediction theory.

It is trained on **real experimental binding data** from ChEMBL —
the world's largest open database of drug-like molecules with measured
biological activity.

### What makes this different
- Trained on real ChEMBL IC50 measurements — not simulated data
- Conformal prediction gives a valid coverage guarantee — not a heuristic
- OOD detection tells you when the molecule is too unusual to trust
- Feature importance explains every prediction

### How to cite
*Premnath, K. (2024). ConformalDock: Calibrated Uncertainty Quantification
for Molecular Docking Scores using Conformal Prediction.
MSc Bioinformatics. conformaldock-kirtana.streamlit.app*

### Links
- Live app: conformaldock-kirtana.streamlit.app
- GitHub: github.com/KirtanaPrem/conformaldock
- Data source: ebi.ac.uk/chembl
    """)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#475569;'>ConformalDock · Built by Kirtana Premnath · MSc Bioinformatics · 2024</p>", unsafe_allow_html=True)
