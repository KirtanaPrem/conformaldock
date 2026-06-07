import streamlit as st
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ConformalDock",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #1c1917; }
[data-testid="stSidebar"] { background-color: #1c1917; border-right: 1px solid #44403c; }
.block-container { padding-top: 4rem; }
section[data-testid="stSidebar"] .stMarkdown p { color: #a8a29e; }
section[data-testid="stSidebar"] .stMarkdown h2 { color: #fafaf9; }
.app-title { font-size: 26px; font-weight: 700; color: #fafaf9; margin-bottom: 3px; }
.app-title span { color: #fb923c; }
.app-sub { color: #78716c; font-size: 12px; margin-bottom: 8px; }
.data-badge { display: inline-flex; align-items: center; gap: 6px; background: #292524; border: 1px solid #44403c; border-radius: 20px; padding: 4px 12px; color: #a8a29e; font-size: 11px; margin-bottom: 16px; }
.dot-green { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; }
.section-label { color: #78716c; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; margin: 14px 0 8px; }
.prop-card { background: #292524; border: 1px solid #44403c; border-radius: 8px; padding: 10px 12px; }
.prop-label { color: #78716c; font-size: 10px; margin-bottom: 4px; }
.prop-value { color: #fafaf9; font-size: 16px; font-weight: 600; }
.prop-unit { color: #57534e; font-size: 10px; }
.lip-row { display: flex; align-items: center; gap: 8px; background: #292524; border: 1px solid #44403c; border-radius: 6px; padding: 7px 10px; margin-bottom: 4px; }
.lip-dot-pass { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; flex-shrink: 0; }
.lip-dot-fail { width: 7px; height: 7px; border-radius: 50%; background: #f87171; flex-shrink: 0; }
.lip-text { color: #a8a29e; font-size: 11px; flex: 1; }
.lip-val { color: #57534e; font-size: 10px; }
.verdict-pass { background: #292524; border-left: 3px solid #4ade80; padding: 8px 12px; color: #86efac; font-size: 12px; margin: 8px 0; font-weight: 500; }
.verdict-fail { background: #292524; border-left: 3px solid #f87171; padding: 8px 12px; color: #fca5a5; font-size: 12px; margin: 8px 0; font-weight: 500; }
.result-card { background: #292524; border: 1px solid #44403c; border-radius: 8px; padding: 12px; }
.result-card.highlight { border-color: #fb923c; }
.result-label { color: #78716c; font-size: 10px; margin-bottom: 6px; }
.result-value { color: #fafaf9; font-size: 18px; font-weight: 700; }
.result-value.accent { color: #fb923c; }
.conf-label { color: #a8a29e; font-size: 11px; }
.conf-val { color: #fb923c; font-size: 11px; font-weight: 600; }
.bar-track { background: #44403c; border-radius: 4px; height: 6px; margin-top: 5px; }
.bar-fill-orange { background: #fb923c; border-radius: 4px; height: 6px; }
.verdict-high { background: #292524; border-left: 3px solid #4ade80; padding: 8px 12px; color: #86efac; font-size: 12px; margin: 8px 0; font-weight: 500; display: flex; align-items: center; gap: 8px; }
.verdict-medium { background: #292524; border-left: 3px solid #facc15; padding: 8px 12px; color: #fde68a; font-size: 12px; margin: 8px 0; font-weight: 500; display: flex; align-items: center; gap: 8px; }
.verdict-low { background: #292524; border-left: 3px solid #f87171; padding: 8px 12px; color: #fca5a5; font-size: 12px; margin: 8px 0; font-weight: 500; display: flex; align-items: center; gap: 8px; }
.vdot-g { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; flex-shrink: 0; }
.vdot-y { width: 7px; height: 7px; border-radius: 50%; background: #facc15; flex-shrink: 0; }
.vdot-r { width: 7px; height: 7px; border-radius: 50%; background: #f87171; flex-shrink: 0; }
.feat-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.feat-name { color: #a8a29e; font-size: 11px; min-width: 130px; }
.feat-track { flex: 1; background: #44403c; border-radius: 3px; height: 5px; }
.feat-fill { background: #fb923c; border-radius: 3px; height: 5px; }
.feat-pct { color: #78716c; font-size: 10px; min-width: 30px; text-align: right; }
.search-found { background: #292524; border: 1px solid #fb923c; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; }
.search-name { color: #fafaf9; font-size: 13px; font-weight: 600; margin-bottom: 2px; }
.search-smiles { color: #57534e; font-size: 9px; font-family: monospace; word-break: break-all; }
.citation-box { background: #292524; border: 1px solid #44403c; border-radius: 8px; padding: 10px 14px; margin-top: 16px; color: #57534e; font-size: 10px; line-height: 1.7; }
.result-table { width: 100%; border-collapse: collapse; }
.result-table th { background: #292524; color: #78716c; font-size: 10px; padding: 8px 10px; text-align: left; border-bottom: 1px solid #44403c; letter-spacing: 1px; text-transform: uppercase; }
.result-table td { color: #a8a29e; font-size: 11px; padding: 8px 10px; border-bottom: 1px solid #292524; }
.result-table tr:hover td { background: #292524; color: #fafaf9; }
</style>
""", unsafe_allow_html=True)

MOLECULE_SMILES = {
    "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", "C9H8O4"),
    "ibuprofen": ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen", "C13H18O2"),
    "caffeine": ("Cn1cnc2c1c(=O)n(c(=O)n2C)C", "Caffeine", "C8H10N4O2"),
    "morphine": ("CN1CCC23c4c(ccc(O)c4OC2(O)C=CC1C3)O", "Morphine", "C17H19NO3"),
    "paracetamol": ("CC(=O)Nc1ccc(O)cc1", "Paracetamol", "C8H9NO2"),
    "acetaminophen": ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen", "C8H9NO2"),
    "marijuana": ("CCCCCc1cc(O)c2c(c1)OC(C)(CCC=C(C)C)c1ccc(O)cc1-2", "THC (Cannabis)", "C21H30O2"),
    "thc": ("CCCCCc1cc(O)c2c(c1)OC(C)(CCC=C(C)C)c1ccc(O)cc1-2", "THC", "C21H30O2"),
    "cocaine": ("COC(=O)C1CC(OC(=O)c2ccccc2)CC1N", "Cocaine", "C17H21NO4"),
    "alcohol": ("CCO", "Ethanol", "C2H6O"),
    "ethanol": ("CCO", "Ethanol", "C2H6O"),
    "dopamine": ("NCCc1ccc(O)c(O)c1", "Dopamine", "C8H11NO2"),
    "serotonin": ("NCCc1c[nH]c2ccc(O)cc12", "Serotonin", "C10H12N2O"),
    "nicotine": ("CN1CCCC1c1cccnc1", "Nicotine", "C10H14N2"),
    "melatonin": ("COc1ccc2[nH]cc(CCNC(C)=O)c2c1", "Melatonin", "C13H16N2O2"),
    "adrenaline": ("CNC(C(O)c1ccc(O)c(O)c1)O", "Adrenaline", "C9H13NO3"),
    "mdma": ("CNC(C)Cc1ccc2c(c1)OCO2", "MDMA", "C11H15NO2"),
    "lsd": ("CCN(CC)C(=O)C1CN(C)C2Cc3c[nH]c4cccc(c34)C2=C1", "LSD", "C20H25N3O"),
    "warfarin": ("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "Warfarin", "C19H16O4"),
    "metformin": ("CN(C)C(=N)NC(N)=N", "Metformin", "C4H11N5"),
    "cholesterol": ("CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C", "Cholesterol", "C27H46O"),
    "testosterone": ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Testosterone", "C19H28O2"),
    "cortisol": ("CC12CCC3C(C1CCC2(C(=O)CO)O)CCC4=CC(=O)CCC34C", "Cortisol", "C21H30O5"),
    "glucose": ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Glucose", "C6H12O6"),
}

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

def search_pubchem_all(name):
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            + requests.utils.quote(name.strip())
            + "/property/CanonicalSMILES,IUPACName,MolecularFormula,"
            "MolecularWeight,IsomericSMILES/JSON"
        )
        headers = {"Accept": "application/json", "User-Agent": "ConformalDock/1.0"}
        resp = requests.get(url, timeout=15, headers=headers)
        if resp.status_code != 200:
            return []
        compounds = resp.json()["PropertyTable"]["Properties"]
        results = []
        seen_formulas = set()
        for c in compounds:
            smiles = c.get("CanonicalSMILES") or c.get("IsomericSMILES") or ""
            formula = c.get("MolecularFormula", "")
            iupac = c.get("IUPACName", "")
            mw = c.get("MolecularWeight", "")
            cid = c.get("CID", "")
            if not smiles:
                continue
            if formula in seen_formulas:
                continue
            seen_formulas.add(formula)
            results.append({
                "CID": str(cid),
                "Name": iupac.title() if iupac else name.title(),
                "Formula": formula,
                "MW": str(mw),
                "smiles": smiles,
            })
        return results
    except:
        return []

def lookup_local(name):
    key = name.strip().lower()
    if key in MOLECULE_SMILES:
        smiles, display_name, formula = MOLECULE_SMILES[key]
        return {"smiles": smiles, "name": display_name, "formula": formula}
    for k, (smiles, display_name, formula) in MOLECULE_SMILES.items():
        if key in k or key in display_name.lower():
            return {"smiles": smiles, "name": display_name, "formula": formula}
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chembl_data():
    from rdkit import Chem
    all_smiles, all_scores = [], []
    targets = [
        "CHEMBL202", "CHEMBL204", "CHEMBL1827", "CHEMBL261",
        "CHEMBL340", "CHEMBL205", "CHEMBL1862", "CHEMBL3784"
    ]
    for target_id in targets:
        try:
            url = (
                "https://www.ebi.ac.uk/chembl/api/data/activity"
                "?target_chembl_id=" + target_id +
                "&standard_type=IC50&standard_relation=%3D"
                "&assay_type=B&pchembl_value__isnull=false"
                "&limit=80&format=json"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            for act in resp.json().get("activities", []):
                smiles = act.get("canonical_smiles")
                pchembl = act.get("pchembl_value")
                if not smiles or not pchembl:
                    continue
                try:
                    score = -(float(pchembl) * 1.364)
                except:
                    continue
                if not (-15 < score < -2):
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.GetNumAtoms() <= 80:
                    all_smiles.append(smiles)
                    all_scores.append(score)
        except:
            continue
    return all_smiles, all_scores

@st.cache_resource(show_spinner=False)
def build_model():
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
    idx = np.random.RandomState(42).permutation(n)
    X_train = X[idx[:n - cal_size]]
    X_cal = X[idx[n - cal_size:]]
    y_train = y[idx[:n - cal_size]]
    y_cal = y[idx[n - cal_size:]]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, random_state=42, subsample=0.8
    )
    model.fit(X_train_s, y_train)
    cal_residuals = np.abs(y_cal - model.predict(X_cal_s))
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

def show_prediction(mol, smiles, model, scaler, cal_residuals, X_train_s, coverage):
    from rdkit.Chem import Descriptors, rdMolDescriptors
    st.success("Molecule loaded successfully!")
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = round(Descriptors.TPSA(mol), 2)
    atoms = mol.GetNumAtoms()

    st.markdown('<div class="section-label">Molecular properties</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="prop-card"><div class="prop-label">Molecular weight</div><div class="prop-value">' + str(mw) + '<span class="prop-unit"> g/mol</span></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="prop-card"><div class="prop-label">LogP</div><div class="prop-value">' + str(logp) + '</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="prop-card"><div class="prop-label">H-bond donors</div><div class="prop-value">' + str(hbd) + '</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="prop-card"><div class="prop-label">H-bond acceptors</div><div class="prop-value">' + str(hba) + '</div></div>', unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown('<div class="prop-card"><div class="prop-label">Rings</div><div class="prop-value">' + str(rings) + '</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="prop-card"><div class="prop-label">Rotatable bonds</div><div class="prop-value">' + str(rotbonds) + '</div></div>', unsafe_allow_html=True)
    with c7:
        st.markdown('<div class="prop-card"><div class="prop-label">TPSA</div><div class="prop-value">' + str(tpsa) + '</div></div>', unsafe_allow_html=True)
    with c8:
        st.markdown('<div class="prop-card"><div class="prop-label">Atoms</div><div class="prop-value">' + str(atoms) + '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Lipinski drug-likeness</div>', unsafe_allow_html=True)
    rule1 = mw <= 500
    rule2 = logp <= 5
    rule3 = hbd <= 5
    rule4 = hba <= 10
    passed = sum([rule1, rule2, rule3, rule4])

    lc1, lc2 = st.columns(2)
    with lc1:
        d1 = "lip-dot-pass" if rule1 else "lip-dot-fail"
        d2 = "lip-dot-pass" if rule2 else "lip-dot-fail"
        st.markdown('<div class="lip-row"><div class="' + d1 + '"></div><div class="lip-text">Molecular weight ≤ 500</div><div class="lip-val">' + str(mw) + '</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="lip-row"><div class="' + d2 + '"></div><div class="lip-text">LogP ≤ 5</div><div class="lip-val">' + str(logp) + '</div></div>', unsafe_allow_html=True)
    with lc2:
        d3 = "lip-dot-pass" if rule3 else "lip-dot-fail"
        d4 = "lip-dot-pass" if rule4 else "lip-dot-fail"
        st.markdown('<div class="lip-row"><div class="' + d3 + '"></div><div class="lip-text">H-bond donors ≤ 5</div><div class="lip-val">' + str(hbd) + '</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="lip-row"><div class="' + d4 + '"></div><div class="lip-text">H-bond acceptors ≤ 10</div><div class="lip-val">' + str(hba) + '</div></div>', unsafe_allow_html=True)

    if passed == 4:
        st.markdown('<div class="verdict-pass">All 4 rules passed — good oral drug candidate</div>', unsafe_allow_html=True)
    elif passed == 3:
        st.markdown('<div class="verdict-pass">3 of 4 rules passed — borderline oral drug candidate</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-fail">Only ' + str(passed) + ' of 4 rules passed — may not be orally bioavailable</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Conformal prediction result</div>', unsafe_allow_html=True)
    features = get_features(mol)
    result_cp = conformal_predict(model, scaler, cal_residuals, features, coverage)
    similarity = get_similarity(features, X_train_s, scaler)
    sim_pct = int(similarity * 100)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown('<div class="result-card"><div class="result-label">Predicted binding score</div><div class="result-value">' + str(result_cp["point"]) + ' kcal/mol</div></div>', unsafe_allow_html=True)
    with rc2:
        st.markdown('<div class="result-card highlight"><div class="result-label">Guaranteed interval (' + str(coverage) + '%)</div><div class="result-value accent">[' + str(result_cp["lower"]) + ', ' + str(result_cp["upper"]) + ']</div></div>', unsafe_allow_html=True)
    with rc3:
        st.markdown('<div class="result-card"><div class="result-label">Interval width</div><div class="result-value">' + str(result_cp["width"]) + ' kcal/mol</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="display:flex;justify-content:space-between;"><span class="conf-label">Similarity to training data</span><span class="conf-val">' + str(sim_pct) + '%</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="bar-track"><div class="bar-fill-orange" style="width:' + str(sim_pct) + '%"></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if similarity > 0.6 and result_cp["width"] < 5.0:
        st.markdown('<div class="verdict-high"><div class="vdot-g"></div>High confidence — molecule is well represented in training data</div>', unsafe_allow_html=True)
    elif similarity > 0.35:
        st.markdown('<div class="verdict-medium"><div class="vdot-y"></div>Medium confidence — prediction is reasonable, treat with some caution</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-low"><div class="vdot-r"></div>Low confidence — molecule is unusual, prediction less reliable</div>', unsafe_allow_html=True)

    with st.expander("What is conformal prediction?"):
        st.markdown("""
Conformal prediction converts any ML model into one with a **mathematically guaranteed** interval.

Instead of assuming the data follows a specific distribution, it uses a held-out calibration set
to measure how wrong the model typically is, then sets interval width to guarantee coverage.

**Conformal quantile:** """ + str(result_cp["q_hat"]) + """ kcal/mol · **Calibration set:** """ + str(result_cp["n_cal"]) + """ molecules
        """)

    with st.expander("Feature importance — what drove this prediction?"):
        importances = model.feature_importances_
        names = get_feature_names()
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:8]
        for name, imp in pairs:
            pct = round(imp * 100, 1)
            st.markdown(
                '<div class="feat-row">'
                '<div class="feat-name">' + name + '</div>'
                '<div class="feat-track"><div class="feat-fill" style="width:' + str(pct) + '%"></div></div>'
                '<div class="feat-pct">' + str(pct) + '%</div>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="citation-box">'
        'Kirtana Premnath (2026). ConformalDock: Calibrated Uncertainty Quantification for '
        'Molecular Docking Scores using Conformal Prediction. '
        'MSc Bioinformatics and Data Science, Sathyabama Institute of Science and Technology, Chennai. '
        'conformaldock-kirtana.streamlit.app'
        '</div>',
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    coverage = st.slider("Coverage level", 80, 95, 90, 5)
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. Search any molecule by name
2. Choose exact compound from results
3. **ChEMBL** provides real binding data
4. **Gradient Boosting** predicts score
5. **Conformal prediction** guarantees interval
    """)
    st.markdown("---")
    st.markdown("**Kirtana Premnath**\nMSc Bioinformatics & Data Science")

st.markdown('<div class="app-title">🔬 <span>Conformal</span>Dock</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Calibrated Uncertainty for Molecular Docking Scores · Trained on real experimental data from ChEMBL</div>', unsafe_allow_html=True)
st.markdown('<div class="data-badge"><div class="dot-green"></div> Model trained on real ChEMBL experimental measurements</div>', unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model, scaler, cal_residuals, X_train_s, n_train, n_cal = build_model()

if model is None:
    st.error("Could not load ChEMBL data. Please refresh.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Benchmark", "📄 About"])

with tab1:
    st.markdown('<div class="section-label">Search molecule</div>', unsafe_allow_html=True)

    search_mode = st.radio(
        "Input method",
        ["Search by name", "Paste SMILES directly"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if search_mode == "Paste SMILES directly":
        smiles_input = st.text_input(
            "Paste SMILES string",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
        )
        if smiles_input:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                st.markdown("---")
                show_prediction(mol, smiles_input, model, scaler, cal_residuals, X_train_s, coverage)
            else:
                st.error("Could not parse SMILES — please check and try again.")

    else:
        st.markdown("**Quick examples:**")
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        with ec1:
            if st.button("Aspirin", use_container_width=True):
                st.session_state["search_query"] = "aspirin"
        with ec2:
            if st.button("Caffeine", use_container_width=True):
                st.session_state["search_query"] = "caffeine"
        with ec3:
            if st.button("Morphine", use_container_width=True):
                st.session_state["search_query"] = "morphine"
        with ec4:
            if st.button("Dopamine", use_container_width=True):
                st.session_state["search_query"] = "dopamine"
        with ec5:
            if st.button("THC", use_container_width=True):
                st.session_state["search_query"] = "thc"

        default_query = st.session_state.get("search_query", "")
        drug_name = st.text_input(
            "Type any drug, molecule, or compound name",
            value=default_query,
            placeholder="e.g. penicillin, cannabidiol, serotonin, sildenafil..."
        )

        if drug_name:
            local = lookup_local(drug_name)
            if local:
                st.markdown(
                    '<div class="search-found">'
                    '<div class="search-name">Found: ' + local["name"] + '</div>'
                    '<div class="search-smiles">Formula: ' + local["formula"] + '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                from rdkit import Chem
                mol = Chem.MolFromSmiles(local["smiles"])
                if mol:
                    st.markdown("---")
                    show_prediction(mol, local["smiles"], model, scaler, cal_residuals, X_train_s, coverage)
            else:
                with st.spinner("Searching PubChem for all compounds matching " + drug_name + "..."):
                    results = search_pubchem_all(drug_name)

                if not results:
                    st.error("No compounds found for \"" + drug_name + "\". Try a different name or spelling.")
                else:
                    st.markdown(
                        '<div class="section-label">Found ' + str(len(results)) + ' compounds — click a row to select</div>',
                        unsafe_allow_html=True
                    )

                    display_df = pd.DataFrame([{
                        "PubChem CID": r["CID"],
                        "Name": r["Name"],
                        "Formula": r["Formula"],
                        "Mol. Weight": r["MW"],
                    } for r in results])

                    selected = st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                    )

                    if selected and selected.selection.rows:
                        row_idx = selected.selection.rows[0]
                        chosen = results[row_idx]
                        st.markdown(
                            '<div class="search-found">'
                            '<div class="search-name">Selected: ' + chosen["Name"] + ' (' + chosen["Formula"] + ')</div>'
                            '<div class="search-smiles">CID: ' + chosen["CID"] + ' · SMILES: ' + chosen["smiles"][:80] + '...</div>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        from rdkit import Chem
                        mol = Chem.MolFromSmiles(chosen["smiles"])
                        if mol:
                            st.markdown("---")
                            show_prediction(mol, chosen["smiles"], model, scaler, cal_residuals, X_train_s, coverage)
                        else:
                            st.error("Could not parse this compound's structure. Try selecting a different one.")

with tab2:
    st.markdown("## Benchmark results")
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Training molecules", str(n_train))
    with mc2:
        st.metric("Calibration molecules", str(n_cal))
    with mc3:
        st.metric("Data source", "ChEMBL")
    with mc4:
        st.metric("Features", "16")
    st.markdown("### Method comparison")
    comparison = pd.DataFrame({
        "Method": ["AutoDock Vina", "Random Forest", "RF + heuristic interval", "ConformalDock (this work)"],
        "Real experimental data": ["No", "Depends", "Depends", "Yes — ChEMBL"],
        "Uncertainty estimate": ["No", "No", "Heuristic", "Guaranteed"],
        "OOD detection": ["No", "No", "No", "Yes"],
        "Explainability": ["No", "Partial", "Partial", "Yes"],
        "Free & open": ["Yes", "Yes", "Yes", "Yes"],
    })
    st.dataframe(comparison, use_container_width=True)
    st.info("Full validation on CASF-2016 (285 complexes) in preparation. Target journal: Journal of Chemical Information and Modeling.")

with tab3:
    st.markdown("## About ConformalDock")
    st.markdown("---")
    st.markdown("""
### The problem
Every molecular docking tool gives a single binding score with no uncertainty.
Drug discovery teams make expensive synthesis decisions based on a number
that could be significantly wrong — with no way to know.

### The solution
ConformalDock wraps any prediction with a **mathematically guaranteed**
confidence interval using conformal prediction theory, trained on real
experimental IC50 binding data from ChEMBL.

### What makes this different
- Search any molecule by name — see all matching compounds and pick the exact one
- Trained on real ChEMBL IC50 measurements, not simulated data
- Conformal prediction gives a valid coverage guarantee, not a heuristic
- OOD detection tells you when the molecule is too unusual to trust
- Feature importance explains every prediction

### How to cite
*Kirtana Premnath (2026). ConformalDock: Calibrated Uncertainty Quantification
for Molecular Docking Scores using Conformal Prediction.
MSc Bioinformatics and Data Science, Sathyabama Institute of Science and Technology, Chennai.
conformaldock-kirtana.streamlit.app*

### Links
- Live app: conformaldock-kirtana.streamlit.app
- GitHub: github.com/KirtanaPrem/conformaldock
- Data source: ebi.ac.uk/chembl
    """)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#44403c;font-size:11px;'>ConformalDock · Kirtana Premnath · MSc Bioinformatics and Data Science · 2026</p>", unsafe_allow_html=True)
