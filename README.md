# ConformalDock

**Calibrated uncertainty quantification for molecular binding score prediction using conformal prediction.**

🔗 [Live App](https://conformaldock-kirtana.streamlit.app) | [bioRxiv Preprint](https://www.biorxiv.org/content/10.1101/2026.06.08.730837) | Submitted to J. Chem. Inf. Model.

## What it does

ConformalDock wraps a gradient-boosting binding score predictor with split conformal prediction to produce mathematically guaranteed prediction intervals for molecular docking scores.

Instead of a single number, you get: **−8.2 kcal/mol [−9.6, −6.8] at 90% coverage**.

## Validation

- Trained on **4,629 experimentally measured IC50 values** across **50 ChEMBL protein targets**
- **90.6% empirical coverage** at 90% nominal level on 926 held-out compounds
- Coverage confirmed at 80%, 85%, 90%, 95% — all within 0.5 pp of target
- Out-of-distribution detection via Euclidean distance to training manifold

## Paper

> Premnath, K. *ConformalDock: Calibrated Uncertainty Quantification for Molecular Binding Score Prediction Using Conformal Prediction.* bioRxiv 2026. Submitted to J. Chem. Inf. Model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

All training data from [ChEMBL](https://www.ebi.ac.uk/chembl/) (release 33), publicly available.

## Author

Kirtana Premnath · Independent Researcher, Chennai, India · kirtanaprem@gmail.com
