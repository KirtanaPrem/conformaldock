import streamlit as st

st.set_page_config(
    page_title="ConformalDock",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 ConformalDock")
st.subheader("Calibrated Uncertainty for Molecular Docking Scores")

st.success("App is live and connected to GitHub!")

st.markdown("""
### What is this?
Most drug discovery tools give you a single score with no indication 
of how trustworthy that score is.

**ConformalDock** adds a guaranteed confidence interval to every 
prediction — so you know not just *what* the score is, but 
*how much to trust it*.

### Coming soon:
- Paste a molecule and get a prediction with confidence interval
- Upload a whole list of molecules and rank them by trustworthiness  
- See which molecules the model is uncertain about
""")

st.info("Built by Kirtana Premnath · MSc Bioinformatics")
