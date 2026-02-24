import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pubchempy as pcp
import plotly.express as px


# ---------- LOAD MODEL SAFELY ----------
model = load_model("model_bioinfo.h5", compile=False)
ohe = joblib.load("ohe_fabric.pkl")
scaler = joblib.load("scaler.pkl")


# ---------- MOLECULAR FEATURES ----------
def mol_to_features(smiles, radius=2, nBits=128):
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fp_arr = np.array(fp)

        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol)
        ]

        return np.concatenate([fp_arr, np.array(desc)])

    return np.zeros((nBits + 5,))


# ---------- NAME → SMILES ----------
def name_to_smiles(name):
    try:
        compound = pcp.get_compounds(name, 'name')
        if compound:
            return compound[0].canonical_smiles
    except:
        return None
    return None


# ---------- PREDICTION FUNCTION ----------
def predict_repellency(molecule_input, fabric, density, absorbency):

    # Convert name to SMILES if needed
    if Chem.MolFromSmiles(molecule_input):
        smiles = molecule_input
    else:
        smiles = name_to_smiles(molecule_input)
        if smiles is None:
            return None, None

    mol_features = mol_to_features(smiles)

    fabric_ohe = ohe.transform([[fabric]])

    fabric_props = np.array([density, absorbency]).reshape(1, -1)

    X = np.hstack([mol_features.reshape(1, -1),
                   fabric_ohe,
                   fabric_props])

    X_scaled = scaler.transform(X)

    score = model.predict(X_scaled)[0][0]

    return smiles, score


# ---------- STREAMLIT UI ----------
st.title("🦟 ANN-Based Mosquito Repellent Predictor")
st.write("Predict repellency effectiveness of compounds on fabrics.")


st.sidebar.header("User Input")

molecule_input = st.sidebar.text_input(
    "Enter Molecule (Name or SMILES)",
    "DEET"
)

fabric = st.sidebar.selectbox(
    "Select Fabric",
    ohe.categories_[0].tolist()
)

density = st.sidebar.slider("Fabric Density", 50, 300, 150)
absorbency = st.sidebar.slider("Absorbency", 0.0, 1.0, 0.5)


if st.sidebar.button("Predict"):

    smiles, score = predict_repellency(
        molecule_input, fabric, density, absorbency
    )

    if smiles is None:
        st.error("❌ Molecule not found.")
    else:
        st.subheader("Prediction Result")

        st.write(f"**Input Molecule:** {molecule_input}")
        st.write(f"**SMILES:** {smiles}")
        st.write(f"**Repellency Score:** {score:.2f}")

        if score > 0.8:
            st.success("Excellent repellency")
        elif score > 0.6:
            st.warning("Moderate repellency")
        else:
            st.error("Low repellency")

        # Comparison plot
        st.subheader("Repellency Across Fabrics")

        scores = []
        fabrics = ohe.categories_[0].tolist()

        for f in fabrics:
            _, s = predict_repellency(
                molecule_input,
                f,
                density,
                absorbency
            )
            scores.append(s)

        df_plot = pd.DataFrame({
            "Fabric": fabrics,
            "Repellency": scores
        })

        fig = px.bar(df_plot, x="Fabric", y="Repellency")
        st.plotly_chart(fig)
