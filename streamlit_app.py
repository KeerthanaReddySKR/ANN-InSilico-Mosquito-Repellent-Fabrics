import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pubchempy as pcp
import plotly.express as px


# ---------------- LOAD MODEL ----------------
model = load_model("model_bioinfo.h5", compile=False)
ohe = joblib.load("ohe_fabric.pkl")
scaler = joblib.load("scaler.pkl")


# ---------------- MOLECULE LIST ----------------
molecule_list = [
    "DEET","Citronella","Limonene","Geraniol","Eucalyptol",
    "Neem oil","Lavender oil","Peppermint oil","Camphor","Thymol",
    "Menthol","Alpha-pinene","Beta-pinene","Linalool","Citral",
    "Borneol","Terpineol","Carvacrol","Eugenol","Myrcene",
    "Ocimene","Fenchone","Sabinene","Cineole","Isoeugenol"
]


# ---------------- MOLECULE FEATURES ----------------
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


# ---------------- NAME TO SMILES ----------------
def name_to_smiles(name):
    try:
        compound = pcp.get_compounds(name, "name")
        if compound:
            return compound[0].canonical_smiles
    except:
        return None
    return None


# ---------------- PREDICTION FUNCTION ----------------
def predict_repellency(molecule_input, fabric, density, absorbency):

    smiles = name_to_smiles(molecule_input)

    if smiles is None:
        return None, None

    mol_features = mol_to_features(smiles)

    fabric_ohe = ohe.transform([[fabric]])

    fabric_props = np.array([density, absorbency]).reshape(1, -1)

    X = np.hstack([
        mol_features.reshape(1, -1),
        fabric_ohe,
        fabric_props
    ])

    X_scaled = scaler.transform(X)

    score = float(model.predict(X_scaled)[0][0])

    # Ensure 0–1 range
    score = max(0, min(1, score))

    return smiles, score


# ---------------- STREAMLIT UI ----------------
st.title("🦟 ANN-Based Mosquito Repellent Predictor")
st.write("Research-grade in silico prediction of repellency effectiveness.")


st.sidebar.header("User Input")

molecule_input = st.sidebar.selectbox(
    "Select Molecule",
    molecule_list
)

fabric = st.sidebar.selectbox(
    "Select Fabric",
    ohe.categories_[0].tolist()
)

density = st.sidebar.slider("Fabric Density (g/m²)", 50, 300, 150)
absorbency = st.sidebar.slider("Absorbency", 0.0, 1.0, 0.5)


if st.sidebar.button("Predict"):

    smiles, score = predict_repellency(
        molecule_input,
        fabric,
        density,
        absorbency
    )

    if smiles is None:
        st.error("Molecule data not found.")
    else:
        st.subheader("Prediction Result")

        st.write(f"**Molecule:** {molecule_input}")
        st.write(f"**SMILES:** {smiles}")
        st.write(f"**Repellency Score:** {score:.2f}")

        # Interpretation
        if score >= 0.75:
            st.success("High repellency")
        elif score >= 0.5:
            st.warning("Moderate repellency")
        else:
            st.error("Low repellency")

        # Fabric comparison
        st.subheader("Repellency Across Fabrics")

        fabrics = ohe.categories_[0].tolist()
        scores = []

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

