import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import pubchempy as pcp
import plotly.express as px

# ---------------- Load Model + Encoders ----------------
model = load_model("model_bioinfo.h5")
ohe = joblib.load("ohe_fabric.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Molecule Features ----------------
def mol_to_features(smiles, radius=2, nBits=256):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        fp_arr = np.array(fp)
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol)
        ]
        return np.concatenate([fp_arr, np.array(desc)])
    else:
        return np.zeros((nBits + 6,))

# ---------------- Name → SMILES ----------------
def name_to_smiles(name):
    try:
        compound = pcp.get_compounds(name, 'name')
        if compound and compound[0].canonical_smiles:
            return compound[0].canonical_smiles
    except:
        return None
    return None

# ---------------- Prediction ----------------
def predict_repellency(molecule_input, fabric, fabric_density=150, absorbency=0.5):
    # Convert name → SMILES if needed
    if Chem.MolFromSmiles(molecule_input):
        smiles = molecule_input
    else:
        smiles = name_to_smiles(molecule_input)
        if smiles is None:
            return None, None
    
    # Molecule features
    mol_features = mol_to_features(smiles)
    
    # Fabric one-hot
    fabric_ohe = ohe.transform([[fabric]])
    
    # Fabric properties
    fabric_props = np.array([fabric_density, absorbency]).reshape(1, -1)
    
    # Combine all
    X = np.hstack([mol_features.reshape(1, -1), fabric_ohe, fabric_props])
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    score = model.predict(X_scaled)[0][0]
    return smiles, score

# ---------------- Streamlit App ----------------
st.title("🦟 ANN-Based Prediction of Mosquito-Repellent Fabrics")

st.sidebar.header("User Input")
molecule_input = st.sidebar.text_input("Enter Molecule (Name or SMILES)", "DEET")
fabric = st.sidebar.selectbox("Select Fabric", ohe.categories_[0].tolist())
fabric_density = st.sidebar.slider("Fabric Density (g/m²)", 50, 300, 150)
absorbency = st.sidebar.slider("Absorbency", 0.0, 1.0, 0.5)

if st.sidebar.button("Predict"):
    smiles, score = predict_repellency(molecule_input, fabric, fabric_density, absorbency)
    
    if smiles is None:
        st.error("❌ Molecule not found in PubChem!")
    else:
        st.subheader("🔬 Molecule Information")
        st.write(f"**Input:** {molecule_input}")
        st.write(f"**SMILES:** {smiles}")
        
        # Molecule visualization
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption="Molecule Structure")
        
        # Repellency result
        st.subheader("🦟 Predicted Repellency")
        st.write(f"**Repellency Score:** {score:.2f}")
        
        if score > 0.8:
            st.success("✅ Excellent repellency")
        elif score > 0.6:
            st.warning("⚠️ Moderate repellency")
        else:
            st.error("❌ Low repellency")
        
        # Heatmap across fabrics
        st.subheader("📊 Repellency Across All Fabrics")
        scores = []
        for f in ohe.categories_[0].tolist():
            _, s = predict_repellency(molecule_input, f, fabric_density, absorbency)
            scores.append(s)
        
        df = pd.DataFrame({"Fabric": ohe.categories_[0].tolist(), "Repellency": scores})
        fig = px.bar(df, x="Fabric", y="Repellency", color="Repellency", range_y=[0,1])
        st.plotly_chart(fig)
