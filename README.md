# 🦟 ANN-InSilico Mosquito-Repellent Fabrics  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)  
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)](https://www.rdkit.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)  



## 📖 Abstract  

Vector-borne diseases such as **malaria, dengue, and Zika** remain major global health challenges. Traditional methods of testing **mosquito repellents** on fabrics are time-consuming, costly, and ethically challenging.  

This project introduces an **Artificial Neural Network (ANN)-based in silico framework** for predicting:  
1. **Mosquito-repellent effectiveness** of compounds on different fabrics.  
2. **Pollinator (bee) safety** for ecological sustainability.  

By integrating **cheminformatics descriptors**, **fabric/material properties**, and **AI-driven modeling**, this project demonstrates how computational methods can accelerate **smart clothing design** for disease prevention.  



## 🎯 Problem Statement  

- 🌍 Millions are affected by mosquito-borne diseases annually.  
- 🧪 Current testing requires **in vivo assays** → costly, slow, ethically challenging.  
- 🐝 Repellent safety for pollinators is often neglected in early design stages.  

**Solution:** An ANN-powered **dual-prediction system** for *repellency* and *bee safety*, with a **Streamlit web interface** for usability.  


## ✨ Features  

- 🧬 **Cheminformatics Integration**: Molecular input via **SMILES** or compound names, processed with **RDKit** & **PubChemPy**.  
- 🧵 **Material Informatics**: Includes **fabric type, weave, density, absorbency, and thickness**.  
- 🤖 **Multi-Task ANN**: Simultaneous prediction of **mosquito repellency** & **bee safety scores**.  
- 📊 **Visualization Dashboard**: Scatter, histogram, bar, and 3D molecular descriptor plots.  
- 💾 **Data Persistence**: SQLite database for prediction history.  
- 📑 **Automated Reporting**: Downloadable **PDF summaries** of results.  
- 🌐 **Colab + Streamlit Deployment**: Accessible via ngrok tunneling for global usability.  



## 🛠️ Skills & Technologies Demonstrated  

- **Bioinformatics**: Data preprocessing, descriptor handling, in silico modeling.  
- **Machine Learning**: ANN design, multi-task learning, optimization.  
- **Cheminformatics**: Descriptor extraction, SMILES parsing, chemical feature engineering.  
- **Material Science Informatics**: Integration of textile parameters.  
- **Visualization**: Interactive 2D/3D plots.  
- **Software Engineering**: Modular ML pipeline + deployment-ready web app.  
- **Reporting & Storage**: SQLite integration & automated PDF generation.  



## 🛠️ Tech Stack  

- **Core ML**: TensorFlow/Keras, scikit-learn  
- **Cheminformatics**: RDKit, PubChemPy  
- **Data Handling**: pandas, numpy  
- **Visualization**: matplotlib, seaborn, plotly  
- **App Development**: Streamlit, pyngrok  
- **Reporting**: ReportLab  
- **Storage**: SQLite  

## ANN-InSilico-Mosquito-Repellent-Fabrics
```
├─ model_bioinfo.py        # Main ANN model
├─ predict.py              # Script for new predictions
├─ visualize_results.py    # Visualization scripts
├─ input_data.csv          # Sample input data
├─ requirements.txt        # Python dependencies
├─ README.md
└─ /outputs               # Folder for results and plots
```

## ⚙️ Installation  

**1. Clone the repository**
```bash
git clone https://github.com/KeerthanaReddySKR/ANN-InSilico-Mosquito-Repellent-Fabrics.git
```
**2. Navigate into the project folder**
```bash
cd ANN-InSilico-Mosquito-Repellent-Fabrics
```
**3. Install dependencies(Ensure you have Python (≥3.8) and pip installed. Then run:)**
```bash
pip install -r requirements.txt
```
**4. (Optional) Create a virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Linux/Mac)
source venv/bin/activate

# Then install dependencies
pip install -r requirements.txt
```

## 📈 Visualizations  

The project includes several data visualizations for interpretation:  

### 1️⃣ Bar Chart – Mean Repellency by Fabric  
```python
import plotly.express as px
fig = px.bar(df, x="fabric", y="repellency", color="fabric",
             title="Mean Repellency by Fabric", text_auto=".2f")
fig.show() 
```
### 2️⃣ Scatter Plot – Repellency vs Bee Safety
```python
fig = px.scatter(df, x="repellency", y="bee_score", color="fabric",
                 size="thickness", hover_data=["molecule","mw","logp"],
                 title="Repellency vs Bee Safety")
fig.show()
```
### 3️⃣ Histogram – Bee Safety Distribution
```python
fig = px.histogram(df, x="bee_score", color="bee_toxicity", nbins=20,
                   title="Bee Safety Distribution")
fig.show()
```
### 4️⃣ 3D Plot – Molecular Descriptor Space
```python
fig = px.scatter_3d(df, x="mw", y="logp", z="repellency",
                    color="fabric", size="thickness",
                    title="3D Molecular Space (MW, LogP, Repellency)")
fig.show()
```
![image_alt](https://github.com/KeerthanaReddySKR/ANN-InSilico-Mosquito-Repellent-Fabrics/blob/main/Images/Database.png?raw=true)
