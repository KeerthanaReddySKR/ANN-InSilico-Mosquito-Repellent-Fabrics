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



## ⚙️ Installation  

```bash
git clone https://github.com/yourusername/ANN-InSilico-Mosquito-Repellent-Fabrics.git
cd ANN-InSilico-Mosquito-Repellent-Fabrics
pip install -r requirements.txt
