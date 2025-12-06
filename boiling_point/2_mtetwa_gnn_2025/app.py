import streamlit as st
import pandas as pd
import torch
from model_loader import load_gcn_model, load_mpnn_model  
from predictor import predict_boiling_point
import numpy as np

# Title
st.title("Normal Boiling Point Prediction Application")

# Sidebar - model selection
model_name = st.sidebar.selectbox("Select Prediction Model", ["GCN", "MPNN"])

# Input type selector
input_type = st.radio("Select Input Option", ["SMILES", "CSV"])

smiles_list = []
if input_type == "SMILES":
    smiles_input = st.text_area("Enter the SMILES (one per line)", height=150)
    if smiles_input:
        smiles_list = [line.strip() for line in smiles_input.strip().split('\n') if line.strip()]
elif input_type == "CSV":
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        if 'SMILES' not in df.columns:
            st.error("CSV must contain a 'SMILES' column")
        else:
            smiles_list = df['SMILES'].dropna().tolist()

# Predict button
if st.button("Predict") and smiles_list:
    with st.spinner("Loading model and predicting..."):

        if model_name == "GCN":
            model = load_gcn_model()
            predictions = np.round(predict_boiling_point(model, smiles_list, mode='gcn'),3)
        else:
            model = load_mpnn_model()
            predictions = predict_boiling_point(model, smiles_list, mode='mpnn')

        result_df = pd.DataFrame({"SMILES": smiles_list, "Predicted Boiling Point (K)": predictions})
        st.success("Prediction Complete")
        st.dataframe(result_df)
        csv_download = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv_download, file_name="predictions.csv", mime="text/csv")
