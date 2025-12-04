"""
Cetane Number Predictor - Streamlit App
Deployed on HuggingFace Spaces
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from rdkit import Chem

# Page config
st.set_page_config(
    page_title="Cetane Number Predictor",
    page_icon="⛽",
    layout="wide"
)

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('cn_predictor_complete.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Validate SMILES
def validate_smiles(smiles):
    """Check if SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Main app
def main():
    st.title("⛽ Cetane Number Predictor")
    st.markdown("""
    Predict cetane numbers from SMILES strings using machine learning.
    
    **Input options:**
    - Single SMILES string
    - CSV file with SMILES column
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check if 'cn_predictor_complete.pkl' exists.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This model predicts cetane numbers for fuel molecules based on their chemical structure (SMILES).
        
        **Features:**
        - Morgan fingerprints (2048 bits)
        - 208 RDKit molecular descriptors
        - Feature selection pipeline
        
        **Model:** Trained on experimental cetane number data
        """)
        
        st.header("Example SMILES")
        st.code("CCCCCCCCCCCCCCCC", language="text")
        st.caption("Hexadecane (CN ≈ 100)")
        
        st.code("CC(C)CCCCC", language="text")
        st.caption("Isoheptane (CN ≈ 30)")
    
    # Main content - tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    # ==================== TAB 1: Single Prediction ====================
    with tab1:
        st.header("Single SMILES Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            smiles_input = st.text_input(
                "Enter SMILES string:",
                placeholder="e.g., CCCCCCCCCCCCCCCC",
                help="Input a valid SMILES representation of your molecule"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
        
        if predict_button and smiles_input:
            with st.spinner("Calculating cetane number..."):
                # Validate SMILES
                if not validate_smiles(smiles_input):
                    st.error("❌ Invalid SMILES string. Please check your input.")
                else:
                    try:
                        # Make prediction
                        cn_value = model.predict_single(smiles_input)
                        
                        # Display result
                        st.success("✅ Prediction Complete!")
                        
                        # Result card
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.metric(
                                label="Predicted Cetane Number",
                                value=f"{cn_value:.2f}",
                                help="Higher cetane numbers indicate better ignition quality"
                            )
                        
                        # Interpretation
                        st.subheader("Interpretation")
                        if cn_value >= 55:
                            st.info("🟢 **Excellent** cetane quality - Premium diesel fuel")
                        elif cn_value >= 40:
                            st.info("🟡 **Good** cetane quality - Standard diesel fuel")
                        else:
                            st.info("🔴 **Low** cetane quality - May require additives")
                        
                        # Show molecular structure (optional)
                        with st.expander("View Molecular Structure"):
                            mol = Chem.MolFromSmiles(smiles_input)
                            from rdkit.Chem import Draw
                            img = Draw.MolToImage(mol, size=(400, 300))
                            st.image(img, caption="Molecular Structure")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
        
        elif predict_button:
            st.warning("⚠️ Please enter a SMILES string first.")
    
    # ==================== TAB 2: Batch Prediction ====================
    with tab2:
        st.header("Batch Prediction from CSV")
        
        st.markdown("""
        Upload a CSV file with a column containing SMILES strings.
        The app will predict cetane numbers for all molecules.
        """)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV must contain a 'SMILES' column"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.write("### Preview of uploaded data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check for SMILES column
                smiles_columns = [col for col in df.columns if 'smiles' in col.lower()]
                
                if not smiles_columns:
                    st.error("❌ No 'SMILES' column found in CSV. Please ensure your CSV has a column named 'SMILES'.")
                else:
                    # Let user select SMILES column if multiple found
                    if len(smiles_columns) > 1:
                        smiles_col = st.selectbox("Select SMILES column:", smiles_columns)
                    else:
                        smiles_col = smiles_columns[0]
                    
                    st.info(f"Using column: **{smiles_col}**")
                    
                    # Predict button
                    if st.button("🔮 Predict All", type="primary"):
                        with st.spinner(f"Predicting cetane numbers for {len(df)} molecules..."):
                            try:
                                # Validate and predict
                                predictions = []
                                valid_smiles = []
                                invalid_indices = []
                                
                                progress_bar = st.progress(0)
                                
                                for idx, smiles in enumerate(df[smiles_col]):
                                    # Update progress
                                    progress_bar.progress((idx + 1) / len(df))
                                    
                                    if pd.isna(smiles) or not validate_smiles(smiles):
                                        predictions.append(np.nan)
                                        invalid_indices.append(idx)
                                    else:
                                        try:
                                            cn = model.predict_single(smiles)
                                            predictions.append(cn)
                                            valid_smiles.append(smiles)
                                        except:
                                            predictions.append(np.nan)
                                            invalid_indices.append(idx)
                                
                                progress_bar.empty()
                                
                                # Add predictions to dataframe
                                df['Predicted_CN'] = predictions
                                
                                # Show results
                                st.success(f"✅ Predictions complete! Valid: {len(valid_smiles)}/{len(df)}")
                                
                                if invalid_indices:
                                    st.warning(f"⚠️ {len(invalid_indices)} invalid SMILES were skipped.")
                                
                                # Display results
                                st.write("### Results")
                                st.dataframe(df, use_container_width=True)
                                
                                # Statistics
                                col1, col2, col3, col4 = st.columns(4)
                                valid_predictions = [p for p in predictions if not np.isnan(p)]
                                
                                with col1:
                                    st.metric("Total Molecules", len(df))
                                with col2:
                                    st.metric("Valid Predictions", len(valid_predictions))
                                with col3:
                                    st.metric("Mean CN", f"{np.mean(valid_predictions):.2f}" if valid_predictions else "N/A")
                                with col4:
                                    st.metric("Std CN", f"{np.std(valid_predictions):.2f}" if valid_predictions else "N/A")
                                
                                # Download button
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name="cetane_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                # Distribution plot
                                if valid_predictions:
                                    st.write("### Distribution of Predicted Cetane Numbers")
                                    
                                    # Create histogram data
                                    hist_data = pd.DataFrame({
                                        'Cetane Number': valid_predictions
                                    })
                                    
                                    st.bar_chart(hist_data['Cetane Number'].value_counts().sort_index())
                                
                            except Exception as e:
                                st.error(f"❌ Batch prediction failed: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
        
        else:
            # Show example CSV format
            st.info("💡 **CSV Format Example:**")
            example_df = pd.DataFrame({
                'SMILES': ['CCCCCCCCCCCCCCCC', 'CC(C)CCCCC', 'c1ccccc1'],
                'Name': ['Hexadecane', 'Isoheptane', 'Benzene']
            })
            st.dataframe(example_df, use_container_width=True)
            
            # Download example CSV
            example_csv = example_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Example CSV",
                data=example_csv,
                file_name="example_smiles.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Model trained on experimental data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()