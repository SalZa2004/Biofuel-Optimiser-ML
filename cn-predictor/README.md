
# Cetane Number Predictor ⛽

Predict cetane numbers from SMILES strings using machine learning.

## Features

- 🔬 **Single Prediction**: Enter one SMILES string to get instant cetane number prediction
- 📊 **Batch Prediction**: Upload CSV files with multiple SMILES for bulk predictions
- 📈 **Visual Analysis**: View molecular structures and prediction distributions
- 📥 **Export Results**: Download predictions as CSV

## How to Use

### Single Prediction
1. Go to the "Single Prediction" tab
2. Enter a SMILES string (e.g., `CCCCCCCCCCCCCCCC`)
3. Click "Predict"
4. View your cetane number prediction!

### Batch Prediction
1. Go to the "Batch Prediction" tab
2. Upload a CSV file with a `SMILES` column
3. Click "Predict All"
4. Download results with predictions

## Model Details

- **Input**: SMILES strings (molecular structure representation)
- **Output**: Cetane number (0-150 scale)
- **Features**: 
  - Morgan fingerprints (2048 bits, radius=2)
  - 208 RDKit molecular descriptors
  - Feature selection (top 300 important features)
- **Training**: Experimental cetane number data from fuel database

## Example SMILES

| SMILES | Name | Typical CN |
|--------|------|-----------|
| `CCCCCCCCCCCCCCCC` | Hexadecane | ~100 |
| `CC(C)CCCCC` | Isoheptane | ~30 |
| `CCCCCCCC` | Octane | ~65 |

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{cn_predictor_2025,
  author = {Your Name},
  title = {Cetane Number Predictor},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/spaces/SalZa2004/cn-predictor}
}
```

