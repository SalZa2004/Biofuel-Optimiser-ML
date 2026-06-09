# Predicting Optimal Biofuel Composition Using Machine Learning

This project aims to develop a machine learning (ML)-based model for predicting the best 
biofuel compositions tailored for certain applications and engine types. With the world turning 
towards green energy, biofuels represent an acceptable substitute for fossil fuels. However, it 
takes time and is costly to experiment to determine the best combination of bio-components 
such as ethanol, biodiesel, and other biomass-derived fuels. By applying data-driven 
approaches, the project seeks to improve the process of finding compositions that achieve 
efficiency maximisation, emissions minimisation, and maintaining engine performance. 

The system will use the past record of fuel compositions, combustion properties, and engine 
performance parameters to train supervised machine learning algorithms. The algorithm will 
learn to map certain fuel compositions to target output values (e.g. energy density, emissions 
profile, ignition delay). The aim is to create a predictive model that can suggest biofuel 
compositions for specific constraints or applications, e.g. heavy transport, air transport, power 
generation. This study has the potential to speed up greener fuel adoption and aid in 
decarbonisation efforts in different industries.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Key Components](#-key-components)
- [Installation](#-installation)
- [Usage](#-usage)

---

## Project Overview

This project develops **AI-powered tools** for designing optimal biofuel molecules that address the critical challenge of balancing multiple fuel properties:

- **Cetane Number (CN)**: Combustion quality
- **Yield Sooting Index (YSI)**: Soot formation (environmental impact)
Constraints:
- **Physical Properties**: Boiling point, Density, Lower heating value, Dynamic viscosity


## 📁 Project Structure
```
Biofuel-Optimiser-ML/
│
├── core/                              # Shared core functionality
│   ├── predictors/                    # Property prediction models
│   │   ├── pure_component/            # ML model predictor logic for pure molecules
│   │   │   ├── generic.py             # Generic predictor wrapper
│   │   │   ├── property_predictor.py  # Batch prediction with optimisation
│   │   │   └── hf_models.py           # Hugging Face model predictor paths
│   │   │
│   │   └── mixture/                  # GNN models for mixtures (future)
│   │
│   ├── evolution/                    # Genetic algorithm components
│   │   ├── molecule.py               # Molecule dataclass with fitness
│   │   ├── population.py             # Population management & Pareto fronts
│   │   └── evolution.py              # Main evolutionary algorithm
│   │
│   ├── blending/                      # Fuel blending logic (future)
│   ├── config.py                      # Configuration dataclasses
│   ├── data_prep.py                   # Data loading utilities
│   └── shared_features.py             # Molecular featurisation (RDKit descriptors)
│
├── applications/                     # User-facing applications
│   ├── pure_predictor/             # Tab 1: Predict properties of pure molecules
|   |   ├── main.py                   # Entry point
│   │   ├── cli.py                    # Command-line interface
│   │   └── results.py                # Results display & export
│   ├── mixture_predictor/          # Tab 2: Predict properties of mixtures (future work)
│   ├── molecule_generator/         # Tab 3: Generate molecules (pure optimization)
│   │   ├── main.py                   # Entry point
│   │   ├── cli.py                    # Command-line interface
│   │   └── results.py                # Results display & export
│   └── mixture_aware_generator/    # Tab 4: Generate molecules (blend optimisation) (future work)
│
├── data/                              # Data files
│   ├── database/                      # SQLite databases
│   │   └── database_main.db           # Main molecular property database
│   │
│   └── fragments/                     # CREM fragment database for molecule mutation
│       └── diesel_fragments.db        # 2000 diesel-relevant fragments
│
├── models/                            # Trained model weights
│   ├── pure_component/                # 6 ML models (CN, YSI, BP, density, LHV, viscosity)
│   │   ├── cn_predictor_model/        # Cetane Number predictor
│   │   ├── ysi_predictor_model/       # YSI predictor
│   │   ├── bp_predictor_model/        # Boiling Point predictor
│   │   ├── density_predictor_model/   # Density predictor
│   │   ├── lhv_predictor_model/       # Lower Heating Value predictor
│   │   └── dynamic_viscosity_predictor_model/ # Dynamic Viscosity predictor
│   │
│   └── mixture/                      # GNN models (future)
│
├── results/                          # Output files
│   ├── final_population.csv          # All generated molecules
│   └── pareto_front.csv              # Non-dominated solutions (CN vs YSI trade-offs)
│
├── docker/                            # Docker deployment
│   ├── Dockerfile
|   ├── .dockerignore
│   └── docker-compose.yml

├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```
---

## 🔑 Key Components Explained

### 1. **Core Module** (`core/`)

The foundation of the project containing all reusable logic.

#### **A. Predictors** (`core/predictors/`)

**Pure Component Predictors:**
- Predict 6 properties for individual molecules using ML models
- **Models**: Cetane Number, YSI, Density, Boiling Point, Dynamic Viscosity, LHV 

**Models Hosted On:**
- Hugging Face Hub (6 models)
- Auto-downloaded on first use

#### **B. Evolution Module** (`core/evolution/`)

**Genetic Algorithm Components:**

1. **`molecule.py`**: Molecule dataclass
   - Stores SMILES, properties, fitness
   - Pareto dominance checking
   - Fitness calculation (single or multi-objective)

2. **`population.py`**: Population management
   - Survivor selection (top 50%)
   - Pareto front extraction
   - Duplicate prevention

3. **`evolution.py`**: Main algorithm
   - Initialization (stratified sampling from training data)
   - Mutation (CREM-based chemical modifications)
   - Fitness evaluation (batch processing)
   - Constraint filtering

**Algorithm Flow:**
```
1. Initialize: 600 diverse molecules → Filter → 100 valid
2. Loop (6 generations):
   a. Select top 50% survivors (Pareto front + best remainder)
   b. Each survivor → 5 mutations (CREM)
   c. Batch predict properties
   d. Filter by constraints
   e. Form new population
3. Output: Final population + Pareto front
```
#### **C. Shared Features** (`core/shared_features.py`)

**Molecular Featurization:**
- Converts SMILES → 200+ RDKit molecular descriptors
- Feature selection (removes low-variance and correlated features)
- Optimised for batch processing
---

### 2. **Applications** (`applications/`)

User-facing tools that combine core components.
#### **Application 1: Pure Component Property Predictor** (Currently Implemented)

**Purpose:** Predicts all properties (Cetane Number, YSI, Boiling Point, Density, Lower Heating Value, Dynamic Viscosity) from the SMILES of the pure component molecule.

**Usage:**
```bash
python -m applications.pure_predictor.main

# Interactive prompts:
# - Single or Batch Prediction: 1 or 2
# - Input SMILES 
# - Outputs property predictions

```

#### **Application 2: Molecule Generator** (Currently Implemented)

**Purpose:** Generate molecules optimised for target cetane number (with optional YSI minimization)

**Features:**
- **Two optimization modes:**
  1. Target CN (minimize error from target)
  2. Maximize CN (find highest possible CN)
- **Multi-objective:** Optionally minimize YSI while optimizing CN
- **Constraints:** BP, density, LHV, viscosity all within fuel specifications
- **Pareto optimization:** Extract non-dominated solutions

**Usage:**
```bash
python -m applications.molecule_generator.main

# Interactive prompts:
# - Target CN: 50
# - Minimize YSI: y
# - Runs 6 generations with 100 molecules
```

**Output:**
- `results/final_population.csv`: All molecules ranked by fitness
- `results/pareto_front.csv`: Optimal CN vs YSI trade-offs

---

### 3. **Models** (`models/pure_component/`)

Six trained ML models, each in its own directory:

| Property | Model Type | R² | Test MAE | Training Samples |Test Samples |
|----------|-----------|-----|-----|-----------------|-----|
| **Cetane Number (CN)** | ExtraTreesRegressor | 0.944 | 3.82 | 973 | 244 |
| **YSI** | ExtraTreesRegressor | 0.91 | 3.1 | 838 | 210
| **Boiling Point (BP)** | ExtraTreesRegressor | 0.9795 | 7.6 °C | 602 | 151 |
| **Density** | ExtraTreesRegressor | 0.99 | 8.0 kg/m³ | 561 | 141
| **LHV** | ExtraTreesRegressor | 0.9572 | 0.5096 MJ/kg | 486 | 122
| **Dynamic Viscosity** | ExtraTreesRegressor | 0.9776 | 21 cP | 522| 130

**Each model directory contains:**
- `model.joblib`: Trained model weights 
- `feature_importances.csv`: Top features ranked
- `evaluation_plots.png`: R², residuals, feature importance plots
- `test_predictions.csv`: Held-out test set predictions

---

### 4. **Data** (`data/`)

#### **A. Database** (`data/database/`)
- `database_main.db`: SQLite database with 1494 molecules
  - Pure component properties
  - Mixture data (for future GNN training)

#### **B. Fragments** (`data/fragments/`)
- `diesel_fragments.db`: CREM database with ~2000 molecular fragments
  - Extracted from diesel compounds
  - Ensures chemically realistic mutations
  - Maintains synthesizability

---

## 🚀 Installation

### Prerequisites
- Python 3.10

### Setup
```bash
# 1. Clone repository
git clone https://github.com/SalZa2004/Biofuel-Optimiser-ML.git
cd biofuel-ml

# 2. Create environment
conda create -n biofuel python=3.10
conda activate biofuel

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install project in development mode
pip install -e .

```

## OR DOCKER Setup

### Prerequisites
- WSL
- Docker Engine



### Setup
```bash
# 1. Go to docker directory
cd docker

# 2. Build the docker image
docker compose build

# 3. Start up docker image
docker compose up -d

# 4. Execute docker
docker exec -it biofuel-ml bash
```


---

## 💻 Usage

### Quick Start: Generate Molecules
```bash
python -m applications.molecule_generator.main
```
---
## 📊 Current Status
### ✅ Completed 
1. **Pure Component Prediction**
   - ✅ 6 ML models trained and validated
   - ✅ Models deployed on Hugging Face Hub
   - ✅ Batch prediction optimized (6× faster)
   - ✅ Feature selection implemented

2. **Molecule Generator (Pure Component)**
   - ✅ Genetic algorithm with CREM mutations
   - ✅ Multi-objective optimization (CN + YSI)
   - ✅ Pareto front extraction
   - ✅ Constraint satisfaction (BP, density, LHV, viscosity)
   - ✅ Two modes: target CN & maximize CN
   - ✅ Validated on 6 generations, 100 molecules

3. **Project Structure**
   - ✅ Modular architecture (core + applications)
   - ✅ Clean separation of concerns
   - ✅ Well-documented code
   - ✅ Ready for Hugging Face deployment

### 🚧 In Progress (Next Week)

1. **Mixture Property Prediction**
   - [ ] Integrate GNN model (MolPool architecture)
   - [ ] Test on blend datasets
   - [ ] Validate accuracy vs linear blending rules

2. **Mixture-Aware Generator**
   - [ ] Implement blend simulator
   - [ ] Fitness evaluation using GNN
   - [ ] Comparison: pure vs mixture-aware optimization

3. **Documentation**
   - [ ] API reference
   - [ ] Tutorial notebooks
   - [ ] Deployment guide


