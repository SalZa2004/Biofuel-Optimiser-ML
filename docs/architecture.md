# System Architecture

## Application Layer

The system is split into four user-facing applications, each built on a shared `core/` library.

```mermaid
graph TD
    subgraph Applications
        A1["pure_predictor\n─────────────\nPredict 6 properties\nfor a single SMILES"]
        A2["molecule_generator\n─────────────\nGenerate molecules\noptimised for CN/YSI"]
        A3["mixture_predictor\n─────────────\nPredict blend DCN\nfrom components + fractions"]
        A4["mixture_aware_generator\n─────────────\nGenerate additives\noptimised as blends"]
    end

    subgraph Core
        SF["shared_features\n(RDKit → 200+ descriptors\n+ feature selection)"]

        subgraph Predictors
            PP["PropertyPredictor\n(batches 6 models)"]
            GP["GenericPredictor\n(single model wrapper)"]
            MP["MixtureDCNPredictor\n(GNN via PyTorch)"]
        end

        subgraph Evolution
            ME["MolecularEvolution\n(pure optimisation)"]
            MAE["MixtureAwareMolecularEvolution\n(blend optimisation)"]
            POP["Population\n(NSGA-II Pareto front\n+ crowding distance)"]
            MOL["Molecule / MixtureAwareMolecule\n(dataclass + dominance check)"]
            UF["EnsembleUncertaintyFilter\n(ExtraTrees variance)"]
        end

        CFG["EvolutionConfig / MixtureConfig\n(dataclasses)"]
        DP["data_prep\n(SQLite → training DataFrame)"]
        BFL["BaseFuelLibrary\n(fossil diesel / biodiesel SMILES)"]
        BL["blending_law\n(YSI linear blending)"]
    end

    subgraph External
        RDKit["RDKit"]
        CREM["CREM\n(fragment database\ndiesel_fragments.db)"]
        HF["HuggingFace Hub\n(6 trained ExtraTrees models)"]
        DB["SQLite\ndatabase_main.db\n(1494 molecules)"]
        PT["PyTorch\n(GNN weights)"]
    end

    A1 --> SF
    A1 --> GP
    A1 --> PP

    A2 --> ME
    A4 --> MAE
    MAE --> ME

    ME --> SF
    ME --> PP
    ME --> POP
    ME --> UF
    ME --> CREM
    POP --> MOL

    MAE --> MP
    MAE --> BFL
    MAE --> BL

    A3 --> MP

    PP --> GP
    GP --> HF
    SF --> RDKit

    ME --> DP
    DP --> DB

    MP --> PT
    ME --> CFG
    MAE --> CFG
```

---

## Data Flow: Pure Component Prediction

```mermaid
flowchart LR
    IN["Input SMILES"] --> FEAT["RDKit featurisation\n200+ descriptors"]
    FEAT --> FS["Feature selection\n(VarianceThreshold\n+ correlation filter)"]
    FS --> M1["CN model\n(ExtraTrees)"]
    FS --> M2["YSI model"]
    FS --> M3["BP model"]
    FS --> M4["Density model"]
    FS --> M5["LHV model"]
    FS --> M6["Viscosity model"]
    M1 & M2 & M3 & M4 & M5 & M6 --> OUT["Property table\n+ Tanimoto similarity"]
```

---

## Data Flow: Mixture DCN Prediction

```mermaid
flowchart LR
    IN2["Component SMILES\n+ mole fractions"] --> GNN["MixtureDCNPredictor\n(GNN — PyTorch)"]
    GNN --> DCN["Blend DCN"]
    DCN --> RES["results CSV\n+ summary table"]
```
