# Machine Learning Surrogate Models for Multi-Objective Metabolic Engineering Optimization

Bioinformatics and machine learning pipeline for generating metabolic engineering datasets from Flux Balance Analysis (FBA) simulations and training surrogate models capable of predicting biomass growth and tryptophan production.

The project evaluates multiple encoding strategies and hybrid machine learning architectures to approximate genome-scale metabolic simulations performed with the *Escherichia coli* iML1515 model.

**Author:** António Araújo  
**Institution:** Centre of Biological Engineering, University of Minho

## Pipeline

The project is organized into three main stages:

- **Stage 1 — Dataset Generation** (`Código/`,`1_fase/`): generates metabolic engineering scenarios using the iML1515 genome-scale metabolic model. Random glucose uptake rates, oxygen uptake rates and gene knockouts are sampled, followed by Flux Balance Analysis (FBA) to obtain biomass growth and tryptophan production values.

- **Stage 2 — Data Preparation** (`Código/`): converts the original dataset into alternative machine learning representations:
  - Integer Encoding
  - Multi-Hot Encoding

  These datasets are subsequently used for model training.

- **Stage 3 — Machine Learning Models** (`Código/`, `3_fase/`): trains and evaluates standalone Artificial Neural Networks (ANNs) and hybrid SVM–ANN pipelines for biomass and tryptophan prediction.

Each modelling approach generates trained models, metadata files, evaluation metrics and prediction reports.

---

## Repository Structure

```text
Projeto_PG59752/
├── README.md
├── requirements.txt
│
├── Código/
│   ├── dataset_generation.py
│   ├── dataset_original_to_integer_encoding.py
│   ├── dataset_original_to_multihot.py
│   ├── ANN_integer_encoding.py
│   ├── ANN_multihot_encoding.py
│   ├── SVM+ANN integer encoding.py
│   ├── SVM+ANN Multihot_encoding.py
│   └── SVM_multihotencoding+ANN_integer encoding.py
│
├── 1_fase/
│   ├── iML1515.xml
│   ├── dataset_knockouts_uptakes.xlsx
│   └── Artigo_intercalar_PG59752.pdf
│
├── 2_fase/
│   └── Apresentação_projeto.pdf
│
├── 3_fase/
│   ├── Integer_Encoding_results/
│   ├── Multihot_Encoding_results/
│   ├── SVM+ANN_integerEncoding_results/
│   ├── SVM+ANN_MultihotEncoding_results/
│   └── SVM_multihot+ANN_integer_results/
└── .gitignore
```

The `3_fase/` directory contains all trained models, preprocessing objects, metadata files and evaluation outputs generated during the experiments.

---

## Setup

The project uses a Python environment (3.13.7) shared across all stages.

```bash
pip install -r requirements.txt
```

---

## Quickstart

To reproduce the complete workflow from dataset generation to model evaluation:

```bash
# Dataset generation
python Código/dataset_generation.py

# Dataset preprocessing
python Código/dataset_original_to_integer_encoding.py
python Código/dataset_original_to_multihot.py

# Standalone ANNs
python Código/ANN_integer_encoding.py
python Código/ANN_multihot_encoding.py

# Hybrid models
python "Código/SVM+ANN integer encoding.py"
python "Código/SVM+ANN Multihot_encoding.py"
python "Código/SVM_multihotencoding+ANN_integer encoding.py"
```

---

## Models Evaluated

| Model | Representation |
|---------|---------|
| ANN | Integer Encoding |
| ANN | Multi-Hot Encoding |
| Hybrid SVM + ANN | Integer Encoding |
| Hybrid SVM + ANN | Multi-Hot Encoding |
| Hybrid SVM + ANN | Cross-Representation |

The hybrid architectures use a Support Vector Machine (SVM) to predict strain viability before performing regression with an Artificial Neural Network.

---

## Dataset Characteristics

| Parameter | Value |
|------------|------------|
| Organism | *Escherichia coli* |
| Metabolic Model | iML1515 |
| Genes | 1515 |
| Gene Knockouts | 1–6 per simulation |
| Glucose Uptake Range | 0.1–20 mmol/gDW/h |
| Oxygen Uptake Range | 0.1–20 mmol/gDW/h |
| Viability Threshold | 10% WT Biomass |

---

## Evaluation Metrics

Models are evaluated using:

- Coefficient of Determination (R²)
- Mean Absolute Error (MAE)
- Classification Accuracy

for both:

- Biomass growth prediction
- Tryptophan production prediction

---

## Design Decisions

- **Two encoding strategies** — Integer Encoding and Multi-Hot Encoding were evaluated to determine the most suitable representation for gene knockout information.
- **Hybrid modelling approach** — viability prediction is treated separately from flux prediction using SVM classifiers combined with ANN regressors.
- **Genome-scale simulations** — all datasets originate from Flux Balance Analysis simulations performed with the iML1515 model.
- **Multi-objective focus** — surrogate models simultaneously predict biomass growth and tryptophan production.
- **Reproducible workflow** — all preprocessing, training and evaluation steps are implemented as independent scripts.

---

## Related Publication

This repository supports the project:

**Artificial Neural Network Surrogate Models for Multi-Objective Metabolic Engineering Optimization**

developed within the MSc in Bioinformatics at the University of Minho.

---

## License

This repository is intended for academic and research purposes.
