# 🏥 Symptom-to-Disease Prediction Pipeline

> 🎯 A robust transformer-based NLP pipeline for mapping patient symptom descriptions to disease diagnoses, featuring modular design, extensive model comparisons, and interactive visualizations.

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#-tech-stack)
- [📊 Dataset](#-dataset)
- [🚀 Quick Start](#-quick-start)
- [📈 Model Performance](#-model-performance)
- [🔧 Project Structure](#-project-structure)
- [📖 Usage Guide](#-usage-guide)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📞 Contact & Support](#-contact--support)

---

## 🎯 Project Overview

This end-to-end pipeline leverages transformer architectures to classify diseases from free-text patient symptom narratives. It streamlines data ingestion, NLP preprocessing, model training, evaluation, and result visualization.

**Objectives:**

- Evaluate multiple pretrained transformers (BERT, RoBERTa, BioBERT, etc.) on a biomedical symptom dataset.
- Report metrics (accuracy, precision, recall, F1-score) and analyze confusion matrices.
- Provide interactive visualizations of training dynamics and comparative model performance.

---

## ✨ Key Features

- 🔄 **Modular Pipeline**: Separate components for data loading, preprocessing, model training, and evaluation.
- 🤖 **Multiple Transformer Support**: Easily plug in BERT, RoBERTa, BioBERT, ClinicalBERT, or custom checkpoints.
- 🔍 **Detailed Metrics**: Compute classification reports, confusion matrices, ROC curves, and AUC.
- 📊 **Interactive Visualizations**: Generate plots with Matplotlib and Plotly for loss/accuracy curves and ROC comparisons.
- ⚡ **Hyperparameter Tuning**: Built-in interfaces for grid search or W&B sweeps on learning rate, batch size, epochs.
- ♻️ **Reproducibility**: Configuration-driven experiments via YAML/JSON; logging with Weights & Biases.

---

## 🛠️ Tech Stack

| Category                | Tools & Libraries          |
| ----------------------- | -------------------------- |
| **Language**            | Python 3.8+                |
| **NLP Framework**       | Hugging Face Transformers  |
| **Deep Learning**       | PyTorch                    |
| **ML Utilities**        | scikit-learn               |
| **Visualization**       | Matplotlib , Plotly        |
| **Experiment Tracking** | Weights & Biases           |

---

## 📊 Dataset

- **File**: `data/symptom_disease.csv`
- **Columns**:
  - `symptoms`: Patient symptom text.
  - `disease_label`: Ground-truth disease category.
- **Preprocessing Steps**:
  1. Text cleaning: lowercasing, punctuation removal, stopword filtering
  2. Tokenization: Hugging Face tokenizer for each model
  3. Optional augmentation: synonym replacement or back-translation

---

## 🚀 Quick Start

1. **Clone repo**:
   ```bash
   git clone https://github.com/your-username/symptom-disease-prediction.git
   cd symptom-disease-prediction
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure experiments**: Edit `config/config.yaml` to select models, batch size, learning rate, epochs.
4. **Run experiment**:
   ```bash
   python src/run_experiment.py --config config/config.yaml
   ```
5. **Visualize**:
   - Launch Jupyter and open `notebooks/compare_models.ipynb`
   - View W&B dashboard for real-time logs

---

## 📈 Model Performance

| Model         | Accuracy | Precision | Recall | F1-score |
| ------------- | -------- | --------- | ------ | -------- |
| BERT-base     | 0.85     | 0.86      | 0.84   | 0.85     |
| RoBERTa-large | 0.88     | 0.89      | 0.87   | 0.88     |
| BioBERT       | 0.90     | 0.91      | 0.89   | 0.90     |
| ClinicalBERT  | 0.89     | 0.90      | 0.88   | 0.89     |

> Metrics vary by split and hyperparameters. See `notebooks/compare_models.ipynb` for detailed plots.

---

## 🔧 Project Structure

```
symptom-disease-prediction/
├── config/                 # YAML/JSON experiment configs
│   └── config.yaml
├── data/                   # Raw and processed datasets
│   └── symptom_disease.csv
├── notebooks/              # Analysis and visualization notebooks
│   └── compare_models.ipynb
├── src/                    # Source modules
│   ├── data_loader.py      # Loading & preprocessing
│   ├── model.py            # Transformer model classes
│   ├── evaluate.py         # Metrics & plotting
│   └── run_experiment.py   # Entry point
├── requirements.txt        # Dependencies
├── LICENSE.md              # MIT license
└── README.md               # This file
```

---

## 📖 Usage Guide

1. **Data Loading**: Customize `src/data_loader.py` to add new preprocessing steps.
2. **Model Configuration**: In `config/config.yaml`, list model names and hyperparameters.
3. **Training**: Execute `run_experiment.py` to train and log results.
4. **Evaluation**: Use `src/evaluate.py` or the notebook for metrics and plots.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a branch: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/YourFeature`
5. Open a PR

Refer to [CODE\_OF\_CONDUCT.md] for guidelines.

---

## 📄 License

This project is licensed under MIT. See [LICENSE.md] for details.

---

## 📞 Contact & Support

- **Email**: [alhadielaf2428@gmail.com](mailto\:alhadielaf2428@gmail.com)
- **LinkedIn**: [linkedin.com/in/alhadi-elaf](https://www.linkedin.com/in/alhadi-elaf/)

---

⭐ **If you find this project helpful, please give it a star on GitHub!**

