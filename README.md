# 🎓 AI-Based EDM Performance Recognition System

An **Educational Data Mining (EDM)** system that identifies at-risk students using machine learning, provides **explainable predictions** via SHAP, enables **Human-in-the-Loop (HITL) teacher overrides**, and generates comprehensive performance reports.

Built on the **[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset)**.

---

## Login Credentials

### Teacher / Admin Portal
- **User ID**: `admin11`
- **Password**: `1122`

### 🎓 Student Portal
- **Student ID**: `28400` *(Other valid IDs: `30268`, `11391`)*
- **Password**: `1234`

---

## 💻 Quick Start & Commands guide

### 1. Prerequisite (macOS only)
If you are on macOS and running XGBoost, install OpenMP runtime to prevent system errors:
```bash
brew install libomp
```

### 2. Setup Virtual Environment & Dependencies

Open your terminal in the project root directory (`edm-system`):

```bash
# Create Virtual Environment
python3 -m venv venv

# Activate Virtual Environment
# On macOS / Linux:
source venv/bin/activate

# On Windows (Command Prompt / PowerShell):
# venv\Scripts\activate

# Install Required Dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

---

### 3. Run the ML Pipeline

To download dataset, process features, train models (XGBoost & Logistic Regression), and generate SHAP explanations:

```bash
python run_pipeline.py
```

---

### 4. Launch Applications

#### A. Flask Backend Server
Runs backend APIs (or falls back to port `5001` if port `5000` is in use):

```bash
python backend/app.py
```
> Access backend / web page at: `http://localhost:5001` (or `http://localhost:5000`)

#### B. Streamlit Multi-page Dashboard
Runs interactive Dashboard UI:

```bash
streamlit run dashboard/Student_Development.py
```
> Access dashboard at: `http://localhost:8501`

---

## 🏗️ Architecture

```
edm-system/
├── src/                    # Core ML pipeline modules
│   ├── data_loader.py      # Download & load OULAD dataset
│   ├── preprocessing.py    # Merge, clean, encode
│   ├── feature_engineering.py  # Engineer 25 ML features
│   ├── model.py            # Train LR baseline + XGBoost
│   ├── explainability.py   # SHAP explanations
│   ├── clustering.py       # KMeans personalization
│   ├── fairness.py         # Algorithmic fairness (Fairlearn)
│   └── report.py           # KPI summary & hypothesis test
├── backend/                # Flask REST API
│   └── app.py              # API server + serves frontend
├── frontend/               # Web application (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── dashboard/              # Streamlit multi-page dashboard
│   ├── Student_Development.py  # Main entry point
│   └── pages/
├── run_pipeline.py         # Master orchestration script
└── requirements.txt        # Python dependencies
```

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **At-Risk Prediction** | XGBoost classifier (93.1% accuracy, 0.979 ROC-AUC) |
| **Baseline Comparison** | Logistic Regression baseline with McNemar's hypothesis test |
| **SHAP Explainability** | Per-student waterfall explanations for transparent predictions |
| **HITL Overrides** | Teachers can confirm or override model predictions |
| **Student Clustering** | KMeans-based behavioral profiling with intervention mapping |
| **Algorithmic Fairness** | Fairlearn-powered demographic parity analysis |
| **25 Engineered Features** | Engagement, assessment, demographic, and registration features |

---

## Technology Stack

- **ML**: scikit-learn, XGBoost, SHAP, Fairlearn
- **Data**: pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Backend**: Flask, Flask-CORS
- **Dashboard**: Streamlit
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)

---

## Dataset

**Open University Learning Analytics Dataset (OULAD)**

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset.
> *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

Contains anonymized data on 32,593 student-module enrollments across 7 tables covering demographics, VLE interactions, and assessments.

---

## License

This project is developed for academic/research purposes as part of a Research-Based Learning (RBL) project.