# Global Health Econometric Analysis & EDA

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF_Generation-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-teal)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

## Overview
This repository contains a validated data science pipeline and interactive Streamlit dashboard that generate an **IEEE-formatted Exploratory Data Analysis (EDA) report** on synthetic global macroeconomic and health indicators.

The objective is to quantify relationships between Gross Domestic Product (GDP) per capita, educational attainment, sanitation infrastructure, healthcare spending, physician density, and public health outcomes (Life Expectancy and Child Mortality).

## Features
- **Calibrated Synthetic Data:** Generates a synthetic cross-sectional dataset (`N=200` by default) with regional log-GDP distributions, realistic structural correlations, and ~5% MCAR missingness.
- **Runtime Guardrails:** Enforces Pydantic parameter bounds, DataFrame schema/domain/finiteness validation, non-zero regression variance checks, and project-root path-traversal protection (`src/guardrails.py`).
- **Econometric Modeling & Diagnostics:** Computes bivariate and multivariate OLS regressions (95% CI, RMSE, MAE, Adjusted R²), Variance Inflation Factors (VIF), Pearson vs. Spearman correlations, Z-score residual outliers (`|Z| > 2`), and Shapiro-Wilk residual normality tests.
- **Automated IEEE Reporting:** Compiles 3 statistical tables and 4 colorblind-accessible Seaborn/Matplotlib figures into a double-column IEEE PDF (`eda_research_report.pdf`), with explicit disclosure of synthetic and cross-sectional limitations.
- **Interactive Dashboard:** Streamlit UI (`app.py`) with region filtering, Preston Curve, Education vs. Child Mortality, Multivariate OLS & Residual Diagnostics, Correlation Heatmap, and PDF export.
- **Self-Contained Docker & Testing:** Non-root Docker image that pre-generates data and report on build, backed by a `pytest` suite covering pipeline, guardrails, and UI pages.

## Repository Structure
```text
global_health_eda/
├── data/                       # Generated CSV dataset (gitignored)
├── src/
│   ├── data_generator.py       # Calibrated synthetic dataset generation
│   ├── generate_eda_report.py  # Econometrics, diagnostics & IEEE PDF generation
│   └── guardrails.py           # Input, domain, and path-safety guardrails
├── tests/
│   ├── test_app.py             # Self-contained Streamlit AppTest suite
│   └── test_pipeline.py        # Pipeline, econometrics & guardrails unit tests
├── app.py                      # Interactive Streamlit dashboard
├── Dockerfile                  # Self-contained non-root container image
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Test & security audit dependencies
└── README.md                   # Project documentation
```

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/GaneshArwan/global_health_eda.git
cd global_health_eda
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
# For testing and security scanning:
pip install -r requirements-dev.txt
```

## Usage

### 1. Generate the Dataset
```bash
python src/data_generator.py
```

### 2. Compile the IEEE PDF Report
```bash
python src/generate_eda_report.py
```

### 3. Run the Test Suite
```bash
pytest --cov=src tests/
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### 5. Run with Docker
```bash
docker build -t global-health-eda .
docker run -p 8501:8501 global-health-eda
```

## Key Findings (Abstract)
Empirical estimates on the synthetic cross-sectional dataset (`N=183` complete cases) align with the Preston Curve paradigm (`R²=0.85`), showing a strong log-linear association between GDP per capita and longevity. Multivariate OLS (`Adj. R²=0.90`, `Max VIF=5.40`) indicates that educational attainment and sanitation access retain strong positive associations alongside income, while raw healthcare spending share and physician density add minimal partial explanatory power in this simulated structure.

## Author
**GaneshArwan (William)**
wiliamzwili135@gmail.com
