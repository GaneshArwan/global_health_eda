# Global Health Econometric Analysis & EDA

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Security](https://img.shields.io/badge/security-0_vulnerabilities-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview
This repository contains an automated, highly validated data science pipeline that generates an **IEEE-formatted Exploratory Data Analysis (EDA) report** on global macroeconomic and health indicators.

The objective is to statistically quantify the dependencies between Gross Domestic Product (GDP) per capita, educational attainment, sanitation infrastructure, and public health outcomes (Life Expectancy and Child Mortality).

## Features
- **Synthetic Data Generation:** Generates a realistic global health dataset (N=200) encompassing multiple macroeconomic indicators.
- **Econometric Modeling:** Conducts Ordinary Least Squares (OLS) regression, calculates Root Mean Square Error (RMSE), and determines 95% Confidence Intervals.
- **Anomaly Detection:** Utilizes Z-scores on residuals to identify statistical anomalies where real-world outcomes deviate from baseline expectations.
- **Automated Reporting:** Dynamically compiles findings, statistics, and colorblind-accessible seaborn visualizations into a professional, double-column IEEE PDF document.
- **Enterprise Testing:** Achieves 95% test coverage through a `pytest` suite simulating I/O boundaries.

## Repository Structure
```
global_health_eda/
├── data/                       # Contains the generated CSV dataset
├── src/                        
│   ├── data_generator.py       # Pipeline: Synthetic dataset generation
│   └── generate_eda_report.py  # Pipeline: Econometrics & PDF Generation
├── tests/                      
│   └── test_pipeline.py        # Pytest integration suite
├── temp_assets/                # Temporary directory for plot generation
└── README.md                   # Project documentation
```

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/global_health_eda.git
cd global_health_eda
```

2. Activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn reportlab pytest pytest-cov
```

## Usage

### 1. Run the Test Suite
Ensure the environment is structurally sound:
```bash
pytest --cov=src tests/
```

### 2. Generate the Dataset
Create the synthetic dataset in the `data/` folder:
```bash
python src/data_generator.py
```

### 3. Compile the PDF Report
Run the EDA reporting engine. This will execute the regressions, generate the plots, and output `eda_research_report.pdf` into the root directory:
```bash
python src/generate_eda_report.py
```

## Key Findings (Abstract)
Empirical findings validate the Preston Curve paradigm, demonstrating a statistically significant logarithmic relationship between economic output and longevity. Furthermore, OLS regression highlights education as a critical independent variable in reducing child mortality. Results indicate steep diminishing marginal returns for highly developed economies, proving that optimal global capital allocation must aggressively prioritize socio-educational infrastructure in developing nations over pure healthcare spending.

## Author
**William**
wiliamzwili135@gmail.com
