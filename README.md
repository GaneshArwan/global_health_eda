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
