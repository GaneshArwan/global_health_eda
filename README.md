# Global Health Econometric Analysis & EDA

> Automated econometric pipeline, IEEE two-column PDF generator, and Streamlit dashboard analyzing wealth, education, sanitation, and mortality on synthetic cross-sectional country data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GaneshArwan/global_health_eda/blob/main/notebooks/global_health_eda.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://global-health-eda.streamlit.app/)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Non--Root-2496ED?style=flat&logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## What This Project Does

This repository runs an end-to-end exploratory data analysis (EDA) workflow over 200 synthetic country records (`N=183` complete cases after listwise filtering). It estimates bivariate and multivariate ordinary least squares (OLS) models, checks multicollinearity (VIF) and residual normality (Shapiro-Wilk), compiles a two-page IEEE PDF report (`eda_research_report.pdf`), and exposes the same calculations in an interactive Streamlit app (`app.py`).

## Verified Econometric Results (`N=183`, Seed `42`)

| Model / Check | Outcome | Key Estimates |
|---|---|---|
| Preston Curve OLS (`ln(GDP)` -> `Life_Expectancy`) | Longevity (years) | `R² = 0.852`, slope `= 5.54` (`95% CI: [5.20, 5.88]`), `RMSE = 2.96`, `p = 4.54e-77` |
| Education OLS (`Education_Index` -> `Child_Mortality_Per_1000`) | Child deaths / 1k | `R² = 0.635`, slope `= -143.9` (`95% CI: [-159.9, -127.9]`), `RMSE = 12.81`, `p = 1.77e-41` |
| Multivariate OLS (5 predictors -> `Life_Expectancy`) | Longevity (years) | `R² = 0.900`, `Adj. R² = 0.897`, `RMSE = 2.44`, `Max VIF = 5.40`, `df = 177` |
| Residual Diagnostics (Preston OLS) | Model fit & tails | `9` outliers (`\|Z\| > 2`), Shapiro-Wilk `W = 0.994` (`p = 0.649`) |
| Boundary Censoring Audit | Floor / ceiling caps | `Life_Expectancy >= 85`: `7` (`3.8%`); `Child_Mortality <= 2`: `41` (`22.4%`) |

Controlling for income, education (`p = 4.38e-07`), and sanitation (`p = 3.49e-08`), raw healthcare spending share (`p = 0.811`) and physician density (`p = 0.622`) add minimal partial explanatory power in the simulated structural equations.

## Quick Start

### Live Demo & Google Colab
- **Live Dashboard:** [global-health-eda.streamlit.app](https://global-health-eda.streamlit.app/)
- **Google Colab:** [notebooks/global_health_eda.ipynb](https://colab.research.google.com/github/GaneshArwan/global_health_eda/blob/main/notebooks/global_health_eda.ipynb) — runs the full pipeline and renders all four diagnostic figures without local setup.

### Local Setup
```bash
git clone https://github.com/GaneshArwan/global_health_eda.git
cd global_health_eda
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# Unix/MacOS: source venv/bin/activate
pip install -r requirements.txt
```

### Run Pipeline & Dashboard
```bash
python src/data_generator.py        # Writes data/global_health_data.csv
python src/generate_eda_report.py   # Builds eda_research_report.pdf and plots in temp_assets/
streamlit run app.py                # Starts dashboard (auto-generates data/PDF if missing)
```

### Docker
```bash
docker build -t global-health-eda .
docker run -p 8501:8501 global-health-eda
```
The image runs as non-root `appuser`, pre-generates the dataset and PDF report during build, and exposes a Python stdlib health check on port `8501`.

### Tests & Security Checks
```bash
pip install -r requirements-dev.txt
pytest --cov=src tests/
bandit -r src app.py
pip-audit --skip-editable
```
GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) uses `workflow_dispatch` only so pushes and pull requests consume zero free-tier runner minutes.

## Repository Structure

```text
global_health_eda/
├── .github/workflows/ci.yml          # Manual-only CI workflow (pytest + bandit)
├── .streamlit/config.toml            # Headless server & telemetry-off settings
├── data/                             # Generated CSV directory (gitignored)
├── notebooks/global_health_eda.ipynb # Self-contained Google Colab notebook
├── src/
│   ├── data_generator.py             # Calibrated synthetic dataset generator
│   ├── generate_eda_report.py        # Econometric summary, plots & IEEE PDF builder
│   └── guardrails.py                 # Path-traversal, Pydantic bounds & DataFrame checks
├── tests/
│   ├── test_app.py                   # Streamlit AppTest suite (auto-seeds missing CSV)
│   └── test_pipeline.py              # Pipeline, econometrics & guardrails unit tests
├── app.py                            # Streamlit dashboard (auto-generates CSV/PDF on demand)
├── Dockerfile                        # Non-root Python 3.11 container
├── requirements.txt                  # Runtime dependencies
├── requirements-dev.txt              # Test & security audit dependencies
└── LICENSE                           # MIT License
```

## Guardrails & Limitations

- **Runtime Guardrails (`src/guardrails.py`):** Blocks path escapes outside the repository root (`resolve_safe_path`), enforces `GenerationConfig` bounds via Pydantic, strips `inf`/`-inf` and invalid numeric values, checks column domain bounds, and requires non-zero predictor variance before fitting OLS.
- **Limitations:** Indicators come from a synthetic parametric generator rather than empirical WHO/World Bank panel records; cross-sectional OLS captures statistical association rather than causation; listwise deletion assumes missingness is independent of outcomes (MCAR); and floor clipping on `Child_Mortality_Per_1000` (`22.4%`) attenuates lower-tail variance.

## Author

**GaneshArwan** — [github.com/GaneshArwan](https://github.com/GaneshArwan)
