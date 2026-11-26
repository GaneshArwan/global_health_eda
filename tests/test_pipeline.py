"""Unit and integration tests for data generation, guardrails, and EDA report pipeline."""

import os
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import data_generator, generate_eda_report, guardrails


def _sample_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Country": [f"C{i}" for i in range(1, 13)],
            "Region": ["Africa", "Asia", "Americas", "Europe", "Oceania", "Africa"] * 2,
            "GDP_Per_Capita": [
                900.0, 1800.0, 3200.0, 5500.0, 9000.0, 14000.0,
                21000.0, 29000.0, 38000.0, 48000.0, 62000.0, 80000.0,
            ],
            "Education_Index": [0.32, 0.41, 0.48, 0.56, 0.63, 0.68, 0.74, 0.79, 0.84, 0.88, 0.91, 0.94],
            "Sanitation_Access_Pct": [25.0, 38.0, 52.0, 61.0, 72.0, 78.0, 86.0, 91.0, 95.0, 97.0, 99.0, 100.0],
            "Physicians_Per_10k": [1.0, 2.5, 4.2, 6.8, 10.5, 13.0, 18.0, 22.0, 27.0, 31.0, 36.0, 40.0],
            "Life_Expectancy": [52.0, 58.5, 63.0, 66.5, 70.0, 72.5, 75.5, 77.5, 79.5, 81.0, 82.0, 83.0],
            "Child_Mortality_Per_1000": [110.0, 85.0, 68.0, 54.0, 40.0, 31.0, 22.0, 16.0, 11.0, 8.0, 5.5, 4.0],
            "Healthcare_Spending_Pct_GDP": [3.2, 4.1, 4.8, 5.2, 6.0, 6.5, 7.2, 8.1, 9.0, 9.8, 10.6, 11.4],
        }
    )


def test_data_generation_shape_and_calibration():
    """Test generator produces expected schema and avoids excessive ceiling/floor pinning."""
    df = data_generator.generate_global_health_data(num_countries=200)
    assert len(df) == 200
    for col in guardrails.REQUIRED_COLUMNS:
        assert col in df.columns

    clean = guardrails.validate_health_dataframe(df)
    assert (clean["Life_Expectancy"] >= 85.0).mean() < 0.25
    assert (clean["Child_Mortality_Per_1000"] <= 2.0).mean() < 0.25


def test_data_generator_nan_injection():
    """Test that NaN values are injected into the synthetic dataset."""
    df = data_generator.generate_global_health_data(num_countries=100)
    assert df.isna().sum().sum() > 0


def test_guardrails_path_traversal_blocked():
    """Test that paths escaping the project root are rejected."""
    with pytest.raises(ValueError, match="escapes project root"):
        guardrails.resolve_safe_path("../outside_secret.txt")


def test_guardrails_config_bounds():
    """Test GenerationConfig rejects invalid country count or negative seed."""
    with pytest.raises(ValueError):
        data_generator.generate_global_health_data(num_countries=2)
    with pytest.raises(ValueError):
        data_generator.generate_global_health_data(num_countries=10, random_seed=-1)


def test_guardrails_dataframe_validation():
    """Test schema, domain bounds, row-count, inf, and variance guardrails."""
    valid = _sample_valid_df()
    cleaned = guardrails.validate_health_dataframe(valid)
    assert len(cleaned) == 12

    with pytest.raises(ValueError, match="empty"):
        guardrails.validate_health_dataframe(valid.iloc[0:0])

    with pytest.raises(ValueError, match="missing required columns"):
        guardrails.validate_health_dataframe(valid.drop(columns=["Country"]))

    bad_gdp = valid.copy()
    bad_gdp.loc[0, "GDP_Per_Capita"] = -50.0
    with pytest.raises(ValueError, match="GDP_Per_Capita"):
        guardrails.validate_health_dataframe(bad_gdp)

    too_few = valid.iloc[:2].copy()
    with pytest.raises(ValueError, match="At least 3"):
        guardrails.validate_health_dataframe(too_few)

    zero_var = valid.copy()
    zero_var["GDP_Per_Capita"] = 5000.0
    with pytest.raises(ValueError, match="sufficient variance"):
        guardrails.validate_health_dataframe(zero_var)

    inf_df = valid.copy()
    inf_df.loc[:, "GDP_Per_Capita"] = np.inf
    with pytest.raises(ValueError, match="valid numeric records"):
        guardrails.validate_health_dataframe(inf_df)


def test_get_95_ci_guard():
    """Test get_95_ci rejects n_obs <= 2 and computes symmetric bounds otherwise."""
    with pytest.raises(ValueError, match="At least 3"):
        generate_eda_report.get_95_ci(1.0, 0.2, 2)
    low, high = generate_eda_report.get_95_ci(2.0, 0.5, 10)
    assert low < 2.0 < high


def test_compute_econometric_summary():
    """Test econometric summary returns valid descriptive, OLS, VIF, and Shapiro diagnostics."""
    df = _sample_valid_df()
    stats = generate_eda_report.compute_econometric_summary(df, initial_rows=12)
    assert stats["n_obs"] == 12
    assert 0.9 < stats["model1"]["r2"] <= 1.0
    assert stats["model3"] is not None
    assert len(stats["model3"]["predictors"]) == 5
    assert all(p["vif"] >= 1.0 for p in stats["model3"]["predictors"])
    assert 0.0 <= stats["model1"]["shapiro_p"] <= 1.0

    # Small sample (n=4) should skip multivariate SEs cleanly without error
    tiny_stats = generate_eda_report.compute_econometric_summary(df.iloc[:4].copy(), initial_rows=4)
    assert tiny_stats["model3"] is None


@patch("matplotlib.pyplot.savefig")
def test_generate_visualizations(mock_savefig):
    """Test all 4 visualizations are saved and figures are closed cleanly."""
    df = _sample_valid_df()
    preston, edu, heat, resid, corr = generate_eda_report.generate_visualizations(df)

    assert preston == "temp_assets/preston_curve.png"
    assert edu == "temp_assets/edu_mort.png"
    assert heat == "temp_assets/heatmap.png"
    assert resid == "temp_assets/residuals_diagnostics.png"
    assert corr is not None
    assert mock_savefig.call_count == 4
    assert len(plt.get_fignums()) == 0


@patch("src.generate_eda_report.BaseDocTemplate.build")
@patch("src.generate_eda_report.generate_visualizations")
@patch("pandas.read_csv")
def test_generate_eda_report_dynamic_text(mock_read_csv, mock_gen_vis, mock_build):
    """Test full report pipeline uses dynamic R-squared, IEEE tables, and clean prose."""
    mock_df = _sample_valid_df()
    mock_read_csv.return_value = mock_df
    mock_gen_vis.return_value = (
        "temp_assets/preston_curve.png",
        "temp_assets/edu_mort.png",
        "temp_assets/heatmap.png",
        "temp_assets/residuals_diagnostics.png",
        mock_df.select_dtypes(include=[float, int]).corr(),
    )

    generate_eda_report.generate_eda_report()

    assert mock_build.call_count == 1
    story = mock_build.call_args[0][0]
    rendered_text = " ".join(getattr(flowable, "text", "") for flowable in story)
    assert "Analytics Reporter" not in rendered_text
    assert "Persona" not in rendered_text
    assert "R²=0.81" not in rendered_text
    assert "R²=0.99" in rendered_text
    assert "TABLE III. MULTIVARIATE OLS FOR LIFE EXPECTANCY" in rendered_text


@patch("pandas.read_csv", side_effect=FileNotFoundError("missing.csv"))
def test_generate_eda_report_raises_on_missing_csv(mock_read_csv):
    """Test report generator propagates missing CSV errors instead of exiting 0."""
    with pytest.raises(FileNotFoundError):
        generate_eda_report.generate_eda_report()
