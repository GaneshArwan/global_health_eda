"""Synthetic global health dataset generator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.guardrails import GenerationConfig, resolve_safe_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from guardrails import GenerationConfig, resolve_safe_path

REGION_LOG_GDP_MEANS: dict[str, float] = {
    "Africa": 7.7,
    "Asia": 8.5,
    "Americas": 9.0,
    "Europe": 10.1,
    "Oceania": 9.8,
}


def generate_global_health_data(num_countries: int = 200, random_seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic global health dataset with calibrated cross-sectional distributions."""
    config = GenerationConfig(num_countries=num_countries, random_seed=random_seed)
    n = config.num_countries
    rng = np.random.default_rng(config.random_seed)

    countries = [f"Country_{i}" for i in range(1, n + 1)]
    regions = list(REGION_LOG_GDP_MEANS.keys())
    region_col = rng.choice(regions, n, p=[0.25, 0.30, 0.20, 0.20, 0.05])

    log_gdp_means = np.array([REGION_LOG_GDP_MEANS[str(r)] for r in region_col])
    gdp_per_capita = np.clip(np.exp(rng.normal(log_gdp_means, 0.85)), 500, 120000)
    log_gdp = np.log(gdp_per_capita)

    education_index = np.clip(0.06 + 0.072 * log_gdp + rng.normal(0, 0.07, n), 0.15, 0.96)
    sanitation_access = np.clip(-32 + 9.8 * log_gdp + 26 * education_index + rng.normal(0, 8.5, n), 10, 100)
    base_life_exp = 22 + 3.5 * log_gdp + 15.5 * education_index + 0.11 * sanitation_access
    life_expectancy = np.clip(base_life_exp + rng.normal(0, 2.4, n), 45, 85)
    child_mortality = np.clip(210 - 2.3 * life_expectancy - 0.32 * sanitation_access + rng.normal(0, 7.0, n), 2, 150)
    hc_spending_pct = np.clip(2.0 + 0.45 * log_gdp + rng.normal(0, 1.5, n), 1.5, 16)
    physicians_per_10k = np.clip(-11 + 3.0 * log_gdp + rng.normal(0, 4.2, n), 0.2, 50)

    df = pd.DataFrame(
        {
            "Country": countries,
            "Region": region_col,
            "GDP_Per_Capita": np.round(gdp_per_capita, 2),
            "Education_Index": np.round(education_index, 3),
            "Sanitation_Access_Pct": np.round(sanitation_access, 1),
            "Physicians_Per_10k": np.round(physicians_per_10k, 1),
            "Life_Expectancy": np.round(life_expectancy, 1),
            "Child_Mortality_Per_1000": np.round(child_mortality, 1),
            "Healthcare_Spending_Pct_GDP": np.round(hc_spending_pct, 1),
        }
    )

    # Inject ~5% MCAR missingness while guaranteeing at least one NaN when n >= 20
    hc_mask = rng.random(n) < 0.05
    san_mask = rng.random(n) < 0.05
    if n >= 20 and not (hc_mask.any() or san_mask.any()):
        hc_mask[0] = True
    df.loc[hc_mask, "Healthcare_Spending_Pct_GDP"] = np.nan
    df.loc[san_mask, "Sanitation_Access_Pct"] = np.nan

    return df


if __name__ == "__main__":
    output_path = resolve_safe_path("data/global_health_data.csv")
    os.makedirs(output_path.parent, exist_ok=True)
    generated_df = generate_global_health_data()
    generated_df.to_csv(output_path, index=False)
    print(f"Data generated successfully: {output_path}")
