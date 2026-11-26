"""Runtime input validation, domain bounds, and path-traversal guardrails."""

from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Country",
    "Region",
    "GDP_Per_Capita",
    "Education_Index",
    "Sanitation_Access_Pct",
    "Physicians_Per_10k",
    "Life_Expectancy",
    "Child_Mortality_Per_1000",
    "Healthcare_Spending_Pct_GDP",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "GDP_Per_Capita",
    "Education_Index",
    "Sanitation_Access_Pct",
    "Physicians_Per_10k",
    "Life_Expectancy",
    "Child_Mortality_Per_1000",
    "Healthcare_Spending_Pct_GDP",
)


class GenerationConfig(BaseModel):
    """Validated parameters for synthetic global health data generation."""

    num_countries: int = Field(default=200, ge=3, le=100_000)
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)


def resolve_safe_path(relative_path: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    """Resolve a path inside base_dir and reject path-traversal escapes."""
    base_resolved = base_dir.resolve()
    candidate = Path(relative_path)
    target = candidate.resolve() if candidate.is_absolute() else (base_resolved / candidate).resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"Path '{relative_path}' escapes project root '{base_resolved}'.") from exc
    return target


def validate_health_dataframe(df: pd.DataFrame, min_rows: int = 3) -> pd.DataFrame:
    """Validate schema, domain bounds, and variance; return cleaned copy."""
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    clean = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    if len(clean) < min_rows:
        raise ValueError(f"At least {min_rows} complete records are required; found {len(clean)}.")

    for col in NUMERIC_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean[list(NUMERIC_COLUMNS)] = clean[list(NUMERIC_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=list(NUMERIC_COLUMNS)).copy()
    if len(clean) < min_rows:
        raise ValueError(f"At least {min_rows} valid numeric records are required; found {len(clean)}.")

    checks = [
        ("GDP_Per_Capita", (clean["GDP_Per_Capita"] > 0).all(), "must be strictly positive"),
        ("Education_Index", clean["Education_Index"].between(0, 1).all(), "must be within [0, 1]"),
        ("Sanitation_Access_Pct", clean["Sanitation_Access_Pct"].between(0, 100).all(), "must be within [0, 100]"),
        ("Life_Expectancy", (clean["Life_Expectancy"] > 0).all(), "must be strictly positive"),
        ("Child_Mortality_Per_1000", (clean["Child_Mortality_Per_1000"] >= 0).all(), "must be non-negative"),
        ("Healthcare_Spending_Pct_GDP", (clean["Healthcare_Spending_Pct_GDP"] >= 0).all(), "must be non-negative"),
        ("Physicians_Per_10k", (clean["Physicians_Per_10k"] >= 0).all(), "must be non-negative"),
    ]
    for col, ok, msg in checks:
        if not bool(ok):
            raise ValueError(f"Column '{col}' {msg}.")

    if clean["GDP_Per_Capita"].nunique() < 2 or clean["Education_Index"].nunique() < 2:
        raise ValueError("Dataset lacks sufficient variance in GDP_Per_Capita or Education_Index for regression.")

    return clean
