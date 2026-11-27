"""Econometric EDA pipeline and IEEE-formatted PDF report generator."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import linregress, pearsonr, shapiro, spearmanr, t, zscore
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from src.guardrails import NUMERIC_COLUMNS, resolve_safe_path, validate_health_dataframe
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from guardrails import NUMERIC_COLUMNS, resolve_safe_path, validate_health_dataframe

SHORT_LABELS: dict[str, str] = {
    "GDP_Per_Capita": "GDP/Cap ($)",
    "Education_Index": "Education Idx",
    "Sanitation_Access_Pct": "Sanitation (%)",
    "Physicians_Per_10k": "Physicians/10k",
    "Life_Expectancy": "Life Exp (yr)",
    "Child_Mortality_Per_1000": "Child Mort/1k",
    "Healthcare_Spending_Pct_GDP": "Health (% GDP)",
}

MULTI_PREDICTORS: list[tuple[str, str]] = [
    ("log_gdp", "ln(GDP/Cap)"),
    ("Education_Index", "Education Idx"),
    ("Sanitation_Access_Pct", "Sanitation (%)"),
    ("Healthcare_Spending_Pct_GDP", "Health Spend (%)"),
    ("Physicians_Per_10k", "Physicians/10k"),
]


def get_95_ci(slope: float, std_err: float, n_obs: int) -> tuple[float, float]:
    """Calculate the 95% confidence interval for an OLS regression slope."""
    if n_obs <= 2:
        raise ValueError("At least 3 observations required for 95% CI.")
    t_val = t.ppf(0.975, df=n_obs - 2)
    margin = t_val * std_err
    return slope - margin, slope + margin


def _fit_multivariate_ols(df_clean: pd.DataFrame) -> dict[str, Any] | None:
    """Fit multivariate OLS for Life_Expectancy and compute SEs, p-values, and VIF."""
    work = df_clean.copy()
    work["log_gdp"] = np.log(work["GDP_Per_Capita"])
    cols = [col for col, _ in MULTI_PREDICTORS]
    labels = [label for _, label in MULTI_PREDICTORS]
    n_obs = len(work)
    k_params = len(cols) + 1
    if n_obs <= k_params:
        return None

    x_mat = np.column_stack([np.ones(n_obs)] + [work[c].to_numpy(dtype=float) for c in cols])
    y_vec = work["Life_Expectancy"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    y_hat = x_mat @ beta
    residuals = y_vec - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - k_params)
    rmse = float(np.sqrt(mean_squared_error(y_vec, y_hat)))
    mae = float(mean_absolute_error(y_vec, y_hat))

    dof = n_obs - k_params
    s2_err = ss_res / dof
    xtx = x_mat.T @ x_mat
    cov = s2_err * (np.linalg.inv(xtx) if np.linalg.cond(xtx) < 1e12 else np.linalg.pinv(xtx))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_vals = 2.0 * (1.0 - t.cdf(np.abs(t_stats), df=dof))

    # Variance Inflation Factors for each predictor
    vifs: list[float] = []
    for idx, col in enumerate(cols):
        other_cols = [c for j, c in enumerate(cols) if j != idx]
        x_other = np.column_stack([np.ones(n_obs)] + [work[c].to_numpy(dtype=float) for c in other_cols])
        y_col = work[col].to_numpy(dtype=float)
        b_j, *_ = np.linalg.lstsq(x_other, y_col, rcond=None)
        ss_res_j = float(np.sum((y_col - x_other @ b_j) ** 2))
        ss_tot_j = float(np.sum((y_col - y_col.mean()) ** 2))
        r2_j = 1.0 - (ss_res_j / ss_tot_j) if ss_tot_j > 0 else 1.0
        vif = float(1.0 / (1.0 - r2_j)) if r2_j < 1.0 - 1e-12 else float("inf")
        vifs.append(vif)

    rows = []
    for name, b_val, se_val, p_val, vif_val in zip(labels, beta[1:], se[1:], p_vals[1:], vifs):
        rows.append(
            {
                "predictor": name,
                "beta": float(b_val),
                "se": float(se_val),
                "p_val": float(p_val),
                "vif": float(vif_val),
            }
        )

    return {
        "intercept": float(beta[0]),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": rmse,
        "mae": mae,
        "dof": dof,
        "predictors": rows,
        "max_vif": max(vifs),
    }


def compute_econometric_summary(
    df_clean: pd.DataFrame,
    initial_rows: int,
    missing_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute descriptive statistics, regional summaries, OLS models, and diagnostics."""
    n_obs = len(df_clean)
    retention_rate = (n_obs / initial_rows) * 100.0 if initial_rows > 0 else 0.0

    # Descriptive statistics
    desc_rows = []
    for col in NUMERIC_COLUMNS:
        series = df_clean[col]
        q75, q25 = np.percentile(series, [75, 25])
        desc_rows.append(
            {
                "column": col,
                "label": SHORT_LABELS[col],
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "median": float(series.median()),
                "iqr": float(q75 - q25),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    # Boundary censoring audit
    censoring = {
        "life_exp_ceiling": int((df_clean["Life_Expectancy"] >= 85.0).sum()),
        "child_mort_floor": int((df_clean["Child_Mortality_Per_1000"] <= 2.0).sum()),
        "sanitation_ceiling": int((df_clean["Sanitation_Access_Pct"] >= 100.0).sum()),
    }

    # Regional breakdown
    region_rows = []
    for region, grp in df_clean.groupby("Region"):
        region_rows.append(
            {
                "region": str(region),
                "n": int(len(grp)),
                "median_gdp": float(grp["GDP_Per_Capita"].median()),
                "mean_life": float(grp["Life_Expectancy"].mean()),
                "mean_mort": float(grp["Child_Mortality_Per_1000"].mean()),
                "mean_san": float(grp["Sanitation_Access_Pct"].mean()),
            }
        )
    region_rows.sort(key=lambda r: r["mean_life"], reverse=True)

    # Model 1: Preston Curve (ln(GDP) -> Life_Expectancy)
    log_gdp = np.log(df_clean["GDP_Per_Capita"].to_numpy(dtype=float))
    life_exp = df_clean["Life_Expectancy"].to_numpy(dtype=float)
    slope_gdp, int_gdp, r_gdp, p_gdp, err_gdp = linregress(log_gdp, life_exp)
    r2_gdp = float(r_gdp**2)
    ci_gdp_low, ci_gdp_high = get_95_ci(float(slope_gdp), float(err_gdp), n_obs)
    fitted_gdp = int_gdp + slope_gdp * log_gdp
    residuals_gdp = life_exp - fitted_gdp
    z_gdp = np.zeros_like(residuals_gdp) if np.std(residuals_gdp) == 0 else zscore(residuals_gdp)
    rmse_gdp = float(np.sqrt(mean_squared_error(life_exp, fitted_gdp)))
    mae_gdp = float(mean_absolute_error(life_exp, fitted_gdp))
    shapiro_w, shapiro_p = shapiro(residuals_gdp)

    outlier_mask = np.abs(z_gdp) > 2.0
    outlier_df = df_clean.loc[outlier_mask, ["Country", "Region", "GDP_Per_Capita", "Life_Expectancy"]].copy()
    outlier_df["Z"] = z_gdp[outlier_mask]
    outlier_df["abs_Z"] = np.abs(outlier_df["Z"])
    outlier_df = outlier_df.sort_values("abs_Z", ascending=False)
    top_outliers = [
        f"{row.Country} ({row.Region}, Z={row.Z:+.2f})" for row in outlier_df.head(3).itertuples(index=False)
    ]

    # Model 2: Education_Index -> Child_Mortality_Per_1000
    edu = df_clean["Education_Index"].to_numpy(dtype=float)
    mort = df_clean["Child_Mortality_Per_1000"].to_numpy(dtype=float)
    slope_edu, int_edu, r_edu, p_edu, err_edu = linregress(edu, mort)
    r2_edu = float(r_edu**2)
    ci_edu_low, ci_edu_high = get_95_ci(float(slope_edu), float(err_edu), n_obs)
    fitted_edu = int_edu + slope_edu * edu
    rmse_edu = float(np.sqrt(mean_squared_error(mort, fitted_edu)))
    mae_edu = float(mean_absolute_error(mort, fitted_edu))

    # Correlations (Pearson vs Spearman)
    corr_matrix = df_clean.select_dtypes(include=[np.number]).corr()
    rank_corrs: dict[str, dict[str, float]] = {}
    for col in ("GDP_Per_Capita", "Education_Index", "Sanitation_Access_Pct", "Healthcare_Spending_Pct_GDP", "Physicians_Per_10k"):
        pr, pp = pearsonr(df_clean[col], df_clean["Life_Expectancy"])
        sr, sp = spearmanr(df_clean[col], df_clean["Life_Expectancy"])
        rank_corrs[col] = {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        }

    return {
        "initial_rows": initial_rows,
        "n_obs": n_obs,
        "retention_rate": retention_rate,
        "missing_counts": missing_counts or {},
        "censoring": censoring,
        "descriptive": desc_rows,
        "regional": region_rows,
        "model1": {
            "slope": float(slope_gdp),
            "intercept": float(int_gdp),
            "r2": r2_gdp,
            "p": float(p_gdp),
            "ci_low": float(ci_gdp_low),
            "ci_high": float(ci_gdp_high),
            "rmse": rmse_gdp,
            "mae": mae_gdp,
            "fitted": fitted_gdp,
            "residuals": residuals_gdp,
            "z_scores": z_gdp,
            "outliers_count": int(outlier_mask.sum()),
            "top_outliers": top_outliers,
            "shapiro_w": float(shapiro_w),
            "shapiro_p": float(shapiro_p),
        },
        "model2": {
            "slope": float(slope_edu),
            "intercept": float(int_edu),
            "r2": r2_edu,
            "p": float(p_edu),
            "ci_low": float(ci_edu_low),
            "ci_high": float(ci_edu_high),
            "rmse": rmse_edu,
            "mae": mae_edu,
        },
        "model3": _fit_multivariate_ols(df_clean),
        "corr_matrix": corr_matrix,
        "rank_corrs": rank_corrs,
    }


def generate_visualizations(
    df: pd.DataFrame, stats: dict[str, Any] | None = None
) -> tuple[str, str, str, str, pd.DataFrame]:
    """Generate and save 4 diagnostic EDA plots as PNGs for the PDF report."""
    assets_dir = resolve_safe_path("temp_assets")
    os.makedirs(assets_dir, exist_ok=True)
    sns.set_palette("colorblind")

    if stats is None:
        log_gdp = np.log(df["GDP_Per_Capita"].to_numpy(dtype=float))
        life_exp = df["Life_Expectancy"].to_numpy(dtype=float)
        slope_g, int_g, _, _, _ = linregress(log_gdp, life_exp)
        fitted = int_g + slope_g * log_gdp
        residuals = life_exp - fitted
    else:
        fitted = stats["model1"]["fitted"]
        residuals = stats["model1"]["residuals"]

    # 1. Preston Curve (Log GDP vs Life Expectancy)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="GDP_Per_Capita", y="Life_Expectancy", hue="Region", alpha=0.8, s=40, ax=ax)
    sns.regplot(
        data=df,
        x="GDP_Per_Capita",
        y="Life_Expectancy",
        scatter=False,
        logx=True,
        color="black",
        line_kws={"linestyle": "--"},
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title("Preston Curve: Life Expectancy vs GDP (Log) with 95% CI", fontsize=10)
    ax.set_xlabel("GDP Per Capita (USD, Log Scale)", fontsize=9)
    ax.set_ylabel("Life Expectancy (Years)", fontsize=9)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    preston_path = "temp_assets/preston_curve.png"
    plt.savefig(resolve_safe_path(preston_path), dpi=300)
    plt.close(fig)

    # 2. Education vs Child Mortality
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(
        data=df,
        x="Education_Index",
        y="Child_Mortality_Per_1000",
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "#d95f02"},
        ax=ax,
    )
    ax.set_title("Educational Attainment vs Child Mortality (with 95% CI)", fontsize=10)
    ax.set_xlabel("Education Index (0 to 1)", fontsize=9)
    ax.set_ylabel("Child Mortality (per 1,000 live births)", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    edu_mort_path = "temp_assets/edu_mort.png"
    plt.savefig(resolve_safe_path(edu_mort_path), dpi=300)
    plt.close(fig)

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="vlag", fmt=".2f", vmin=-1, vmax=1, annot_kws={"size": 6}, ax=ax)
    ax.set_title("Multivariate Pearson Correlation Matrix", fontsize=10)
    ax.tick_params(axis="x", labelsize=6.5, labelrotation=45)
    ax.tick_params(axis="y", labelsize=6.5)
    plt.tight_layout()
    heatmap_path = "temp_assets/heatmap.png"
    plt.savefig(resolve_safe_path(heatmap_path), dpi=300)
    plt.close(fig)

    # 4. Preston Curve Residual Diagnostics (Residuals vs Fitted + Distribution)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    res_std = float(np.std(residuals))
    ax1.scatter(fitted, residuals, alpha=0.7, s=22, color="#0072B2")
    ax1.axhline(0, color="black", linewidth=1)
    if res_std > 0:
        ax1.axhline(2 * res_std, color="#D55E00", linestyle="--", linewidth=0.9, label="+/-2 SD")
        ax1.axhline(-2 * res_std, color="#D55E00", linestyle="--", linewidth=0.9)
        ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("Residuals vs Fitted (Preston OLS)", fontsize=8.5)
    ax1.set_xlabel("Fitted Life Expectancy (yr)", fontsize=8)
    ax1.set_ylabel("Residual (yr)", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=14, color="#009E73", alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_title("Residual Distribution", fontsize=8.5)
    ax2.set_xlabel("Residual (yr)", fontsize=8)
    ax2.set_ylabel("Count", fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    resid_path = "temp_assets/residuals_diagnostics.png"
    plt.savefig(resolve_safe_path(resid_path), dpi=300)
    plt.close(fig)

    return preston_path, edu_mort_path, heatmap_path, resid_path, corr


def _build_ieee_table(data: list[list[str]], col_widths: list[float]) -> Table:
    """Create a compact IEEE-styled ReportLab table within column width."""
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("LEADING", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def generate_eda_report() -> None:
    """Load validated health data, compute econometric diagnostics, and build the IEEE PDF."""
    csv_path = resolve_safe_path("data/global_health_data.csv")
    df = pd.read_csv(csv_path)

    initial_rows = len(df)
    missing_counts = {str(col): int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}
    df_clean = validate_health_dataframe(df, min_rows=3)
    stats = compute_econometric_summary(df_clean, initial_rows, missing_counts)

    preston_path, edu_mort_path, heatmap_path, resid_path, corr_matrix = generate_visualizations(df_clean, stats)
    m1 = stats["model1"]
    m2 = stats["model2"]
    m3 = stats["model3"]
    cens = stats["censoring"]
    n_obs = stats["n_obs"]

    corr_hc_life = float(corr_matrix.loc["Healthcare_Spending_Pct_GDP", "Life_Expectancy"])
    corr_san_life = float(corr_matrix.loc["Sanitation_Access_Pct", "Life_Expectancy"])
    gdp_rc = stats["rank_corrs"]["GDP_Per_Capita"]

    # Document Layout - IEEE Two Column Format
    output_pdf = str(resolve_safe_path("eda_research_report.pdf"))
    doc = BaseDocTemplate(
        output_pdf,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    title_frame = Frame(
        doc.leftMargin,
        doc.height + doc.bottomMargin - 1.85 * inch,
        doc.width,
        1.85 * inch,
        id="title",
        showBoundary=0,
    )
    col1_first = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width / 2 - 0.15 * inch,
        doc.height - 1.85 * inch,
        id="col1_first",
        showBoundary=0,
    )
    col2_first = Frame(
        doc.leftMargin + doc.width / 2 + 0.15 * inch,
        doc.bottomMargin,
        doc.width / 2 - 0.15 * inch,
        doc.height - 1.85 * inch,
        id="col2_first",
        showBoundary=0,
    )
    first_page = PageTemplate(id="FirstPage", frames=[title_frame, col1_first, col2_first])

    col1 = Frame(doc.leftMargin, doc.bottomMargin, doc.width / 2 - 0.15 * inch, doc.height, id="col1", showBoundary=0)
    col2 = Frame(
        doc.leftMargin + doc.width / 2 + 0.15 * inch,
        doc.bottomMargin,
        doc.width / 2 - 0.15 * inch,
        doc.height,
        id="col2",
        showBoundary=0,
    )
    later_pages = PageTemplate(id="LaterPages", frames=[col1, col2])
    doc.addPageTemplates([first_page, later_pages])

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Normal"], fontName="Times-Roman", fontSize=20, spaceAfter=12, leading=24, alignment=1
    )
    author_style = ParagraphStyle(
        "AuthorStyle", parent=styles["Normal"], fontName="Times-Roman", fontSize=10.5, spaceAfter=0, alignment=1, leading=13
    )
    abstract_style = ParagraphStyle(
        "AbstractStyle", parent=styles["Normal"], fontName="Times-Bold", fontSize=8.5, spaceAfter=8, alignment=4, leading=10.5
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Normal"], fontName="Times-Roman", fontSize=10, spaceAfter=6, spaceBefore=10, alignment=1
    )
    normal_style = ParagraphStyle(
        "NormalStyle", parent=styles["Normal"], fontName="Times-Roman", fontSize=9.5, spaceAfter=5, leading=11.5, alignment=4
    )
    ref_style = ParagraphStyle(
        "RefStyle", parent=styles["Normal"], fontName="Times-Roman", fontSize=8, spaceAfter=4, leading=10, alignment=4
    )
    caption_style = ParagraphStyle(
        "CaptionStyle", parent=styles["Normal"], fontName="Times-Roman", fontSize=8, spaceAfter=6, alignment=1, textColor=colors.black
    )
    subheading_style = ParagraphStyle("Subheading", parent=normal_style, fontName="Times-Italic")

    missing_summary = (
        ", ".join(f"{SHORT_LABELS.get(k, k)}: {v}" for k, v in stats["missing_counts"].items())
        if stats["missing_counts"]
        else "None"
    )
    multi_abstract = f", and multivariate OLS achieves Adj. R²={m3['adj_r2']:.2f}" if m3 else ""

    # Table I: Descriptive Statistics (total width = 3.20 in)
    desc_table_data = [["Feature", "Mean (SD)", "Med (IQR)", "Range"]]
    for row in stats["descriptive"]:
        col = row["column"]
        if col == "GDP_Per_Capita":
            mean_sd = f"${row['mean']:,.0f} ({row['std']:,.0f})"
            med_iqr = f"${row['median']:,.0f} ({row['iqr']:,.0f})"
            rng_txt = f"{row['min']:,.0f}-{row['max']:,.0f}"
        elif col == "Education_Index":
            mean_sd = f"{row['mean']:.2f} ({row['std']:.2f})"
            med_iqr = f"{row['median']:.2f} ({row['iqr']:.2f})"
            rng_txt = f"{row['min']:.2f}-{row['max']:.2f}"
        else:
            mean_sd = f"{row['mean']:.1f} ({row['std']:.1f})"
            med_iqr = f"{row['median']:.1f} ({row['iqr']:.1f})"
            rng_txt = f"{row['min']:.1f}-{row['max']:.1f}"
        desc_table_data.append([row["label"], mean_sd, med_iqr, rng_txt])
    table_desc = _build_ieee_table(desc_table_data, [0.92 * inch, 0.79 * inch, 0.79 * inch, 0.65 * inch])

    # Table II: Regional Breakdown (total width = 3.15 in)
    reg_table_data = [["Region", "n", "Med GDP", "Life Exp", "Mort/1k", "San (%)"]]
    for r in stats["regional"]:
        reg_table_data.append(
            [
                r["region"],
                str(r["n"]),
                f"${r['median_gdp']:,.0f}",
                f"{r['mean_life']:.1f}",
                f"{r['mean_mort']:.1f}",
                f"{r['mean_san']:.1f}",
            ]
        )
    table_reg = _build_ieee_table(
        reg_table_data, [0.66 * inch, 0.27 * inch, 0.66 * inch, 0.52 * inch, 0.52 * inch, 0.52 * inch]
    )

    if abs(corr_san_life) >= abs(corr_hc_life):
        corr_comparison = (
            f"Sanitation Access exhibits a stronger linear correlation with Life Expectancy (r={corr_san_life:.2f}) "
            f"than Healthcare Spending as % of GDP (r={corr_hc_life:.2f})."
        )
    else:
        corr_comparison = (
            f"Healthcare Spending as % of GDP exhibits a stronger linear correlation with Life Expectancy "
            f"(r={corr_hc_life:.2f}) than Sanitation Access (r={corr_san_life:.2f})."
        )

    outliers_note = (
        f" Notable high-|Z| cases include: {', '.join(m1['top_outliers'])}."
        if m1["top_outliers"]
        else ""
    )

    story: list[Any] = [
        Paragraph(
            "Empirical Assessment of Macroeconomic and<br/>Educational Determinants on Global Health Outcomes",
            title_style,
        ),
        Paragraph("GaneshArwan<br/>github.com/GaneshArwan", author_style),
        FrameBreak(),
        NextPageTemplate("LaterPages"),
        Paragraph(
            f"<b><i>Abstract</i>—This study presents an econometric exploratory data analysis (EDA) of global health "
            f"indicators on a synthetic cross-sectional dataset (N={n_obs} complete cases from {initial_rows} generated "
            f"records; retention {stats['retention_rate']:.1f}%). Using bivariate and multivariate ordinary least squares "
            f"(OLS) regression, we quantify associations between GDP per capita, educational attainment, sanitation access, "
            f"healthcare spending, physician density, and mortality outcomes. The log-linear Preston Curve captures "
            f"R²={m1['r2']:.2f} of life expectancy variance, education explains R²={m2['r2']:.2f} of child mortality "
            f"variance{multi_abstract}. Diagnostic checks highlight boundary censoring, multicollinearity (VIF), and "
            f"residual structure, underscoring why socio-educational covariates complement income growth.</b>",
            abstract_style,
        ),
        Paragraph(
            "<b><i>Index Terms</i>—Econometrics, Preston Curve, OLS Regression, Multicollinearity, Residual Diagnostics.</b>",
            abstract_style,
        ),
        Paragraph("I. INTRODUCTION", heading_style),
        Paragraph(
            "The positive, concave association between national income per capita and longevity—known as the Preston "
            "Curve—is a foundational empirical regularity in development economics [1]. Subsequent scholarship argues that "
            "human capital and public infrastructure, particularly education and sanitation, mediate much of the mortality "
            "reduction attributed to wealth alone [2].",
            normal_style,
        ),
        Paragraph(
            "This report evaluates a synthetic cross-sectional dataset modeled after World Health Organization (WHO) "
            "macro-health indicators [3]. Rather than reporting only headline R² values, we present full descriptive "
            "distributions, regional stratification, multivariate partial slopes with variance inflation factors (VIF), "
            "residual normality tests, and methodological limitations.",
            normal_style,
        ),
        Paragraph("II. DATA QUALITY & DESCRIPTIVE PROFILE", heading_style),
        Paragraph(
            f"The raw table contained {initial_rows} synthetic country records. Missing values occurred in [{missing_summary}]. "
            f"Applying schema, domain-bound, and complete-case filtering retained N={n_obs} records ({stats['retention_rate']:.1f}%). "
            f"Boundary audit found {cens['life_exp_ceiling']} records at the 85.0-year life expectancy ceiling, "
            f"{cens['child_mort_floor']} at the 2.0/1,000 child mortality floor, and {cens['sanitation_ceiling']} at 100% sanitation.",
            normal_style,
        ),
        Paragraph("TABLE I. DESCRIPTIVE STATISTICS (N=" + str(n_obs) + ")", caption_style),
        table_desc,
        Spacer(1, 6),
        Paragraph("TABLE II. REGIONAL STRATIFICATION (MEDIAN GDP & MEANS)", caption_style),
        table_reg,
        Spacer(1, 6),
        Paragraph("III. EMPIRICAL RESULTS & DIAGNOSTICS", heading_style),
        Paragraph("A. Preston Curve: Log-Linear Income vs. Longevity", subheading_style),
        Image(str(resolve_safe_path(preston_path)), width=3.15 * inch, height=2.07 * inch),
        Paragraph("Fig. 1. Preston Curve: Life Expectancy vs. log(GDP per capita) with 95% CI.", caption_style),
        Paragraph(
            f"As shown in Fig. 1, OLS regression of Life Expectancy on ln(GDP) yields R²={m1['r2']:.3f} "
            f"(p={m1['p']:.2e}), RMSE={m1['rmse']:.2f} years, and MAE={m1['mae']:.2f} years. The slope is "
            f"{m1['slope']:.2f} years per unit ln(GDP) (95% CI: [{m1['ci_low']:.2f}, {m1['ci_high']:.2f}]), meaning "
            f"doubling GDP per capita is associated with +{m1['slope'] * np.log(2):.2f} years of life expectancy. "
            f"Because raw GDP is right-skewed, linear Pearson r={gdp_rc['pearson_r']:.2f} understates the relationship "
            f"relative to rank-based Spearman rho={gdp_rc['spearman_rho']:.2f}.",
            normal_style,
        ),
        Paragraph("B. Education & Child Mortality", subheading_style),
        Image(str(resolve_safe_path(edu_mort_path)), width=3.15 * inch, height=2.07 * inch),
        Paragraph("Fig. 2. OLS Regression of Education Index vs. Child Mortality (95% CI).", caption_style),
        Paragraph(
            f"In Fig. 2, Education Index explains R²={m2['r2']:.3f} of Child Mortality variance (p={m2['p']:.2e}, "
            f"RMSE={m2['rmse']:.2f}, MAE={m2['mae']:.2f} deaths/1,000). The slope is {m2['slope']:.1f} "
            f"(95% CI: [{m2['ci_low']:.1f}, {m2['ci_high']:.1f}]), corresponding to {abs(m2['slope']) * 0.1:.2f} "
            f"fewer deaths per 1,000 live births for each +0.10 increase in Education Index.",
            normal_style,
        ),
    ]

    # Section III.C: Multivariate OLS + Table III
    story.append(Paragraph("C. Multivariate OLS & Multicollinearity (VIF)", subheading_style))
    if m3 is not None:
        multi_table_data = [["Predictor", "Beta", "SE", "p-val", "VIF"]]
        for p_row in m3["predictors"]:
            p_str = f"{p_row['p_val']:.1e}" if p_row["p_val"] < 0.001 else f"{p_row['p_val']:.3f}"
            vif_str = "inf" if not np.isfinite(p_row["vif"]) else f"{p_row['vif']:.2f}"
            multi_table_data.append(
                [
                    p_row["predictor"],
                    f"{p_row['beta']:+.3f}",
                    f"{p_row['se']:.3f}",
                    p_str,
                    vif_str,
                ]
            )
        table_multi = _build_ieee_table(
            multi_table_data, [1.08 * inch, 0.51 * inch, 0.47 * inch, 0.59 * inch, 0.50 * inch]
        )
        story.append(Paragraph("TABLE III. MULTIVARIATE OLS FOR LIFE EXPECTANCY", caption_style))
        story.append(table_multi)
        story.append(Spacer(1, 5))
        story.append(
            Paragraph(
                f"Controlling jointly for all five covariates (intercept={m3['intercept']:.2f}) increases explained "
                f"variance to R²={m3['r2']:.3f} (Adj. R²={m3['adj_r2']:.3f}, RMSE={m3['rmse']:.2f} yr, "
                f"MAE={m3['mae']:.2f} yr, df={m3['dof']}). Maximum VIF across predictors is {m3['max_vif']:.2f}, "
                f"quantifying shared variance among wealth, education, sanitation, and physician density.",
                normal_style,
            )
        )
    else:
        story.append(
            Paragraph(
                f"Sample size (N={n_obs}) is insufficient for full 6-parameter multivariate inference.",
                normal_style,
            )
        )

    # Section III.D: Correlation & Residual Diagnostics
    story.extend(
        [
            Paragraph("D. Correlation Structure & Residual Diagnostics", subheading_style),
            Image(str(resolve_safe_path(heatmap_path)), width=3.15 * inch, height=2.46 * inch),
            Paragraph("Fig. 3. Multivariate Pearson correlation matrix.", caption_style),
            Paragraph(corr_comparison, normal_style),
            Image(str(resolve_safe_path(resid_path)), width=3.15 * inch, height=1.58 * inch),
            Paragraph("Fig. 4. Preston Curve residual spread and histogram.", caption_style),
            Paragraph(
                f"Residual diagnostics in Fig. 4 show {m1['outliers_count']} records with |Z| > 2.0.{outliers_note} "
                f"A Shapiro-Wilk test on Preston OLS residuals yields W={m1['shapiro_w']:.3f} (p={m1['shapiro_p']:.3f}), "
                f"{'consistent with' if m1['shapiro_p'] >= 0.05 else 'indicating departure from'} normality at alpha=0.05.",
                normal_style,
            ),
            Paragraph("IV. DISCUSSION, LIMITATIONS & KPIs", heading_style),
            Paragraph(
                "<b>Methodological Limitations:</b> (1) <i>Synthetic Data:</i> indicators are simulated from parametric "
                "structural equations and cannot substitute for empirical WHO/World Bank panel data. (2) <i>Cross-Sectional "
                "Design:</i> OLS estimates measure statistical association, not causal treatment effects, and omit "
                "unobserved confounders (e.g., governance, conflict, disease burden). (3) <i>Complete-Case Filtering:</i> "
                "listwise deletion assumes missingness is independent of outcomes (MCAR). (4) <i>Boundary Clipping:</i> "
                "upper/lower caps attenuate tails and can introduce heteroscedasticity.",
                normal_style,
            ),
            Paragraph(
                "<b>Analytical Takeaways & Monitoring KPIs:</b> Within the model, sanitation and education retain "
                "strong associations alongside income. Policy dashboards tracking cross-country progress should monitor "
                "three empirical KPIs: (i) population sanitation coverage (%), (ii) secondary education attainment index, "
                "and (iii) residual life-expectancy Z-scores to identify countries outperforming or lagging income peers.",
                normal_style,
            ),
            Paragraph("REFERENCES", heading_style),
            Paragraph(
                '[1] S. H. Preston, "The changing relation between mortality and level of economic development," '
                "<i>Population Studies</i>, vol. 29, no. 2, pp. 231-248, 1975.",
                ref_style,
            ),
            Paragraph("[2] A. Sen, <i>Development as Freedom</i>. Oxford University Press, 1999.", ref_style),
            Paragraph(
                '[3] World Health Organization, "Global Health Observatory Data Repository," Geneva, 2026.',
                ref_style,
            ),
        ]
    )

    doc.build(story)
    print("EDA research report generated successfully: eda_research_report.pdf")


if __name__ == "__main__":
    generate_eda_report()
