"""Interactive Streamlit dashboard for Global Health Econometric EDA."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_generator import generate_global_health_data
from src.generate_eda_report import compute_econometric_summary, generate_eda_report
from src.guardrails import resolve_safe_path, validate_health_dataframe

st.set_page_config(page_title="Global Health EDA", layout="wide", page_icon="🌍")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and validate the global health CSV dataset, generating it if absent."""
    csv_path = resolve_safe_path("data/global_health_data.csv")
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generate_global_health_data().to_csv(csv_path, index=False)
    raw_df = pd.read_csv(csv_path)
    return validate_health_dataframe(raw_df, min_rows=3)


def main() -> None:
    """Render the Global Health EDA Streamlit application."""
    st.title("🌍 Global Health Econometric Dashboard")
    st.markdown("Exploratory data analysis of synthetic macroeconomic and public health indicators.")

    try:
        df = load_data()
    except (FileNotFoundError, ValueError, OSError):
        st.warning(
            "Unable to load a valid dataset. "
            "Run `python src/data_generator.py` to regenerate `data/global_health_data.csv`."
        )
        st.stop()

    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Go to:",
        [
            "Overview",
            "Preston Curve (Wealth vs Health)",
            "Education & Child Mortality",
            "Multivariate OLS & Diagnostics",
            "Multivariate Heatmap",
            "Report Download",
        ],
    )

    regions = sorted(df["Region"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect("Filter by Region:", options=regions, default=regions)
    filtered_df = df[df["Region"].isin(selected_regions)].copy()

    if len(filtered_df) < 3 or filtered_df["GDP_Per_Capita"].nunique() < 2 or filtered_df["Education_Index"].nunique() < 2:
        st.warning("Select regions containing at least 3 records with distinct GDP and Education values.")
        st.stop()

    stats = compute_econometric_summary(filtered_df, initial_rows=len(filtered_df))
    m1 = stats["model1"]
    m2 = stats["model2"]
    m3 = stats["model3"]

    if section == "Overview":
        st.header("Dataset Overview")
        st.dataframe(filtered_df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Countries Analyzed", len(filtered_df))
        col2.metric("Avg Life Expectancy", f"{filtered_df['Life_Expectancy'].mean():.1f} Years")
        col3.metric("Avg GDP Per Capita", f"${filtered_df['GDP_Per_Capita'].mean():,.0f}")

    elif section == "Preston Curve (Wealth vs Health)":
        st.header("The Preston Curve")
        st.markdown("Logarithmic relationship between GDP per capita and life expectancy.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{m1['r2']:.3f}")
        c2.metric("Log Slope", f"{m1['slope']:.2f}")
        c3.metric("RMSE", f"{m1['rmse']:.2f} yrs")
        c4.metric("Outliers (|Z| > 2)", m1["outliers_count"])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_palette("colorblind")
        sns.scatterplot(
            data=filtered_df, x="GDP_Per_Capita", y="Life_Expectancy", hue="Region", alpha=0.8, s=60, ax=ax
        )
        sns.regplot(
            data=filtered_df,
            x="GDP_Per_Capita",
            y="Life_Expectancy",
            scatter=False,
            logx=True,
            color="black",
            line_kws={"linestyle": "--"},
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_title(f"Life Expectancy vs GDP (Log) — p={m1['p']:.2e}", fontsize=14)
        ax.set_xlabel("GDP Per Capita (USD, Log Scale)")
        ax.set_ylabel("Life Expectancy (Years)")
        st.pyplot(fig)
        plt.close(fig)

    elif section == "Education & Child Mortality":
        st.header("Educational Attainment & Child Mortality")

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{m2['r2']:.3f}")
        c2.metric("Slope", f"{m2['slope']:.1f} / unit")
        c3.metric("RMSE", f"{m2['rmse']:.2f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            data=filtered_df,
            x="Education_Index",
            y="Child_Mortality_Per_1000",
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "#d95f02"},
            ax=ax,
        )
        ax.set_title(f"Educational Attainment vs Child Mortality — p={m2['p']:.2e}", fontsize=14)
        ax.set_xlabel("Education Index (0 to 1)")
        ax.set_ylabel("Child Mortality (per 1,000 live births)")
        st.pyplot(fig)
        plt.close(fig)

    elif section == "Multivariate OLS & Diagnostics":
        st.header("Multivariate OLS & Residual Diagnostics")
        if m3 is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{m3['r2']:.3f}")
            c2.metric("Adj. R²", f"{m3['adj_r2']:.3f}")
            c3.metric("RMSE", f"{m3['rmse']:.2f} yrs")
            c4.metric("Max VIF", f"{m3['max_vif']:.2f}")

            pred_table = pd.DataFrame(
                [
                    {
                        "Predictor": row["predictor"],
                        "Beta": round(row["beta"], 4),
                        "SE": round(row["se"], 4),
                        "p-value": f"{row['p_val']:.2e}" if row["p_val"] < 0.001 else f"{row['p_val']:.4f}",
                        "VIF": "inf" if not np.isfinite(row["vif"]) else round(row["vif"], 2),
                    }
                    for row in m3["predictors"]
                ]
            )
            st.dataframe(pred_table, use_container_width=True, hide_index=True)
        else:
            st.info("At least 7 filtered records are required to estimate full 6-parameter multivariate OLS.")

        st.subheader("Preston Curve Residual Diagnostics")
        norm_status = "consistent with normality" if m1["shapiro_p"] >= 0.05 else "departs from normality"
        st.caption(f"Shapiro-Wilk W={m1['shapiro_w']:.3f}, p={m1['shapiro_p']:.3f} ({norm_status} at alpha=0.05).")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        residuals = m1["residuals"]
        fitted = m1["fitted"]
        res_std = float(np.std(residuals))
        ax1.scatter(fitted, residuals, alpha=0.7, s=30, color="#0072B2")
        ax1.axhline(0, color="black", linewidth=1)
        if res_std > 0:
            ax1.axhline(2 * res_std, color="#D55E00", linestyle="--", linewidth=1, label="+/-2 SD")
            ax1.axhline(-2 * res_std, color="#D55E00", linestyle="--", linewidth=1)
            ax1.legend(loc="upper right")
        ax1.set_title("Residuals vs Fitted (Preston OLS)")
        ax1.set_xlabel("Fitted Life Expectancy (Years)")
        ax1.set_ylabel("Residual (Years)")
        ax1.grid(True, alpha=0.3)

        ax2.hist(residuals, bins=14, color="#009E73", alpha=0.75, edgecolor="white")
        ax2.axvline(0, color="black", linewidth=1)
        ax2.set_title("Residual Distribution")
        ax2.set_xlabel("Residual (Years)")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif section == "Multivariate Heatmap":
        st.header("Correlation Matrix")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(stats["corr_matrix"], annot=True, cmap="vlag", fmt=".2f", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    elif section == "Report Download":
        st.header("Download IEEE Report")
        st.markdown("Download the compiled econometric PDF report below.")
        pdf_path = resolve_safe_path("eda_research_report.pdf")
        if not pdf_path.exists():
            with st.spinner("Generating IEEE PDF report..."):
                generate_eda_report()
        if pdf_path.exists():
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file.read(),
                    file_name="eda_research_report.pdf",
                    mime="application/pdf",
                )
        else:
            st.error("Unable to generate PDF report.")


if __name__ == "__main__":
    main()
