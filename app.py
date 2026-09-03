import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, zscore
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="Global Health EDA", layout="wide", page_icon="🌍")

@st.cache_data
def load_data():
    return pd.read_csv('data/global_health_data.csv').dropna()

def main():
    st.title("🌍 Global Health Econometric Dashboard")
    st.markdown("An interactive exploratory data analysis of macroeconomic and health indicators.")

    try:
        df = load_data()
    except Exception:
        st.warning("Data not found. Please run `python src/data_generator.py` to generate the dataset.")
        st.stop()

    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Go to:", ["Overview", "Preston Curve (Wealth vs Health)", "Education & Child Mortality", "Multivariate Heatmap", "Report Download"])

    if section == "Overview":
        st.header("Dataset Overview")
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Countries Analyzed", len(df))
        col2.metric("Avg Life Expectancy", f"{df['Life_Expectancy'].mean():.1f} Years")
        col3.metric("Avg GDP Per Capita", f"${df['GDP_Per_Capita'].mean():,.0f}")

    elif section == "Preston Curve (Wealth vs Health)":
        st.header("The Preston Curve")
        st.markdown("The relationship between national wealth and citizen longevity adheres strictly to a logarithmic transformation.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_palette("colorblind")
        sns.scatterplot(data=df, x='GDP_Per_Capita', y='Life_Expectancy', hue='Region', alpha=0.8, s=60, ax=ax)
        sns.regplot(data=df, x='GDP_Per_Capita', y='Life_Expectancy', scatter=False, color='black', line_kws={'linestyle':'--'}, ax=ax)
        ax.set_xscale('log')
        ax.set_title('Life Expectancy vs GDP (Log)', fontsize=14)
        ax.set_xlabel('GDP Per Capita (USD, Log Scale)')
        ax.set_ylabel('Life Expectancy (Years)')
        st.pyplot(fig)
        
    elif section == "Education & Child Mortality":
        st.header("Socio-Educational Determinants of Mortality")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(data=df, x='Education_Index', y='Child_Mortality_Per_1000', scatter_kws={'alpha':0.6}, line_kws={'color':'#d95f02'}, ax=ax)
        ax.set_title('Educational Attainment vs Child Mortality', fontsize=14)
        ax.set_xlabel('Education Index (0 to 1)')
        ax.set_ylabel('Child Mortality (per 1,000 live births)')
        st.pyplot(fig)

    elif section == "Multivariate Heatmap":
        st.header("Correlation Matrix")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='vlag', fmt=".2f", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)

    elif section == "Report Download":
        st.header("Download Official IEEE Report")
        st.markdown("The formal econometric PDF report can be downloaded below. Make sure you've generated it first.")
        if os.path.exists("eda_research_report.pdf"):
            with open("eda_research_report.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name="eda_research_report.pdf",
                    mime="application/pdf",
                )
        else:
            st.error("Report not found. Run `python src/generate_eda_report.py` to generate the PDF.")

if __name__ == "__main__":
    main()
