import pandas as pd
import numpy as np
import os

def generate_global_health_data(num_countries=200, random_seed=42):
    """
    Generates a synthetic global health dataset with realistic correlations.
    """
    np.random.seed(random_seed)
    
    # Generate mock country names
    countries = [f"Country_{i}" for i in range(1, num_countries + 1)]
    
    # Regions
    regions = ['Africa', 'Asia', 'Europe', 'Americas', 'Oceania']
    region_col = np.random.choice(regions, num_countries, p=[0.25, 0.3, 0.2, 0.2, 0.05])
    
    # GDP per capita drives many health indicators
    gdp_per_capita = np.random.lognormal(mean=9, sigma=1.2, size=num_countries)
    gdp_per_capita = np.clip(gdp_per_capita, 500, 120000)
    
    # Education Index (0 to 1) correlates with GDP
    education_index = 0.2 + 0.07 * np.log(gdp_per_capita) + np.random.normal(0, 0.08, num_countries)
    education_index = np.clip(education_index, 0.1, 0.99)

    # Sanitation Access (%)
    sanitation_access = 10 + 9 * np.log(gdp_per_capita) + 20 * education_index + np.random.normal(0, 10, num_countries)
    sanitation_access = np.clip(sanitation_access, 5, 100)

    # Life expectancy correlates positively with log(GDP) and Education
    base_life_exp = 30 + 4 * np.log(gdp_per_capita) + 15 * education_index + 0.1 * sanitation_access
    life_expectancy = base_life_exp + np.random.normal(0, 2, num_countries)
    life_expectancy = np.clip(life_expectancy, 45, 85)
    
    # Child mortality correlates negatively with Life Expectancy and Sanitation
    child_mortality = 250 - 2.5 * life_expectancy - 0.5 * sanitation_access + np.random.normal(0, 8, num_countries)
    child_mortality = np.clip(child_mortality, 2, 150)
    
    # Healthcare spending % GDP
    hc_spending_pct = 3 + 0.4 * np.log(gdp_per_capita) + np.random.normal(0, 1.5, num_countries)
    hc_spending_pct = np.clip(hc_spending_pct, 1, 15)

    # Physicians per 10,000
    physicians_per_10k = 1 + 3 * np.log(gdp_per_capita) + np.random.normal(0, 5, num_countries)
    physicians_per_10k = np.clip(physicians_per_10k, 0.1, 50)
    
    # Build dataframe
    df = pd.DataFrame({
        'Country': countries,
        'Region': region_col,
        'GDP_Per_Capita': np.round(gdp_per_capita, 2),
        'Education_Index': np.round(education_index, 3),
        'Sanitation_Access_Pct': np.round(sanitation_access, 1),
        'Physicians_Per_10k': np.round(physicians_per_10k, 1),
        'Life_Expectancy': np.round(life_expectancy, 1),
        'Child_Mortality_Per_1000': np.round(child_mortality, 1),
        'Healthcare_Spending_Pct_GDP': np.round(hc_spending_pct, 1)
    })
    
    # Introduce some missing values randomly
    mask = np.random.rand(num_countries) < 0.05
    df.loc[mask, 'Healthcare_Spending_Pct_GDP'] = np.nan
    df.loc[np.random.rand(num_countries) < 0.05, 'Sanitation_Access_Pct'] = np.nan
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate_global_health_data()
    output_path = os.path.join('data', 'global_health_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data generated successfully: {output_path}")
