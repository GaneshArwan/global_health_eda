import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the actual project scripts
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import data_generator
import generate_eda_report

def test_data_generation_shape():
    """Test if data generator produces correct shape and columns."""
    df = data_generator.generate_global_health_data(num_countries=50)
    assert len(df) == 50
    assert 'Country' in df.columns
    assert 'GDP_Per_Capita' in df.columns
    assert 'Life_Expectancy' in df.columns
    assert 'Education_Index' in df.columns

def test_data_generator_nan_injection():
    """Test that NaN values are injected into the dataset as per requirements."""
    df = data_generator.generate_global_health_data(num_countries=100)
    # Check if there are any NaNs in the dataset
    assert df.isna().sum().sum() > 0

@patch('matplotlib.pyplot.savefig')
def test_generate_visualizations(mock_savefig):
    """Test the visualization generation function independently."""
    # Create a dummy dataframe
    data = {
        'GDP_Per_Capita': [1000, 2000, 3000, 4000],
        'Life_Expectancy': [50, 60, 70, 80],
        'Region': ['A', 'A', 'B', 'B'],
        'Education_Index': [0.2, 0.4, 0.6, 0.8],
        'Child_Mortality_Per_1000': [100, 80, 40, 20],
        'Healthcare_Spending_Pct_GDP': [2, 4, 6, 8],
        'Sanitation_Access_Pct': [20, 40, 60, 80]
    }
    df = pd.DataFrame(data)
    
    # Run visualization generator
    preston, edu, heat, corr = generate_eda_report.generate_visualizations(df)
    
    # Assert return values
    assert preston == 'temp_assets/preston_curve.png'
    assert edu == 'temp_assets/edu_mort.png'
    assert heat == 'temp_assets/heatmap.png'
    assert corr is not None
    assert mock_savefig.call_count == 3

@patch('generate_eda_report.BaseDocTemplate.build')
@patch('generate_eda_report.generate_visualizations')
@patch('pandas.read_csv')
def test_generate_eda_report(mock_read_csv, mock_gen_vis, mock_build):
    """Test full report generation pipeline."""
    # Mock data
    mock_df = pd.DataFrame({
        'GDP_Per_Capita': [1000, 5000, 15000, 40000, 60000],
        'Life_Expectancy': [50, 65, 75, 80, 82],
        'Region': ['Africa', 'Asia', 'South America', 'Europe', 'North America'],
        'Education_Index': [0.3, 0.5, 0.7, 0.9, 0.95],
        'Child_Mortality_Per_1000': [120, 60, 25, 5, 4],
        'Healthcare_Spending_Pct_GDP': [3, 5, 7, 10, 12],
        'Sanitation_Access_Pct': [20, 50, 80, 99, 100]
    })
    
    # Mock return values
    mock_read_csv.return_value = mock_df
    # Mock visualizations return (paths and correlation matrix)
    mock_gen_vis.return_value = ('test.png', 'test2.png', 'test3.png', mock_df.select_dtypes(include=[float, int]).corr())
    
    # Call the main function
    generate_eda_report.generate_eda_report()
    
    # Assert it called build() on the PDF document
    assert mock_build.call_count == 1
