import pytest
from streamlit.testing.v1 import AppTest

def test_app_loads_correctly():
    """Test that the Streamlit app loads without exceptions and shows the title."""
    at = AppTest.from_file("../app.py").run()
    assert not at.exception
    assert "Global Health Econometric Dashboard" in at.title[0].value

def test_app_navigation():
    """Test sidebar navigation works and doesn't crash."""
    at = AppTest.from_file("../app.py").run()
    # Check default section
    assert at.header[0].value == "Dataset Overview"
