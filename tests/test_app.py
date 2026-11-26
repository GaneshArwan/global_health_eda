"""Streamlit UI integration tests for Global Health EDA dashboard."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_generator import generate_global_health_data
from src.guardrails import resolve_safe_path

APP_PATH = str(ROOT_DIR / "app.py")


@pytest.fixture(scope="session", autouse=True)
def _ensure_dataset():
    """Ensure synthetic CSV exists so AppTest passes on a fresh clone."""
    csv_path = resolve_safe_path("data/global_health_data.csv")
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generate_global_health_data().to_csv(csv_path, index=False)


def test_app_all_pages():
    """Test every page in the Streamlit app renders without exceptions."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    assert "Global Health Econometric Dashboard" in at.title[0].value

    sections = [
        "Overview",
        "Preston Curve (Wealth vs Health)",
        "Education & Child Mortality",
        "Multivariate OLS & Diagnostics",
        "Multivariate Heatmap",
        "Report Download",
    ]

    for section in sections:
        at.sidebar.radio[0].set_value(section).run()
        assert not at.exception


def test_report_download_button():
    """Test that the Report Download section shows a download button or missing-report error."""
    at = AppTest.from_file(APP_PATH).run()
    at.sidebar.radio[0].set_value("Report Download").run()
    assert not at.exception

    has_button = len(at.download_button) > 0
    has_error = len(at.error) > 0
    assert has_button or has_error


def test_region_filter_guardrail():
    """Test that clearing all regions triggers the guardrail warning instead of crashing."""
    at = AppTest.from_file(APP_PATH).run()
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    assert len(at.warning) > 0
