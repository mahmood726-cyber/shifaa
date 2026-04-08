"""Shared pytest fixtures."""
from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def tmp_data(tmp_path):
    for sub in ("cache", "output"):
        (tmp_path / sub).mkdir()
    return tmp_path

@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR
