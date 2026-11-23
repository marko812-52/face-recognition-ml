"""Спільні фікстури: дані Digits і підготовлені дані облич."""

from pathlib import Path

import pytest

from devops_lab7.ml_models import DatasetSplit, load_digits_split

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def digits() -> DatasetSplit:
    return load_digits_split()


@pytest.fixture(autouse=True)
def project_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Відносні шляхи data/ і output/ рахуються від кореня проєкту."""
    monkeypatch.chdir(PROJECT)
