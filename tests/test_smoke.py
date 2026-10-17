"""Every pinned runtime dependency imports on this platform (spec 01, Req 1.5)."""

import importlib

import pytest

PACKAGES = [
    "rdkit.Chem",
    "duckdb",
    "polars",
    "pyarrow",
    "lightgbm",
    "sklearn",
    "scipy",
    "numpy",
    "pandas",
    "mlflow",
    "fastapi",
    "uvicorn",
    "pydantic",
    "streamlit",
    "anthropic",
    "openpyxl",
    "matplotlib",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_import(name):
    importlib.import_module(name)


def test_config_paths():
    from oncora import config

    assert config.ROOT.joinpath("pyproject.toml").exists()
    assert set(config.RAW) >= {"ccle", "dss3", "compoundinfo", "cellinfo"}
