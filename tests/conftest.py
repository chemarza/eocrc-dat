from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PACKAGE = ROOT / "eocrc_dat"


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS
