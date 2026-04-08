"""
Migration tests: validate parity between the old data_generation module and the
new datasets module.

TODO: Delete this file before merging into main — once data_generation is removed
      these tests will break by design.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from muniverse.data_generation import generate_recording as old_generate_recording
from muniverse.datasets import generate_synthetic_recording
from muniverse.utils.containers import verify_container_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "neuromotion.json"

# New pipeline config schema (datasets module)
_NEW_MOVEMENT_OVERRIDE = {
    "EffortProfile": "Trapezoid",
    "AngleProfile": "Constant",
    "InitialAngle": 0,
    "TargetAngle": 0,
    "TargetEffort": 10,
    "NRepetitions": 1,
    "RestDuration": 0.5,
    "RampDuration": 0.5,
    "HoldDuration": 0.5,
    "SinFrequency": 0,
    "MovementDuration": 2,
}

# Old pipeline config schema (data_generation module)
# Uses EffortLevel instead of TargetEffort, and includes SinAmplitude
_OLD_MOVEMENT_OVERRIDE = {
    "EffortProfile": "Trapezoid",
    "AngleProfile": "Constant",
    "TargetAngle": 0,
    "EffortLevel": 10,
    "NRepetitions": 1,
    "RestDuration": 0.5,
    "RampDuration": 0.5,
    "HoldDuration": 0.5,
    "SinFrequency": 0,
    "SinAmplitude": 0,
    "MovementDuration": 2,
}

CONTAINER = "pranavm19/muniverse:neuromotion"


def _pick_engine():
    for engine in ("singularity", "docker"):
        try:
            if verify_container_engine(engine):
                return engine
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    return None


def _load_new_short_config():
    with open(DEFAULT_CONFIG_PATH) as f:
        config = json.load(f)
    config["MovementConfiguration"]["MovementProfileParameters"].update(
        _NEW_MOVEMENT_OVERRIDE
    )
    return config


def _load_old_short_config():
    """Load config using old schema for data_generation module.
    Also fixes DOF case bug in old _run_neuromotion.py (expects lowercase 'd').
    """
    with open(DEFAULT_CONFIG_PATH) as f:
        config = json.load(f)
    config["MovementConfiguration"]["MovementProfileParameters"] = _OLD_MOVEMENT_OVERRIDE.copy()
    config["MovementConfiguration"]["MovementDOF"] = "Radial-Ulnar-deviation"
    return config


def _has_container():
    return _pick_engine() is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    e = _pick_engine()
    if e is None:
        pytest.skip("No container engine available")
    return e


@pytest.fixture(scope="module")
def new_short_config(tmp_path_factory):
    cfg = _load_new_short_config()
    p = tmp_path_factory.mktemp("cfg") / "new_config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


@pytest.fixture(scope="module")
def old_short_config(tmp_path_factory):
    cfg = _load_old_short_config()
    p = tmp_path_factory.mktemp("cfg") / "old_config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_container(), reason="No container engine available")
def test_old_vs_new_emg_shape(engine, old_short_config, new_short_config, tmp_path):
    """Old and new pipelines produce EMG with the same shape for the same config."""
    old_cfg_path = old_short_config

    old_out = str(tmp_path / "old_out")
    os.makedirs(old_out)
    old_run_dir = old_generate_recording(
        config={
            "input_config": old_cfg_path,
            "output_dir": old_out,
            "engine": engine,
            "container": CONTAINER,
            "cache_dir": str(tmp_path / "cache"),
        }
    )
    old_emg_files = list(Path(old_run_dir).rglob("*.npy")) + list(Path(old_run_dir).rglob("*.npz"))
    assert old_emg_files, f"Old pipeline produced no output files in {old_run_dir}"

    # New pipeline
    new_out = str(tmp_path / "new_out")
    os.makedirs(new_out)
    new_results = generate_synthetic_recording(
        input_config=new_short_config,
        output_dir=new_out,
        engine=engine,
        container=CONTAINER,
        cache_dir=None,
    )
    new_emg = new_results["emg"]
    assert new_emg.ndim == 3

    # Compare shapes if old pipeline output is readable
    old_npz_files = list(Path(old_run_dir).rglob("emg_data.npz"))
    if old_npz_files:
        old_data = np.load(old_npz_files[0], allow_pickle=True)
        if "emg" in old_data:
            old_emg = old_data["emg"]
            assert old_emg.shape == new_emg.shape, (
                f"EMG shape mismatch: old={old_emg.shape}, new={new_emg.shape}"
            )
