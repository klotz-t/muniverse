"""
regenerate_neuromotion_test.py

Loads each trial from the Neuromotion-Test dataset, translates its simulation
config to the current datasets-module format, regenerates the data, and
produces comparison plots to assess reproducibility.

Usage:
    python scripts/regenerate_neuromotion_test.py \\
        --old_data /path/to/Neuromotion-Test/sub-sim02_ses-02_ECRB \\
        --container /path/to/muniverse_neuromotion.sif \\
        --engine singularity \\
        --output data/neuromotion_regen

Optional flags:
    --skip_regen   Skip regeneration; only plot from already-saved .npz files.
    --no_noise     Zero out NoiseLeveldb/NoiseSeed before regenerating.
"""

import argparse
import copy
import json
import os
import sys
import tempfile

import edfio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------

def translate_config(old_config: dict) -> dict:
    """
    Convert an old-format simulation config to the current datasets-module format.

    Changes made:
    - Container-specific top-level keys removed (PathToBioMimeWeights, etc.)
    - EffortLevel → TargetEffort
    - MovementDOF case normalised ("Radial-Ulnar-deviation" → "Radial-Ulnar-Deviation")
    - InitialAngle added for Sinusoid/Triangular angle profiles:
        Sinusoid: derived from old TargetAngle (centre) and SinAmplitude (half-range)
        Triangular: defaults to 0 (neutral position)
    - SinAmplitude removed (absorbed into InitialAngle/TargetAngle)
    - _description fields stripped
    """
    config = copy.deepcopy(old_config)

    # Strip container-internal top-level keys
    for key in ("PathToBioMimeWeights", "MorphMUAPS", "PathToMUAPFile"):
        config.pop(key, None)

    # Strip internal description annotations
    for section in ("SubjectConfiguration", "RecordingConfiguration", "MovementConfiguration"):
        config.get(section, {}).pop("_description", None)

    movement_cfg = config["MovementConfiguration"]
    movement_cfg.pop("MovementType", None)

    # Fix MovementDOF case
    dof = movement_cfg.get("MovementDOF", "")
    if dof.lower() == "radial-ulnar-deviation":
        movement_cfg["MovementDOF"] = "Radial-Ulnar-Deviation"

    params = movement_cfg["MovementProfileParameters"]

    # EffortLevel → TargetEffort
    if "EffortLevel" in params and "TargetEffort" not in params:
        params["TargetEffort"] = params.pop("EffortLevel")
    elif "EffortLevel" in params:
        params.pop("EffortLevel")

    # Derive InitialAngle for profiles that need it
    angle_profile = params.get("AngleProfile", "Constant")
    target_angle_old = params.get("TargetAngle", 0)
    sin_amplitude = params.pop("SinAmplitude", 0)

    if angle_profile == "Sinusoid":
        # Old convention: TargetAngle = centre, SinAmplitude = signed half-range.
        # New _sinusoid_angle_profile uses:
        #   offset = (target + initial)/2, amplitude = (target - initial)/2
        # So: initial = centre + sin_amplitude, target = centre - sin_amplitude
        params["InitialAngle"] = target_angle_old + sin_amplitude
        params["TargetAngle"] = target_angle_old - sin_amplitude
    elif angle_profile == "Triangular":
        # Old code moved from neutral (0) to TargetAngle and back.
        if "InitialAngle" not in params:
            params["InitialAngle"] = 0
    # Constant profile: no InitialAngle needed.

    return config


def summarise_translation(old_cfg: dict, new_cfg: dict) -> list[str]:
    """Return a list of human-readable lines describing config changes."""
    lines = []
    old_params = old_cfg["MovementConfiguration"]["MovementProfileParameters"]
    new_params = new_cfg["MovementConfiguration"]["MovementProfileParameters"]

    old_dof = old_cfg["MovementConfiguration"]["MovementDOF"]
    new_dof = new_cfg["MovementConfiguration"]["MovementDOF"]
    if old_dof != new_dof:
        lines.append(f"  MovementDOF: '{old_dof}' → '{new_dof}'")

    if "EffortLevel" in old_params:
        lines.append(f"  EffortLevel={old_params['EffortLevel']} → TargetEffort={new_params['TargetEffort']}")

    old_angle = old_params.get("AngleProfile", "Constant")
    if old_angle == "Sinusoid":
        lines.append(
            f"  Sinusoid angle: TargetAngle={old_params['TargetAngle']}, "
            f"SinAmplitude={old_params.get('SinAmplitude', 0)} → "
            f"InitialAngle={new_params['InitialAngle']}, TargetAngle={new_params['TargetAngle']}"
        )
    elif old_angle == "Triangular":
        lines.append(f"  Triangular angle: added InitialAngle={new_params.get('InitialAngle', 0)}")

    return lines or ["  (no structural changes)"]


# ---------------------------------------------------------------------------
# Load old data
# ---------------------------------------------------------------------------

def load_old_emg(edf_path: str, n_rows: int = 10) -> np.ndarray:
    """
    Read EDF and return emg of shape (n_rows, n_cols, n_samples).
    The number of columns is inferred from total signal count.
    """
    edf = edfio.read_edf(edf_path)
    signals = np.stack([s.data for s in edf.signals], axis=0)  # (n_ch, n_samples)
    n_ch, n_samples = signals.shape
    n_cols = n_ch // n_rows
    return signals.reshape(n_rows, n_cols, n_samples)


def load_old_spikes(tsv_path: str) -> list:
    """
    Read spikes TSV and return a list of spike-time arrays, one per motor unit.
    source_id is the MU index; spike_time is in samples.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    n_units = int(df["source_id"].max()) + 1
    spikes = [np.array(df[df["source_id"] == i]["spike_time"].values, dtype=int)
              for i in range(n_units)]
    return spikes


def load_old_effort_angle(config: dict, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive old effort and angle profiles from the old config params."""
    from muniverse.datasets.movement import (
        generate_effort_profile, generate_angle_profile,
    )
    effort, _ = generate_effort_profile(config)
    angle, _ = generate_angle_profile(config)
    return effort, angle


# ---------------------------------------------------------------------------
# Regenerate
# ---------------------------------------------------------------------------

def regenerate_trial(new_config: dict, output_path: str, engine: str, container: str) -> dict:
    """
    Generate a recording with the current datasets module and cache result.
    Returns the result dict (emg, spikes, effort_profile, angle_profile, ...).
    """
    if os.path.exists(output_path):
        print(f"    [cache] Loading existing {output_path}")
        return dict(np.load(output_path, allow_pickle=True))

    from muniverse.datasets import generate_synthetic_recording

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(new_config, f)
        tmp_cfg = f.name

    out_dir = os.path.dirname(output_path)
    try:
        results = generate_synthetic_recording(
            input_config=tmp_cfg,
            output_dir=out_dir,
            engine=engine,
            container=container,
            cache_dir=None,
            verbose=True,
        )
        # generate_synthetic_recording saves to out_dir/emg_data.npz; rename to our path.
        default_path = os.path.join(out_dir, "emg_data.npz")
        if default_path != output_path and os.path.exists(default_path):
            os.rename(default_path, output_path)
        else:
            np.savez_compressed(output_path, **results)
    finally:
        os.unlink(tmp_cfg)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _first_active(spikes: list) -> int:
    for i, s in enumerate(spikes):
        if len(s) > 0:
            return i
    return 0


def plot_trial_comparison(
    trial_name: str,
    old_emg: np.ndarray,
    new_results: dict,
    old_spikes: list,
    old_effort: np.ndarray,
    old_angle: np.ndarray,
    new_config: dict,
    fs: float,
    save_path: str,
):
    """
    Four-panel comparison: effort profile, angle profile, spike raster, EMG channel.
    """
    new_emg = new_results["emg"]       # (n_rows, n_cols, T) or (n_ch, T)
    new_spikes_raw = new_results["spikes"]
    new_effort = new_results.get("effort_profile", np.array([]))
    new_angle = new_results.get("angle_profile", np.array([]))

    # Normalise spike list
    def _to_list(s):
        if isinstance(s, np.ndarray) and s.dtype == object:
            return [s[i] for i in range(len(s))]
        return list(s)

    new_spikes = _to_list(new_spikes_raw)

    # Flatten EMG to (n_ch, T) for easy channel access
    def _flatten(emg):
        if emg.ndim == 3:
            nr, nc, t = emg.shape
            return emg.reshape(nr * nc, t)
        return emg if emg.shape[0] < emg.shape[1] else emg.T

    old_emg_flat = _flatten(old_emg)
    new_emg_flat = _flatten(new_emg)

    t_old = np.arange(old_emg_flat.shape[1]) / fs
    t_new = np.arange(new_emg_flat.shape[1]) / fs
    t_eff_old = np.arange(len(old_effort)) / fs
    t_eff_new = np.arange(len(new_effort)) / fs
    t_ang_old = np.arange(len(old_angle)) / fs
    t_ang_new = np.arange(len(new_angle)) / fs

    fig, axes = plt.subplots(4, 1, figsize=(13, 12))
    fig.suptitle(f"Reproducibility: {trial_name}", fontsize=11, fontweight="bold")

    # 1. Effort profile
    ax = axes[0]
    ax.plot(t_eff_old, old_effort * 100, lw=1.2, label="old data (from config)")
    ax.plot(t_eff_new, new_effort * 100, lw=1.2, ls="--", label="regen")
    ax.set_ylabel("Effort (%MVC)")
    ax.set_title("Effort profile")
    ax.legend(loc="upper right", fontsize=8)

    # 2. Angle profile
    ax = axes[1]
    ax.plot(t_ang_old, old_angle, lw=1.2, label="old data (from config)")
    ax.plot(t_ang_new, new_angle, lw=1.2, ls="--", label="regen")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Angle profile")
    ax.legend(loc="upper right", fontsize=8)

    # 3. Spike raster (first active unit from each)
    ax = axes[2]
    mu_old = _first_active(old_spikes)
    mu_new = _first_active(new_spikes)
    if len(old_spikes[mu_old]) > 0:
        ax.eventplot(old_spikes[mu_old] / fs, lineoffsets=1.0, linelengths=0.7,
                     linewidths=0.8, label=f"old MU{mu_old}")
    if len(new_spikes) > mu_new and len(new_spikes[mu_new]) > 0:
        ax.eventplot(new_spikes[mu_new] / fs, lineoffsets=0.0, linelengths=0.7,
                     linewidths=0.8, colors="tab:orange", label=f"regen MU{mu_new}")
    ax.set_title("Spike trains (first active MU)")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["regen", "old"])
    ax.legend(loc="upper right", fontsize=8)

    # 4. EMG channel 0
    ax = axes[3]
    ax.plot(t_old, old_emg_flat[0], lw=0.5, label="old ch0")
    ax.plot(t_new, new_emg_flat[0], lw=0.5, alpha=0.8, color="tab:orange", label="regen ch0")
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlabel("Time (s)")
    ax.set_title("EMG channel 0")
    ax.legend(loc="upper right", fontsize=8)

    # Stats annotation
    min_t = min(old_emg_flat.shape[1], new_emg_flat.shape[1])
    corr = float(np.corrcoef(old_emg_flat[0, :min_t], new_emg_flat[0, :min_t])[0, 1])
    rms_old = float(np.sqrt(np.mean(old_emg_flat[0, :min_t] ** 2)))
    rms_new = float(np.sqrt(np.mean(new_emg_flat[0, :min_t] ** 2)))
    fig.text(
        0.99, 0.01,
        f"ch0 corr={corr:.3f}  RMS old={rms_old:.4f}  regen={rms_new:.4f}",
        ha="right", va="bottom", fontsize=8, color="gray",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    [plot] Saved {save_path}")
    return corr, rms_old, rms_new


def plot_summary(summary_rows: list, output_dir: str):
    """Bar chart summarising per-trial channel-0 correlation."""
    if not summary_rows:
        return
    names = [r["name"] for r in summary_rows]
    corrs = [r["corr"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4))
    bars = ax.bar(range(len(names)), corrs, color="steelblue")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r (ch0, old vs regen)")
    ax.set_title("EMG reproducibility per trial")
    ax.set_ylim([-1.05, 1.1])
    for bar, val in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_reproducibility.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[summary] Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_trials(old_data_dir: str) -> list[dict]:
    """Return list of dicts with paths for each trial."""
    trials = []
    for fname in sorted(os.listdir(old_data_dir)):
        if not fname.endswith("_simulation.json"):
            continue
        base = fname.replace("_simulation.json", "")
        trial = {
            "name": base,
            "sim_json": os.path.join(old_data_dir, fname),
            "edf": os.path.join(old_data_dir, f"{base}_emg.edf"),
            "tsv": os.path.join(old_data_dir, f"{base}_spikes.tsv"),
        }
        if os.path.exists(trial["edf"]) and os.path.exists(trial["tsv"]):
            trials.append(trial)
        else:
            print(f"[warn] Skipping {base}: missing EDF or TSV")
    return trials


def main():
    parser = argparse.ArgumentParser(description="Regenerate Neuromotion-Test trials and compare to old data.")
    parser.add_argument("--old_data", required=True,
                        help="Path to old data directory (sub-sim02_ses-02_ECRB)")
    parser.add_argument("--container", default=None,
                        help="Container image name (Docker) or .sif path (Singularity)")
    parser.add_argument("--engine", default="singularity", choices=["docker", "singularity"])
    parser.add_argument("--output", default="data/neuromotion_regen",
                        help="Output directory for regenerated data and plots")
    parser.add_argument("--skip_regen", action="store_true",
                        help="Skip regeneration; only plot from cached .npz files")
    parser.add_argument("--no_noise", action="store_true",
                        help="Set NoiseLeveldb and NoiseSeed to None before regenerating")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    trials = discover_trials(os.path.abspath(args.old_data))
    print(f"Found {len(trials)} trials in {args.old_data}\n")

    summary_rows = []

    for trial in trials:
        name = trial["name"]
        print(f"=== {name} ===")

        # Load old simulation config
        with open(trial["sim_json"]) as f:
            sim_json = json.load(f)
        old_config = sim_json["InputData"]["Configuration"]

        # Translate to new format
        new_config = translate_config(old_config)

        if args.no_noise:
            new_config["RecordingConfiguration"]["NoiseLeveldb"] = None
            new_config["RecordingConfiguration"]["NoiseSeed"] = None

        print("  Config translation:")
        for line in summarise_translation(old_config, new_config):
            print(line)

        fs = new_config["RecordingConfiguration"]["SamplingFrequency"]

        # Load old data
        print("  Loading old data...")
        old_emg = load_old_emg(trial["edf"])
        old_spikes = load_old_spikes(trial["tsv"])
        old_effort, old_angle = load_old_effort_angle(old_config, fs)

        # Regenerate
        trial_output_dir = os.path.join(output_dir, name)
        os.makedirs(trial_output_dir, exist_ok=True)
        npz_path = os.path.join(trial_output_dir, "regen_emg_data.npz")

        if args.skip_regen:
            if not os.path.exists(npz_path):
                print(f"  [skip_regen] No cached result at {npz_path}, skipping trial.")
                continue
            print("  Loading cached regenerated data...")
            new_results = dict(np.load(npz_path, allow_pickle=True))
        else:
            if args.container is None:
                print("  [error] --container is required unless --skip_regen is set.")
                sys.exit(1)
            print("  Regenerating...")
            new_results = regenerate_trial(new_config, npz_path, args.engine, args.container)

        # Save translated config next to regen result for reference
        cfg_path = os.path.join(trial_output_dir, "translated_config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as f:
                json.dump(new_config, f, indent=2)

        # Plot comparison
        plot_path = os.path.join(trial_output_dir, "comparison.png")
        corr, rms_old, rms_new = plot_trial_comparison(
            trial_name=name,
            old_emg=old_emg,
            new_results=new_results,
            old_spikes=old_spikes,
            old_effort=old_effort,
            old_angle=old_angle,
            new_config=new_config,
            fs=fs,
            save_path=plot_path,
        )
        summary_rows.append({"name": name, "corr": corr, "rms_old": rms_old, "rms_new": rms_new})
        print()

    # Summary plot
    plot_summary(summary_rows, output_dir)

    # Summary CSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"[summary] Saved {csv_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
