"""Export stage1_collect outputs to stage2_visualize with auto-generated directory name."""

import shutil
from pathlib import Path


STATION_NAME = "stage1_collect"
NEXT_STATION_NAME = "stage2_visualize"


def _read_metadata(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _get_experiment_config(data_dir: Path) -> dict | None:
    """Extract config from the first valid metadata.txt found under data_dir."""
    if not data_dir.exists():
        print(f"Error: data/ directory not found: {data_dir}")
        return None

    required = ["distribution", "num_rounds", "local_epochs", "run_id"]

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        meta_path = subdir / "metadata.txt"
        if not meta_path.exists():
            continue
        meta = _read_metadata(meta_path)
        if all(k in meta for k in required):
            return {
                "distribution": meta["distribution"],
                "num_rounds":   meta["num_rounds"],
                "local_epochs": meta["local_epochs"],
                "run_id":       meta["run_id"],
            }

    print("Error: No valid metadata.txt found in data/ (directory may be empty)")
    return None


def _make_dir_name(config: dict) -> str:
    return (
        f"{config['distribution']}-"
        f"Round-{config['num_rounds']}-"
        f"Epoch-{config['local_epochs']}-"
        f"{config['run_id']}"
    )


def export() -> bool:
    station_dir = Path(__file__).resolve().parent
    project_root = station_dir.parent

    if station_dir.name != STATION_NAME:
        print(f"Warning: expected to run from '{STATION_NAME}', got '{station_dir.name}'")

    print("\n" + "=" * 60)
    print("Stage 1 → Stage 2  Export")
    print("=" * 60)

    data_dir = station_dir / "data"
    charts_dir = station_dir / "charts"

    if not data_dir.exists():
        print("Error: data/ directory not found")
        return False
    if not charts_dir.exists():
        print("Error: charts/ directory not found")
        return False

    print("  data/   found")
    print("  charts/ found")

    config = _get_experiment_config(data_dir)
    if not config:
        return False

    print(f"\n  Distribution : {config['distribution']}")
    print(f"  Rounds       : {config['num_rounds']}")
    print(f"  Local Epochs : {config['local_epochs']}")
    print(f"  Run ID       : {config['run_id']}")

    dir_name = _make_dir_name(config)
    print(f"\n  Output folder: {dir_name}")

    next_station = project_root / NEXT_STATION_NAME
    next_station.mkdir(exist_ok=True)

    target = next_station / dir_name
    if target.exists():
        answer = input(f"\n  '{target.name}' already exists. Overwrite? (y/n): ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return False
        shutil.rmtree(target)

    target.mkdir(parents=True)

    # Copy data/
    print("\nCopying data/ ...")
    shutil.copytree(data_dir, target / "data")
    print("  done")

    # Copy charts/
    print("Copying charts/ ...")
    shutil.copytree(charts_dir, target / "charts")
    print("  done")

    # Write SUMMARY.txt
    summary = target / "SUMMARY.txt"
    summary.write_text(
        "=" * 60 + "\n"
        "STAGE 1 RESULTS SUMMARY\n"
        "=" * 60 + "\n\n"
        f"Directory   : {dir_name}\n\n"
        "Configuration:\n"
        f"  Distribution  : {config['distribution']}\n"
        f"  Rounds        : {config['num_rounds']}\n"
        f"  Local Epochs  : {config['local_epochs']}\n"
        f"  Run ID        : {config['run_id']}\n\n"
        "Contents:\n"
        "  data/   - raw per-strategy metrics (CSV + JSON + metadata)\n"
        "  charts/ - accuracy charts per strategy\n",
        encoding="utf-8",
    )
    print("  SUMMARY.txt written")

    print("\n" + "=" * 60)
    print("Export complete.")
    print("=" * 60)
    print(f"\nLocation: {target.relative_to(project_root)}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if export() else 1)
