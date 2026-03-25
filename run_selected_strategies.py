import re
import subprocess
import sys
import time
from pathlib import Path


# Use lowercase run keys and map them to StrategyName enum attributes.
STRATEGIES_IN_ORDER = [
    # "fedavg",
    # "fedavgm",
    # "fedadagrad",
    # "fedadam",
    "fedprox",
    "fedyogi",
    "bulyan",
    "krum",
    "multikrum",
    "fedmedian",
    "fedtrimmedavg",
    "differentialprivacyclientsideadaptiveclipping",
    "differentialprivacyclientsidefixedclipping",
    "differentialprivacyserversideadaptiveclipping",
    "differentialprivacyserversidefixedclipping",
    "fedxgbbagging",
    "fedxgbcyclic",
    "qfedavg",
]

# Map registry keys to StrategyName enum attribute names (uppercase)
STRATEGY_KEY_TO_VALUE = {
    "fedavg": "FEDAVG",
    "fedavgm": "FEDAVGM",
    "fedadagrad": "FEDADAGRAD",
    "fedadam": "FEDADAM",
    "fedprox": "FEDPROX",
    "fedyogi": "FEDYOGI",
    "bulyan": "BULYAN",
    "krum": "KRUM",
    "multikrum": "MULTIKRUM",
    "fedmedian": "FEDMEDIAN",
    "fedtrimmedavg": "FEDTRIMMEDAVG",
    "differentialprivacyclientsideadaptiveclipping": "DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING",
    "differentialprivacyclientsidefixedclipping": "DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING",
    "differentialprivacyserversideadaptiveclipping": "DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING",
    "differentialprivacyserversidefixedclipping": "DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING",
    "fedxgbbagging": "FEDXGBBAGGING",
    "fedxgbcyclic": "FEDXGBCYCLIC",
    "qfedavg": "QFEDAVG",
}


def update_strategy_in_config(config_path: Path, strategy_name: str) -> None:
    """Directly edit config/__init__.py to set STRATEGY_NAME before subprocess runs."""
    content = config_path.read_text(encoding='utf-8')

    # Pattern: STRATEGY_NAME = StrategyName.SOMETHING
    pattern = r'(STRATEGY_NAME\s*=\s*StrategyName\.)\w+'
    replacement = f'\\1{strategy_name}'

    new_content, count = re.subn(pattern, replacement, content, count=1)
    if count != 1:
        raise RuntimeError(
            f"Failed to update STRATEGY_NAME in {config_path}. Expected exactly one match."
        )
    config_path.write_text(new_content, encoding='utf-8')


def _latest_res_dir(project_root: Path) -> Path | None:
    """Return latest timestamped result directory under res/, if any."""
    res_root = project_root / "res"
    if not res_root.exists():
        return None

    candidates = [p for p in res_root.iterdir() if p.is_dir() and re.match(r"^\d{8}-\d{6}", p.name)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def cleanup_ray_runtime(project_root: Path) -> None:
    """Best-effort cleanup to avoid stale Ray sessions between strategy runs."""
    ray_cli = Path(sys.executable).parent / "Scripts" / "ray.exe"
    stop_cmd = [str(ray_cli), "stop", "--force"] if ray_cli.exists() else ["ray", "stop", "--force"]

    subprocess.run(
        stop_cmd,
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)


def run_one_strategy(strategy_key: str, project_root: Path, config_path: Path) -> int:
    """Run simulation for one strategy by updating config and running run.py."""
    strategy_value = STRATEGY_KEY_TO_VALUE.get(strategy_key, strategy_key)

    # Avoid Ray runtime leftovers causing worker registration failures on Windows.
    cleanup_ray_runtime(project_root)

    # Update config.py with the new strategy
    update_strategy_in_config(config_path, strategy_value)

    print("\n" + "=" * 80)
    print(f"[START] {strategy_key} (value: {strategy_value})")
    print("=" * 80)

    before_latest = _latest_res_dir(project_root)
    started = time.perf_counter()
    # Run run.py as-is; it will read the updated config package config/__init__.py
    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=str(project_root)
    )
    elapsed = time.perf_counter() - started
    after_latest = _latest_res_dir(project_root)

    status = "OK" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
    print(f"[END] {strategy_key} -> {status} | elapsed={elapsed:.2f}s")

    if after_latest is not None and after_latest != before_latest:
        print(f"[RESULT_DIR] {after_latest}")

    return result.returncode


def main() -> int:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config" / "__init__.py"
    failures = []
    original_config_content = config_path.read_text(encoding="utf-8")

    print("Running strategies in fixed order:")
    for idx, key in enumerate(STRATEGIES_IN_ORDER, start=1):
        value = STRATEGY_KEY_TO_VALUE[key]
        print(f"{idx}. {key} -> {value}")

    try:
        for strategy_key in STRATEGIES_IN_ORDER:
            code = run_one_strategy(strategy_key, project_root, config_path)
            if code != 0:
                failures.append((strategy_key, code))
    finally:
        # Always restore the original config to avoid side effects on manual runs.
        config_path.write_text(original_config_content, encoding="utf-8")
        print("[RESTORE] config/__init__.py restored to original content")

    print("\n" + "-" * 80)
    if failures:
        print("Completed with failures:")
        for key, code in failures:
            print(f"- {key}: exit={code}")
        return 1

    print("All strategies completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
