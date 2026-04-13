"""Run multiple FL strategies back-to-back.

Edit STRATEGIES_TO_RUN to select which strategies to execute.
Each strategy patches config/__init__.py, runs run.py in a subprocess,
then restores the original config.
"""

import re
import subprocess
import sys
import time
from pathlib import Path


# ─── Edit this list to choose which strategies to run ────────────────────────
STRATEGIES_TO_RUN = [
    # "fedavg",
    # "fedavgm",
    # "fedadagrad",
    # "fedadam",
    # "fedprox",
    # "fedyogi",
    # "bulyan",
    # "krum",
    "multikrum",
    # "fedmedian",
    # "fedtrimmedavg",
    # "dp_client_adaptive",
    "dp_client_fixed",
    "dp_server_adaptive",
    "dp_server_fixed",
    # "fedxgb_bagging",
    # "fedxgb_cyclic",
    # "qfedavg",
]

# Maps human-friendly keys to StrategyName enum attribute names.
_KEY_TO_ENUM = {
    "fedavg":             "FEDAVG",
    "fedavgm":            "FEDAVGM",
    "fedadagrad":         "FEDADAGRAD",
    "fedadam":            "FEDADAM",
    "fedprox":            "FEDPROX",
    "fedyogi":            "FEDYOGI",
    "bulyan":             "BULYAN",
    "krum":               "KRUM",
    "multikrum":          "MULTIKRUM",
    "fedmedian":          "FEDMEDIAN",
    "fedtrimmedavg":      "FEDTRIMMEDAVG",
    "dp_client_adaptive": "DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING",
    "dp_client_fixed":    "DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING",
    "dp_server_adaptive": "DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING",
    "dp_server_fixed":    "DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING",
    "fedxgb_bagging":     "FEDXGBBAGGING",
    "fedxgb_cyclic":      "FEDXGBCYCLIC",
    "qfedavg":            "QFEDAVG",
}


def _patch_config(config_path: Path, enum_attr: str) -> None:
    """Overwrite STRATEGY_NAME in config/__init__.py."""
    content = config_path.read_text(encoding="utf-8")
    new_content, count = re.subn(
        r"(STRATEGY_NAME\s*=\s*StrategyName\.)\w+",
        rf"\g<1>{enum_attr}",
        content,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Failed to patch STRATEGY_NAME in {config_path}")
    config_path.write_text(new_content, encoding="utf-8")


def _latest_results_dir(root: Path) -> Path | None:
    res = root / "results" / "runs"
    if not res.exists():
        return None
    candidates = [
        p for p in res.iterdir()
        if p.is_dir() and re.match(r"^\d{8}-\d{6}", p.name)
    ]
    return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None


def _stop_ray(root: Path) -> None:
    """Best-effort Ray teardown to avoid stale sessions between runs."""
    ray_exe = Path(sys.executable).parent / "Scripts" / "ray.exe"
    cmd = [str(ray_exe), "stop", "--force"] if ray_exe.exists() else ["ray", "stop", "--force"]
    subprocess.run(cmd, cwd=str(root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)


def _run_one(key: str, root: Path, config_path: Path) -> int:
    enum_attr = _KEY_TO_ENUM[key]
    _stop_ray(root)
    _patch_config(config_path, enum_attr)

    print(f"\n{'='*60}")
    print(f"[RUN] {key}  →  StrategyName.{enum_attr}")
    print(f"{'='*60}")

    before = _latest_results_dir(root)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, "run.py"], cwd=str(root))
    elapsed = time.perf_counter() - t0
    after = _latest_results_dir(root)

    status = "OK" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
    print(f"[END] {key} → {status}  ({elapsed:.1f}s)")
    if after and after != before:
        print(f"[OUTPUT] {after}")

    return result.returncode


def main() -> int:
    root = Path(__file__).resolve().parent
    config_path = root / "src" / "config" / "__init__.py"
    original = config_path.read_text(encoding="utf-8")
    failures: list[tuple[str, int]] = []

    print("Strategies queued:")
    for i, key in enumerate(STRATEGIES_TO_RUN, 1):
        print(f"  {i:2d}. {key}  →  {_KEY_TO_ENUM[key]}")

    try:
        for key in STRATEGIES_TO_RUN:
            code = _run_one(key, root, config_path)
            if code != 0:
                failures.append((key, code))
    finally:
        config_path.write_text(original, encoding="utf-8")
        print("\n[RESTORED] config/__init__.py reset to original")

    print(f"\n{'─'*60}")
    if failures:
        print(f"Finished with {len(failures)} failure(s):")
        for key, code in failures:
            print(f"  {key}: exit={code}")
        return 1

    print(f"All {len(STRATEGIES_TO_RUN)} strategies completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
