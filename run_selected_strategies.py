import subprocess
import sys
import time
from pathlib import Path


STRATEGIES_IN_ORDER = [
    "FedTrimmedAvg",
    "DifferentialPrivacyClientSideAdaptiveClipping",
    "DifferentialPrivacyClientSideFixedClipping",
    "DifferentialPrivacyServerSideAdaptiveClipping",
    "DifferentialPrivacyServerSideFixedClipping",
    "FedXgbBagging",
    "FedXgbCyclic",
    "QFedAvg",
]


def run_one_strategy(strategy_name: str, project_root: Path) -> int:
    python_code = (
        "import config, run; "
        f"config.STRATEGY_NAME = '{strategy_name}'; "
        "run.run_simulation()"
    )

    cmd = [sys.executable, "-c", python_code]
    print("\n" + "=" * 80)
    print(f"[START] {strategy_name}")
    print("=" * 80)

    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(project_root))
    elapsed = time.perf_counter() - started

    status = "OK" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
    print(f"[END] {strategy_name} -> {status} | elapsed={elapsed:.2f}s")
    return result.returncode


def main() -> int:
    project_root = Path(__file__).resolve().parent
    failures = []

    print("Running strategies in fixed order:")
    for idx, name in enumerate(STRATEGIES_IN_ORDER, start=1):
        print(f"{idx}. {name}")

    for strategy_name in STRATEGIES_IN_ORDER:
        code = run_one_strategy(strategy_name, project_root)
        if code != 0:
            failures.append((strategy_name, code))

    print("\n" + "-" * 80)
    if failures:
        print("Completed with failures:")
        for name, code in failures:
            print(f"- {name}: exit={code}")
        return 1

    print("All strategies completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
