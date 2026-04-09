"""
P053 — Post-Simulation Update Script
======================================
Run this AFTER copying simulation results back from Colab/S3 to the workspace.
Regenerates all plots, updates the dashboard, and prints a checklist.

Usage:
    python -m src.post_simulation_update

What it does:
    1. Validate simulation_timeline.json is present and complete
    2. Regenerate simulation charts (plot_simulation_results.py)
    3. Copy new charts to assets/
    4. Print dashboard update instructions (web/dashboard.html auto-reads the JSON)
    5. Print report update checklist
    6. Show what to commit
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
TIMELINE_PATH = DATA_DIR / "simulation_timeline.json"
DRIFT_REPORT_DIR = DATA_DIR / "drift_reports"


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "✅" if ok else "❌"
    line = f"  {status} {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def validate_simulation_results() -> bool:
    """Validate expected simulation artifacts are present and complete."""
    print("\n[1/5] Validating simulation results...")
    all_ok = True

    # Timeline
    if TIMELINE_PATH.exists():
        with open(TIMELINE_PATH) as f:
            tl = json.load(f)
        days = len(tl.get("days", []))
        retrains = len(tl.get("retrain_events", []))
        elapsed = tl.get("total_elapsed_min", 0)
        all_ok &= _check(
            "simulation_timeline.json",
            days > 0,
            f"{days} days, {retrains} retrain events, {elapsed:.1f} min",
        )
        if days < 40:
            print(f"      ⚠️  Only {days}/40 days present. Did the simulation complete?")
    else:
        all_ok &= _check("simulation_timeline.json", False, "NOT FOUND — copy from Google Drive")

    # Drift reports
    drift_files = list(DRIFT_REPORT_DIR.glob("day_*.json")) if DRIFT_REPORT_DIR.exists() else []
    all_ok &= _check(
        f"drift_reports/",
        len(drift_files) >= 30,
        f"{len(drift_files)} day reports found",
    )

    # MLflow DB
    mlflow_db = PROJECT_ROOT / "mlflow.db"
    all_ok &= _check(
        "mlflow.db",
        mlflow_db.exists(),
        f"{mlflow_db.stat().st_size / 1e3:.0f} KB" if mlflow_db.exists() else "NOT FOUND",
    )

    # Benchmark files
    bench_files = list(DATA_DIR.glob("benchmark_*.json"))
    all_ok &= _check(
        "benchmark_*.json",
        len(bench_files) > 0,
        f"{len(bench_files)} files",
    )

    return all_ok


def regenerate_charts() -> bool:
    """Run plot_simulation_results.py to regenerate all simulation charts."""
    print("\n[2/5] Regenerating simulation charts...")

    # Check if timeline exists first
    if not TIMELINE_PATH.exists():
        print("  ❌ Cannot generate charts — simulation_timeline.json missing.")
        return False

    result = subprocess.run(
        [sys.executable, "-m", "src.plot_simulation_results"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        # Count new assets
        generated = list(ASSETS_DIR.glob("p53_3*.png"))
        print(f"  ✅ Charts regenerated: {len(generated)} PNG files in assets/")
        for p in sorted(generated):
            print(f"     {p.name}")
        return True
    else:
        print(f"  ❌ Chart generation failed (exit {result.returncode})")
        if result.stderr:
            print(f"     STDERR: {result.stderr[-400:]}")
        return False


def print_dashboard_status() -> None:
    """The dashboard reads simulation_timeline.json directly — no update needed."""
    print("\n[3/5] Dashboard status...")
    dashboard = PROJECT_ROOT / "web" / "dashboard.html"
    if dashboard.exists():
        print("  ✅ web/dashboard.html exists")
        print("     The dashboard reads data/simulation_timeline.json directly via fetch().")
        print("     Open it in a browser to see updated results — no file changes needed.")
    else:
        print("  ❌ web/dashboard.html not found")


def print_report_checklist() -> None:
    """Print what needs manual attention for the HTML report."""
    print("\n[4/5] Report update checklist...")
    report = PROJECT_ROOT / "docs" / "Memory_Yield_Predictor_Report.html"
    if report.exists():
        size_mb = report.stat().st_size / 1e6
        print(f"  ✅ docs/Memory_Yield_Predictor_Report.html ({size_mb:.1f} MB)")
        print("     The report has PLACEHOLDER simulation sections.")
        print("     To embed real charts, ask Copilot to update the report.")
    print()
    print("  Manual steps after this script:")
    print("    1. Open web/dashboard.html → verify timeline looks correct")
    print("    2. Check assets/p53_33_drift_timeline.png, p53_34_retrain_story.png")
    print("    3. Tell Copilot: 'embed simulation charts in docs/Memory_Yield_Predictor_Report.html'")
    print("    4. Tell Copilot: 'generate PDF report with Playwright'")


def print_git_checklist() -> None:
    """Print what to commit."""
    print("\n[5/5] Git commit checklist...")
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        print("  Modified / new files:")
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
        print()
        print("  Suggested commit:")
        print("    git add data/simulation_timeline.json data/drift_reports/")
        print("           data/benchmark_*.json mlflow.db mlruns/ assets/")
        print("    git commit -m 'feat: add 40-day simulation results + regenerated charts'")
        print("    git push")
    else:
        print("  No uncommitted changes detected.")


def main() -> None:
    print("=" * 70)
    print("P053 — Post-Simulation Update")
    print("=" * 70)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Timeline:     {TIMELINE_PATH}")

    validate_simulation_results()
    charts_ok = regenerate_charts()
    print_dashboard_status()
    print_report_checklist()
    print_git_checklist()

    print("\n" + "=" * 70)
    if charts_ok:
        print("✅ Post-simulation update complete.")
    else:
        print("⚠️  Partial update — check errors above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
