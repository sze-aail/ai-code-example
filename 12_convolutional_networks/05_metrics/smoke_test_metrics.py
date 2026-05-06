"""Smoke test for Lec12 metrics visualization scripts."""
import subprocess
import sys
from pathlib import Path
SCRIPTS = [
    "01_confusion_matrix_prf1.py",
    "02_topk_accuracy_visualization.py",
    "03_roc_pr_curves.py",
    "04_iou_bbox_visualization.py",
    "05_map_kuszobonkent.py",
    "06_szegmentacio_overlay.py",
    "07_miou_per_class.py",
    "08_pq_bontas.py",
    "09_interaktiv_kuszob.py",
]
def main():
    base = Path(__file__).resolve().parent
    failed = []
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        result = subprocess.run([sys.executable, str(base / script), "--smoke"], check=False)
        if result.returncode != 0:
            failed.append(script)
    print("\n=== Smoke result ===")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)
    print("All Lec12 metrics scripts completed.")
if __name__ == "__main__":
    main()
