"""Smoke test for Lec13 generative and NLP metrics scripts."""
import subprocess
import sys
from pathlib import Path
SCRIPTS = [
    "01_fid_toy.py",
    "02_is_kid_toy.py",
    "03_clip_score_toy.py",
    "04_bleu_rouge_meteor_toy.py",
    "05_bertscore_toy.py",
    "06_perplexity_toy.py",
    "07_lm_eval_harness_wrapper.py",
    "08_human_ab_tester.py",
    "09_llm_intrinsic_eval.py",
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
    print("All Lec13 metrics scripts completed.")
if __name__ == "__main__":
    main()
