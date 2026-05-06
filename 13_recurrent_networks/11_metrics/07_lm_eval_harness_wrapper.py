"""Wrapper guidance for lm-evaluation-harness style benchmarking."""
import argparse
import shutil
import subprocess
def parse_args():
    parser = argparse.ArgumentParser(description="lm-eval harness wrapper")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--tasks", default="hellaswag")
    return parser.parse_args()
def main():
    args = parse_args()
    exe = shutil.which("lm_eval")
    if exe is None:
        print("Missing optional dependency: lm-evaluation-harness CLI (lm_eval)")
        print("Install and run: pip install lm-eval")
        return
    cmd = [
        exe,
        "--model",
        "hf",
        "--model_args",
        f"pretrained={args.model}",
        "--tasks",
        args.tasks,
        "--batch_size",
        "1",
    ]
    if args.smoke:
        print("Smoke mode command preview:")
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=False)
if __name__ == "__main__":
    main()
