"""LLM intrinsic metric mini-demo using HuggingFace evaluate."""
import argparse
from metrics_common import missing_dep
def parse_args():
    parser = argparse.ArgumentParser(description="LLM intrinsic evaluate demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    _ = parse_args()
    try:
        import evaluate
    except Exception:
        missing_dep("evaluate")
        return
    preds = ["Paris", "4", "The capital of Hungary is Budapest."]
    refs = ["Paris", "4", "Budapest is the capital of Hungary."]
    try:
        exact = evaluate.load("exact_match")
        em = exact.compute(predictions=preds, references=refs)
        print(f"Exact match = {em.get('exact_match', 0.0):.3f}")
    except Exception as exc:
        print(f"exact_match metric unavailable: {exc}")
    try:
        bleu = evaluate.load("bleu")
        bleu_out = bleu.compute(predictions=[preds[-1]], references=[[refs[-1]]])
        print(f"evaluate BLEU = {bleu_out.get('bleu', 0.0):.3f}")
    except Exception as exc:
        print(f"bleu metric unavailable: {exc}")
if __name__ == "__main__":
    main()
