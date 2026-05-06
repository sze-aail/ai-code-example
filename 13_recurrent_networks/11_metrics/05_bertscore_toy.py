"""Toy BERTScore example with multilingual model."""
import argparse
from metrics_common import missing_dep
def parse_args():
    parser = argparse.ArgumentParser(description="BERTScore toy demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    _ = parse_args()
    cands = ["A macska a szonyegen pihen."]
    refs = ["A macska a szonyegen ul."]
    try:
        from bert_score import score
        _, _, f1 = score(cands, refs, lang="hu", model_type="xlm-roberta-large")
        print(f"BERTScore F1 = {float(f1.mean()):.3f}")
    except Exception:
        missing_dep("bert-score")
if __name__ == "__main__":
    main()
