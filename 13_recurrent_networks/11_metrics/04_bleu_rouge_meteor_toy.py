"""Toy BLEU, ROUGE, and METEOR demos with robust fallbacks."""
import argparse
from metrics_common import missing_dep
def parse_args():
    parser = argparse.ArgumentParser(description="BLEU/ROUGE/METEOR toy demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    _ = parse_args()
    refs = [["The cat sat on the mat."]]
    hyps = ["The cat is on the mat."]
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, refs)
        print(f"BLEU(sacrebleu) = {bleu.score:.2f}")
    except Exception:
        missing_dep("sacrebleu")
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(refs[0][0], hyps[0])
        for key, value in scores.items():
            print(f"{key}: P={value.precision:.3f} R={value.recall:.3f} F1={value.fmeasure:.3f}")
    except Exception:
        missing_dep("rouge-score")
    try:
        import nltk
        from nltk.translate.meteor_score import single_meteor_score
        nltk.download("wordnet", quiet=True)
        meteor = single_meteor_score(refs[0][0].split(), hyps[0].split())
        print(f"METEOR = {meteor:.3f}")
    except Exception:
        missing_dep("nltk")
if __name__ == "__main__":
    main()
