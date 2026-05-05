"""
09_huggingface_pipelines_demo.py

Minimal HuggingFace pipeline usage for text generation and sentiment.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace pipeline demo")
    parser.add_argument("--gen-model", default="distilgpt2")
    parser.add_argument("--sentiment-model", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_new_tokens = min(args.max_new_tokens, 20)

    try:
        from transformers import pipeline
    except ImportError:
        print("Missing optional dependency: transformers")
        return

    model_kwargs = {"local_files_only": args.local_files_only} if args.local_files_only else {}

    try:
        gen = pipeline("text-generation", model=args.gen_model, **model_kwargs)
        out = gen("Mi a self-attention egy mondatban?", max_new_tokens=args.max_new_tokens)
        print("Generated text:")
        print(out[0]["generated_text"])
    except Exception as exc:
        print(f"Text generation skipped: {exc}")

    try:
        sa = pipeline("sentiment-analysis", model=args.sentiment_model, **model_kwargs)
        print("Sentiment:")
        print(sa(["A vizsga kozeledik.", "A diasor jo lett."]))
    except Exception as exc:
        print(f"Sentiment skipped: {exc}")


if __name__ == "__main__":
    main()

