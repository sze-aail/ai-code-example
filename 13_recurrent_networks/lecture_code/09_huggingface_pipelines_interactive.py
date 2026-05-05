"""Interactive CLI wrapper around HuggingFace pipelines."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive HuggingFace pipeline demo")
    parser.add_argument("--gen-model", default="distilgpt2")
    parser.add_argument("--sentiment-model", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import pipeline
    except ImportError:
        print("Missing optional dependency: transformers")
        return

    kwargs = {"local_files_only": True} if args.local_files_only else {}

    try:
        gen = pipeline("text-generation", model=args.gen_model, **kwargs)
    except Exception as exc:
        gen = None
        print(f"Generation pipeline unavailable: {exc}")

    try:
        sa = pipeline("sentiment-analysis", model=args.sentiment_model, **kwargs)
    except Exception as exc:
        sa = None
        print(f"Sentiment pipeline unavailable: {exc}")

    if args.smoke:
        if gen is not None:
            print(gen("Explain self-attention in one sentence:", max_new_tokens=20)[0]["generated_text"])
        if sa is not None:
            print(sa(["The lecture slides are clear.", "The deadline is close."]))
        return

    print("Interactive mode. Commands: gen <text>, sent <text>, empty line exits.")
    while True:
        line = input("hf> ").strip()
        if not line:
            break
        if line.startswith("gen ") and gen is not None:
            print(gen(line[4:], max_new_tokens=40)[0]["generated_text"])
        elif line.startswith("sent ") and sa is not None:
            print(sa([line[5:]]))
        else:
            print("Unknown command or unavailable pipeline.")


if __name__ == "__main__":
    main()

