"""Perplexity demo with stride evaluation for GPT-style models."""
import argparse
from metrics_common import missing_dep
def parse_args():
    parser = argparse.ArgumentParser(description="Perplexity toy demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    args = parse_args()

    if args.smoke:
        # Keep smoke fast and offline-friendly.
        synthetic_nll = [1.15, 1.04, 1.27, 1.11]
        import math

        ppl = math.exp(sum(synthetic_nll) / len(synthetic_nll))
        print(f"PPL (synthetic smoke) = {ppl:.2f}")
        return

    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except Exception:
        missing_dep("transformers")
        return
    model_id = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    text = "This is a tiny perplexity evaluation example for stride-based scoring. " * 8
    enc = tokenizer(text, return_tensors="pt")
    max_length = model.config.n_positions
    stride = 256
    nlls = []
    end_loc = 0
    for i in range(0, enc.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, enc.input_ids.size(1))
        trg_len = end_loc - i
        input_ids = enc.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(f"PPL = {float(ppl):.2f}")
if __name__ == "__main__":
    main()
