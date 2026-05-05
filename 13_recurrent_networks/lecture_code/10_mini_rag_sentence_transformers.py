"""
10_mini_rag_sentence_transformers.py

Mini RAG pipeline: embed -> retrieve -> answer.
"""

import argparse
import numpy as np

DOCS = [
    "A self-attention sulyokat a softmax(QK^T/sqrt(d_k)) adja.",
    "A VAE veszteseg ket tagja: rekonstrukcios hiba + KL divergencia.",
    "A forget gate bias LSTM-ben gyakran 1-re inicializalt.",
    "A diffusion model training celja az epsilon zaj becslese.",
    "A pre-norm transformer stabilabb tanitast ad, mint a post-norm.",
]


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int):
    doc_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    sims = doc_norm @ q_norm
    idx = np.argsort(sims)[-k:][::-1]
    return idx, sims[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Mini RAG demo")
    parser.add_argument("--query", default="Mi a VAE ket fo veszteseg-tagja?")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Missing optional dependency: sentence-transformers")
        return

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_emb = emb.encode(DOCS)
    q_emb = emb.encode([args.query])[0]

    idx, sims = cosine_topk(q_emb, doc_emb, k=args.k)
    retrieved = [DOCS[i] for i in idx]

    print("Retrieved docs:")
    for rank, (i, s) in enumerate(zip(idx, sims), start=1):
        print(f"{rank}. sim={s:.3f} -> {DOCS[i]}")

    prompt = "Kontextus:\n" + "\n".join(retrieved) + f"\n\nKerdes: {args.query}\nValasz:"

    try:
        from transformers import pipeline

        llm = pipeline("text-generation", model="distilgpt2")
        out = llm(prompt, max_new_tokens=80)
        print("\nLLM answer:")
        print(out[0]["generated_text"])
    except Exception:
        # Fallback when LLM pipeline/model is unavailable.
        print("\nFallback answer:")
        print("A valasz a retrieval kontextus alapjan: " + " ".join(retrieved))


if __name__ == "__main__":
    main()

