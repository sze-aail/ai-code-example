"""Interactive mini-RAG demo."""

import argparse
import importlib
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
    parser = argparse.ArgumentParser(description="Interactive mini-RAG")
    parser.add_argument("--query", default="Mi a VAE ket fo veszteseg-tagja?")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def answer(query: str, emb_model, k: int):
    doc_emb = emb_model.encode(DOCS)
    q_emb = emb_model.encode([query])[0]
    idx, sims = cosine_topk(q_emb, doc_emb, k=k)
    retrieved = [str(DOCS[int(i)]) for i in idx]
    print("Retrieved docs:")
    for rank, (i, score) in enumerate(zip(idx, sims), start=1):
        print(f"{rank}. sim={score:.3f} -> {DOCS[i]}")
    try:
        pipeline = importlib.import_module("transformers").pipeline
        llm = pipeline("text-generation", model="distilgpt2")
        prompt = "Kontextus:\n" + "\n".join(retrieved) + f"\n\nKerdes: {query}\nValasz:"
        print(llm(prompt, max_new_tokens=70)[0]["generated_text"])
    except Exception:
        print("Fallback answer:")
        print(" ".join(retrieved))


def main():
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Missing optional dependency: sentence-transformers")
        return

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if args.smoke:
        answer(args.query, emb, args.k)
        return

    print("Interactive mode. Empty line exits.")
    while True:
        query = input("rag> ").strip()
        if not query:
            break
        answer(query, emb, args.k)


if __name__ == "__main__":
    main()


