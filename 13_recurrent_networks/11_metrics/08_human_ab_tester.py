"""Minimal Streamlit A/B evaluation app for human preference collection."""
import argparse
import json
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser(description="Human A/B tester (Streamlit)")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    args = parse_args()
    if args.smoke:
        sample = {
            "pairs": [
                {"a": "Model A: concise answer", "b": "Model B: verbose answer"},
                {"a": "Model A: factual answer", "b": "Model B: creative answer"},
            ]
        }
        out = Path(__file__).resolve().with_name("08_ab_pairs.json")
        out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
        print(f"Saved: {out.name}")
        print("Run interactive app: streamlit run 08_human_ab_tester.py")
        return
    try:
        import streamlit as st
    except Exception:
        print("Missing optional dependency: streamlit")
        return
    pairs_path = Path(__file__).resolve().with_name("08_ab_pairs.json")
    if pairs_path.exists():
        pairs = json.loads(pairs_path.read_text(encoding="utf-8")).get("pairs", [])
    else:
        pairs = [
            {"a": "Model A answer example", "b": "Model B answer example"},
            {"a": "Model A second example", "b": "Model B second example"},
        ]
    st.title("Lec13 - Human A/B preference mini-tester")
    idx = st.session_state.get("idx", 0)
    if idx >= len(pairs):
        st.success("Finished all pairs.")
        return
    pair = pairs[idx]
    c1, c2 = st.columns(2)
    c1.write(pair["a"])
    c2.write(pair["b"])
    choice = st.radio("Which one is better?", ["Left", "Right", "Tie"], key=f"choice_{idx}")
    if st.button("Next"):
        st.write(f"Logged: pair={idx}, choice={choice}")
        st.session_state["idx"] = idx + 1
if __name__ == "__main__":
    main()
