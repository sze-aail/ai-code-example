"""Toy CLIP score demo with optional open_clip_torch."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from metrics_common import missing_dep
def parse_args():
    parser = argparse.ArgumentParser(description="CLIP score toy demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def make_toy_image(path: Path):
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[..., 0] = 180
    arr[60:170, 70:160, 1] = 220
    Image.fromarray(arr).save(path)
def main():
    _ = parse_args()
    try:
        import torch
        import open_clip
    except Exception:
        missing_dep("open-clip-torch")
        return
    img_path = Path(__file__).resolve().with_name("03_clip_toy_image.png")
    make_toy_image(img_path)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    prompt = "a simple synthetic geometric shape"
    image = preprocess(Image.open(img_path)).unsqueeze(0)
    text = tokenizer([prompt])
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).item()
    print(f"CLIP score = {score:.4f}")
if __name__ == "__main__":
    main()
