"""Evaluate a trained model with BLEU-1..4 on Flickr8k test images.

    python training/evaluate_bleu.py --captions Flickr8k.token.txt --images Flicker8k_Dataset \
        --test-list Flickr_8k.testImages.txt --limit 1000
"""
import argparse
import os
import sys

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from caption import Captioner, MODEL_PATH  # noqa: E402
from train_flickr8k import clean, load_captions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--test-list", help="Flickr_8k.testImages.txt (default: images not in storage/train_images.txt)")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--resnet-weights", default="imagenet")
    args = ap.parse_args()

    caps = load_captions(args.captions)
    if args.test_list:
        with open(args.test_list) as f:
            names = [l.strip() for l in f if l.strip() in caps]
    else:
        train_file = os.path.join(ROOT, "storage", "train_images.txt")
        train = set(open(train_file).read().split()) if os.path.exists(train_file) else set()
        names = sorted(n for n in caps if n not in train)
    names = [n for n in names if os.path.exists(os.path.join(args.images, n))][: args.limit or None]

    cap = Captioner(model_path=args.model, resnet_weights=args.resnet_weights)
    refs, hyps = [], []
    for i, n in enumerate(names, 1):
        hyps.append(cap.caption_image(os.path.join(args.images, n)).split())
        refs.append([clean(c).split() for c in caps[n]])
        print(f"\r{i}/{len(names)}", end="", flush=True)
    print()
    sm = SmoothingFunction().method1
    for k, w in enumerate([(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25,) * 4], 1):
        print(f"BLEU-{k}: {corpus_bleu(refs, hyps, weights=w, smoothing_function=sm):.4f}")


if __name__ == "__main__":
    main()
