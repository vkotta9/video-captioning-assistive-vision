"""Compare captioning models with BLEU-1..4 on the same held-out test images.

    # original 2021 model vs. a retrained one, on the retrained model's test split
    python training/evaluate_bleu.py --captions annotations/captions_train2017.json --images train2017.zip \
        --models model_weights/model_9.h5 retrained_coco/model_weights/best.keras \
        --test-list retrained_coco/storage/test_images.txt --limit 1000
"""
import argparse
import os
import sys

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from caption import MODEL_PATH, Captioner, build_encoder  # noqa: E402
from data import ImageSource, clean, extract_features, load_captions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--images", required=True, help="image folder or .zip")
    ap.add_argument("--models", nargs="+", default=[MODEL_PATH], help="one or more .h5 / .keras models")
    ap.add_argument("--test-list", required=True, help="file with test image names (e.g. <out>/storage/test_images.txt)")
    ap.add_argument("--limit", type=int, default=1000, help="number of test images (0 = all)")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--show", type=int, default=5, help="print captions for the first N images")
    ap.add_argument("--resnet-weights", default="imagenet")
    args = ap.parse_args()

    caps = load_captions(args.captions)
    source = ImageSource(args.images)
    with open(args.test_list) as f:
        names = [l.strip() for l in f if l.strip() in caps and l.strip() in source][: args.limit or None]
    refs = [[clean(c).split() for c in caps[n]] for n in names]

    encoder = build_encoder(args.resnet_weights)
    print(f"Encoding {len(names)} test images ...")
    feats = extract_features(names, source, encoder, cache_dir=os.path.join(os.path.dirname(args.test_list), f"test_features_{len(names)}"))

    results = {}
    for path in args.models:
        cap = Captioner(model_path=path, encoder=encoder)
        hyps = []
        for i in range(len(names)):
            hyps.append(cap.predict_caption(feats[i:i + 1].astype(np.float32), args.beam).split())
            print(f"\r  {os.path.basename(path)}: {i + 1}/{len(names)}", end="", flush=True)
        print()
        results[path] = hyps

    sm = SmoothingFunction().method1
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1 / 3, 1 / 3, 1 / 3, 0), (0.25,) * 4]
    print(f"\nBLEU on {len(names)} test images (beam {args.beam}):")
    print(f"{'model':45} " + "  ".join(f"BLEU-{k}" for k in range(1, 5)))
    for path, hyps in results.items():
        scores = [corpus_bleu(refs, hyps, weights=w, smoothing_function=sm) for w in weights]
        print(f"{path[-45:]:45} " + "  ".join(f"{s:6.3f}" for s in scores))

    for i, n in enumerate(names[: args.show]):
        print(f"\n{n}\n  reference: {' '.join(refs[i][0])}")
        for path, hyps in results.items():
            print(f"  {os.path.basename(path):12}: {' '.join(hyps[i])}")


if __name__ == "__main__":
    main()
