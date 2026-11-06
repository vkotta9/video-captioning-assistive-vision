"""Train the CNN (ResNet50) + RNN (LSTM) captioning model - same design as the 2021 model_9.h5 -
on Flickr8k, Flickr30k or MS COCO.

  * captions cleaned to lowercase letters, 1-letter words dropped, wrapped in startseq ... endseq
  * vocabulary = words seen more than --threshold times in the training captions
  * image -> ResNet50 (ImageNet, last layer removed) -> 2048 features -> Dropout 0.3 -> Dense 256
  * caption so far -> Embedding 50 (GloVe, frozen, if --glove is given) -> Dropout 0.3 -> LSTM 256
  * add -> Dense 256 relu -> Dense vocab softmax (next word)

MS COCO (recommended, ~118k images; run on a GPU, e.g. Google Colab - see training/train_colab.ipynb):
    python training/train.py --captions annotations/captions_train2017.json --images train2017.zip \
        --glove glove.6B.50d.txt --out retrained_coco

Flickr8k (reproduces the original model):
    python training/train.py --captions Flickr8k.token.txt --images Flicker8k_Dataset \
        --train-list Flickr_8k.trainImages.txt --test-list Flickr_8k.testImages.txt \
        --glove glove.6B.50d.txt --images-per-batch 3 --out retrained_flickr8k

Output (in --out):  model_weights/best.keras (lowest validation loss), model_weights/model_<epoch>.keras,
storage/word_to_idx.pkl + idx_to_word.pkl, storage/{train,val,test}_images.txt, history.csv.
Re-run the same command to resume after an interruption.

Use the result with:   python live_caption.py --model retrained_coco/model_weights/best.keras
"""
import argparse
import csv
import json
import os
import pickle
import random
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import keras
from keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from caption import build_encoder  # noqa: E402
from data import ImageSource, clean, extract_features, load_captions  # noqa: E402


# ---------------------------------------------------------------- vocabulary
def build_vocab(captions, threshold):
    counts = {}
    for c in captions:
        for w in c.split():
            counts[w] = counts.get(w, 0) + 1
    words = [w for w, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if n > threshold]
    word_to_idx = {w: i + 1 for i, w in enumerate(words)}           # 0 is padding
    for tok in ("startseq", "endseq"):
        word_to_idx[tok] = len(word_to_idx) + 1
    return word_to_idx, {i: w for w, i in word_to_idx.items()}


def glove_matrix(word_to_idx, vocab_size, path, dim):
    m = np.zeros((vocab_size, dim), dtype="float32")
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in word_to_idx and len(parts) == dim + 1:
                m[word_to_idx[parts[0]]] = np.asarray(parts[1:], dtype="float32")
                found += 1
    print(f"GloVe vectors found for {found}/{vocab_size - 1} words")
    return m


# ---------------------------------------------------------------- model (same design as model_9.h5)
def build_model(vocab_size, max_len, emb=None, emb_dim=50, units=256, lr=1e-3):
    input_img_features = keras.Input(shape=(2048,), name="image_features")
    inp_img1 = layers.Dropout(0.3)(input_img_features)
    inp_img2 = layers.Dense(units, activation="relu")(inp_img1)

    input_captions = keras.Input(shape=(max_len,), dtype="int32", name="caption_so_far")
    embed = layers.Embedding(vocab_size, emb_dim, mask_zero=True, trainable=emb is None)
    inp_cap1 = embed(input_captions)
    inp_cap2 = layers.Dropout(0.3)(inp_cap1)
    inp_cap3 = layers.LSTM(units)(inp_cap2)

    decoder1 = layers.add([inp_img2, inp_cap3])
    decoder2 = layers.Dense(units, activation="relu")(decoder1)
    outputs = layers.Dense(vocab_size, activation="softmax", name="next_word")(decoder2)

    model = keras.Model(inputs=[input_img_features, input_captions], outputs=outputs)
    if emb is not None:
        embed.set_weights([emb])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr))
    return model


# ---------------------------------------------------------------- training pairs
def encode_captions(names, descriptions, word_to_idx, max_len, row_of):
    """Token matrix (n_captions, max_len + 1), post-padded, plus the feature row of each caption."""
    end = word_to_idx["endseq"]
    toks, rows = [], []
    for n in names:
        for d in descriptions[n]:
            ids = [word_to_idx[w] for w in f"startseq {d} endseq".split() if w in word_to_idx]
            if len(ids) > max_len + 1:
                ids = ids[:max_len] + [end]
            toks.append(ids + [0] * (max_len + 1 - len(ids)))
            rows.append(row_of[n])
    return np.array(toks, dtype=np.int32), np.array(rows, dtype=np.int64)


class CaptionPairs(keras.utils.PyDataset):
    """Caption 'startseq a dog runs endseq' -> ([img, 'startseq'] -> 'a'), ([img, 'startseq a'] -> 'dog'), ..."""

    def __init__(self, tokens, rows, features, batch_size, shuffle=True, **kw):
        super().__init__(**kw)
        self.tokens, self.rows, self.features = tokens, rows, features
        self.max_len = tokens.shape[1] - 1
        self.batch_size, self.shuffle = batch_size, shuffle
        lengths = (tokens > 0).sum(axis=1)
        self.samples = np.stack([np.repeat(np.arange(len(tokens)), lengths - 1),
                                 np.concatenate([np.arange(1, n) for n in lengths])], axis=1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __getitem__(self, i):
        b = self.samples[i * self.batch_size:(i + 1) * self.batch_size]
        ci, t = b[:, 0], b[:, 1]
        seq = self.tokens[ci, :self.max_len] * (np.arange(self.max_len)[None, :] < t[:, None])
        return (self.features[self.rows[ci]].astype(np.float32), seq.astype(np.int32)), self.tokens[ci, t]


def read_list(path, valid):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() in valid]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", required=True, help="COCO captions json, Flickr30k results.csv or Flickr8k token file")
    ap.add_argument("--images", required=True, help="image folder or .zip (e.g. train2017.zip)")
    ap.add_argument("--out", required=True, help="output folder (use a Google Drive folder on Colab)")
    ap.add_argument("--cache-dir", help="where to cache ResNet50 features (default: <out>/features)")
    ap.add_argument("--train-list", help="optional file with training image names")
    ap.add_argument("--test-list", help="optional file with test image names")
    ap.add_argument("--val-size", type=int, default=1000, help="images held out for validation")
    ap.add_argument("--test-size", type=int, default=1000, help="images held out for testing (if no --test-list)")
    ap.add_argument("--max-images", type=int, default=0, help="use only N images (0 = all), for quick runs")
    ap.add_argument("--glove", help="glove.6B.50d.txt; if omitted the embedding is learned")
    ap.add_argument("--threshold", type=int, default=10, help="keep words seen more than this many times")
    ap.add_argument("--max-len", type=int, default=35, help="longest caption in words incl. start/end (2021: 35)")
    ap.add_argument("--units", type=int, default=256, help="LSTM / Dense size (2021: 256)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024, help="(image, partial caption) pairs per batch")
    ap.add_argument("--images-per-batch", type=int, default=0,
                    help="2021-style batching by images (e.g. 3); overrides --batch-size with ~images*55 pairs")
    ap.add_argument("--patience", type=int, default=3, help="stop after N epochs without validation improvement")
    ap.add_argument("--resnet-weights", default="imagenet")
    args = ap.parse_args()

    out_w, out_s = os.path.join(args.out, "model_weights"), os.path.join(args.out, "storage")
    os.makedirs(out_w, exist_ok=True)
    os.makedirs(out_s, exist_ok=True)

    # ---------------- captions and splits
    source = ImageSource(args.images)
    raw = {k: v for k, v in load_captions(args.captions).items() if k in source}
    if not raw:
        raise SystemExit("No captioned images found - check --captions and --images.")
    descriptions = {k: [clean(c) for c in v] for k, v in raw.items()}

    split_file = os.path.join(out_s, "splits.json")
    if os.path.exists(split_file):                                  # resume: keep the same split
        splits = json.load(open(split_file))
    else:
        names = sorted(descriptions)
        random.Random(42).shuffle(names)
        if args.max_images:
            names = names[:args.max_images]
        test = read_list(args.test_list, descriptions) if args.test_list else names[:args.test_size]
        rest = [n for n in names if n not in set(test)]
        if args.train_list:
            train_set = set(read_list(args.train_list, descriptions))
            train = [n for n in rest if n in train_set]
            val = [n for n in rest if n not in train_set][:args.val_size]
        else:
            val, train = rest[:args.val_size], rest[args.val_size:]
        splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
        json.dump(splits, open(split_file, "w"))
        for k, v in splits.items():
            with open(os.path.join(out_s, f"{k}_images.txt"), "w") as f:
                f.write("\n".join(v))

    vocab_file = os.path.join(out_s, "word_to_idx.pkl")
    if os.path.exists(vocab_file):
        word_to_idx = pickle.load(open(vocab_file, "rb"))
        idx_to_word = pickle.load(open(os.path.join(out_s, "idx_to_word.pkl"), "rb"))
    else:
        word_to_idx, idx_to_word = build_vocab(
            [c for n in splits["train"] for c in descriptions[n]], args.threshold)
        pickle.dump(word_to_idx, open(vocab_file, "wb"))
        pickle.dump(idx_to_word, open(os.path.join(out_s, "idx_to_word.pkl"), "wb"))
    vocab_size = len(idx_to_word) + 1
    observed = max(len(c.split()) + 2 for n in splits["train"] for c in descriptions[n])
    max_len = min(observed, args.max_len)
    print(f"images train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}   "
          f"vocab_size: {vocab_size}   max_len: {max_len}")

    # ---------------- image features (cached, resumable)
    feat_names = splits["train"] + splits["val"]
    print("Extracting ResNet50 features ...")
    features = extract_features(feat_names, source, build_encoder(args.resnet_weights),
                                args.cache_dir or os.path.join(args.out, "features"))
    row_of = {n: i for i, n in enumerate(feat_names)}

    batch = args.images_per_batch * 55 if args.images_per_batch else args.batch_size
    train_tok, train_rows = encode_captions(splits["train"], descriptions, word_to_idx, max_len, row_of)
    val_tok, val_rows = encode_captions(splits["val"], descriptions, word_to_idx, max_len, row_of)
    train_data = CaptionPairs(train_tok, train_rows, features, batch, workers=4)
    val_data = CaptionPairs(val_tok, val_rows, features, batch, shuffle=False, workers=4)
    print(f"training pairs per epoch: {len(train_data.samples):,}   batches: {len(train_data):,}")

    # ---------------- model (new or resumed)
    hist_path = os.path.join(args.out, "history.csv")
    history = list(csv.DictReader(open(hist_path))) if os.path.exists(hist_path) else []
    start_epoch = len(history)
    last = os.path.join(out_w, f"model_{start_epoch - 1}.keras")
    if history and os.path.exists(last):
        print(f"Resuming from {last}")
        model = keras.models.load_model(last)
    else:
        history, start_epoch = [], 0
        emb = glove_matrix(word_to_idx, vocab_size, args.glove, 50) if args.glove else None
        model = build_model(vocab_size, max_len, emb, units=args.units)
    model.summary()

    best = min((float(h["val_loss"]) for h in history), default=float("inf"))
    bad = 0
    for h in reversed(history):                                     # epochs since last improvement
        if float(h["val_loss"]) <= best:
            break
        bad += 1

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
        print(f"\nEpoch {epoch + 1}/{args.epochs}  (learning rate {lr:.2e})")
        h = model.fit(train_data, validation_data=val_data, epochs=1, verbose=1)
        loss, val_loss = h.history["loss"][-1], h.history["val_loss"][-1]

        if val_loss < best:
            best, bad = val_loss, 0
            model.save(os.path.join(out_w, "best.keras"))
            print(f"  new best model (val_loss {val_loss:.4f}) -> {out_w}/best.keras")
        else:
            bad += 1
            model.optimizer.learning_rate.assign(lr * 0.5)           # reduce LR on plateau
            print(f"  no improvement ({bad}/{args.patience}); learning rate halved")
        # checkpoint for resuming (saved after the LR change so a resumed run keeps it)
        model.save(os.path.join(out_w, f"model_{epoch}.keras"))
        old = os.path.join(out_w, f"model_{epoch - 2}.keras")       # keep only the last two epochs
        if os.path.exists(old):
            os.remove(old)

        history.append({"epoch": epoch, "loss": f"{loss:.4f}", "val_loss": f"{val_loss:.4f}",
                        "lr": f"{lr:.2e}", "minutes": f"{(time.time() - t0) / 60:.1f}"})
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        if bad >= args.patience:
            print("Early stopping.")
            break

    print(f"\nDone. Best validation loss {best:.4f}. Model: {out_w}/best.keras")


if __name__ == "__main__":
    main()
