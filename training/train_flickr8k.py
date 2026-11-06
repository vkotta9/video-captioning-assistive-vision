"""Re-train the captioning model on Flickr8k (reconstruction of the lost 2021 training notebook).

It reproduces the exact architecture and vocabulary of model_weights/model_9.h5:
  * captions cleaned to lowercase letters, 1-letter words dropped, wrapped in startseq ... endseq
  * vocabulary = words appearing more than 10 times (1845 words) + startseq + endseq -> 1848 with padding
  * image features: ResNet50 (ImageNet) without its last layer -> 2048-d vector
  * decoder: GloVe 50-d frozen embedding -> Dropout 0.3 -> LSTM 256  (+)  Dropout 0.3 -> Dense 256
             -> add -> Dense 256 relu -> Dense vocab softmax
  * captions post-padded to max_len (35 for Flickr8k), greedy decoding at inference

Dataset (from Kaggle "Flickr8k" or the original release):
    Flickr8k.token.txt (or captions.txt), the image folder, optionally Flickr_8k.trainImages.txt
GloVe:  glove.6B.50d.txt  (https://nlp.stanford.edu/projects/glove/) - optional but recommended

    python training/train_flickr8k.py --captions Flickr8k.token.txt --images Flicker8k_Dataset \
        --train-list Flickr_8k.trainImages.txt --glove glove.6B.50d.txt --epochs 20

Outputs go to retrained/model_weights/model_<epoch>.keras and retrained/storage/*.pkl, so the original
model is never overwritten. Use a retrained model with:
    python live_caption.py --model retrained/model_weights/model_9.keras
"""
import argparse
import collections
import csv
import os
import pickle
import random
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import keras
from keras import layers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from caption import build_encoder, IMG_SIZE  # noqa: E402
from keras.applications.resnet50 import preprocess_input  # noqa: E402


# ---------------------------------------------------------------- data
def load_captions(path):
    caps = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "\t" in first:                                  # Flickr8k.token.txt: "img.jpg#0<TAB>caption"
            for line in f:
                if line.strip():
                    key, cap = line.rstrip("\n").split("\t", 1)
                    caps[key.split("#")[0]].append(cap)
        else:                                               # Kaggle captions.txt: "image,caption"
            for row in csv.DictReader(f):
                caps[row["image"]].append(row["caption"])
    return caps


def clean(sentence):
    sentence = re.sub("[^a-z]+", " ", sentence.lower())
    return " ".join(w for w in sentence.split() if len(w) > 1)


def build_vocab(all_captions, threshold=10):
    counts = collections.Counter(w for c in all_captions for w in c.split())
    words = [w for w, n in sorted(counts.items(), key=lambda x: -x[1]) if n > threshold]
    word_to_idx = {w: i + 1 for i, w in enumerate(words)}
    for tok in ("startseq", "endseq"):
        word_to_idx[tok] = len(word_to_idx) + 1
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word


def encode_images(names, image_dir, encoder, batch=32):
    feats = {}
    for s in range(0, len(names), batch):
        chunk = names[s:s + batch]
        arr = np.stack([keras.utils.img_to_array(keras.utils.load_img(os.path.join(image_dir, n), target_size=IMG_SIZE))
                        for n in chunk])
        out = encoder(preprocess_input(arr), training=False).numpy()
        feats.update({n: out[i] for i, n in enumerate(chunk)})
        print(f"\r  encoded {min(s + batch, len(names))}/{len(names)} images", end="", flush=True)
    print()
    return feats


def embedding_matrix(word_to_idx, vocab_size, glove_path, dim=50):
    m = np.zeros((vocab_size, dim), dtype="float32")
    if not glove_path:
        return None
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in word_to_idx and len(parts) == dim + 1:
                m[word_to_idx[parts[0]]] = np.asarray(parts[1:], dtype="float32")
    return m


# ---------------------------------------------------------------- model (identical to model_9.h5)
def build_model(vocab_size, max_len, emb=None, emb_dim=50):
    input_img_features = keras.Input(shape=(2048,))
    inp_img1 = layers.Dropout(0.3)(input_img_features)
    inp_img2 = layers.Dense(256, activation="relu")(inp_img1)

    input_captions = keras.Input(shape=(max_len,))
    embed = layers.Embedding(vocab_size, emb_dim, mask_zero=True, trainable=emb is None)
    inp_cap1 = embed(input_captions)
    inp_cap2 = layers.Dropout(0.3)(inp_cap1)
    inp_cap3 = layers.LSTM(256)(inp_cap2)

    decoder1 = layers.add([inp_img2, inp_cap3])
    decoder2 = layers.Dense(256, activation="relu")(decoder1)
    outputs = layers.Dense(vocab_size, activation="softmax")(decoder2)

    model = keras.Model(inputs=[input_img_features, input_captions], outputs=outputs)
    if emb is not None:
        embed.set_weights([emb])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


class CaptionBatches(keras.utils.PyDataset):
    """Each caption 'startseq a dog runs endseq' -> ([img, 'startseq'] -> 'a'), ([img, 'startseq a'] -> 'dog'), ..."""

    def __init__(self, captions, features, word_to_idx, max_len, images_per_batch=3, **kw):
        super().__init__(**kw)
        self.names = list(captions)
        self.captions, self.features = captions, features
        self.w2i, self.max_len, self.ipb = word_to_idx, max_len, images_per_batch

    def __len__(self):
        return int(np.ceil(len(self.names) / self.ipb))

    def on_epoch_end(self):
        random.shuffle(self.names)

    def __getitem__(self, i):
        X1, X2, y = [], [], []
        for name in self.names[i * self.ipb:(i + 1) * self.ipb]:
            for desc in self.captions[name]:
                seq = [self.w2i[w] for w in desc.split() if w in self.w2i]
                for t in range(1, len(seq)):
                    inp = seq[:t][: self.max_len]
                    X1.append(self.features[name])
                    X2.append(inp + [0] * (self.max_len - len(inp)))
                    y.append(seq[t])
        return (np.array(X1, "float32"), np.array(X2, "int32")), np.array(y, "int32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="Flickr8k.token.txt or captions.txt")
    ap.add_argument("--images", required=True, help="folder with the Flickr8k images")
    ap.add_argument("--train-list", help="Flickr_8k.trainImages.txt (default: random 6000 images)")
    ap.add_argument("--glove", help="glove.6B.50d.txt (if omitted, the embedding is learned)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--images-per-batch", type=int, default=3)
    ap.add_argument("--threshold", type=int, default=10, help="keep words seen more than this many times")
    ap.add_argument("--resnet-weights", default="imagenet")
    ap.add_argument("--out", default=os.path.join(ROOT, "retrained"),
                    help="folder to write model_weights/ and storage/ into (default: retrained/)")
    args = ap.parse_args()

    raw = load_captions(args.captions)
    raw = {k: v for k, v in raw.items() if os.path.exists(os.path.join(args.images, k))}
    descriptions = {k: [clean(c) for c in v] for k, v in raw.items()}

    if args.train_list:
        with open(args.train_list) as f:
            train_names = [l.strip() for l in f if l.strip() in descriptions]
    else:
        names = sorted(descriptions)
        random.Random(42).shuffle(names)
        train_names = names[:6000] if len(names) > 7000 else names[: int(len(names) * 0.85)]

    word_to_idx, idx_to_word = build_vocab([c for cs in descriptions.values() for c in cs], args.threshold)
    vocab_size = len(idx_to_word) + 1
    train_desc = {n: [f"startseq {c} endseq" for c in descriptions[n]] for n in train_names}
    max_len = max(len(c.split()) for cs in train_desc.values() for c in cs)
    print(f"images: {len(descriptions)}  train: {len(train_names)}  vocab_size: {vocab_size}  max_len: {max_len}")

    print("Encoding training images with ResNet50 ...")
    features = encode_images(train_names, args.images, build_encoder(args.resnet_weights))

    emb = embedding_matrix(word_to_idx, vocab_size, args.glove)
    model = build_model(vocab_size, max_len, emb)
    model.summary()

    os.makedirs(os.path.join(args.out, "model_weights"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "storage"), exist_ok=True)
    with open(os.path.join(args.out, "storage", "word_to_idx.pkl"), "wb") as f:
        pickle.dump(word_to_idx, f)
    with open(os.path.join(args.out, "storage", "idx_to_word.pkl"), "wb") as f:
        pickle.dump(idx_to_word, f)
    with open(os.path.join(args.out, "storage", "train_images.txt"), "w") as f:
        f.write("\n".join(train_names))

    data = CaptionBatches(train_desc, features, word_to_idx, max_len, args.images_per_batch)
    for epoch in range(args.epochs):
        if epoch == 10:                                   # lower the learning rate for the second half
            model.optimizer.learning_rate.assign(1e-4)
        model.fit(data, epochs=1, verbose=1)
        path = os.path.join(args.out, "model_weights", f"model_{epoch}.keras")
        model.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
