"""Dataset helpers shared by train.py and evaluate_bleu.py.

Supported caption files (format is detected automatically):
  * MS COCO      annotations/captions_train2017.json
  * Flickr30k    results.csv  ("image_name| comment_number| comment")  or Kaggle captions.txt
  * Flickr8k     Flickr8k.token.txt ("img.jpg#0<TAB>caption")          or Kaggle captions.txt ("image,caption")

Images can be a folder or a .zip file (e.g. COCO's train2017.zip), so huge datasets
don't have to be extracted first.
"""
import collections
import csv
import io
import json
import os
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image


def clean(sentence):
    """Same cleaning as the 2021 project: lowercase letters only, drop 1-letter words."""
    sentence = re.sub("[^a-z]+", " ", sentence.lower())
    return " ".join(w for w in sentence.split() if len(w) > 1)


def load_captions(path):
    """Return {image_file_name: [raw caption, ...]}."""
    caps = collections.defaultdict(list)
    if path.endswith(".json"):                                         # MS COCO
        with open(path) as f:
            data = json.load(f)
        id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"]:
            caps[id_to_file[ann["image_id"]]].append(ann["caption"])
        return dict(caps)

    with open(path, encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "\t" in first:                                              # Flickr8k.token.txt
            for line in f:
                if line.strip():
                    key, cap = line.rstrip("\n").split("\t", 1)
                    caps[key.split("#")[0]].append(cap)
        elif "|" in first:                                             # Flickr30k results.csv
            next(f)
            for line in f:
                parts = [p.strip() for p in line.rstrip("\n").split("|")]
                if len(parts) >= 3 and parts[2]:
                    caps[parts[0]].append("|".join(parts[2:]))
        else:                                                          # Kaggle captions.txt
            for row in csv.DictReader(f):
                caps[row["image"]].append(row["caption"])
    return dict(caps)


class ImageSource:
    """Read images by file name from a folder or from inside a .zip archive."""

    def __init__(self, path):
        self.path = path
        self.is_zip = path.lower().endswith(".zip")
        self._local = threading.local()
        if self.is_zip:
            with zipfile.ZipFile(path) as z:
                self.members = {os.path.basename(n): n for n in z.namelist() if not n.endswith("/")}
        else:
            self.members = {n: os.path.join(path, n) for n in os.listdir(path)}

    def __contains__(self, name):
        return name in self.members

    def _zip(self):
        if not hasattr(self._local, "zip"):          # one handle per thread: ZipFile is not thread-safe
            self._local.zip = zipfile.ZipFile(self.path)
        return self._local.zip

    def load(self, name, size=(224, 224)):
        """RGB float32 array resized like keras.utils.load_img(target_size=...) (nearest), as in 2021."""
        if self.is_zip:
            img = Image.open(io.BytesIO(self._zip().read(self.members[name])))
        else:
            img = Image.open(self.members[name])
        return np.asarray(img.convert("RGB").resize(size, Image.NEAREST), dtype="float32")


def extract_features(names, source, encoder, cache_dir, batch=64, part_size=4096, workers=8):
    """ResNet50 features (float16, N x 2048) for every image in `names`, in that order.

    Features are cached in cache_dir as part_XXXX.npy files of `part_size` images. Each part is written
    in one go, so the cache works on Google Drive and extraction resumes after a Colab disconnect.
    """
    import tensorflow as tf
    from keras.applications.resnet50 import preprocess_input

    os.makedirs(cache_dir, exist_ok=True)
    names_path = os.path.join(cache_dir, "feature_names.json")
    if os.path.exists(names_path) and json.load(open(names_path)) != names:
        raise SystemExit(f"{cache_dir} holds features for a different image list - use another --cache-dir")
    with open(names_path, "w") as f:
        json.dump(names, f)

    encode = tf.function(lambda x: encoder(x, training=False), autograph=False,
                         input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    n_parts = (len(names) + part_size - 1) // part_size
    parts = []
    with ThreadPoolExecutor(workers) as pool:
        for p in range(n_parts):
            path = os.path.join(cache_dir, f"part_{p:04d}.npy")
            part_names = names[p * part_size:(p + 1) * part_size]
            if os.path.exists(path):
                arr = np.load(path)
                if len(arr) == len(part_names):
                    parts.append(arr)
                    continue
            out = np.zeros((len(part_names), 2048), dtype=np.float16)
            for s in range(0, len(part_names), batch):
                chunk = part_names[s:s + batch]
                imgs = np.stack(list(pool.map(source.load, chunk)))
                out[s:s + len(chunk)] = encode(tf.constant(preprocess_input(imgs))).numpy()
                done = p * part_size + s + len(chunk)
                print(f"\r  encoded {done}/{len(names)} images", end="", flush=True)
            tmp = path + ".tmp.npy"
            np.save(tmp, out)
            os.replace(tmp, path)                    # appears only when complete
            parts.append(out)
    print(f"\r  features ready for {len(names)} images" + " " * 20)
    return np.concatenate(parts) if parts else np.zeros((0, 2048), np.float16)
