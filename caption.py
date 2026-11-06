"""Image captioning with the original 2021 model (ResNet50 encoder + LSTM decoder).

    from caption import Captioner
    cap = Captioner()
    cap.caption_image("photo.jpg")      # from a file
    cap.caption_frame(frame)            # from an OpenCV (BGR) webcam frame

Command line:
    python caption.py photo.jpg [more.jpg ...]
"""
import argparse
import os
import pickle
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "model_weights", "model_9.h5")
STORAGE_DIR = os.path.join(ROOT, "storage")
MAX_LEN = 35          # longest training caption (Flickr8k), fixed by the trained model
IMG_SIZE = (224, 224)


def build_encoder(weights="imagenet"):
    """ResNet50 pretrained on ImageNet, with the 1000-class output layer removed -> 2048-d feature vector.

    `weights` can be "imagenet" (downloaded automatically by Keras the first time)
    or a path to a local ResNet50 weights file for offline use.
    """
    if weights in (None, "imagenet"):
        base = ResNet50(weights=weights, input_shape=(*IMG_SIZE, 3))
    else:
        base = ResNet50(weights=None, input_shape=(*IMG_SIZE, 3))
        base.load_weights(weights)
    return keras.Model(base.input, base.layers[-2].output)


class Captioner:
    def __init__(self, model_path=MODEL_PATH, storage_dir=None, resnet_weights="imagenet"):
        if storage_dir is None:
            # vocabulary lives in <project>/storage next to <project>/model_weights/<model>
            sibling = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(model_path))), "storage")
            storage_dir = sibling if os.path.exists(os.path.join(sibling, "word_to_idx.pkl")) else STORAGE_DIR
        self.decoder = keras.models.load_model(model_path, compile=False)
        self.encoder = build_encoder(resnet_weights)
        with open(os.path.join(storage_dir, "word_to_idx.pkl"), "rb") as f:
            self.word_to_idx = pickle.load(f)
        with open(os.path.join(storage_dir, "idx_to_word.pkl"), "rb") as f:
            self.idx_to_word = pickle.load(f)
        # the decoder has two inputs: image features (2048) and the partial caption (max_len)
        self._img_first = self.decoder.inputs[0].shape[-1] == 2048
        seq_input = self.decoder.inputs[1] if self._img_first else self.decoder.inputs[0]
        self.max_len = seq_input.shape[1] or MAX_LEN

        # Compiled (graph-mode) versions of both networks: ~10x faster than calling the Keras models eagerly,
        # because the decoder is called once per generated word.
        self._encode_fn = tf.function(
            lambda x: self.encoder(x, training=False),
            input_signature=[tf.TensorSpec([None, *IMG_SIZE, 3], tf.float32)], autograph=False)
        self._decode_fn = tf.function(
            lambda img, seq: self.decoder([img, seq] if self._img_first else [seq, img], training=False),
            input_signature=[tf.TensorSpec([None, 2048], tf.float32), tf.TensorSpec([None, self.max_len], tf.int32)], autograph=False)

    # ---------- encoder ----------
    def _encode_rgb(self, rgb_224):
        x = preprocess_input(np.expand_dims(rgb_224.astype("float32"), 0))
        return self._encode_fn(tf.constant(x)).numpy().reshape(1, -1)

    def encode_image(self, path):
        img = keras.utils.load_img(path, target_size=IMG_SIZE)
        return self._encode_rgb(keras.utils.img_to_array(img))

    def encode_frame(self, frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._encode_rgb(cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA))

    # ---------- decoder ----------
    def _pad(self, seqs):
        out = np.zeros((len(seqs), self.max_len), dtype="int32")      # post-padding, as in training
        for i, s in enumerate(seqs):
            s = s[: self.max_len]
            out[i, : len(s)] = s
        return out

    def _next_word_probs(self, photo, seqs):
        img = np.repeat(np.asarray(photo, dtype="float32").reshape(1, -1), len(seqs), axis=0)
        return self._decode_fn(tf.constant(img), tf.constant(self._pad(seqs))).numpy()

    def predict_caption(self, photo, beam_width=1):
        """Greedy search (beam_width=1, same as the original) or beam search (beam_width > 1)."""
        start, end = self.word_to_idx["startseq"], self.word_to_idx["endseq"]
        if beam_width <= 1:
            seq = [start]
            for _ in range(self.max_len):
                word = int(self._next_word_probs(photo, [seq])[0].argmax())
                seq.append(word)
                if word == end:
                    break
        else:
            beams, finished = [([start], 0.0)], []
            for _ in range(self.max_len):
                probs = self._next_word_probs(photo, [s for s, _ in beams])
                candidates = []
                for (s, score), p in zip(beams, probs):
                    for w in np.argsort(p)[-beam_width:]:
                        candidates.append((s + [int(w)], score + float(np.log(p[w] + 1e-12))))
                candidates.sort(key=lambda c: c[1] / len(c[0]), reverse=True)   # length-normalised
                beams = []
                for s, score in candidates[:beam_width]:
                    (finished if s[-1] == end else beams).append((s, score))
                if not beams:
                    break
            pool = finished or beams
            seq = max(pool, key=lambda c: c[1] / len(c[0]))[0]
        words = [self.idx_to_word[i] for i in seq[1:] if i != end and i in self.idx_to_word]
        return " ".join(words)

    def caption_image(self, path, beam_width=1):
        return self.predict_caption(self.encode_image(path), beam_width)

    def caption_frame(self, frame_bgr, beam_width=1):
        return self.predict_caption(self.encode_frame(frame_bgr), beam_width)


_default = None


def caption_this_image(input_img):
    """Same function name the original 2021 Flask app used."""
    global _default
    if _default is None:
        _default = Captioner()
    return _default.caption_image(input_img)


def main():
    ap = argparse.ArgumentParser(description="Caption one or more images.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=MODEL_PATH, help="decoder model (.h5 original or a retrained .keras)")
    ap.add_argument("--beam", type=int, default=1, help="beam width (1 = greedy, as in the original)")
    ap.add_argument("--resnet-weights", default="imagenet",
                    help="'imagenet' (auto-download) or a path to local ResNet50 weights")
    args = ap.parse_args()
    cap = Captioner(model_path=args.model, resnet_weights=args.resnet_weights)
    for path in args.images:
        print(f"{path}: {cap.caption_image(path, args.beam)}")


if __name__ == "__main__":
    main()
