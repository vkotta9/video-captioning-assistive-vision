# A Systematic Video Captioning for Assistive Vision

Live video captioning with an AI voice. The laptop camera is captured continuously; every 3 seconds a frame is
described in a sentence by a **CNN (ResNet50) + RNN (LSTM)** model, and the sentence is **spoken aloud**, so a
visually impaired person can hear what is in front of the camera.

> B.Tech mini-project, Department of CSE, GMR Institute of Technology, Rajam (2021).
> Team: K. Vijay Kumar, P. Vijay Lakshmana Rao, M. Tejaswi, K. Naga Nimesh. Guide: Dr. K. Lakshmana Rao.
>
> This repository is a 2026 rebuild of the original project. It uses the **original trained model**
> (`model_weights/model_9.h5`) and vocabulary, with the code updated to run on current Python / TensorFlow.

```
 laptop camera ──► frame every 3 s ──► ResNet50 (ImageNet) ──► 2048-d features ─┐
                                                                                 ├─► LSTM decoder ─► "brown dog is running through the grass" ─► 🔊 pyttsx3 voice
                                      startseq + words so far ─► GloVe embedding ┘        (word by word until endseq)
```

## Example output

Captions produced by the original model on test images in this rebuild:

```
brown dog is running through the grass
boy in blue shirt is running through the grass
race car in race
two people are running on the beach
man and woman pose for picture          <- live webcam, 2021 notebook output
```

## Quick start

Python 3.10 – 3.12 recommended.

```bash
git clone https://github.com/vkotta9/video-captioning-assistive-vision.git
cd video-captioning-assistive-vision
python -m venv venv
venv\Scripts\activate            # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

The first run downloads the ResNet50 ImageNet weights (~100 MB) automatically.

### 1. Live video captioning with voice (main demo)

```bash
python live_caption.py
```

A window shows the camera with the latest caption at the bottom; each new caption is spoken.
Press **q** or **Esc** to stop.

| Option | Meaning |
|---|---|
| `--interval 1` | caption every second (default 3, as in the report) |
| `--beam 1` | greedy decoding exactly like the 2021 code (default 3 = beam search, fewer repeated words) |
| `--source video.mp4` | use a video file instead of the webcam (`--source 1` for a second camera) |
| `--no-voice` | captions on screen only |
| `--voice 1` | choose a voice (Windows SAPI5: 0 = male, 1 = female) |
| `--rate 140` | speaking speed |
| `--repeat` | speak the caption even when it is the same as the previous one |
| `--save-frames` | save every captioned frame to `data/` like the original notebook |

Linux needs `espeak` for the voice (`sudo apt install espeak-ng`); Windows and macOS work out of the box.

### 2. Caption an image from the command line

```bash
python caption.py photo.jpg another.jpg
```

### 3. Web app (upload an image, get its caption)

```bash
python app.py
```
Open http://127.0.0.1:5000, choose an image and press **Submit**.

## How it works

**Encoder (CNN).** ResNet50 pretrained on ImageNet with its final 1000-class layer removed. Every frame is resized
to 224 × 224 and turned into a 2048-dimensional feature vector.

**Decoder (RNN).** The trained model (`model_9.h5`) has two inputs:

| Branch | Layers |
|---|---|
| image | 2048 features → Dropout 0.3 → Dense 256 (ReLU) |
| caption so far | 35 word indices → Embedding 50-d (GloVe, frozen) → Dropout 0.3 → LSTM 256 |
| merge | add → Dense 256 (ReLU) → Dense 1848 (softmax = next word) |

Captions start with `startseq`; the model predicts the next word, which is appended, until `endseq` or 35 words.
The 2021 code took the single most likely word each time (greedy). The rebuild uses beam search by default:
it keeps the 3 best partial sentences and returns the most likely complete one, which avoids repeats such as
"leather jacket and leather jacket".

**Speed.** Both networks run as compiled TensorFlow graphs (`tf.function`), so one caption takes about 0.1 s
(greedy) to 0.3 s (beam search) on a laptop CPU, about 10x faster than calling the Keras models directly.

**Dataset.** Flickr8k (8,091 images, 5 captions each). Captions were lower-cased, punctuation and 1-letter words
removed, and words seen more than 10 times kept: 1,845 words + `startseq` + `endseq` (`storage/*.pkl`).

**Voice.** `pyttsx3`, an offline text-to-speech library (SAPI5 on Windows, espeak on Linux; `say` on macOS).
Captioning and speech run in background threads so the video never freezes.

## Re-training on a bigger dataset (MS COCO)

The 2021 model learned from only 6,000 Flickr8k photos with a 1,848-word vocabulary, which limits how
accurate and varied its captions can be. `training/train.py` trains **the same CNN + LSTM design** on
MS COCO (~118,000 images, ~590,000 captions, ~9,000-word vocabulary), Flickr30k or Flickr8k.

**Easiest: Google Colab (free GPU).** Open
[`training/train_colab.ipynb`](training/train_colab.ipynb) in Colab
([open directly](https://colab.research.google.com/github/vkotta9/video-captioning-assistive-vision/blob/main/training/train_colab.ipynb)),
choose *Runtime → Change runtime type → T4 GPU* and run all cells. It downloads COCO, trains, compares the
new model with the 2021 one using BLEU scores, and downloads `retrained_coco.zip`. Progress is saved to Google
Drive, so a disconnected session resumes where it stopped.

Unzip `retrained_coco.zip` inside this project folder, then:

```bash
python live_caption.py --model retrained_coco/model_weights/best.keras
python caption.py photo.jpg --model retrained_coco/model_weights/best.keras
CAPTION_MODEL=retrained_coco/model_weights/best.keras python app.py   # Windows: set CAPTION_MODEL=... then python app.py
```

**Command line (any machine with a GPU):**

```bash
# MS COCO 2017 - images are read straight from the zip, no need to extract 19 GB
python training/train.py --captions annotations/captions_train2017.json --images train2017.zip \
    --glove glove.6B.50d.txt --out retrained_coco

# Flickr30k (Kaggle results.csv) or Flickr8k (reproduces the 2021 setup)
python training/train.py --captions results.csv --images flickr30k_images --glove glove.6B.50d.txt --out retrained_f30k
python training/train.py --captions Flickr8k.token.txt --images Flicker8k_Dataset \
    --train-list Flickr_8k.trainImages.txt --test-list Flickr_8k.testImages.txt \
    --glove glove.6B.50d.txt --images-per-batch 3 --out retrained_flickr8k

# BLEU-1..4 for several models on the same test images
python training/evaluate_bleu.py --captions annotations/captions_train2017.json --images train2017.zip \
    --models model_weights/model_9.h5 retrained_coco/model_weights/best.keras \
    --test-list retrained_coco/storage/test_images.txt
```

What the training script does: cleans captions exactly like the 2021 code, keeps words seen more than 10 times,
caches ResNet50 features (resumable), holds out 1,000 validation and 1,000 test images, saves `best.keras` by
validation loss, halves the learning rate when validation stops improving and stops early after 3 bad epochs.
Re-running the same command resumes an interrupted run. The original `model_weights/model_9.h5` is never touched.

## Repository layout

```
live_caption.py        live webcam captioning + voice (main demo)
caption.py             Captioner: ResNet50 encoder + LSTM decoder, greedy / beam search
app.py, templates/     Flask web app for single images
model_weights/         model_9.h5 - the original trained decoder (2021)
storage/               word_to_idx.pkl, idx_to_word.pkl - original vocabulary
training/              train.py (COCO / Flickr30k / Flickr8k), evaluate_bleu.py, train_colab.ipynb
original_2021/         the original code exactly as it was (notebook, caption.py, app.py)
```

## What changed from the 2021 code

* Runs on TensorFlow 2.16+ / Keras 3 (removed `_make_predict_function`, `keras.preprocessing` imports).
* The webcam loop captions every 3 seconds, as described in the report. In the original notebook
  `currentframe += 1` sat outside the loop, so every frame was captioned.
* Video no longer freezes while a caption is generated or spoken (background threads).
* About 10x faster captioning (compiled `tf.function` graphs) and optional beam search.
* Caption shown on the video window; command-line options for interval, voice, source and more.
* Flask app: safe upload names, file-type check, uploads kept in `static/uploads/`.

## References

* L. Gao, Z. Guo, H. Zhang, X. Xu, H. T. Shen, "Video Captioning with Attention-based LSTM and Semantic Consistency," *IEEE Transactions on Multimedia*, 2017.
* S. Amirian, K. Rasheed, T. R. Taha, H. R. Arabnia, "Automatic Image and Video Caption Generation With Deep Learning: A Concise Review and Algorithmic Overlap," *IEEE Access*, 2020.
* M. Amaresh, S. Chitrakala, "Video Captioning using Deep Learning: An Overview of Methods, Datasets and Metrics," ICCSP 2019.
* V. Iashin, E. Rahtu, "Multi-modal Dense Video Captioning," CVPR Workshops 2020.
* N. Guo, H. Liu, L. Jiang, "Attention-based Visual-Audio Fusion for Video Caption Generation," ICARM 2019.
* V. P. Campos, L. M. G. Gonçalves, T. M. U. de Araújo, "Applying Audio Description for Context Understanding of Surveillance Videos by People With Visual Impairments," IEEE AVSS 2018.
* K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016.
* M. Hodosh, P. Young, J. Hockenmaier, "Framing Image Description as a Ranking Task" (Flickr8k), JAIR 2013.
