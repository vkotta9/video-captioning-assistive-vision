"""Flask web app: upload an image and get its caption (the 2021 "Deploy Image Captioning" app).

    python app.py          then open http://127.0.0.1:5000
"""
import os
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from caption import MODEL_PATH, Captioner

UPLOAD_DIR = os.path.join("static", "uploads")
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads
_captioner = None


def get_captioner():
    global _captioner
    if _captioner is None:
        _captioner = Captioner(model_path=os.environ.get("CAPTION_MODEL", MODEL_PATH),
                               resnet_weights=os.environ.get("RESNET_WEIGHTS", "imagenet"))
    return _captioner


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_file():
    img = request.files.get("image")
    if img is None or not img.filename:
        return render_template("index.html", error="Please choose an image first.")
    ext = os.path.splitext(img.filename)[1].lower()
    if ext not in ALLOWED:
        return render_template("index.html", error=f"Unsupported file type {ext!r}.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_{secure_filename(img.filename)}"
    path = os.path.join(UPLOAD_DIR, name)
    img.save(path)

    caption = get_captioner().caption_image(path)
    results = {"image": "/" + path.replace(os.sep, "/"), "description": caption}
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=False)
