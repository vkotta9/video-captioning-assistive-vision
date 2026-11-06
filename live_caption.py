"""Live video captioning with an AI voice - the main demo of the project.

The laptop camera is captured continuously. Every few seconds (3 s, as in the 2021 report)
the current frame is captioned by ResNet50 + LSTM and the caption is spoken aloud.

    python live_caption.py                     # webcam 0, caption every 3 seconds, with voice
    python live_caption.py --interval 5        # caption every 5 seconds
    python live_caption.py --source video.mp4  # use a video file instead of the webcam
    python live_caption.py --no-voice          # captions on screen only

Press  q  or  Esc  in the video window to stop.

Compared with the 2021 notebook, captioning and speech run in background threads, so the
video keeps playing while a caption is generated or spoken.
"""
import argparse
import os
import platform
import queue
import threading
import time

import cv2

from caption import MODEL_PATH, Captioner


class Speaker:
    """Offline text-to-speech with pyttsx3 (Windows SAPI5 / espeak) or `say` on macOS, in its own thread."""

    def __init__(self, rate=160, voice=None, enabled=True):
        self.rate, self.voice, self.enabled = rate, voice, enabled
        self._q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        if not self.enabled or not text:
            return
        while not self._q.empty():           # only the newest caption matters
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(text)

    def _speak(self, text):
        if platform.system() == "Darwin":
            import subprocess
            subprocess.run(["say", "-r", str(self.rate), text], check=False)
            return
        import pyttsx3
        # a new engine per sentence avoids pyttsx3 hanging when runAndWait() is reused from a thread
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        if self.voice is not None:
            voices = engine.getProperty("voices")
            engine.setProperty("voice", voices[self.voice % len(voices)].id)  # 0 = male, 1 = female on Windows
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _run(self):
        if platform.system() == "Windows":
            try:
                import comtypes
                comtypes.CoInitialize()      # SAPI5 needs COM initialised in this thread
            except Exception:
                pass
        while True:
            text = self._q.get()
            try:
                self._speak(text)
            except Exception as e:           # audio problems must never stop the video
                print(f"[voice] {e}")


def draw_caption(frame, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, text or "...", (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "q / Esc: quit", (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Live video captioning with voice output.")
    ap.add_argument("--source", default="0", help="camera index (0 = laptop webcam) or a video file path")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between captions (default 3)")
    ap.add_argument("--no-voice", action="store_true", help="do not speak captions")
    ap.add_argument("--voice", type=int, default=None, help="voice index (Windows: 0 male, 1 female)")
    ap.add_argument("--rate", type=int, default=160, help="speech rate in words per minute")
    ap.add_argument("--repeat", action="store_true", help="speak a caption even if it is the same as the last one")
    ap.add_argument("--save-frames", action="store_true", help="save each captioned frame to ./data like the original")
    ap.add_argument("--no-window", action="store_true", help="run without a video window (print captions only)")
    ap.add_argument("--max-captions", type=int, default=0, help="stop after N captions (0 = run until quit)")
    ap.add_argument("--model", default=MODEL_PATH, help="decoder model (.h5 original or a retrained .keras)")
    ap.add_argument("--resnet-weights", default="imagenet", help="'imagenet' or path to local ResNet50 weights")
    args = ap.parse_args()

    print("Loading models (the first run downloads ResNet50 ImageNet weights, ~100 MB)...")
    captioner = Captioner(model_path=args.model, resnet_weights=args.resnet_weights)
    speaker = Speaker(rate=args.rate, voice=args.voice, enabled=not args.no_voice)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}. Is the camera in use by another app?")
    is_file = not isinstance(source, int)
    frame_delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25) if is_file else 0
    if args.save_frames:
        os.makedirs("data", exist_ok=True)

    state = {"caption": "", "count": 0, "busy": False}
    jobs = queue.Queue(maxsize=1)

    def caption_worker():
        last_spoken = None
        while True:
            item = jobs.get()
            if item is None:
                break
            idx, frame = item
            t0 = time.time()
            text = captioner.caption_frame(frame)
            state["caption"] = text
            state["count"] += 1
            print(f"[{time.strftime('%H:%M:%S')}] frame {idx}: {text}   ({time.time() - t0:.1f}s)", flush=True)
            if args.repeat or text != last_spoken:
                speaker.say(text)
                last_spoken = text
            state["busy"] = False

    worker = threading.Thread(target=caption_worker, daemon=True)
    worker.start()

    frame_idx, last_capture = 0, -1e9
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            if now - last_capture >= args.interval and not state["busy"]:
                last_capture = now
                state["busy"] = True
                if args.save_frames:
                    cv2.imwrite(os.path.join("data", f"frame{frame_idx}.jpg"), frame)
                jobs.put((frame_idx, frame.copy()))
            frame_idx += 1

            if not args.no_window:
                draw_caption(frame, state["caption"])
                cv2.imshow("LIVE - Video Captioning for Assistive Vision", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if args.max_captions and state["count"] >= args.max_captions:
                break
            if frame_delay:
                time.sleep(frame_delay)
    finally:
        # let an in-progress caption finish so it is not lost
        while state["busy"] and worker.is_alive():
            time.sleep(0.05)
        jobs.put(None)
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
