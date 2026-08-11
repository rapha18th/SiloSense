"""Minimal local inference server — run this INSIDE Termux's proot-distro
Ubuntu on the phone. Serves the static demo page and a /predict endpoint that
runs the INT8 (Arm-optimized delegate) model on an uploaded audio clip.

    cd SiloSense/app
    python3 server.py
    # then open http://127.0.0.1:5000 in Chrome on the same phone
"""
import io
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from features import logmel, resample_linear, SAMPLE_RATE  # noqa: E402

import soundfile as sf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "silosense_int8.onnx")

app = Flask(__name__, static_folder=APP_DIR, static_url_path="")

_session = None
_providers_used = None


def get_session():
    global _session, _providers_used
    if _session is None:
        providers = ["XnnpackExecutionProvider", "CPUExecutionProvider"]
        try:
            _session = ort.InferenceSession(MODEL_PATH, providers=providers)
        except Exception:
            _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        _providers_used = _session.get_providers()
        print(f"loaded {MODEL_PATH} with providers: {_providers_used}")
    return _session


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file uploaded"}), 400

    # The browser sends a WAV blob it built itself via the Web Audio API
    # (see index.html) specifically so this server only ever has to decode
    # plain WAV — soundfile can't read the MediaRecorder's native
    # audio/webm+Opus output at all, and we're avoiding an ffmpeg/librosa
    # dependency on the phone, so the conversion happens client-side instead.
    audio_bytes = request.files["audio"].read()
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = resample_linear(y, sr, SAMPLE_RATE)

    feat = logmel(y)[None, :, :].astype(np.float32)

    sess = get_session()
    input_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    logits = sess.run(None, {input_name: feat})[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = np.exp(logits[0] - logits[0].max())
    probs = probs / probs.sum()
    label = int(np.argmax(probs))

    return jsonify({
        "label": "infested" if label == 1 else "clean",
        "confidence": float(probs[label]),
        "probs": {"clean": float(probs[0]), "infested": float(probs[1])},
        "latency_ms": round(latency_ms, 2),
        "providers": _providers_used,
    })


if __name__ == "__main__":
    get_session()  # fail fast / warm up on startup
    app.run(host="127.0.0.1", port=5000)
