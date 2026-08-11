"""Run this INSIDE Termux's proot-distro Ubuntu, on the phone itself.

Loads both the FP32 and INT8 ONNX models, times inference over many runs
(discarding a warmup batch so first-call JIT/cache effects don't skew the
number), and re-checks accuracy on-device — this script's stdout output is
the actual evidence for the submission's optimization claims, so it prints a
plain-text table plus writes benchmark_results.json.

Usage (inside the phone's proot Ubuntu, from the project's models/ + data/
folders copied over):
    python3 benchmark_on_device.py
"""
import csv
import json
import os
import time

import numpy as np
import onnxruntime as ort

from features import logmel, load_audio, CLIP_SECONDS


def accuracy_and_f1(labels, preds):
    # Hand-rolled so the phone doesn't need scikit-learn installed just for
    # two numbers — keeps the on-device dependency list to the bare minimum.
    labels = np.array(labels)
    preds = np.array(preds)
    acc = float((labels == preds).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return acc, f1, precision, recall

MODEL_DIR = "models"
FP32_ONNX = os.path.join(MODEL_DIR, "silosense_fp32.onnx")
INT8_ONNX = os.path.join(MODEL_DIR, "silosense_int8.onnx")
MANIFEST = "data/manifest.csv"
N_WARMUP = 5
N_TIMED = 50


def load_all_val_rows():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                rows.append(row)
    return rows


def load_val_clips_for_latency(n=N_WARMUP + N_TIMED):
    rows = load_all_val_rows()
    if not rows:
        raise SystemExit("no val rows found in manifest")
    if len(rows) < n:
        # repeat if the val split is small — we're timing steady-state
        # inference here, not measuring on unique data (accuracy is checked
        # separately, over every unique val row, with no repetition).
        rows = (rows * (n // len(rows) + 1))[:n]
    return rows[:n]


def precompute_features(rows):
    feats = []
    for r in rows:
        y = load_audio(r["path"], offset=float(r["start_offset_sec"]), duration=CLIP_SECONDS)
        feats.append(logmel(y)[None, :, :].astype(np.float32))
    return feats


def bench(model_path, feats, providers):
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    actual_providers = sess.get_providers()

    for f in feats[:N_WARMUP]:
        sess.run(None, {input_name: f})

    times_ms = []
    for f in feats[N_WARMUP:N_WARMUP + N_TIMED]:
        t0 = time.perf_counter()
        sess.run(None, {input_name: f})
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    return {
        "providers_requested": providers,
        "providers_actual": actual_providers,
        "mean_ms": float(times_ms.mean()),
        "median_ms": float(np.median(times_ms)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "std_ms": float(times_ms.std()),
        "n_runs": len(times_ms),
    }


def check_accuracy(model_path, rows, providers):
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    preds, labels = [], []
    for r in rows:
        y = load_audio(r["path"], offset=float(r["start_offset_sec"]), duration=CLIP_SECONDS)
        feat = logmel(y)[None, :, :].astype(np.float32)
        logits = sess.run(None, {input_name: feat})[0]
        preds.append(int(np.argmax(logits[0])))
        labels.append(int(r["label"]))
    acc, f1, prec, rec = accuracy_and_f1(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision_infested": prec, "recall_infested": rec, "n": len(rows)}


def main():
    all_val_rows = load_all_val_rows()  # every unique val row, no repetition, for accuracy
    rows_for_latency = load_val_clips_for_latency()
    feats = precompute_features(rows_for_latency)

    results = {}
    results["fp32"] = bench(FP32_ONNX, feats, ["CPUExecutionProvider"])
    results["fp32"]["model_size_mb"] = os.path.getsize(FP32_ONNX) / 1e6
    results["fp32"]["accuracy_check"] = check_accuracy(FP32_ONNX, all_val_rows, ["CPUExecutionProvider"])

    int8_providers = ["XnnpackExecutionProvider", "CPUExecutionProvider"]
    try:
        results["int8"] = bench(INT8_ONNX, feats, int8_providers)
        acc_providers = results["int8"]["providers_actual"]
    except Exception as e:
        print(f"XNNPACK EP unavailable on this device ({e}); falling back to CPU-only INT8 run")
        results["int8"] = bench(INT8_ONNX, feats, ["CPUExecutionProvider"])
        acc_providers = ["CPUExecutionProvider"]
    results["int8"]["model_size_mb"] = os.path.getsize(INT8_ONNX) / 1e6
    results["int8"]["accuracy_check"] = check_accuracy(INT8_ONNX, all_val_rows, acc_providers)

    speedup = results["fp32"]["mean_ms"] / results["int8"]["mean_ms"]
    size_reduction = results["fp32"]["model_size_mb"] / results["int8"]["model_size_mb"]

    print("\n=== SiloSense on-device benchmark (this phone's ARM CPU) ===")
    print(f"{'':10s} {'size(MB)':>10s} {'mean(ms)':>10s} {'median(ms)':>12s} {'p90(ms)':>10s} {'acc':>8s} {'f1':>8s}")
    for k in ("fp32", "int8"):
        r = results[k]
        ac = r["accuracy_check"]
        print(f"{k:10s} {r['model_size_mb']:10.3f} {r['mean_ms']:10.3f} {r['median_ms']:12.3f} "
              f"{r['p90_ms']:10.3f} {ac['accuracy']:8.4f} {ac['f1']:8.4f}")
    print(f"\nspeedup (mean latency): {speedup:.2f}x")
    print(f"size reduction: {size_reduction:.2f}x")
    print(f"accuracy delta (int8 - fp32): {results['int8']['accuracy_check']['accuracy'] - results['fp32']['accuracy_check']['accuracy']:+.4f}")
    print(f"INT8 execution providers actually used: {results['int8']['providers_actual']}")
    print(f"(accuracy check ran over {len(all_val_rows)} unique val windows)")

    results["speedup_x"] = speedup
    results["size_reduction_x"] = size_reduction
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote benchmark_results.json")


if __name__ == "__main__":
    main()
