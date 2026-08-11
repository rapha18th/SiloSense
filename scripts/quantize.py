"""Post-training INT8 quantization of the FP32 ONNX model.

Uses STATIC quantization (not dynamic): this model is dominated by Conv2d
layers (3 conv blocks + 1 tiny final Gemm), and ONNX Runtime's dynamic
quantization only touches MatMul/Gemm/LSTM ops by default — it would have
left every Conv layer, i.e. almost the entire model, still in FP32. Static
quantization (with a calibration pass over real training-set audio) actually
quantizes the Conv layers too, which is the honest, substantive version of
the "INT8 optimization" claim for a CNN like this one.

Also verifies INT8 accuracy/F1 on the *same* val split as the FP32 baseline
(Section 8.3's requirement: prove the optimization didn't quietly break the
model) and reports the size delta.
"""
import csv
import json
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, QuantType, QuantFormat, CalibrationDataReader, CalibrationMethod
)
from onnxruntime.quantization.preprocess import quant_pre_process
from sklearn.metrics import accuracy_score, f1_score

from features import logmel, load_audio, CLIP_SECONDS

MODEL_DIR = "models"
FP32_ONNX = os.path.join(MODEL_DIR, "silosense_fp32.onnx")
PREPROCESSED_ONNX = os.path.join(MODEL_DIR, "silosense_fp32_preproc.onnx")
INT8_ONNX = os.path.join(MODEL_DIR, "silosense_int8.onnx")
MANIFEST = "data/manifest.csv"
METRICS_JSON = os.path.join(MODEL_DIR, "int8_metrics.json")

N_CALIBRATION_SAMPLES = 100


def load_rows(split):
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append(row)
    return rows


def make_feature(row):
    y = load_audio(row["path"], offset=float(row["start_offset_sec"]), duration=CLIP_SECONDS)
    return logmel(y)[None, :, :].astype(np.float32)


class ManifestCalibrationReader(CalibrationDataReader):
    def __init__(self, rows, input_name):
        self.rows = rows
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.rows):
            return None
        feat = make_feature(self.rows[self.idx])
        self.idx += 1
        return {self.input_name: feat}


def eval_onnx(model_path, rows, providers):
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    preds, labels = [], []
    for r in rows:
        feat = make_feature(r)
        logits = sess.run(None, {input_name: feat})[0]
        preds.append(int(np.argmax(logits[0])))
        labels.append(int(r["label"]))
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    return acc, f1


def main():
    quant_pre_process(FP32_ONNX, PREPROCESSED_ONNX, skip_symbolic_shape=True)

    train_rows = load_rows("train")
    rng = np.random.RandomState(42)
    calib_rows = list(rng.choice(train_rows, size=min(N_CALIBRATION_SAMPLES, len(train_rows)), replace=False))

    probe_sess = ort.InferenceSession(PREPROCESSED_ONNX, providers=["CPUExecutionProvider"])
    input_name = probe_sess.get_inputs()[0].name

    calib_reader = ManifestCalibrationReader(calib_rows, input_name)
    quantize_static(
        PREPROCESSED_ONNX, INT8_ONNX, calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    fp32_size = os.path.getsize(FP32_ONNX) / 1e6
    int8_size = os.path.getsize(INT8_ONNX) / 1e6
    print(f"FP32 size: {fp32_size:.4f} MB   INT8 size: {int8_size:.4f} MB   ratio: {fp32_size/int8_size:.2f}x")

    val_rows = load_rows("val")

    providers = ["XnnpackExecutionProvider", "CPUExecutionProvider"]
    try:
        acc, f1 = eval_onnx(INT8_ONNX, val_rows, providers)
        actual_sess = ort.InferenceSession(INT8_ONNX, providers=providers)
        providers_actual = actual_sess.get_providers()
    except Exception as e:
        print(f"XNNPACK EP unavailable ({e}); falling back to CPUExecutionProvider only")
        providers = ["CPUExecutionProvider"]
        acc, f1 = eval_onnx(INT8_ONNX, val_rows, providers)
        providers_actual = providers

    print(f"INT8 val_acc={acc:.4f} val_f1={f1:.4f} (providers_actual={providers_actual})")

    with open(METRICS_JSON, "w") as f:
        json.dump({
            "val_accuracy": acc, "val_f1": f1,
            "model_size_mb": int8_size,
            "fp32_size_mb": fp32_size,
            "size_reduction_ratio": fp32_size / int8_size,
            "quantization_method": "static QDQ, MinMax calibration, Conv+Gemm quantized",
            "n_calibration_samples": len(calib_rows),
            "providers_requested": providers,
            "providers_actual": providers_actual,
        }, f, indent=2)


if __name__ == "__main__":
    main()
