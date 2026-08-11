"""Build a small, self-contained folder to copy onto the phone in one shot:
scripts + models + app + a size-capped, class-balanced sample of val clips
(not the full ~1GB of raw SPID/ESC-50 audio, and not the full val split's
larger files) with a matching trimmed manifest.csv.

Output: phone_package/  (and phone_package.zip, if `zipfile` succeeds)
"""
import csv
import os
import shutil
import zipfile

MANIFEST = "data/manifest.csv"
OUT_DIR = "phone_package"
PER_CLASS_BUDGET_BYTES = 8_000_000  # ~8MB per class -> ~16MB total audio


def pick_clips():
    rows_by_file = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            rows_by_file.setdefault(row["path"], []).append(row)

    file_sizes = [(path, os.path.getsize(path), rows[0]["label"]) for path, rows in rows_by_file.items()]
    file_sizes.sort(key=lambda t: t[1])  # smallest first, so we get more variety per MB spent

    selected_paths = set()
    budget_used = {"0": 0, "1": 0}
    for path, size, label in file_sizes:
        if budget_used[label] + size <= PER_CLASS_BUDGET_BYTES:
            selected_paths.add(path)
            budget_used[label] += size

    selected_rows = []
    for path in selected_paths:
        selected_rows.extend(rows_by_file[path])

    return selected_rows, selected_paths, budget_used


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(os.path.join(OUT_DIR, "scripts"))
    os.makedirs(os.path.join(OUT_DIR, "models"))
    os.makedirs(os.path.join(OUT_DIR, "app"))
    os.makedirs(os.path.join(OUT_DIR, "data", "clips"))

    for fn in ("features.py", "model.py", "benchmark_on_device.py"):
        shutil.copy(os.path.join("scripts", fn), os.path.join(OUT_DIR, "scripts", fn))
    for fn in ("silosense_fp32.onnx", "silosense_int8.onnx"):
        shutil.copy(os.path.join("models", fn), os.path.join(OUT_DIR, "models", fn))
    for fn in ("server.py", "index.html"):
        shutil.copy(os.path.join("app", fn), os.path.join(OUT_DIR, "app", fn))

    selected_rows, selected_paths, budget_used = pick_clips()

    path_remap = {}
    for path in selected_paths:
        base = os.path.basename(path)
        new_rel = os.path.join("data", "clips", base)
        shutil.copy(path, os.path.join(OUT_DIR, new_rel))
        path_remap[path] = new_rel

    out_manifest = os.path.join(OUT_DIR, "data", "manifest.csv")
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "start_offset_sec", "label", "source", "material", "noise", "split"])
        w.writeheader()
        for r in selected_rows:
            r = dict(r)
            r["path"] = path_remap[r["path"]]
            w.writerow(r)

    print(f"selected {len(selected_paths)} files / {len(selected_rows)} windows")
    print(f"budget used: clean={budget_used['0']/1e6:.2f}MB, infested={budget_used['1']/1e6:.2f}MB")

    total_size = sum(os.path.getsize(os.path.join(root, fn))
                      for root, _, files in os.walk(OUT_DIR) for fn in files)
    print(f"phone_package/ total size: {total_size/1e6:.2f} MB")

    zip_path = OUT_DIR + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for fn in files:
                full = os.path.join(root, fn)
                arcname = os.path.relpath(full, ".")
                zf.write(full, arcname)
    print(f"wrote {zip_path} ({os.path.getsize(zip_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
