"""Build a train/val manifest from the downloaded SPID session subset + ESC-50
hard negatives.

Two methodology points that matter for honest numbers (Section 8.3 of the
doc explicitly asks for this):

1. Split is done at the FILE level, not the window level. Several SPID
   sessions are long (~7min) recordings that would otherwise yield hundreds
   of near-identical consecutive windows — splitting those into train/val at
   the window level would leak near-duplicate audio across the split and
   inflate val accuracy.
2. Windows per file are capped (spread evenly across the file's duration)
   so a handful of long recordings can't dominate the dataset relative to
   the many short, event-triggered infested clips.
"""
import csv
import os
import random

import soundfile as sf

from features import CLIP_SECONDS

SELECTED_CSV = "data/spid_selected.csv"  # from download_subset.py
ESC50_DIR = "data/esc50"
OUT_MANIFEST = "data/manifest.csv"

VAL_FRACTION = 0.2
RNG_SEED = 42
MAX_WINDOWS_PER_FILE = 25
MAX_ESC50_FILES = 180  # cap so 2000 generic clips don't swamp the ~50 real infested clips

# aspids_log.csv 'target' values that mean "no insect present" (clean class),
# including non-insect acoustic-test labels that show up in the same column.
CLEAN_TARGETS = {
    "no insects (unconfirmed)",
    "talking",
    "sweep",
    "silence",
}


def label_from_target(target: str) -> int:
    t = (target or "").strip().lower()
    if t in CLEAN_TARGETS or t.startswith("no insect"):
        return 0
    return 1  # any named insect species (or "a lot of bugs") = infested


def file_duration(path):
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception as e:
        print(f"skip unreadable {path}: {e}")
        return None


def offsets_for_file(duration, clip_seconds=CLIP_SECONDS, max_windows=MAX_WINDOWS_PER_FILE):
    n_possible = max(1, int(duration // clip_seconds))
    n = min(n_possible, max_windows)
    if n_possible <= max_windows:
        return [round(i * clip_seconds, 3) for i in range(n_possible)]
    # Evenly spaced sample across the file rather than every consecutive
    # block, so we don't just get 25 near-identical adjacent windows.
    step = (n_possible - 1) / (n - 1) if n > 1 else 0
    return [round(round(i * step) * clip_seconds, 3) for i in range(n)]


def collect_files():
    """Returns list of dicts: path, label, source, material, noise, duration."""
    files = []
    if os.path.exists(SELECTED_CSV):
        with open(SELECTED_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dur = file_duration(row["local_path"])
                if dur is None:
                    continue
                files.append({
                    "path": row["local_path"],
                    "label": label_from_target(row.get("target", "")),
                    "source": "spid",
                    "material": row.get("material", ""),
                    "noise": row.get("noise", ""),
                    "duration": dur,
                })
    else:
        print(f"WARNING: {SELECTED_CSV} not found yet — run download_subset.py first")

    if os.path.isdir(ESC50_DIR):
        esc50_files = sorted(fn for fn in os.listdir(ESC50_DIR) if fn.endswith(".wav"))
        rng = random.Random(RNG_SEED)
        rng.shuffle(esc50_files)
        esc50_files = esc50_files[:MAX_ESC50_FILES]
        for fn in esc50_files:
            path = os.path.join(ESC50_DIR, fn)
            dur = file_duration(path)
            if dur is None:
                continue
            files.append({
                "path": path, "label": 0, "source": "esc50",
                "material": "", "noise": "", "duration": dur,
            })
    return files


def main():
    random.seed(RNG_SEED)
    files = collect_files()
    if not files:
        raise SystemExit("no input files found — run download_subset.py / fetch_esc50.py first")

    # File-level stratified split by (source, label).
    by_group = {}
    for fdesc in files:
        by_group.setdefault((fdesc["source"], fdesc["label"]), []).append(fdesc)

    train_files, val_files = [], []
    for group, flist in by_group.items():
        random.shuffle(flist)
        n_val = max(1, round(len(flist) * VAL_FRACTION)) if len(flist) > 4 else (1 if len(flist) > 1 else 0)
        val_files.extend(flist[:n_val])
        train_files.extend(flist[n_val:])

    rows = []
    for split, flist in (("train", train_files), ("val", val_files)):
        for fdesc in flist:
            for off in offsets_for_file(fdesc["duration"]):
                rows.append({
                    "path": fdesc["path"],
                    "start_offset_sec": off,
                    "label": fdesc["label"],
                    "source": fdesc["source"],
                    "material": fdesc["material"],
                    "noise": fdesc["noise"],
                    "split": split,
                })

    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "start_offset_sec", "label", "source", "material", "noise", "split"])
        w.writeheader()
        w.writerows(rows)

    def summarize(name, flist, wrows):
        n_clean = sum(1 for r in wrows if r["label"] == 0)
        n_inf = sum(1 for r in wrows if r["label"] == 1)
        print(f"{name}: {len(flist)} files -> {len(wrows)} windows (clean={n_clean}, infested={n_inf})")

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    summarize("train", train_files, train_rows)
    summarize("val", val_files, val_rows)
    print(f"total files: {len(files)} (spid={sum(1 for f in files if f['source']=='spid')}, "
          f"esc50={sum(1 for f in files if f['source']=='esc50')})")


if __name__ == "__main__":
    main()
