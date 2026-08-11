"""Download the ~80 candidate files found by index_files.py (total ~340MB,
channel 0 = the piezo channel that directly senses insect movement/acoustic
noise in the material, per the dataset's own channel description) and join
each to its label by matching the file's true recording timestamp against
aspids_log.csv's [start, end) windows (same day; nearest-start tiebreak for
event-triggered clips that land between/at logged window edges).
"""
import csv
import os
from datetime import datetime, timedelta

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "dkadyrov/stored-product-insect-database-spidb-aspids"
CANDIDATES_CSV = "data/spid_candidates.csv"
LOG_CSV = "data/aspids_log.csv"
DOWNLOAD_DIR = "data/spid_downloads"
OUT_CSV = "data/spid_selected.csv"


def load_log():
    rows = []
    with open(LOG_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("start"):
                continue
            rows.append({
                "start": datetime.strptime(row["start"], "%Y-%m-%d %H:%M:%S"),
                "end": datetime.strptime(row["end"], "%Y-%m-%d %H:%M:%S") if row.get("end") else None,
                "target": row.get("target", ""),
                "material": row.get("material", ""),
                "noise": row.get("noise", ""),
            })
    return rows


def match_label(file_dt, log_rows):
    same_day = [r for r in log_rows if r["start"].date() == file_dt.date()]
    if not same_day:
        return None
    # Prefer a row whose [start, end) actually contains the file time.
    contained = [r for r in same_day if r["end"] and r["start"] <= file_dt < r["end"]]
    if contained:
        contained.sort(key=lambda r: (r["end"] - r["start"]))  # tightest window wins
        return contained[0]
    # Fall back to nearest start time within the same day (handles
    # event-triggered clips just outside a logged window's exact bounds).
    same_day.sort(key=lambda r: abs((r["start"] - file_dt).total_seconds()))
    best = same_day[0]
    if abs((best["start"] - file_dt).total_seconds()) <= 600:  # within 10 min
        return best
    return None


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    log_rows = load_log()

    api = KaggleApi()
    api.authenticate()

    selected = []
    with open(CANDIDATES_CSV, encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))

    for i, c in enumerate(candidates):
        file_dt = datetime.strptime(f"{c['date']} {c['time']}", "%Y-%m-%d %H:%M:%S")
        label_row = match_label(file_dt, log_rows)
        if label_row is None:
            print(f"[{i+1}/{len(candidates)}] no label match for {c['path']}, skipping")
            continue

        local_name = os.path.basename(c["path"])
        local_path = os.path.join(DOWNLOAD_DIR, local_name)
        if not os.path.exists(local_path):
            print(f"[{i+1}/{len(candidates)}] downloading {c['path']} ({int(c['bytes'])/1e6:.2f} MB) -> {c['target'] if False else label_row['target']}")
            api.dataset_download_file(DATASET, c["path"], path=DOWNLOAD_DIR, force=True)
            # kaggle CLI/py sometimes zips single files; handle both cases.
            zip_path = local_path + ".zip"
            if os.path.exists(zip_path):
                import zipfile
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(DOWNLOAD_DIR)
                os.remove(zip_path)
                # extracted file keeps original name (no nested dirs expected for single -f download)
        else:
            print(f"[{i+1}/{len(candidates)}] already have {local_path}")

        selected.append({
            "path": c["path"],
            "local_path": local_path,
            "date": c["date"],
            "time": c["time"],
            "target": label_row["target"],
            "material": label_row["material"],
            "noise": label_row["noise"],
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "local_path", "date", "time", "target", "material", "noise"])
        w.writeheader()
        w.writerows(selected)

    print(f"\nDONE: {len(selected)}/{len(candidates)} files labeled and downloaded to {DOWNLOAD_DIR}")
    from collections import Counter
    print("target distribution:", Counter(s["target"] for s in selected))


if __name__ == "__main__":
    main()
