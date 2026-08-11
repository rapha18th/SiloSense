"""Paginate the SPID/A-SPIDS Kaggle file listing (106GB / 12.9k files - far too
large to download in full) looking specifically for files whose *inner* session
timestamp (the true recording date, per the spidb docs' documented structure
`YYYY/MM_DD/YYYY_MM_DD_HH_MM_SS.fff_descriptor/...`) falls on one of the dates
that actually appear in aspids_log.csv. The outer `YYYY/MM/DD` path prefix seen
on Kaggle is an export/batch-date wrapper around the original folder tree, not
the recording date, and does not reliably match the inner timestamp.

Stops early once we've collected enough candidate sessions (one file per
session, first chunk of channel 0 only) so we don't have to paginate all ~645
pages. Retries with exponential backoff on 429 rate-limit errors.
"""
import csv
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "dkadyrov/stored-product-insect-database-spidb-aspids"
LOG_CSV = "data/aspids_log.csv"
OUT_CSV = "data/spid_candidates.csv"
STATE_CSV = "data/spid_scan_progress.csv"  # every file seen, for resumability/debugging

WANT_CHANNEL = "0"
WANT_CHUNK = "00001"
MAX_PAGES = 700
TARGET_CANDIDATES = 80  # stop early once we have this many distinct-session candidates per-date-balanced

PATH_RE = re.compile(
    r"(?P<y>\d{4})_(?P<mo>\d{2})_(?P<d>\d{2})_(?P<h>\d{2})_(?P<mi>\d{2})_(?P<s>\d{2})\.\d+_(?P<desc>t\d+)_Ch(?P<channel>\d+)_(?P<chunk>\d+)\.wav$"
)


def load_log_dates():
    dates = set()
    with open(LOG_CSV, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("start"):
                d = row["start"].split(" ")[0]
                dates.add(d)
    return dates


def main():
    log_dates = load_log_dates()
    print(f"log covers {len(log_dates)} distinct dates: {sorted(log_dates)[:5]}...", file=sys.stderr)

    api = KaggleApi()
    api.authenticate()

    candidates = []
    seen_sessions = set()
    per_date_count = defaultdict(int)

    page_token = None
    page_num = 0
    backoff = 5
    all_seen_f = open(STATE_CSV, "w", newline="", encoding="utf-8")
    seen_writer = csv.writer(all_seen_f)
    seen_writer.writerow(["path", "bytes"])

    while page_num < MAX_PAGES and len(candidates) < TARGET_CANDIDATES:
        try:
            if page_token:
                res = api.dataset_list_files(DATASET, page_token=page_token)
            else:
                res = api.dataset_list_files(DATASET)
        except Exception as e:
            print(f"error on page {page_num+1}: {e}; backing off {backoff}s", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
            continue

        backoff = 5
        page_num += 1
        files = res.files
        if not files:
            break

        for f in files:
            seen_writer.writerow([f.name, f.total_bytes])
            m = PATH_RE.search(f.name)
            if not m:
                continue
            date_str = f"{m.group('y')}-{m.group('mo')}-{m.group('d')}"
            if date_str not in log_dates:
                continue
            if m.group("channel") != WANT_CHANNEL or m.group("chunk") != WANT_CHUNK:
                continue
            session_key = f"{date_str}_{m.group('h')}{m.group('mi')}{m.group('s')}_{m.group('desc')}"
            if session_key in seen_sessions:
                continue
            seen_sessions.add(session_key)
            per_date_count[date_str] += 1
            candidates.append({
                "path": f.name,
                "bytes": f.total_bytes,
                "date": date_str,
                "time": f"{m.group('h')}:{m.group('mi')}:{m.group('s')}",
                "descriptor": m.group("desc"),
            })

        if page_num % 10 == 0 or len(candidates) != len(candidates):
            print(
                f"page {page_num}: candidates so far {len(candidates)} "
                f"(dates hit: {len(per_date_count)})",
                file=sys.stderr,
            )

        page_token = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if not page_token:
            print("reached end of listing", file=sys.stderr)
            break
        time.sleep(1.0)

    all_seen_f.close()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "date", "time", "descriptor"])
        w.writeheader()
        w.writerows(candidates)

    print(f"DONE: {len(candidates)} candidates across {len(per_date_count)} dates, {page_num} pages scanned", file=sys.stderr)
    print(f"per-date counts: {dict(per_date_count)}", file=sys.stderr)


if __name__ == "__main__":
    main()
