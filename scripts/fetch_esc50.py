"""Download ESC-50 (environmental sound / hard-negative background noise)
directly from the original author's GitHub release as a zip of plain WAV
files — avoids the Hugging Face `datasets` Audio feature, which needs
torchcodec+FFmpeg to decode (not available on this machine and unnecessary
extra weight for a hackathon script).
"""
import os
import zipfile
import urllib.request

ZIP_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
OUT_DIR = "data/esc50"
ZIP_PATH = "data/esc50_master.zip"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"downloading {ZIP_URL} ...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print("extracting wav files...")

    n = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            if name.endswith(".ogg") or name.endswith(".wav"):
                base = os.path.basename(name)
                target = os.path.join(OUT_DIR, base)
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                n += 1

    print(f"wrote {n} audio files to {OUT_DIR}")


if __name__ == "__main__":
    main()
