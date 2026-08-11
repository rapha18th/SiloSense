# Archived: Termux + proot-distro deployment path

**Status: abandoned in favor of a native Android Studio app.** Kept here for
reference, not deleted, because it reached a fully working, verified state —
see below. If you ever want the fastest possible "no IDE" path again, this
is a working starting point; otherwise ignore this folder.

## Why abandoned

Not because it stopped working — because the user has prior (rusty but real)
Android dev experience, making native Android Studio development a more
comfortable and ultimately more capable path than fighting Termux/proot
quirks (see the friction log below). Once that's true, Android Studio's
bigger upfront setup cost is worth paying for a more familiar, more
extensible environment (proper native mic APIs, no proot/ptrace layer, no
webm→WAV client-side workaround needed, room to grow past a single-screen
demo). See `../NEXT_STEPS.md` for the pivot plan.

## Final working state (verified before archiving)

- `scripts/features.py` (in the main `scripts/` folder, **not** archived —
  it's still the reference implementation) was rewritten to be pure
  numpy + soundfile, **no librosa** — librosa pulls in numba/llvmlite, which
  has no reliable prebuilt aarch64 Linux wheel and tried to build from
  source inside the phone's proot Ubuntu, which is what actually motivated
  reconsidering this whole path in the first place.
- Models were retrained against the new feature code (has to be — changing
  feature extraction, even slightly, means the model must be retrained on
  matching features): **98.99% val accuracy, F1 0.985** (FP32), **100%**
  accuracy/F1 (INT8) on the small held-out set — see `../models/*_metrics.json`.
- `app/server.py` + `app/index.html`: a Flask backend + browser mic-recorder
  UI. Fixed a real bug before archiving — the browser's `MediaRecorder`
  produces `audio/webm` (Opus), which plain `soundfile` cannot decode at
  all. Fix: `index.html` now decodes the recording via the Web Audio API
  (`AudioContext.decodeAudioData`, which Chrome handles natively) and
  hand-encodes a plain WAV file client-side before upload, so the server
  never has to touch a codec it can't read. Verified via a browser JS unit
  test of the `encodeWav()` function (correct WAV header, correct PCM16
  sample values) and a live page load with no console errors.
- `benchmark_on_device.py`: also fixed a real bug — an early version called
  a "load all rows" helper with `n=10**9`, which fed into repeat-padding
  logic meant for small numbers and tried to build a ~10-million-times-too
  -long list (8GB+ RAM, runaway process). Fixed by splitting "get every
  unique val row" from "get N rows, repeating if short" into two functions.
- `package_for_phone.py`: builds a self-contained ~12MB zip (both ONNX
  models, scripts, app, and a small class-balanced sample of real val
  clips) so getting everything onto the phone is one file transfer instead
  of a dozen individual `curl`s. `phone_package.zip` here is the last
  generated copy, already fixed and regenerated after the WAV bug fix.

## If you ever want to run this again

These files were moved from their original locations (`app/`,
`scripts/benchmark_on_device.py`, `scripts/package_for_phone.py`), so their
`sys.path`/relative-import assumptions point at paths that no longer exist
(e.g. `server.py` does `sys.path.insert(0, ".../scripts")` relative to its
old location one level up from repo root). Either move them back before
running, or fix the relative paths — they were not rewritten to work
in-place here, since this is meant as a reference archive, not a
maintained second deployment target.

The full original Termux + proot-distro setup instructions (still accurate)
were removed from the main README when this was archived — ask a past
version of this session's transcript, or reconstruct from: install Termux
(F-Droid), `pkg install proot-distro`, `proot-distro install ubuntu`,
`proot-distro login ubuntu`, `pip install onnxruntime numpy soundfile
flask` (no librosa needed after the fix above), unzip `phone_package.zip`,
`python3 benchmark_on_device.py`.
