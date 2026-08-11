# SiloSense

On-device, Arm-optimized audio classifier for stored-grain insect infestation. Built for the **Arm Create: AI Optimization Challenge 2026** (Mobile AI track).

## Why this matters

Postharvest grain loss in Zimbabwe and the wider SADC region runs at 30% or more in many years. Most of it is invisible until the damage is already visible: frass, holes, a musty smell. By the time any of those show up, a meaningful share of the harvest is already gone.

Most grain production in the region comes from smallholder farmers. They store their own maize, groundnuts, and small grains at home, or sell through informal traders. The storage is a sack or a brick-and-mud room, not a sealed commercial silo. Extension visits and lab testing rarely reach that scale.

Weevils and grain borers work from inside the kernel outward. The damage stays hidden for weeks before it's obvious by eye. This matters most in years when rainfall has already been poor: postharvest loss erases part of whatever the season's climate variability left standing. Protecting a harvest that already exists is one of the highest-leverage interventions available against food insecurity. The crop is already grown. The only job left is to stop losing it.

SiloSense treats infestation detection as an audio problem instead of a visual one. Insects moving and feeding inside stored grain produce a real acoustic signature, present weeks before any visual sign of damage. A phone's microphone held against a sack picks it up long before a human eye could.

Three choices make this usable in the places that need it most:

- Fully on-device. No connection, no data cost, nothing to upload.
- Runs on hardware people already own, an ordinary Android phone. No added sensor, no added cost.
- Private by construction. Audio never leaves the phone.

The whole pipeline, from a 1.5-second recording to a result, runs on-device, offline, in under two seconds.

## Architecture

Training happens offline in Python. Inference happens on-device in Kotlin. Both stages compute the same log-mel feature spectrogram from the same spec, and that equivalence is checked, not assumed (see [Porting & parity](#porting--parity) below).

```mermaid
flowchart TB
    subgraph Training["Offline training (Python)"]
        A["SPID/A-SPIDS + ESC-50 audio"] --> B["prepare_dataset.py<br/>1.5s windows, file-level split"]
        B --> C["features.py<br/>log-mel spectrogram (40, 151)"]
        C --> D["train.py<br/>SiloSenseNet CNN, FP32"]
        D --> E["quantize.py<br/>static QDQ INT8"]
        E --> F["models/*.onnx"]
    end

    subgraph OnDevice["On-device inference (Kotlin)"]
        G["AudioRecord<br/>16kHz mono mic"] --> H["AudioFeatures.kt<br/>log-mel spectrogram (40, 151)"]
        F --> I["SiloSenseClassifier.kt<br/>ONNX Runtime session"]
        H --> I
        I --> J["MainActivity.kt<br/>Likely Clean / Uncertain / Likely Infested"]
    end

    C -. "same feature spec<br/>verified to 4.2e-7" .-> H
```

## Dataset & pipeline

Training data is the [SPID / A-SPIDS dataset](https://www.kaggle.com/datasets/dkadyrov/stored-product-insect-database-spidb-aspids) (Kaggle, Daniel Kadyrov, MIT-licensed): real acoustic recordings of three stored-grain pests (*Callosobruchus maculatus*, *Tribolium confusum*, *Tenebrio molitor*) across five materials, plus real background-noise recordings and a genuine no-insects negative class. The full dataset is 106GB across roughly 12,900 files. `index_files.py` matches the dataset's own label log (`aspids_log.csv`) against each session's inner recording timestamp, not the outer date folder Kaggle shows, which is a batch export date rather than the true recording date. Getting that join right is what makes the ~246MB downloaded subset actually label-verified rather than guessed.

`fetch_esc50.py` adds ESC-50 environmental clips (capped at 180 files) to broaden the clean class. `prepare_dataset.py` slices everything into 1.5-second windows and splits train/validation **at the file level**, not the window level, so near-identical adjacent windows from the same recording can't leak across the split.

| Stage | Script | Output |
|---|---|---|
| Discover | `scripts/index_files.py` | Sessions matching real logged dates in the 106GB listing |
| Download | `scripts/download_subset.py` | 246MB label-verified subset, joined against `aspids_log.csv` |
| Negatives | `scripts/fetch_esc50.py` | ESC-50 clips for clean-class diversity |
| Prepare | `scripts/prepare_dataset.py` | `data/manifest.csv`, 1.5s windows, file-level train/val split |
| Features | `scripts/features.py` | (40, 151) log-mel spectrogram, pure NumPy, no librosa |
| Model | `scripts/model.py` | SiloSenseNet CNN definition |
| Train | `scripts/train.py` | FP32 baseline, exported to ONNX |
| Quantize | `scripts/quantize.py` | Static QDQ INT8, accuracy re-verified |

## Model

`SiloSenseNet` is a four-block CNN: `Conv2d → BatchNorm → ReLU → MaxPool`, repeated at 16, 32, 64, and 128 channels, ending in adaptive average pooling and a linear classifier to two classes (clean, infested). About 98,000 parameters, 0.39MB in FP32.

The size is deliberate. An earlier ~6,000-parameter version ran fast enough that fixed kernel-launch and quantization overhead made INT8 look *slower* than FP32 in local testing. The model was sized up so there's enough real multiply-accumulate work for an Arm INT8 dot-product path to show a genuine, measurable win.

Trained for 15 epochs with Adam (lr 1e-3, weight decay 1e-4), loss class-weighted inversely to each class's frequency in the training split. The checkpoint with the best validation F1 is kept and exported to ONNX with a dynamic batch axis.

## Quantization

Static QDQ INT8 (`scripts/quantize.py`), MinMax calibration over 100 real training clips. Static, not dynamic: ONNX Runtime's dynamic quantization only touches MatMul/Gemm by default, which on this model (three Conv2d blocks, one small final Gemm) would leave almost everything in FP32. Static quantization with a real calibration pass actually quantizes the convolution layers, which is what Arm's INT8 SIMD/dot-product paths accelerate.

## Results

| | FP32 | INT8 (static QDQ) | Change |
|---|---|---|---|
| Model size | 0.393 MB | 0.111 MB | 3.54x smaller |
| Validation accuracy | 98.99% | 100.00% | no loss (small val set: the INT8 number edging out FP32 is noise) |
| Validation F1 | 0.985 | 1.000 | no loss |
| On-device inference (Galaxy M16, Arm CPU) | 2.78 ms avg | 1.17 ms avg | 2.38x faster |

On-device numbers: both models loaded and benchmarked in the same run for a matched pair, ten warm-up runs discarded, fifty timed runs averaged, timing isolated to `session.run()` only. A separate desktop CPU-only cross-check (AMD Ryzen 5 PRO 8540U, x86_64, no XNNPACK available) showed a smaller 1.28x speedup (0.226ms to 0.176ms), in `models/x86_cpu_benchmark.json`. The two aren't directly comparable (different chip classes, different execution paths), but together they show the INT8 win holds independent of Arm-specific acceleration, and is larger *with* it.

### What actually executed the model

Registering NNAPI as an execution provider is not the same as NNAPI running a node. `SiloSenseClassifier.kt` enables ONNX Runtime's session profiling, runs real inferences, and parses the resulting trace for each node's actual provider. On the test device, for both models, **NNAPI registered but never appears against a single node.** The real work ran on `XnnpackExecutionProvider` and `CPUExecutionProvider`. FP32: 80 nodes on XNNPACK, 50 on CPU. INT8: 70 on XNNPACK, 100 on CPU.

Model load time is also measured directly, not assumed: FP32 loads in 8.9ms; INT8 loads in 472.4ms, about 50x slower despite being a third of the size, most likely because NNAPI/XNNPACK compile the QDQ quantized graph into their own representation on first load. INT8 wins decisively on steady-state latency and size. It does not win on cold start, on this device. Reported because it's true, not smoothed over.

Also measured, not assumed: process memory (PSS) at baseline (47,829 KB), after both models load (126,554 KB, +78,725 KB), and after the benchmark run (149,185 KB, +22,631 KB more); and thermal status via `PowerManager` immediately before and after the benchmark (`NONE` both times: real but narrow evidence against throttling during this specific short run, not a claim about sustained use).

Battery drain is measured too, via `BatteryManager.getIntProperty(BATTERY_PROPERTY_CURRENT_NOW)` read from inside the app process (`/sys/class/power_supply/battery/current_now` is blocked, `Permission denied`, for the shell user on this device). Off USB power: idle baseline averaged -149,400 uA over 30s (15 samples), sustained INT8 inference averaged -332,437 uA over 60s (30 samples, 66,181 inferences run), a marginal delta of -183,037 uA (~183mA). That load rate, ~1,100 predictions/sec, is a synthetic stress test, not the app's real usage pattern of one prediction per ~1.5s recording — so this is a worst-case upper bound on drain, not a typical-use estimate.

## Porting & parity

`scripts/features.py`'s log-mel pipeline (reflect-pad, periodic Hann window, real FFT, 40-band Slaney-style mel filterbank, per-clip relative dB normalization) is reimplemented natively in `AudioFeatures.kt`, including a hand-rolled radix-2 FFT. No external DSP dependency on-device.

That port is checked, not assumed correct. `AudioFeaturesParityTest.kt` diffs a Kotlin-computed log-mel tensor against the Python reference on the same real audio clip: **maximum absolute difference 4.2×10⁻⁷**, effectively floating-point exact. A silent mismatch here wouldn't crash anything. It would just quietly degrade accuracy with no obvious symptom, which is the failure mode this test exists to catch.

```bash
cd android
./gradlew testDebugUnitTest --tests "com.silosense.app.AudioFeaturesParityTest"
```

## App architecture

| File | Role |
|---|---|
| `AudioFeatures.kt` | Log-mel feature extraction, pure Kotlin, verified against `features.py` |
| `SiloSenseClassifier.kt` | Wraps an ONNX Runtime session per model (INT8 for live predictions, FP32 loaded separately for benchmarking), execution-provider profiling, benchmark timing |
| `DeviceDiagnostics.kt` | Real process memory (PSS), thermal-status, and battery current-draw readings via Android's own APIs |
| `MainActivity.kt` | Record → classify → three-tier result UI, plus Source, Full Results, and Battery test dialogs |

The three-tier result (Likely Clean / Uncertain / Likely Infested) isn't a 50% cutoff. The bounds (0.45 / 0.65) come from where the model's own validation data stops separating cleanly: true clean clips top out at P(infested)=0.492, true infested clips start at 0.643. There is no external standard behind this threshold. Real grading standards speak in insects per kilogram or percent insect-damaged kernels, not model confidence, and this model has never been trained or evaluated against either unit. Calibrating against a real standard, such as Zimbabwe's Grain Marketing Board thresholds, is open work, not a claim made here.

```mermaid
sequenceDiagram
    participant U as User
    participant UI as MainActivity (UI thread)
    participant BG as Background thread
    participant AF as AudioFeatures
    participant I8 as SiloSenseClassifier (INT8)

    U->>UI: Tap Record
    UI->>BG: launch
    BG->>BG: AudioRecord.read() (1.5s, 16kHz mono)
    BG->>AF: logMel(pcm)
    AF-->>BG: (40, 151) tensor
    BG->>I8: predict(features)
    I8-->>BG: label, confidence, tier
    BG->>UI: post result
    UI-->>U: Likely Clean / Uncertain / Likely Infested
```

## Build & run

Requires JDK 17 and the Android SDK (platform 35, build-tools 35).

```bash
# from the android/ directory
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Both `.onnx` models and a real, MIT-licensed infested-grain sample clip are bundled as app assets, so the app builds and runs standalone. No dataset download or training run required first.

To regenerate the models from scratch:

```bash
python scripts/prepare_dataset.py
python scripts/train.py
python scripts/quantize.py
```

## Project structure

```
scripts/              training pipeline (Python): dataset prep, features, model, train, quantize
android/               native Android app (Kotlin)
  app/src/main/java/com/silosense/app/
    AudioFeatures.kt         log-mel feature extraction
    SiloSenseClassifier.kt   ONNX Runtime wrapper + benchmarking + EP profiling
    DeviceDiagnostics.kt     real memory/thermal measurement
    MainActivity.kt          UI
  app/src/test/kotlin/       Python-vs-Kotlin feature parity test
models/                trained FP32/INT8 .onnx models + metrics + x86 cross-check
```

## License

MIT.
