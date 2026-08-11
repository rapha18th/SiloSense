const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, VerticalAlign,
} = require("docx");
const fs = require("fs");

const US_LETTER = { width: 12240, height: 15840 };
const MARGIN = 1440;

const INK = "1C1C1E";
const SUBTLE = "6E6E73";
const ACCENT = "0A66C2";
const GREEN = "1E7A46";
const HAIRLINE = "D8D8DC";
const CARD_BG = "F5F5F7";

function hr() {
  return new Paragraph({
    spacing: { before: 200, after: 200 },
    border: { bottom: { color: HAIRLINE, space: 1, style: BorderStyle.SINGLE, size: 6 } },
  });
}

function kicker(text) {
  return new Paragraph({
    spacing: { before: 0, after: 60 },
    children: [new TextRun({ text: text.toUpperCase(), bold: true, color: ACCENT, size: 18, characterSpacing: 20 })],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 480, after: 200 },
    children: [new TextRun({ text, color: INK, bold: true, size: 32 })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, color: INK, bold: true, size: 24 })],
  });
}

function body(text) {
  return new Paragraph({ spacing: { after: 160, line: 320 }, children: [new TextRun({ text, color: INK, size: 22 })] });
}

function mono(text) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    shading: { type: ShadingType.CLEAR, fill: CARD_BG },
    indent: { left: 200, right: 200 },
    children: [new TextRun({ text, color: INK, size: 18, font: "Consolas" })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "main-bullets", level: 0 },
    spacing: { after: 90, line: 300 },
    children: [new TextRun({ text, color: INK, size: 22 })],
  });
}

function caption(text) {
  return new Paragraph({ spacing: { before: 40, after: 240 }, children: [new TextRun({ text, color: SUBTLE, italics: true, size: 18 })] });
}

function resultsTable() {
  const headerCell = (text) => new TableCell({
    width: { size: 2400, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: INK },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 120, bottom: 120, left: 140, right: 140 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 19 })] })],
  });
  const cell = (text, opts = {}) => new TableCell({
    width: { size: 2400, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: opts.fill || "FFFFFF" },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 120, bottom: 120, left: 140, right: 140 },
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, size: 20, bold: !!opts.bold, color: opts.color || INK })],
    })],
  });

  return new Table({
    width: { size: 9600, type: WidthType.DXA },
    columnWidths: [2400, 2400, 2400, 2400],
    rows: [
      new TableRow({ children: [headerCell(""), headerCell("FP32 baseline"), headerCell("INT8 static QDQ"), headerCell("Change")] }),
      new TableRow({ children: [cell("Model size", { bold: true }), cell("0.393 MB", { center: true }), cell("0.111 MB", { center: true }), cell("3.54x smaller", { center: true, bold: true, color: GREEN })] }),
      new TableRow({ children: [cell("Validation accuracy", { bold: true }), cell("98.99%", { center: true }), cell("100.00%", { center: true }), cell("no loss", { center: true })] }),
      new TableRow({ children: [cell("Validation F1", { bold: true }), cell("0.985", { center: true }), cell("1.000", { center: true }), cell("no loss", { center: true })] }),
      new TableRow({ children: [cell("On-device inference, Arm CPU", { bold: true }), cell("2.78 ms avg", { center: true }), cell("1.17 ms avg", { center: true, bold: true, color: GREEN }), cell("2.38x faster", { center: true, bold: true, color: GREEN })] }),
    ],
  });
}

function desktopTable() {
  const headerCell = (text) => new TableCell({
    width: { size: 3200, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: INK },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 120, bottom: 120, left: 140, right: 140 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 19 })] })],
  });
  const cell = (text, opts = {}) => new TableCell({
    width: { size: 3200, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: opts.fill || "FFFFFF" },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 120, bottom: 120, left: 140, right: 140 },
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, size: 20, bold: !!opts.bold, color: opts.color || INK })],
    })],
  });

  return new Table({
    width: { size: 9600, type: WidthType.DXA },
    columnWidths: [3200, 3200, 3200],
    rows: [
      new TableRow({ children: [headerCell(""), headerCell("FP32 baseline"), headerCell("INT8 static QDQ")] }),
      new TableRow({ children: [cell("Mean latency", { bold: true }), cell("0.226 ms", { center: true }), cell("0.176 ms", { center: true, bold: true, color: GREEN })] }),
      new TableRow({ children: [cell("Median latency", { bold: true }), cell("0.216 ms", { center: true }), cell("0.173 ms", { center: true })] }),
      new TableRow({ children: [cell("p90 latency", { bold: true }), cell("0.259 ms", { center: true }), cell("0.189 ms", { center: true })] }),
      new TableRow({ children: [cell("Accuracy / F1", { bold: true }), cell("98.99% / 0.985", { center: true }), cell("100.00% / 1.000", { center: true })] }),
    ],
  });
}

function pipelineTable() {
  const headerCell = (text) => new TableCell({
    width: { size: 3200, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: INK },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 18 })] })],
  });
  const cell = (text) => new TableCell({
    width: { size: 3200, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, size: 19, color: INK })] })],
  });
  const rows = [
    ["Stage", "Script / component", "What it does"],
    ["Discover", "scripts/index_files.py", "Scans the 106GB SPID/A-SPIDS Kaggle listing for sessions matching real logged dates."],
    ["Download", "scripts/download_subset.py", "Pulls a 246MB label-verified subset, joined against aspids_log.csv."],
    ["Negatives", "scripts/fetch_esc50.py", "Adds ESC-50 environmental clips as extra clean-class diversity."],
    ["Prepare", "scripts/prepare_dataset.py", "Slices clips into 1.5s windows, splits train/val by file to prevent leakage."],
    ["Features", "scripts/features.py", "Computes the 40x151 log-mel spectrogram. Pure NumPy, no librosa."],
    ["Model", "scripts/model.py", "Defines the 4-block CNN, about 98K parameters."],
    ["Train", "scripts/train.py", "Trains the FP32 baseline, exports to ONNX."],
    ["Quantize", "scripts/quantize.py", "Static QDQ INT8 quantization, verifies accuracy held."],
    ["Port", "android/…/AudioFeatures.kt", "Kotlin reimplementation of the feature pipeline, checked against Python to 4.2e-7."],
    ["Infer", "android/…/SiloSenseClassifier.kt", "ONNX Runtime session, NNAPI + XNNPACK + CPU."],
    ["Interface", "android/…/MainActivity.kt", "Record, classify, show the result."],
  ];
  return new Table({
    width: { size: 9600, type: WidthType.DXA },
    columnWidths: [3200, 3200, 3200],
    rows: rows.map((r, i) => new TableRow({ children: i === 0 ? r.map(headerCell) : r.map(cell) })),
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "main-bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: US_LETTER, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children: [
      new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "SILOSENSE", bold: true, size: 56, color: INK })] }),
      new Paragraph({ spacing: { after: 20 }, children: [new TextRun({ text: "On-device grain infestation detection for postharvest loss prevention", size: 24, color: SUBTLE })] }),
      new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "Arm Create: AI Optimization Challenge 2026. Mobile AI Track. Submission write-up.", size: 18, color: SUBTLE, italics: true })] }),
      hr(),

      // ---- 1. Problem ----
      kicker("The problem"),
      h1("Grain doesn't announce that it's being eaten"),
      body("Postharvest grain loss in Zimbabwe and the wider SADC region runs at 30% or more in many years. Most of it is invisible until the damage is visible. Frass, holes, a musty smell: by the time any of these show up, a meaningful share of the harvest is already gone."),
      body("Most grain production in the region comes from smallholder farmers. They store their own maize, groundnuts, and small grains at home, or sell through informal traders. The storage is a sack or a brick-and-mud room, not a sealed commercial silo. Extension visits and lab testing rarely reach that scale."),
      body("Weevils and grain borers work from inside the kernel outward. That is the whole problem. The damage stays hidden for weeks before it is obvious by eye."),
      body("This matters most in years when rainfall has already been poor. Postharvest loss erases part of whatever the season's climate variability left standing. Protecting the harvest that already exists is one of the highest-leverage interventions available against food insecurity. The crop is already grown. The only job left is to stop losing it."),

      // ---- 2. Approach ----
      kicker("The approach"),
      h1("Listen to the grain"),
      body("SiloSense treats infestation detection as an audio problem. Insects moving and feeding inside stored grain produce a real acoustic signature. That signature is present weeks before any visual sign of damage. A phone's microphone held against a sack picks it up long before a human eye could."),
      body("Three choices make this usable in the places that need it most."),
      bullet("Fully on-device. No connection, no data cost, nothing to upload."),
      bullet("Runs on hardware people already own. An ordinary Android phone. No added sensor, no added cost."),
      bullet("Private by construction. Audio never leaves the phone."),

      // ---- 3. Dataset ----
      kicker("Finding the data"),
      h1("A 106GB dataset, and 246MB of it that mattered"),
      body("Training data comes from the SPID / A-SPIDS dataset (Kaggle, Daniel Kadyrov, MIT-licensed): real acoustic recordings of three stored-grain pests, Callosobruchus maculatus, Tribolium confusum, and Tenebrio molitor, captured across rice, flour, oatmeal, corn flakes, and wheat groats, plus real background-noise recordings and a genuine no-insects negative class."),
      body("The full dataset is 106GB across roughly 12,900 files, too large to download in full. scripts/index_files.py scans the Kaggle file listing for sessions whose inner recording timestamp matches a date that appears in the dataset's own label log, aspids_log.csv. The outer date folder Kaggle shows is a batch export date, not the true recording date. Matching on the wrong date silently returns unlabeled or mislabeled files, so getting this join right was the actual work."),
      body("scripts/download_subset.py pulls the matched sessions and joins their labels, producing a label-verified subset of about 246MB. scripts/fetch_esc50.py adds ESC-50 environmental recordings, capped at 180 files, to broaden the clean class without letting generic negatives swamp the roughly 50 real infested clips."),

      // ---- 4. Processing ----
      kicker("Processing it"),
      h1("Turning raw recordings into a training set"),
      body("The task is binary: infested or clean. scripts/prepare_dataset.py maps every SPID label to one of the two classes. Named insect species map to infested. Talking, sweep, silence, and no insects map to clean."),
      body("Long recordings, some run seven minutes, are sliced into 1.5-second windows, capped at 25 windows per file and spread evenly across its duration, so a handful of long clips cannot dominate the many short, event-triggered infested clips."),
      body("The train/validation split happens at the file level, not the window level. Splitting at the window level would put near-identical adjacent windows from the same recording on both sides of the split, inflating validation accuracy without meaning anything. The split is stratified by source and label, so both SPID and ESC-50, and both classes, are represented proportionally in each half."),

      // ---- 5. Features ----
      kicker("The features"),
      h1("A log-mel spectrogram, computed the same way twice"),
      body("Each 1.5-second clip becomes a fixed (40, 151) log-mel spectrogram: 40 mel bins across 151 time frames, computed from 16kHz mono audio. The training-side pipeline is pure NumPy and soundfile, no librosa. librosa pulls in numba and llvmlite, which have no reliable prebuilt wheel for Arm targets and try to compile from source on-device. That is exactly what made the abandoned Termux-on-phone path fragile in the first place. The native Android app carries none of that dependency chain either way: it is Kotlin, not Python, so there is no librosa/numba/llvmlite question on-device at all. Its own cold-start cost is a different, separately measured number, reported in Arm Optimization Evidence below rather than assumed from the training-side decision."),
      body("The pipeline: reflect-pad the signal, frame it into overlapping windows, apply a periodic Hann window, take the real FFT and its power spectrum, project through a 40-band Slaney-style mel filterbank, convert to decibels relative to the clip's own peak, floor at minus 80dB, normalize to roughly [-1, 1]."),
      body("This pipeline runs twice: once in Python during training, once in Kotlin on the phone during inference. Both have to produce the same numbers, or the model's accuracy does not transfer to the device. That equivalence was not assumed. It was measured. See the porting section below."),

      // ---- 6. Model ----
      kicker("The model"),
      h1("A CNN sized for the win it needs to show"),
      body("SiloSenseNet is a four-block convolutional network: Conv2d, BatchNorm, ReLU, MaxPool, repeated at 16, 32, 64, and 128 channels, ending in adaptive average pooling and a linear classifier to two classes. About 98,000 parameters. 0.39MB in FP32."),
      body("That size is deliberate. An earlier version at roughly 6,000 parameters ran in about 0.1ms and made INT8 look slower than FP32 in local testing, because fixed kernel-launch and quantization overhead dominated at that scale. The model was sized up moderately so there is enough real multiply-accumulate work for an Arm INT8 dot-product path to show a genuine, measurable win. Still small by any normal standard. The optimization story here is quantization and an Arm-optimized execution path."),

      // ---- 7. Training ----
      kicker("Training"),
      h1("Fifteen epochs, class-weighted"),
      body("The FP32 baseline trains for 15 epochs with Adam, learning rate 1e-3, weight decay 1e-4. Clean and infested clips are not balanced in the raw data, so the loss is class-weighted inversely to each class's frequency in the training split. The checkpoint with the best validation F1 is kept, then re-evaluated and exported to ONNX with a dynamic batch axis."),
      body("Result: 98.99% validation accuracy, F1 0.985."),

      // ---- 8. Quantization ----
      kicker("Quantization"),
      h1("Static quantization, and why it matters"),
      body("scripts/quantize.py applies static QDQ INT8 quantization with MinMax calibration over 100 real training clips. Quantization here is static. ONNX Runtime's dynamic quantization only touches MatMul and Gemm operations by default. This model is three Conv2d blocks and one small final Gemm. Dynamic quantization would have left almost the entire model in FP32. Static quantization, with a calibration pass over real audio, actually quantizes the convolution layers. That is the substantive version of an INT8 claim for a CNN shaped like this one, and it is what Arm's INT8 dot-product and SIMD paths are built to accelerate."),
      body("Result: 100.00% validation accuracy, F1 1.000, on the same held-out split as the FP32 baseline. The small validation set means the INT8 model edging out FP32 is noise. The claim is no loss."),
      resultsTable(),
      caption("On-device latency measured live on a Samsung Galaxy M16, Arm CPU, FP32 and INT8 both loaded and benchmarked in the same run for a matched pair, not separate sessions. ONNX Runtime configured for NNAPI, XNNPACK, and CPU execution providers, all three registering successfully. Timing isolated to session.run() only, ten warm-up runs discarded, fifty timed runs following, averaged. See the Arm Optimization Evidence section below for what ONNX Runtime's own execution trace shows actually ran those fifty runs."),
      body("Earlier, smaller benchmark passes during development (5-20 runs instead of 50, sometimes INT8-only) produced speedups anywhere from 1.68x to 4.92x run to run. The 50-run figure above is the one treated as authoritative here, and even it is one phone, one session, not a claim of a fixed constant."),

      h2("A desktop cross-check"),
      body("Both models were also benchmarked on the development machine itself: an AMD Ryzen 5 PRO 8540U, x86_64, Windows 11, using legacy_termux_proot/benchmark_on_device.py's own methodology, five warm-up runs discarded, fifty timed runs following, run fresh against the exact models shipped in this repository, not the smaller, now-abandoned architecture an earlier x86 dry-run of this same script used before the model was sized up."),
      desktopTable(),
      caption("CPU-only on both models. The desktop ONNX Runtime build here does not include XNNPACK, so this is plain CPUExecutionProvider for FP32 and INT8 alike, not the NNAPI+XNNPACK+CPU path the phone number above uses. It answers a narrower question than the Arm number: does INT8 quantization win on its own, from a smaller model and less compute, with no hardware-specific dot-product path active at all. It does, 1.28x here, which is a different and smaller effect than anything Arm-specific."),
      body("The two numbers are not directly comparable and are not presented as if they were. The x86 chip is a full laptop-class CPU, not a phone SoC, so its absolute latency being lower than the phone's is expected and says nothing about Arm. What it does say: the INT8 win is real on two different instruction set architectures, under two different execution paths, which is a stronger claim than either number alone."),

      h2("Arm optimization evidence"),
      body("Registering an execution provider is not the same as a node actually running on it, and this submission does not treat them as equivalent. ONNX Runtime's Java API confirms what registered on the session, but not what executed each op. Session profiling does: enabling it, running real inferences, then reading ONNX Runtime's own Chrome-trace output node by node shows the actual provider behind each Conv and Gemm."),
      body("On the test device, for both models, NNAPI registered successfully but never appears against a single node in the trace. The convolution and matmul work actually ran on XNNPACK and plain CPU: FP32 split 80 nodes to XnnpackExecutionProvider and 50 to CPUExecutionProvider; INT8 split 70 to XnnpackExecutionProvider and 100 to CPUExecutionProvider. Whatever accelerated this model on this phone, it was XNNPACK's Arm NEON SIMD kernels, not the SoC's NPU or DSP through NNAPI. That is a materially different and more specific claim than \"NNAPI accelerated this,\" and it only exists because it was checked rather than assumed."),
      body("Model load time was measured the same way, wall-clock around asset read plus session creation. FP32 loaded in 8.9ms. INT8 loaded in 472.4ms, roughly fifty times slower to load despite being a third of the size. The most likely explanation is that NNAPI and XNNPACK do real work compiling the QDQ quantized graph into their own representation on first load, a cost the plain FP32 graph does not pay. INT8 still wins decisively on steady-state inference latency and on model size. It does not win on cold start, on this device, and that tradeoff is reported rather than left out because it complicates the headline number."),
      body("Process memory (PSS) was sampled at three points: an app baseline of 47,829 KB, 126,554 KB after both models were loaded (+78,725 KB), and 149,185 KB after the fifty-run benchmark completed (+22,631 KB more). Total footprint stays under 150MB including the whole app process and the Android runtime around it, not just the model."),
      body("Thermal status came from Android's own PowerManager API, sampled immediately before and after the benchmark run: NONE both times, the OS's lowest reportable thermal state. That is real, if narrow, evidence against thermal throttling during this specific short benchmark. It is not a claim about sustained multi-minute or multi-hour use, which was not tested."),
      body("Battery drain was not measured. A meaningful number needs a sustained load test over minutes, not a benchmark pass that completes in well under a second of actual inference time. Named here as an open gap rather than estimated."),

      // ---- 9. Porting ----
      kicker("Getting it onto a phone"),
      h1("Two paths, one that stuck"),
      h2("The first attempt: Termux and proot"),
      body("The first deployment target was Termux with a proot-distro Ubuntu userland on Android: a Flask server and a browser mic-recorder UI, no IDE, no SDK install. It reached a fully working state. Two real bugs were caught and fixed along the way: the browser's MediaRecorder produces audio/webm, which soundfile cannot decode, fixed by decoding client-side through the Web Audio API and hand-encoding a plain WAV before upload; and a benchmark helper that fed a placeholder value of one billion into repeat-padding logic meant for small numbers, which tried to build an eight-gigabyte list and had to be split into two correctly-scoped functions."),
      body("Native Android replaced it. The Termux path worked. Native development gave a more direct route to real mic APIs, no proot layer, and room to grow past a single-screen demo."),
      h2("The second attempt: native Android"),
      body("The Android project was a hand-written Gradle skeleton, never opened in an IDE, no SDK available in the environment that generated it. Bringing it to life meant a full toolchain build from nothing: JDK 17, Android SDK command-line tools, platform 35, build-tools 35, and a Gradle wrapper that did not exist yet and had to be generated."),
      body("With the toolchain in place, three pieces of real code closed the gap between skeleton and working app."),
      pipelineTable(),
      body("The feature pipeline's Kotlin port was checked against the Python reference, not assumed correct. A Python-computed log-mel tensor from a real audio clip was diffed numerically against the Kotlin implementation's output on the same clip, through an automated unit test. Maximum absolute difference: 4.2 times ten to the minus seven. Effectively floating-point exact. A silent mismatch here would not have crashed anything. It would have quietly degraded accuracy with no obvious symptom, which is the failure mode this test exists to rule out."),
      body("The classifier wraps an ONNX Runtime session around each model, configured to attempt NNAPI first, then XNNPACK, falling back to plain CPU only if neither registers. On the test device, all three registered successfully, but registering is not the same as running: see Arm Optimization Evidence above for what ONNX Runtime's own profiling trace shows actually executed each node. Benchmarking follows the same discipline as the old Termux script: warm up, discard the first several runs, time only session.run(), never feature extraction, report the average across many runs rather than one number."),
      body("A second, smaller path was added alongside the live microphone: a bundled real infested-grain clip, one of the actual SPID recordings, MIT-licensed, verified to predict Infested at 69% confidence through the exact same pipeline that runs on live audio. It ships as a raw 16kHz PCM asset and loads through a link on the main screen. Real infested grain was not available to demo live. This closes that gap without faking a result: the audio and the model are both real, only the recording moment is pre-captured instead of live."),

      // ---- 10. Interface ----
      kicker("The interface"),
      h1("One screen, three states"),
      body("The interface is deliberately spare. A true-black background. One card showing a status word and a confidence number that changes size and color with the result. A monospace chip below it showing the actual latency and which execution providers ran. One circular button at the bottom. Tap it, wait a second and a half, get an answer."),
      body("The result is not a flat clean-or-infested split. A 50% cutoff is the default a two-class model falls back to, and it has no grounding beyond that. The app instead reports one of three states: Likely Clean, Uncertain, and Likely Infested. Uncertain sits in the band where the model's own validation data stops separating cleanly, P(infested) between 0.45 and 0.65, rather than forcing every borderline reading into a hard color. Both raw probabilities, clean and infested, are always shown underneath, so the number behind the color is never hidden."),
      body("Nothing about the design explains itself, because it does not need to. The whole interaction is the same fifteen seconds a farmer would spend checking a sack by hand, just with a real answer, or a real acknowledgment of uncertainty, at the end of it instead of a guess."),

      // ---- 10b. Open problems ----
      kicker("Open problems"),
      h1("What it would take to reach a grading standard"),
      body("The three-tier system above is honest about its own limits. It is drawn from this model's validation gap, not from any external authority. It is worth being just as direct about what stands between that and a number a grain-grading body like Zimbabwe's Grain Marketing Board would actually recognize."),
      body("Real grading standards do not speak in model confidence. They speak in insects per kilogram, or percent insect-damaged kernels, measured by count or by sieve. SiloSense's model has never been trained or evaluated against either unit. Closing that gap is a real, unfinished research problem."),
      bullet("Graded ground truth. The training data labels a clip infested or clean. Reaching an insects-per-kilogram estimate needs clips paired with an actual measured count or damage percentage on the same grain, across a range of severities, not just presence or absence."),
      bullet("A severity model, not a binary one. Once graded data exists, the model itself would need to move from two-class classification to a regression or ordinal target, an estimated count or damage percentage rather than a clean-or-infested label."),
      bullet("The actual GMB numbers. This project does not have Zimbabwe's published grading thresholds in hand. Obtaining and referencing the real standard, likely through the GMB directly or through the regional standards it aligns with, is a first step, not an assumption to design around in advance."),
      bullet("Field conditions that match the deployment. SPID/A-SPIDS was recorded under study conditions, across several pest species and storage materials. A tool meant to inform a real grading decision needs recordings matched to the crop, sack material, and moisture range it will actually be used on, primarily maize in smallholder and trader storage in Zimbabwe."),
      bullet("An agreement study against the official method. Before any acoustic estimate could sit next to a GMB inspection result, it needs a formal comparison against that method on the same samples, the kind of agreement analysis regulatory and clinical tools are held to before they are trusted as more than a screening aid."),
      body("Until that work is done, the honest framing is a screening tool. It tells a household or trader where to look closer. It is not a substitute for the standard it would need to be validated against, and this document does not claim otherwise."),

      // ---- 11. Real-world deployment ----
      kicker("In practice"),
      h1("What a real deployment looks like"),
      body("The technology is already at the point where a real rollout looks unremarkable. The app installs like any other Android app, on a phone the household or trader already owns. No new hardware, no subscription, no network requirement. A check takes fifteen seconds: open the app, hold the phone to the sack, tap once."),
      body("The natural unit of use is the household store or the trader's warehouse. A smallholder checks their own stored maize on a rolling basis through the weeks between harvest and sale. A trader holding grain on behalf of several farmers runs the same check across every sack coming in, at the point of purchase, before an infested sack contaminates the rest of the store."),
      body("Three extensions turn this from a single check into a monitoring system, and none of them require a new model, only more interface around the one that already works."),
      bullet("A log of past checks per sack or per storage room, so the trend across checks drives the decision to act."),
      bullet("A batch mode for warehouses: scan every sack in a session, get a summary instead of one result at a time."),
      bullet("Guidance tied to the result: what a clean reading means for re-check timing, what an infested reading means for separating and treating that sack before it spreads."),
      body("None of this requires connectivity, a server, or a subscription. The entire product is the model already sitting in this repository, running on a phone."),

      // ---- 12. Quantifiable gains ----
      kicker("What adoption is worth"),
      h1("An honest estimate"),
      caption("The figures below are an illustrative estimate built on the 30% loss figure already cited, with the assumptions stated. They are not a field measurement."),
      body("Start from the regional baseline: 30% or more postharvest loss in many years. Some of that loss comes from moisture, mold, and rodents, which no audio check catches. Assume, conservatively, that half of the 30% is insect infestation that early detection could plausibly catch before it spreads through a store. That is 15 percentage points of a household's harvest."),
      body("For a smallholder household storing roughly half a tonne of maize through a season, 15 percentage points is on the order of 75kg recovered, simply by catching an infestation at week six instead of week ten or twelve, when it is one sack instead of a third of the store. Scaled to a trader holding grain from a dozen households, the same math scales roughly linearly with volume checked."),
      body("The honest version of this claim is directional, not precise: earlier detection converts a slow, invisible loss into a fast, visible decision, at a scale that tracks how much grain gets checked. The tool costs nothing beyond a phone the household already owns. The upside scales with adoption, not with any further engineering."),

      // ---- 13. Status ----
      kicker("Status"),
      h1("What is verified, and what is not"),
      body("Verified, measured, real:"),
      bullet("The model accuracy, F1, and size numbers above, reproducible from the training scripts in this repository."),
      bullet("The full pipeline, record, extract features, run inference, show a result, running end to end on a physical Arm Android phone."),
      bullet("A matched on-device FP32-vs-INT8 latency benchmark, both models loaded and timed in the same run, fifty timed runs after ten warm-up runs, on the phone the app was demoed on."),
      bullet("Which execution provider actually ran each node, not just which registered, from ONNX Runtime's own profiling trace: XNNPACK and CPU only, on both models, on this device."),
      bullet("Real model-load (cold-start) time for both models, and real process memory (PSS) at baseline, after model load, and after the benchmark run."),
      bullet("A real thermal-status snapshot before and after the benchmark, via Android's PowerManager API: NONE both times on this device, for this specific short run."),
      bullet("Numerical parity between the Python training-time feature pipeline and the on-device Kotlin implementation."),
      bullet("The Infested path, demoed through a bundled real infested clip, since live infested grain was not available at demo time."),
      bullet("The three-tier bands, drawn directly from the validation set's own P(infested) distribution: clean tops out at 0.492, infested starts at 0.643, checked on the actual held-out data, not assumed."),
      bullet("A desktop CPU cross-check, FP32 vs INT8, run fresh on the exact shipped models on the development machine (AMD Ryzen 5 PRO 8540U): 1.28x mean-latency speedup, plain CPUExecutionProvider on both, no Arm-specific path involved. Confirms the INT8 win holds independent of Arm hardware acceleration, on top of the Arm number, not instead of it."),
      body("Not yet done, and not claimed as done:"),
      bullet("Battery drain over sustained use. Would need a multi-minute-plus load test; a single benchmark pass does not produce a meaningful number."),
      bullet("A live microphone test against real infested grain, rather than a bundled clip."),
      bullet("A multi-device Arm benchmark. The numbers above are one phone, a Samsung Galaxy M16, not a spread across chipsets. The 1.68x-4.92x spread seen across separate smaller benchmark passes during development, even on this one phone, is itself a reason not to over-read a single run."),
      bullet("Any calibration against an official grading standard. See Open Problems above. The tiering is heuristic, grounded in this model's own data, not in insects per kilogram or a GMB threshold."),
      bullet("The production extensions described above: logging, batch scanning, treatment guidance. All out of scope for this submission."),

      hr(),
      new Paragraph({ spacing: { before: 100 }, children: [new TextRun({ text: "License: MIT. Dataset: SPID / A-SPIDS (Daniel Kadyrov, MIT-licensed), ESC-50.", size: 17, color: SUBTLE })] }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("SiloSense_Submission_Writeup.docx", buf);
  console.log("wrote SiloSense_Submission_Writeup.docx", buf.length, "bytes");
});
