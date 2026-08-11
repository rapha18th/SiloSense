package com.silosense.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * Records ~1.5s of 16kHz mono audio, runs it through [AudioFeatures.logMel]
 * and [SiloSenseClassifier], and shows the prediction plus an on-device
 * latency benchmark (see PORTING_NOTES.md / NEXT_STEPS.md for the spec this
 * follows).
 *
 * Also ships a bundled real infested-grain clip (assets/sample_infested.pcm,
 * a genuine SPID/A-SPIDS recording, MIT-licensed, verified to predict
 * "Infested" — see the "Try a real infested-grain sample" link) so the
 * Infested path can be demoed without needing live infested grain on hand.
 *
 * Loads both the INT8 model (used for every live prediction) and the FP32
 * model (loaded only to benchmark, never used for a live result) so the
 * Results dialog can show a real matched on-device FP32-vs-INT8 pair,
 * alongside the desktop cross-check numbers from README/the submission doc.
 */
class MainActivity : AppCompatActivity() {

    private val requestAudioPermission =
        registerForActivityResult(androidx.activity.result.contract.ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) showReady() else showMessage("PERMISSION DENIED", "Microphone access is required to listen to grain audio", accent())
        }

    private lateinit var statusWord: TextView
    private lateinit var confidenceText: TextView
    private lateinit var detailText: TextView
    private lateinit var latencyChip: TextView
    private lateinit var recordButton: Button
    private lateinit var sampleLink: TextView
    private lateinit var sourceLink: TextView
    private lateinit var resultsLink: TextView

    private var int8Classifier: SiloSenseClassifier? = null
    private var fp32Classifier: SiloSenseClassifier? = null
    private var deviceInt8Bench: SiloSenseClassifier.BenchmarkResult? = null
    private var deviceFp32Bench: SiloSenseClassifier.BenchmarkResult? = null

    // Real, measured diagnostics — see DeviceDiagnostics.kt and
    // SiloSenseClassifier.diagnoseExecutionProviders(). Populated once,
    // during initial model load, before the official benchmark runs.
    private var baselinePssKb: Int? = null
    private var afterLoadPssKb: Int? = null
    private var afterBenchPssKb: Int? = null
    private var thermalBefore: String? = null
    private var thermalAfter: String? = null
    private var int8ProviderSummary: String? = null
    private var fp32ProviderSummary: String? = null

    private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusWord = findViewById(R.id.statusWord)
        confidenceText = findViewById(R.id.confidenceText)
        detailText = findViewById(R.id.detailText)
        latencyChip = findViewById(R.id.latencyChip)
        recordButton = findViewById(R.id.recordButton)
        sampleLink = findViewById(R.id.sampleLink)
        sourceLink = findViewById(R.id.sourceLink)
        resultsLink = findViewById(R.id.resultsLink)

        recordButton.isEnabled = false
        showMessage("LOADING MODEL…", "", secondary())

        thread {
            baselinePssKb = DeviceDiagnostics.pssKb()

            val int8 = SiloSenseClassifier(applicationContext, "silosense_int8.onnx")
            int8.warmUp()
            int8Classifier = int8
            runOnUiThread { showReady() }

            val fp32 = SiloSenseClassifier(applicationContext, "silosense_fp32.onnx")
            fp32.warmUp()
            fp32Classifier = fp32

            afterLoadPssKb = DeviceDiagnostics.pssKb()

            // Matched on-device benchmark: same real features, both models, same discipline.
            val sampleFeatures = AudioFeatures.logMel(loadSampleClip())

            // One-shot diagnostic pass: identify which execution provider
            // actually ran each node, via ORT's own profiling trace, before
            // running the clean (profiling-off) official benchmark below.
            int8ProviderSummary = int8.diagnoseExecutionProviders(sampleFeatures)
            fp32ProviderSummary = fp32.diagnoseExecutionProviders(sampleFeatures)

            thermalBefore = DeviceDiagnostics.thermalStatus(applicationContext)

            // Official benchmark: steadier sample (10 warmup / 50 timed) than
            // the 5/20 default used for the quick live-tap number, since this
            // is the figure that goes into the write-up and README.
            deviceInt8Bench = int8.benchmark(sampleFeatures, warmupRuns = 10, timedRuns = 50)
            deviceFp32Bench = fp32.benchmark(sampleFeatures, warmupRuns = 10, timedRuns = 50)

            thermalAfter = DeviceDiagnostics.thermalStatus(applicationContext)
            afterBenchPssKb = DeviceDiagnostics.pssKb()
        }

        recordButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                requestAudioPermission.launch(Manifest.permission.RECORD_AUDIO)
            } else {
                runPipeline { recordClip() }
            }
        }

        sampleLink.setOnClickListener {
            runPipeline { loadSampleClip() }
        }

        sourceLink.setOnClickListener { showSourceDialog() }
        resultsLink.setOnClickListener { showResultsDialog() }
    }

    private fun showSourceDialog() {
        AlertDialog.Builder(this)
            .setTitle(R.string.source_dialog_title)
            .setMessage(R.string.source_dialog_body)
            .setPositiveButton(R.string.source_dialog_close, null)
            .show()
    }

    private fun showResultsDialog() {
        val int8Bench = deviceInt8Bench
        val fp32Bench = deviceFp32Bench
        val body = if (int8Bench == null || fp32Bench == null) {
            "Still benchmarking on this device. Try again in a moment."
        } else {
            buildResultsTable(int8Bench, fp32Bench)
        }

        val textView = TextView(this).apply {
            text = body
            typeface = Typeface.MONOSPACE
            textSize = 12f
            setTextColor(ContextCompat.getColor(this@MainActivity, R.color.silo_text))
            setPadding(48, 32, 48, 16)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.results_dialog_title)
            .setView(textView)
            .setPositiveButton(R.string.source_dialog_close, null)
            .show()
    }

    private fun buildResultsTable(
        int8Bench: SiloSenseClassifier.BenchmarkResult,
        fp32Bench: SiloSenseClassifier.BenchmarkResult,
    ): String {
        // Built with string templates, not String.format on the whole block —
        // the static text below has literal '%' characters (98.99%, 100.00%)
        // that String.format would misparse as format-specifier flags.
        val deviceFp32 = "%.2fms".format(fp32Bench.avgMs)
        val deviceInt8 = "%.2fms".format(int8Bench.avgMs)
        val deviceSpeedup = "%.2fx".format(fp32Bench.avgMs / int8Bench.avgMs)
        val fp32Eps = fp32Classifier?.configuredProviders ?: "?"
        val int8Eps = int8Classifier?.configuredProviders ?: "?"
        val fp32Load = "%.1fms".format(fp32Classifier?.loadTimeMs ?: 0.0)
        val int8Load = "%.1fms".format(int8Classifier?.loadTimeMs ?: 0.0)

        val loadDeltaKb = (afterLoadPssKb ?: 0) - (baselinePssKb ?: 0)
        val benchDeltaKb = (afterBenchPssKb ?: 0) - (afterLoadPssKb ?: 0)

        return """
            TRAINING (offline validation)
                          FP32       INT8
            size        0.393MB    0.111MB
            accuracy     98.99%    100.00%
            F1            0.985      1.000
            size reduction: 3.54x

            ON-DEVICE (this phone, live, N=${int8Bench.runs})
            fp32 avg    $deviceFp32
            int8 avg    $deviceInt8
            speedup     $deviceSpeedup
            fp32 load   $fp32Load
            int8 load   $int8Load
            fp32 EPs    $fp32Eps
            int8 EPs    $int8Eps
            fp32 nodes  ${fp32ProviderSummary ?: "not run yet"}
            int8 nodes  ${int8ProviderSummary ?: "not run yet"}

            MEMORY (process PSS, this phone)
            baseline    ${baselinePssKb ?: "?"} KB
            after load  ${afterLoadPssKb ?: "?"} KB  (+$loadDeltaKb KB)
            after bench ${afterBenchPssKb ?: "?"} KB  (+$benchDeltaKb KB)

            THERMAL (this phone, PowerManager)
            before      ${thermalBefore ?: "?"}
            after       ${thermalAfter ?: "?"}

            battery     not measured

            DESKTOP CPU (AMD Ryzen 5 PRO 8540U, x86_64)
            fp32 avg    0.226ms
            int8 avg    0.176ms
            speedup     1.28x
            providers   CPU only, no XNNPACK
        """.trimIndent()
    }

    private fun showReady() {
        setButtonColor(accent())
        recordButton.isEnabled = true
        showMessage("READY", "Tap to record 1.5s of grain-sack audio", secondary())
        latencyChip.text = ""
    }

    private fun showMessage(status: String, detail: String, color: Int) {
        statusWord.text = status
        statusWord.setTextColor(color)
        confidenceText.text = "—"
        confidenceText.setTextColor(secondary())
        detailText.text = detail
    }

    /** Shared pipeline: get PCM (live mic or bundled sample) -> features -> predict -> show. */
    private fun runPipeline(getPcm: () -> FloatArray) {
        val activeClassifier = int8Classifier
        if (activeClassifier == null) {
            showMessage("MODEL LOADING", "Try again in a moment", secondary())
            return
        }
        if (busy) return
        busy = true

        recordButton.isEnabled = false
        sampleLink.isEnabled = false
        setButtonColor(amber())
        showMessage("LISTENING…", "", amber())
        latencyChip.text = ""

        thread {
            try {
                val pcm = getPcm()
                runOnUiThread { showMessage("ANALYZING…", "", amber()) }

                val features = AudioFeatures.logMel(pcm)
                val prediction = activeClassifier.predict(features)
                val bench = activeClassifier.benchmark(features)

                val (tierLabel, resultColor, shownPct) = when (prediction.tier) {
                    SiloSenseClassifier.Tier.LIKELY_CLEAN -> Triple("LIKELY CLEAN", cleanColor(), prediction.cleanProb)
                    SiloSenseClassifier.Tier.LIKELY_INFESTED -> Triple("LIKELY INFESTED", infestedColor(), prediction.infestedProb)
                    SiloSenseClassifier.Tier.UNCERTAIN -> Triple("UNCERTAIN: RECHECK", amber(), prediction.infestedProb)
                }
                val detail = "clean %.1f%%  ·  infested %.1f%%".format(
                    prediction.cleanProb * 100,
                    prediction.infestedProb * 100,
                )
                val latency = "%.2fms avg (%d runs)  ·  %s".format(
                    bench.avgMs,
                    bench.runs,
                    activeClassifier.configuredProviders,
                )

                runOnUiThread {
                    statusWord.text = tierLabel
                    statusWord.setTextColor(resultColor)
                    confidenceText.text = "%.0f%%".format(shownPct * 100)
                    confidenceText.setTextColor(resultColor)
                    detailText.text = detail
                    latencyChip.text = latency
                    setButtonColor(accent())
                    recordButton.isEnabled = true
                    sampleLink.isEnabled = true
                    busy = false
                }
            } catch (e: Exception) {
                runOnUiThread {
                    showMessage("ERROR", e.message ?: "unknown error", infestedColor())
                    setButtonColor(accent())
                    recordButton.isEnabled = true
                    sampleLink.isEnabled = true
                    busy = false
                }
            }
        }
    }

    private fun setButtonColor(color: Int) {
        (recordButton.background.mutate() as GradientDrawable).setColor(color)
    }

    private fun accent() = ContextCompat.getColor(this, R.color.silo_accent)
    private fun secondary() = ContextCompat.getColor(this, R.color.silo_text_secondary)
    private fun amber() = ContextCompat.getColor(this, R.color.silo_amber)
    private fun cleanColor() = ContextCompat.getColor(this, R.color.silo_green)
    private fun infestedColor() = ContextCompat.getColor(this, R.color.silo_red)

    // Permission is verified by the caller (runPipeline's mic branch is only
    // reachable from the click listener's else-branch, after checkSelfPermission).
    @Suppress("MissingPermission")
    private fun recordClip(): FloatArray {
        val sampleRate = AudioFeatures.SAMPLE_RATE
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val bufferBytes = maxOf(minBuf, AudioFeatures.CLIP_SAMPLES * 2)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferBytes,
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            throw IllegalStateException("AudioRecord failed to initialize (sampleRate=$sampleRate)")
        }

        val pcm16 = ShortArray(AudioFeatures.CLIP_SAMPLES)
        try {
            recorder.startRecording()
            var offset = 0
            while (offset < pcm16.size) {
                val read = recorder.read(pcm16, offset, pcm16.size - offset)
                if (read <= 0) break
                offset += read
            }
        } finally {
            recorder.stop()
            recorder.release()
        }
        return FloatArray(pcm16.size) { i -> pcm16[i] / 32768f }
    }

    /** Loads the bundled real infested-grain clip (16kHz mono PCM16, same layout AudioRecord produces). */
    private fun loadSampleClip(): FloatArray {
        val bytes = assets.open(SAMPLE_ASSET).use { it.readBytes() }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 2) { i -> buf.getShort(i * 2) / 32768f }
    }

    override fun onDestroy() {
        super.onDestroy()
        int8Classifier?.close()
        fp32Classifier?.close()
    }

    private companion object {
        const val SAMPLE_ASSET = "sample_infested.pcm"
    }
}
