package com.silosense.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONArray
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Wraps an ONNX Runtime OrtSession around one of the bundled .onnx models
 * (assets/silosense_int8.onnx by default, or assets/silosense_fp32.onnx —
 * see [modelAsset]). Label convention (see scripts/prepare_dataset.py):
 * logits index 0 = clean, index 1 = infested.
 *
 * Benchmark methodology follows legacy_termux_proot/benchmark_on_device.py:
 * time only session.run() (not feature extraction), warm up and discard the
 * first few runs before reporting numbers — see [warmUp] and [benchmark].
 *
 * Tiering: there is no recognized standard mapping this model's P(infested)
 * to a grain-grading threshold. Real grading standards (GMB-style, USDA
 * FGIS, IPM action thresholds) measure insects per kilogram or percent
 * insect-damaged kernels, not acoustic-model confidence. [CLEAN_BOUND] and
 * [INFESTED_BOUND] instead come from this model's own held-out validation
 * set: P(infested) tops out at 0.492 for true clean clips and starts at
 * 0.643 for true infested ones (see scripts/quantize.py's val split). The
 * band between is reported as UNCERTAIN rather than forced into a color.
 *
 * Execution-provider profiling: [configuredProviders] only reports what was
 * successfully *registered* on the session, not which provider actually ran
 * a given node — ONNX Runtime's simple accessors don't expose that. Session
 * profiling does: [diagnoseExecutionProviders] enables it, runs a few real
 * inferences, ends profiling, and parses the resulting Chrome-trace JSON for
 * each node's actual "provider" field. Call it once, early, before relying
 * on [benchmark] for a clean timing number — profiling adds bookkeeping
 * overhead per call, so it's deliberately a separate, one-shot pass.
 */
class SiloSenseClassifier(context: Context, modelAsset: String = "silosense_int8.onnx") : AutoCloseable {

    enum class Tier { LIKELY_CLEAN, UNCERTAIN, LIKELY_INFESTED }

    data class Prediction(
        val label: String,
        val confidence: Float,
        val cleanProb: Float,
        val infestedProb: Float,
        val tier: Tier,
        val inferenceMs: Double,
    )

    data class BenchmarkResult(
        val minMs: Double,
        val avgMs: Double,
        val maxMs: Double,
        val runs: Int,
    )

    /** Execution providers successfully configured on the session (registered, not necessarily what ran a given node). */
    val configuredProviders: String

    /** Wall-clock time to read the model asset and create the OrtSession — real measured cold start, not asserted. */
    val loadTimeMs: Double

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val profilePrefix: String
    private var profilingEnded = false

    init {
        val loadStart = System.nanoTime()
        val modelBytes = context.assets.open(modelAsset).use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        val providers = mutableListOf<String>()
        try {
            options.addNnapi()
            providers += "NNAPI"
        } catch (e: OrtException) {
            // NNAPI EP unavailable on this build/device — falls through.
        }
        try {
            options.addXnnpack(emptyMap())
            providers += "XNNPACK"
        } catch (e: OrtException) {
            // XNNPACK EP unavailable — falls through.
        }
        providers += "CPU"
        configuredProviders = providers.joinToString("+")

        profilePrefix = File(context.cacheDir, "ort_profile_${modelAsset.substringBeforeLast('.')}_").absolutePath
        try {
            options.enableProfiling(profilePrefix)
        } catch (e: OrtException) {
            // Profiling unavailable — diagnoseExecutionProviders() will report that explicitly.
        }

        session = env.createSession(modelBytes, options)
        loadTimeMs = (System.nanoTime() - loadStart) / 1_000_000.0
    }

    /** Runs [count] inferences on dummy zero features and discards the results/timings. */
    fun warmUp(count: Int = 5) {
        val dummy = Array(AudioFeatures.N_MELS) { FloatArray(AudioFeatures.N_FRAMES) }
        repeat(count) { runOnce(dummy) }
    }

    /**
     * One-shot diagnostic: runs [count] real inferences with profiling active,
     * ends profiling, and returns a summary of which execution provider
     * actually executed each profiled node (e.g. "CPUExecutionProvider=6" or
     * "XnnpackExecutionProvider=4, CPUExecutionProvider=2"), pulled straight
     * from ONNX Runtime's own trace output. Must be called before [benchmark]
     * if you want an official timing number uncontaminated by profiling
     * overhead — this ends profiling as its last step.
     */
    fun diagnoseExecutionProviders(features: Array<FloatArray>, count: Int = 5): String {
        if (profilingEnded) return "profiling already ended for this session"
        repeat(count) { runOnce(features) }
        val filePath = try {
            session.endProfiling()
        } catch (e: OrtException) {
            return "profiling unavailable (${e.message})"
        } finally {
            profilingEnded = true
        }
        return try {
            summarizeProviders(filePath)
        } catch (e: Exception) {
            "profile written but could not be parsed (${e.message})"
        }
    }

    private fun summarizeProviders(profileJsonPath: String): String {
        val text = File(profileJsonPath).readText()
        val events = JSONArray(text)
        val counts = linkedMapOf<String, Int>()
        for (i in 0 until events.length()) {
            val event = events.getJSONObject(i)
            val args = event.optJSONObject("args") ?: continue
            val provider = args.optString("provider", "")
            if (provider.isNotEmpty()) {
                counts[provider] = (counts[provider] ?: 0) + 1
            }
        }
        if (counts.isEmpty()) return "no per-node provider field found in profile trace"
        return counts.entries.joinToString(", ") { "${it.key}=${it.value}" }
    }

    private fun runOnce(features: Array<FloatArray>): FloatArray {
        val flat = FloatArray(AudioFeatures.N_MELS * AudioFeatures.N_FRAMES)
        for (i in features.indices) {
            System.arraycopy(features[i], 0, flat, i * AudioFeatures.N_FRAMES, AudioFeatures.N_FRAMES)
        }
        val shape = longArrayOf(1, AudioFeatures.N_MELS.toLong(), AudioFeatures.N_FRAMES.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape).use { input ->
            session.run(mapOf(INPUT_NAME to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val logits = (result[0].value as Array<FloatArray>)[0]
                return logits
            }
        }
    }

    /** Single timed inference — the number shown live in the UI after a recording. */
    fun predict(features: Array<FloatArray>): Prediction {
        val start = System.nanoTime()
        val logits = runOnce(features)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000.0

        val maxLogit = maxOf(logits[0], logits[1])
        val expClean = exp((logits[0] - maxLogit).toDouble())
        val expInfested = exp((logits[1] - maxLogit).toDouble())
        val sum = expClean + expInfested
        val cleanProb = (expClean / sum).toFloat()
        val infestedProb = (expInfested / sum).toFloat()
        val infested = infestedProb > cleanProb
        val tier = when {
            infestedProb < CLEAN_BOUND -> Tier.LIKELY_CLEAN
            infestedProb > INFESTED_BOUND -> Tier.LIKELY_INFESTED
            else -> Tier.UNCERTAIN
        }

        return Prediction(
            label = if (infested) "Infested" else "Clean",
            confidence = if (infested) infestedProb else cleanProb,
            cleanProb = cleanProb,
            infestedProb = infestedProb,
            tier = tier,
            inferenceMs = elapsedMs,
        )
    }

    /**
     * Runs inference back-to-back for [durationMs], no timing or warm-up
     * discipline — this exists purely to create a real, sustained load for
     * battery current-draw sampling (see DeviceDiagnostics.currentDrawMicroamps),
     * not to produce a latency number. Returns the number of inferences
     * completed in that window, for context.
     */
    fun stressRun(features: Array<FloatArray>, durationMs: Long): Int {
        val end = System.nanoTime() + durationMs * 1_000_000
        var count = 0
        while (System.nanoTime() < end) {
            runOnce(features)
            count++
        }
        return count
    }

    /** Repeated timed inference on the same real (already-extracted) features, warm-up discarded — this produces the reportable on-device benchmark number. */
    fun benchmark(features: Array<FloatArray>, warmupRuns: Int = 5, timedRuns: Int = 20): BenchmarkResult {
        repeat(warmupRuns) { runOnce(features) }
        val timesMs = DoubleArray(timedRuns)
        for (i in 0 until timedRuns) {
            val start = System.nanoTime()
            runOnce(features)
            timesMs[i] = (System.nanoTime() - start) / 1_000_000.0
        }
        return BenchmarkResult(
            minMs = timesMs.min(),
            avgMs = timesMs.average(),
            maxMs = timesMs.max(),
            runs = timedRuns,
        )
    }

    override fun close() {
        session.close()
    }

    private companion object {
        const val INPUT_NAME = "logmel"
        // Empirical val-set gap (clean max 0.492, infested min 0.643), padded to round numbers.
        const val CLEAN_BOUND = 0.45f
        const val INFESTED_BOUND = 0.65f
    }
}
