package com.silosense.app

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Kotlin port of scripts/features.py's log-mel spectrogram extraction.
 * See ../../../../PORTING_NOTES.md at the repo root for the exact spec this
 * follows — constants, padding mode, window, and dB normalization must match
 * the Python reference exactly or the model's accuracy won't transfer.
 *
 * Pure Kotlin, no Android framework dependency, so it can be exercised from
 * a plain JVM unit test (see app/src/test/.../AudioFeaturesParityTest.kt).
 */
object AudioFeatures {

    const val SAMPLE_RATE = 16000
    const val CLIP_SECONDS = 1.5
    const val CLIP_SAMPLES = (SAMPLE_RATE * CLIP_SECONDS).toInt() // 24000
    const val N_MELS = 40
    const val N_FFT = 512
    const val HOP_LENGTH = 160
    const val N_FRAMES = 1 + CLIP_SAMPLES / HOP_LENGTH // 151
    private const val N_FREQS = N_FFT / 2 + 1 // 257

    // Periodic Hann window (not symmetric — omits the implicit final wraparound sample).
    private val WINDOW: DoubleArray = DoubleArray(N_FFT) { n ->
        0.5 - 0.5 * cos(2.0 * PI * n / N_FFT)
    }

    private val MEL_FILTERBANK: Array<DoubleArray> = buildMelFilterbank()

    private fun hzToMel(hz: Double): Double = 2595.0 * log10(1.0 + hz / 700.0)

    private fun melToHz(mel: Double): Double = 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0)

    private fun buildMelFilterbank(): Array<DoubleArray> {
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(SAMPLE_RATE / 2.0)
        val melPoints = DoubleArray(N_MELS + 2) { i -> melMin + (melMax - melMin) * i / (N_MELS + 1) }
        val hzPoints = melPoints.map { melToHz(it) }
        val bins = hzPoints.map { hz ->
            val bin = Math.floor((N_FFT + 1) * hz / SAMPLE_RATE).toInt()
            bin.coerceIn(0, N_FREQS - 1)
        }

        val fb = Array(N_MELS) { DoubleArray(N_FREQS) }
        for (i in 0 until N_MELS) {
            val left = bins[i]
            var center = bins[i + 1]
            var right = bins[i + 2]
            if (center <= left) center = left + 1
            if (right <= center) right = center + 1

            for (k in left until min(center, N_FREQS)) {
                fb[i][k] = (k - left).toDouble() / (center - left)
            }
            for (k in center until min(right, N_FREQS)) {
                fb[i][k] = (right - k).toDouble() / (right - center)
            }
        }
        return fb
    }

    /** Pad or truncate to exactly CLIP_SAMPLES, zero-padding on the right if shorter. */
    fun fixedLength(y: FloatArray, nSamples: Int = CLIP_SAMPLES): FloatArray {
        if (y.size == nSamples) return y
        if (y.size > nSamples) return y.copyOf(nSamples)
        val out = FloatArray(nSamples)
        System.arraycopy(y, 0, out, 0, y.size)
        return out
    }

    /**
     * Reflect-pad (numpy mode="reflect": mirrors without repeating the edge
     * sample) by [pad] samples on both sides.
     */
    private fun reflectPad(y: FloatArray, pad: Int): FloatArray {
        val n = y.size
        val out = FloatArray(n + 2 * pad)
        for (i in 0 until pad) out[i] = y[pad - i]
        System.arraycopy(y, 0, out, pad, n)
        for (j in 0 until pad) out[pad + n + j] = y[n - 2 - j]
        return out
    }

    /** In-place iterative radix-2 Cooley-Tukey FFT (size must be a power of 2). */
    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wr = cos(ang)
            val wi = sin(ang)
            var i = 0
            while (i < n) {
                var curWr = 1.0
                var curWi = 0.0
                for (k in 0 until len / 2) {
                    val ur = re[i + k]
                    val ui = im[i + k]
                    val vr = re[i + k + len / 2] * curWr - im[i + k + len / 2] * curWi
                    val vi = re[i + k + len / 2] * curWi + im[i + k + len / 2] * curWr
                    re[i + k] = ur + vr
                    im[i + k] = ui + vi
                    re[i + k + len / 2] = ur - vr
                    im[i + k + len / 2] = ui - vi
                    val nwr = curWr * wr - curWi * wi
                    val nwi = curWr * wi + curWi * wr
                    curWr = nwr
                    curWi = nwi
                }
                i += len
            }
            len = len shl 1
        }
    }

    /** Windowed real FFT power spectrum (N_FREQS bins) of one N_FFT-length frame. */
    private fun framePower(frame: DoubleArray): DoubleArray {
        val re = DoubleArray(N_FFT)
        val im = DoubleArray(N_FFT)
        for (n in 0 until N_FFT) re[n] = frame[n] * WINDOW[n]
        fft(re, im)
        val power = DoubleArray(N_FREQS)
        for (k in 0 until N_FREQS) power[k] = re[k] * re[k] + im[k] * im[k]
        return power
    }

    private fun powerToDb(mel: Array<DoubleArray>, topDb: Double = 80.0, amin: Double = 1e-10): Array<DoubleArray> {
        var globalMax = 0.0
        for (row in mel) for (v in row) if (v > globalMax) globalMax = v
        val refDb = 10.0 * log10(max(amin, globalMax))

        var dbMax = -Double.MAX_VALUE
        val out = Array(mel.size) { i -> DoubleArray(mel[i].size) { k -> 10.0 * log10(max(amin, mel[i][k])) - refDb } }
        for (row in out) for (v in row) if (v > dbMax) dbMax = v
        val floor = dbMax - topDb
        for (row in out) for (k in row.indices) row[k] = max(row[k], floor)
        return out
    }

    /**
     * PCM samples (any length, mono, float in roughly [-1, 1]) -> fixed
     * (N_MELS, N_FRAMES) log-mel spectrogram, normalized to roughly [-1, 1].
     * Matches scripts/features.py's `logmel()` exactly.
     */
    fun logMel(y: FloatArray): Array<FloatArray> {
        val fixed = fixedLength(y)
        val pad = N_FFT / 2
        val padded = reflectPad(fixed, pad)

        val nFrames = 1 + (padded.size - N_FFT) / HOP_LENGTH
        val mel = Array(N_MELS) { DoubleArray(nFrames) }

        for (f in 0 until nFrames) {
            val start = f * HOP_LENGTH
            val frame = DoubleArray(N_FFT) { k -> padded[start + k].toDouble() }
            val power = framePower(frame)
            for (i in 0 until N_MELS) {
                var sum = 0.0
                val fb = MEL_FILTERBANK[i]
                for (k in 0 until N_FREQS) sum += fb[k] * power[k]
                mel[i][f] = sum
            }
        }

        val logSpec = powerToDb(mel)

        // Python normalizes first, then zero-pads (constant mode) if short —
        // dead in practice since fixedLength() always yields exactly
        // N_FRAMES frames, kept only to mirror logmel()'s safety padding.
        val out = Array(N_MELS) { FloatArray(N_FRAMES) }
        for (i in 0 until N_MELS) {
            for (f in 0 until N_FRAMES) {
                out[i][f] = if (f < nFrames) ((logSpec[i][f] + 40.0) / 40.0).toFloat() else 0f
            }
        }
        return out
    }
}
