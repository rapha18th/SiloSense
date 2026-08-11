package com.silosense.app

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Numerically diffs AudioFeatures.logMel() against a reference log-mel
 * tensor computed by the Python implementation (scripts/features.py) on the
 * same real, already-16kHz-mono audio clip. Per PORTING_NOTES.md, a silent
 * mismatch here wouldn't crash anything — it would just quietly degrade
 * accuracy — so this is checked numerically, not by eyeballing the code.
 *
 * Fixtures (test/resources/test_audio_16k.pcm, test_logmel_ref.bin) are
 * little-endian float32 arrays generated from
 * legacy_termux_proot/phone_package/data/clips/1-17808-B-12.wav via:
 *   y = features.load_audio(path); y_fixed = features.fixed_length(y)
 *   logmel_ref = features.logmel(y)
 */
class AudioFeaturesParityTest {

    private fun readFloats(resourceName: String): FloatArray {
        val stream = javaClass.classLoader!!.getResourceAsStream(resourceName)
            ?: error("missing test resource: $resourceName")
        val bytes = DataInputStream(stream).use { it.readBytes() }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) out[i] = buf.getFloat(i * 4)
        return out
    }

    @Test
    fun logMelMatchesPythonReference() {
        val pcm = readFloats("test_audio_16k.pcm")
        require(pcm.size == AudioFeatures.CLIP_SAMPLES) { "fixture PCM length ${pcm.size} != ${AudioFeatures.CLIP_SAMPLES}" }

        val refFlat = readFloats("test_logmel_ref.bin")
        require(refFlat.size == AudioFeatures.N_MELS * AudioFeatures.N_FRAMES)

        val actual = AudioFeatures.logMel(pcm)

        var maxAbsDiff = 0.0
        var sumAbsDiff = 0.0
        for (i in 0 until AudioFeatures.N_MELS) {
            for (f in 0 until AudioFeatures.N_FRAMES) {
                val ref = refFlat[i * AudioFeatures.N_FRAMES + f]
                val diff = kotlin.math.abs(actual[i][f] - ref).toDouble()
                maxAbsDiff = maxOf(maxAbsDiff, diff)
                sumAbsDiff += diff
            }
        }
        val meanAbsDiff = sumAbsDiff / (AudioFeatures.N_MELS * AudioFeatures.N_FRAMES)
        println("logMel parity: maxAbsDiff=$maxAbsDiff meanAbsDiff=$meanAbsDiff")

        assertTrue("max abs diff too large: $maxAbsDiff", maxAbsDiff < 1e-3)
        assertTrue("mean abs diff too large: $meanAbsDiff", meanAbsDiff < 1e-4)
    }
}
