"""Shared audio feature extraction: fixed-length log-mel spectrograms.

Pure numpy + soundfile — no librosa. librosa pulls in numba/llvmlite (and
historically resampy), which frequently has no prebuilt aarch64 Linux wheel
and tries to build from source — exactly the kind of fragile dependency that
breaks inside Termux's proot-distro Ubuntu on a phone. STFT, mel filterbank,
and dB conversion are hand-rolled below (same numeric convention as
librosa's defaults: reflect-padded, centered STFT; Slaney-style mel scale;
power_to_db relative to the clip's own peak, floored at -80dB) so training
(Windows) and on-device inference (phone) run identical, dependency-light
code — reliability on the target hardware mattered more here than matching
librosa's exact output.
"""
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.5
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160  # 10ms at 16kHz

# Fixed number of frames so every clip produces an identically-shaped feature.
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH

_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic Hann


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    n_freqs = n_fft // 2 + 1
    mel_min, mel_max = _hz_to_mel(0.0), _hz_to_mel(sr / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.clip(np.floor((n_fft + 1) * hz_points / sr).astype(int), 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, n_freqs)):
            fb[i, k] = (k - left) / (center - left)
        for k in range(center, min(right, n_freqs)):
            fb[i, k] = (right - k) / (right - center)
    return fb


_MEL_FB = _build_mel_filterbank()  # (N_MELS, N_FFT//2 + 1), built once at import time


def resample_linear(y, orig_sr, target_sr):
    # Naive linear-interpolation resample (no anti-aliasing lowpass) — a
    # deliberate simplification for this prototype; adequate for broadband
    # insect/environmental audio, avoids pulling in scipy/resampy.
    if orig_sr == target_sr or len(y) == 0:
        return y.astype(np.float32)
    duration = len(y) / orig_sr
    n_target = max(1, int(round(duration * target_sr)))
    x_orig = np.linspace(0.0, duration, num=len(y), endpoint=False)
    x_target = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_target, x_orig, y).astype(np.float32)


def load_audio(path, offset=0.0, duration=None):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        start_frame = int(offset * sr)
        f.seek(start_frame)
        n_frames = int(duration * sr) if duration is not None else -1
        y = f.read(frames=n_frames, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return resample_linear(np.asarray(y, dtype=np.float32), sr, SAMPLE_RATE)


def fixed_length(y, n_samples=CLIP_SAMPLES):
    if len(y) >= n_samples:
        return y[:n_samples]
    pad = n_samples - len(y)
    return np.pad(y, (0, pad), mode="constant")


def _power_to_db(power, top_db=80.0, amin=1e-10):
    log_spec = 10.0 * np.log10(np.maximum(amin, power))
    log_spec -= 10.0 * np.log10(np.maximum(amin, power.max()))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def logmel(y):
    y = fixed_length(y)
    pad = N_FFT // 2
    y_padded = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(y_padded) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        y_padded,
        shape=(N_FFT, n_frames),
        strides=(y_padded.strides[0], y_padded.strides[0] * HOP_LENGTH),
    )
    spec = np.fft.rfft(frames * _WINDOW[:, None], n=N_FFT, axis=0)
    power = spec.real ** 2 + spec.imag ** 2  # (N_FFT//2+1, n_frames)

    mel = _MEL_FB @ power  # (N_MELS, n_frames)
    logmel_db = _power_to_db(mel)
    # Normalize to roughly [-1, 1] for stable training.
    logmel_db = (logmel_db + 40.0) / 40.0

    out = logmel_db[:, :N_FRAMES].astype(np.float32)
    if out.shape[1] < N_FRAMES:
        out = np.pad(out, ((0, 0), (0, N_FRAMES - out.shape[1])), mode="constant")
    return out  # shape: (N_MELS, N_FRAMES)
