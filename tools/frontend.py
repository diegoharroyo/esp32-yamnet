"""YAMNet's log-mel front-end in plain numpy, the reference for firmware/main/frontend.c.

Same maths as yamnet_ref/features.py (tf.signal.stft + linear_to_mel_weight_matrix):
16 kHz, 25 ms periodic Hann window, 10 ms hop, 512-point FFT magnitude, 64 HTK-mel bands
125-7500 Hz, log(mel + 0.001), patches of 96 frames every 48 frames.
"""
import numpy as np

SR = 16000
WIN = 400
HOP = 160
NFFT = 512
NBINS = NFFT // 2 + 1
NMEL = 64
PATCH = 96
PATCH_HOP = 48
LOG_OFFSET = 0.001


def hann():
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)


def mel_matrix():
    """tf.signal.linear_to_mel_weight_matrix, float32, shape (NBINS, NMEL)."""
    def hz_to_mel(f):
        return 1127.0 * np.log1p(np.asarray(f, np.float64) / 700.0)

    bins_hz = np.linspace(0.0, SR / 2, NBINS)[1:]
    spec_mel = hz_to_mel(bins_hz)[:, None]
    edges = np.linspace(hz_to_mel(125.0), hz_to_mel(7500.0), NMEL + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    w = np.maximum(0.0, np.minimum((spec_mel - lower) / (center - lower),
                                   (upper - spec_mel) / (upper - center)))
    return np.pad(w, [[1, 0], [0, 0]]).astype(np.float32)


def log_mel(wav):
    """(n_frames, 64) log-mel spectrogram of a float32 waveform in [-1, 1]."""
    wav = np.asarray(wav, np.float32)
    n = 1 + (len(wav) - WIN) // HOP
    idx = np.arange(WIN)[None] + HOP * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(wav[idx] * hann(), NFFT)).astype(np.float32)
    return np.log(mag @ mel_matrix() + LOG_OFFSET).astype(np.float32)


def pad(wav):
    """yamnet_ref.features.pad_waveform: silence up to a whole number of patches."""
    min_n = int((0.96 + 0.025 - 0.010) * SR)
    n = max(len(wav), min_n)
    hops = int(np.ceil((n - min_n) / (PATCH_HOP * HOP)))
    return np.pad(wav, (0, min_n + hops * PATCH_HOP * HOP - len(wav)))


def log_mel_patches(wav):
    lm = log_mel(pad(wav))
    n = 1 + (len(lm) - PATCH) // PATCH_HOP
    return [lm[i * PATCH_HOP: i * PATCH_HOP + PATCH] for i in range(n)]


def load_wav(path):
    import resampy
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return resampy.resample(wav, sr, SR).astype(np.float32) if sr != SR else wav
