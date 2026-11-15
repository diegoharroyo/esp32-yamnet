"""Build YAMNet's core (log-mel patch -> 521 scores) from the official weights and
export it as FP32 and full-INT8 TFLite models.

    python tools/convert.py              # writes build_py/yamnet_core_{fp32,int8}.tflite
    python tools/convert.py --naive      # plain INT8, no logit clip or equalization, to compare
    python tools/convert.py --alpha 0.3  # another equalization strength (0 = logit clip only)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "yamnet_ref"))
import params as yparams  # noqa: E402
import yamnet as ymodel  # noqa: E402
from frontend import log_mel, log_mel_patches, load_wav, pad  # noqa: E402

OUT = ROOT / "build_py"
P = yparams.Params()
# Logits span roughly -680..+70, which leaves int8 a step of ~3 and makes -1 and +1 the
# same value. sigmoid(+-8) already lands on the int8 output's lowest or highest code, so
# clipping there changes nothing the INT8 model can output and gives the logits a 0.06 step.
LOGIT_CLIP = 8.0


def core_model(clip=LOGIT_CLIP):
    # The official .h5 holds the full waveform model: load it there, then copy the
    # weights layer by layer (same order) into a model that starts at the log-mel patch.
    full = ymodel.yamnet_frames_model(P)
    full.load_weights(str(OUT / "yamnet.h5"))
    inp = tf.keras.layers.Input(shape=(P.patch_frames, P.patch_bands), batch_size=1, name="log_mel")
    scores, _ = ymodel.yamnet(inp, P)
    m = tf.keras.Model(inp, scores)
    src = [l for l in full.layers if l.weights]
    dst = [l for l in m.layers if l.weights]
    assert len(src) == len(dst)
    for s, d in zip(src, dst):
        d.set_weights(s.get_weights())
    logits = m.layers[-2].output
    clipped = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -clip, clip))(logits)
    scores = tf.keras.layers.Activation("sigmoid")(clipped)
    return tf.keras.Model(inp, scores), full


def equalize(m, calib, alpha=0.5):
    """Rescale every channel of every ReLU output towards its layer's widest channel.

    From layer 12's pointwise output to layer 14's depthwise one the widest channel is 24-160x
    the median one, so one int8 scale per tensor rounds most values of the narrow ones to 0 or
    1 step. The scales come from the activations on the calibration clips (99.99th percentile).
    ReLU commutes with positive scaling: multiplying a channel's conv weights and BN mean/beta
    by t and the next layer's weights on that channel by 1/t leaves the FP32 model unchanged.
    alpha < 1 only moves part of the way, because the 1/t lands in the next layer's weights:
    free for a depthwise layer (one scale per channel), but a pointwise or dense layer shares
    one scale across input channels.
    """
    convs = [l for l in m.layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))]
    bns = [l for l in m.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    relus = [l for l in m.layers if isinstance(l, tf.keras.layers.ReLU)]
    assert len(convs) == len(bns) == len(relus)
    acts = tf.keras.Model(m.input, [r.output for r in relus]).predict(calib, batch_size=1, verbose=0)
    prev = np.ones(1, np.float32)  # scale already applied to each input channel
    for conv, bn, a in zip(convs, bns, acts):
        r = np.percentile(a.reshape(-1, a.shape[-1]), 99.99, axis=0)
        t = np.where(r > 1e-6, (r.max() / np.maximum(r, 1e-6)) ** alpha, 1.0).astype(np.float32)
        (k,) = conv.get_weights()
        if isinstance(conv, tf.keras.layers.DepthwiseConv2D):
            k = k * (t / prev)[None, None, :, None]
        else:
            k = k * t[None, None, None, :] / prev[None, None, :, None]
        conv.set_weights([k])
        beta, mean, var = bn.get_weights()
        bn.set_weights([beta * t, mean * t, var])
        prev = t
    dense = [l for l in m.layers if isinstance(l, tf.keras.layers.Dense)][-1]
    k, b = dense.get_weights()
    dense.set_weights([k / prev[:, None], b])


def calib_patches(n_max=600):
    """Log-mel patches from the calibration clips (ESC-50 folds 1-4)."""
    import csv
    meta = ROOT / "data" / "ESC-50-master" / "meta" / "esc50.csv"
    rows = [r for r in csv.DictReader(open(meta)) if r["fold"] != "5"]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    out = []
    for r in rows:
        wav = load_wav(ROOT / "data" / "ESC-50-master" / "audio" / r["filename"])
        out.extend(log_mel_patches(wav))
        if len(out) >= n_max:
            break
    return np.stack(out[:n_max]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", action="store_true")
    # On ESC-50 fold 4 (FOLD=4 tools/evaluate.py) 0.3-0.7 all land within 10 clips (69.7-72.3%),
    # with 0.4 and 0.6 highest; 0.5 is the middle.
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    naive = args.naive
    clip = np.inf if naive else LOGIT_CLIP
    m, full = core_model(clip)
    wav = np.random.default_rng(0).standard_normal(40000).astype(np.float32) * 0.1
    # tools/frontend.py against YAMNet's own TF front-end, on noise and on a real clip
    for w in (wav, load_wav(ROOT / "data" / "ESC-50-master" / "audio" / "5-9032-A-0.wav")):
        lm = full(w)[2].numpy()
        assert np.abs(lm - log_mel(pad(w))).max() < 1e-3
    ref = full(wav)[0].numpy()
    got = m(np.stack(log_mel_patches(wav))).numpy()
    ref = np.clip(ref, 1 / (1 + np.exp(clip)), 1 / (1 + np.exp(-clip)))
    assert np.abs(ref - got).max() < 1e-3, np.abs(ref - got).max()

    calib = calib_patches()
    if not naive:
        conv = tf.lite.TFLiteConverter.from_keras_model(m)
        (OUT / "yamnet_core_fp32.tflite").write_bytes(conv.convert())
        before = m(calib[:8]).numpy()
        equalize(m, calib, args.alpha)
        assert np.abs(m(calib[:8]).numpy() - before).max() < 1e-4
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([p[None]] for p in calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tag = "_naive" if naive else "" if args.alpha == 0.5 else f"_a{args.alpha:g}"
    out = OUT / f"yamnet_core_int8{tag}.tflite"
    out.write_bytes(conv.convert())
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
