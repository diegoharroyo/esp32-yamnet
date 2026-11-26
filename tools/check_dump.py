"""Check a firmware dump (CONFIG_YAMNET_DUMP=y) against the Python pipeline.

For every patch the firmware printed, it compares
  - the int8 input from the C front-end with tools/frontend.py on the same clip, and
  - the int8 output of TFLite Micro with TFLite's reference kernels on that same input.

TFLite Micro's kernels (esp-nn's on the ESP32-S3, and its own) can round a value
differently from TFLite's reference kernels. One step of difference in a logit (0.063)
moves the sigmoid output by up to 0.063 * 0.25 = 4 steps of 1/256, plus 1 for the
output's own rounding, so up to 5 is allowed: a rule of thumb, since a rounding
difference further up can grow on its way to the logits. The C front-end's int8 inputs
may differ from Python's by 1 step (a value within float error of a rounding boundary).
The check fails on a bigger difference, or when more than 0.1% of the values differ.
Outputs at the lowest code (score < 0.002) can't show a logit error, so it also prints how
many are above it.

    python tools/check_dump.py firmware/qemu.log   (or the board's serial log)
"""
import struct
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from frontend import log_mel_patches  # noqa: E402


def read_clips():
    blob = (ROOT / "firmware" / "main" / "clips.bin").read_bytes()
    (count,), p, out = struct.unpack_from("<I", blob), 4, []
    for _ in range(count):
        name, n = struct.unpack_from("<32sI", blob, p)
        out.append((name.rstrip(b"\0").decode(), np.frombuffer(blob, "<i2", n, p + 36)))
        p += 36 + 2 * n
    return out


def read_dump(path):
    dump = {}
    for line in open(path, errors="replace"):
        f = line.split()
        # A line cut short or garbled by the log (QEMU quit mid-print) is skipped; its patch then
        # counts as missing.
        if len(f) == 4 and f[0] in ("IN", "OUT") and len(f[3]) == 2 * (96 * 64 if f[0] == "IN" else 521):
            try:
                dump[f[0], int(f[1]), int(f[2])] = np.frombuffer(bytes.fromhex(f[3]), np.int8)
            except ValueError:
                continue
    if not dump:
        sys.exit("no IN/OUT lines in the log: build with sdkconfig.dump (CONFIG_YAMNET_DUMP=y)")
    return dump


def main():
    dump = read_dump(sys.argv[1])
    it = tf.lite.Interpreter(model_path=str(ROOT / "build_py" / "yamnet_core_int8.tflite"),
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    patches = in_diff = in_max = out_diff = out_max = live = 0
    for c, (name, pcm) in enumerate(read_clips()):
        ref = log_mel_patches(pcm / np.float32(32768))
        assert ("IN", c, len(ref)) not in dump, f"{name}: firmware made more patches than Python"
        for k, patch in enumerate(ref):
            if ("IN", c, k) not in dump or ("OUT", c, k) not in dump:
                sys.exit(f"{name}: patch {k} missing, the log ends early")
            got_in, got_out = dump[("IN", c, k)], dump[("OUT", c, k)]
            want_in = np.clip(np.round(patch / s) + z, -128, 127).astype(np.int8).ravel()
            d = np.abs(got_in.astype(int) - want_in)
            in_diff += np.count_nonzero(d)
            in_max = max(in_max, d.max())
            it.set_tensor(i["index"], got_in.reshape(i["shape"]))
            it.invoke()
            want_out = it.get_tensor(o["index"]).ravel().astype(int)
            live += np.count_nonzero((want_out > -128) | (got_out > -128))
            d = np.abs(want_out - got_out)
            out_diff += np.count_nonzero(d)
            out_max = max(out_max, d.max())
            patches += 1
    n_in = patches * i["shape"].prod()
    print(f"{patches} patches")
    print(f"front-end: {in_diff} of {n_in} input values differ from Python, by at most {in_max}")
    n_out = patches * 521
    print(f"model:     {out_diff} of {n_out} output values differ from TFLite, by at most {out_max}; "
          f"{live} are above the lowest code")
    sys.exit(1 if in_max > 1 or out_max > 5 or in_diff > n_in / 1000 or out_diff > n_out / 1000 else 0)


if __name__ == "__main__":
    main()
