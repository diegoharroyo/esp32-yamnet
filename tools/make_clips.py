"""Pack a few ESC-50 fold-5 clips into firmware/main/clips.bin for the firmware to classify.

ESC-50 is CC BY-NC, so the clips are not in the repo: download ESC-50 into data/ and run
this before building. For each home-sound category it takes the first fold-5 clip that the
INT8 model gets right (tools/evaluate.py gives the accuracy over all of them).

Layout, little-endian: u32 count, then per clip: char name[32], u32 n_samples, int16 samples
(16 kHz mono, padded to an even count so the next header stays 4-byte aligned).
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import evaluate as ev  # noqa: E402
from frontend import log_mel_patches  # noqa: E402

CATEGORIES = ["dog", "glass_breaking", "door_wood_knock", "crying_baby", "siren", "clock_alarm"]
OUT = ROOT / "firmware" / "main" / "clips.bin"


def main():
    model = ev.Model(ROOT / "build_py" / "yamnet_core_int8.tflite")
    idx = ev.map_indices()
    cats = list(ev.MAP)
    meta = ROOT / "data" / "ESC-50-master" / "meta" / "esc50.csv"
    rows = [r for r in ev.csv.DictReader(open(meta)) if r["fold"] == "5"]
    blob = bytearray(struct.pack("<I", len(CATEGORIES)))
    for cat in CATEGORIES:
        for r in (r for r in rows if r["category"] == cat):
            wav = ev.load_wav(ROOT / "data" / "ESC-50-master" / "audio" / r["filename"])
            pcm = np.clip(np.round(wav * 32768), -32768, 32767).astype("<i2")
            clip = np.mean([model(p) for p in log_mel_patches(pcm / np.float32(32768))], axis=0)
            per_cat = np.array([clip[idx[c]].max() for c in cats])
            j = cats.index(cat)
            if per_cat[j] > np.delete(per_cat, j).max():  # right as in evaluate.py: a tie is wrong
                break
        else:
            raise SystemExit(f"no fold-5 {cat} clip is classified right")
        pcm = np.pad(pcm, (0, len(pcm) % 2))
        blob += struct.pack("<32sI", cat.encode(), len(pcm)) + pcm.tobytes()
        print(f"{cat:16s} {r['filename']}")
    OUT.write_bytes(blob)
    print(OUT, len(blob), "bytes")


if __name__ == "__main__":
    main()
