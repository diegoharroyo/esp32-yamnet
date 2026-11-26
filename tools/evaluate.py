"""Accuracy of the FP32 and INT8 models on ESC-50 fold 5 (clips never used for calibration
or tuning; FOLD=4 evaluates on fold 4 instead, where the equalization strength was picked,
and which, like folds 1-3, also supplies calibration clips).

YAMNet was trained on AudioSet, not ESC-50, so each ESC-50 class is scored as the highest
mean score among the AudioSet classes that name the same sound (MAP below), and a clip is
right when its true class scores strictly highest (a tie counts as wrong).
Classes with no counterpart among YAMNet's 521 are left out.

    python tools/evaluate.py [build_py/yamnet_core_int8.tflite ...]
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from frontend import load_wav, log_mel_patches  # noqa: E402

MAP = {
    "airplane": ["Aircraft", "Fixed-wing aircraft", "Aircraft engine", "Jet engine", "Propeller"],
    "breathing": ["Breathing", "Pant", "Gasp", "Wheeze"],
    "brushing_teeth": ["Toothbrush", "Electric toothbrush"],
    "car_horn": ["Vehicle horn", "Toot", "Air horn"],
    "cat": ["Cat", "Meow", "Purr", "Caterwaul"],
    "chainsaw": ["Chainsaw"],
    "chirping_birds": ["Bird", "Bird vocalization", "Chirp"],
    "church_bells": ["Church bell", "Change ringing"],
    "clapping": ["Clapping", "Applause"],
    "clock_alarm": ["Alarm clock"],
    "clock_tick": ["Tick", "Tick-tock"],
    "coughing": ["Cough"],
    "cow": ["Cattle", "Moo"],
    "crackling_fire": ["Fire", "Crackle"],
    "crickets": ["Cricket"],
    "crow": ["Crow", "Caw"],
    "crying_baby": ["Baby cry"],
    "dog": ["Dog", "Bark", "Bow-wow"],
    "door_wood_creaks": ["Creak", "Squeak"],
    "door_wood_knock": ["Knock"],
    "engine": ["Engine", "Idling", "Medium engine", "Heavy engine"],
    "fireworks": ["Fireworks", "Firecracker"],
    "footsteps": ["Walk"],
    "frog": ["Frog", "Croak"],
    "glass_breaking": ["Glass", "Shatter", "Breaking"],
    "hand_saw": ["Sawing"],
    "helicopter": ["Helicopter"],
    "hen": ["Chicken", "Cluck"],
    "insects": ["Insect", "Fly", "Buzz", "Mosquito"],
    "keyboard_typing": ["Typing", "Computer keyboard"],
    "laughing": ["Laughter", "Giggle", "Chuckle", "Belly laugh"],
    "mouse_click": ["Clicking"],
    "pig": ["Pig", "Oink"],
    "pouring_water": ["Pour", "Fill (with liquid)"],
    "rain": ["Rain", "Raindrop", "Rain on surface"],
    "rooster": ["Crowing"],
    "sea_waves": ["Waves", "Ocean"],
    "sheep": ["Sheep", "Bleat"],
    "siren": ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine", "Emergency vehicle"],
    "sneezing": ["Sneeze"],
    "snoring": ["Snoring"],
    "thunderstorm": ["Thunderstorm", "Thunder"],
    "toilet_flush": ["Toilet flush"],
    "train": ["Train", "Rail transport", "Railroad car", "Train wheels squealing"],
    "vacuum_cleaner": ["Vacuum cleaner"],
    "water_drops": ["Drip"],
    "wind": ["Wind", "Wind noise (microphone)"],
}


def class_names():
    with open(ROOT / "tools" / "yamnet_ref" / "yamnet_class_map.csv") as f:
        return [r["display_name"] for r in csv.DictReader(f)]


def map_indices():
    names = class_names()
    out = {}
    for esc, targets in MAP.items():
        idx = []
        for t in targets:  # "Train" also matches "Train, ..." and "Train (...)", nothing else
            hit = [i for i, n in enumerate(names) if n == t or n.startswith((t + ",", t + " ("))]
            assert len(hit) == 1, (esc, t, [names[i] for i in hit])
            idx += hit
        out[esc] = idx
    return out


class Model:
    def __init__(self, path):
        # TFLite's reference kernels: on the board and in the emulator, TFLite Micro gives the same outputs
        # but for 3 of 31,260, off by at most 3 steps (tools/check_dump.py). TFLite's default
        # (optimized) PC kernels round differently.
        self.it = tf.lite.Interpreter(model_path=str(path),
                                      experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def __call__(self, patch):
        x = patch[None]
        if self.i["dtype"] == np.int8:
            s, z = self.i["quantization"]
            x = np.clip(np.round(x / s) + z, -128, 127).astype(np.int8)
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.o["index"])[0]
        if self.o["dtype"] == np.int8:
            s, z = self.o["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y


def clips(fold=os.environ.get("FOLD", "5")):
    meta = ROOT / "data" / "ESC-50-master" / "meta" / "esc50.csv"
    rows = [r for r in csv.DictReader(open(meta)) if r["fold"] == fold and r["category"] in MAP]
    for r in rows:
        wav = load_wav(ROOT / "data" / "ESC-50-master" / "audio" / r["filename"])
        yield r["category"], np.stack(log_mel_patches(wav))


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [ROOT / "build_py" / f"yamnet_core_{k}.tflite"
                                                for k in ("fp32", "int8")]
    models = [Model(p) for p in paths]
    idx = map_indices()
    cats = list(MAP)
    right = np.zeros(len(models))
    top1 = []  # 521-class top-1 of each model, to compare models with each other
    n = 0
    for cat, patches in clips():
        n += 1
        tops = []
        for k, m in enumerate(models):
            # A clip's score per class is the mean over its patches, as in YAMNet's own inference.
            clip = np.mean([m(p) for p in patches], axis=0)
            per_cat = np.array([clip[idx[c]].max() for c in cats])
            j = cats.index(cat)
            right[k] += per_cat[j] > np.delete(per_cat, j).max()
            tops.append(int(np.argmax(clip)))  # a tie goes to the lower class index
        top1.append(tops)
    if not n:
        sys.exit(f"no clips in fold {os.environ.get('FOLD', '5')}")
    top1 = np.array(top1)
    print(f"{n} clips, {len(cats)} classes")
    for k, p in enumerate(paths):
        agree = (top1[:, k] == top1[:, 0]).mean()
        print(f"{p.name:30s} accuracy {100 * right[k] / n:5.1f}%   "
              f"same top-1 as {paths[0].name}: {100 * agree:5.1f}%")


if __name__ == "__main__":
    main()
