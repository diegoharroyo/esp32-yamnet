# YAMNet sound classifier for the ESP32-S3, INT8, no cloud

Google's YAMNet sound classifier (521 sound classes, from dog barks and glass breaking to sirens and a baby
crying) built for the ESP32-S3: the log-mel front-end in C and the network in INT8 with TensorFlow Lite Micro,
checked number by number, in Espressif's ESP32-S3 emulator, against the same model running on a PC. No cloud, no
phone, no audio leaves the device. It's the core of a home or security sound detector that tells you *what* it
heard, not just that something was loud. This demo classifies recorded clips stored in flash; feeding it from an
I2S microphone is the next step. Nothing has run on a physical board yet (see Limits).

```
16 kHz audio → log-mel spectrogram (C) → YAMNet INT8 (TFLite Micro) → 521 sound scores → top 3
```

Quantizing YAMNet to INT8 the usual way breaks it: accuracy falls from 71.5% to 27.9%. Two fixes, described
below, bring it back to 70.5%, 4 clips of 376 below the float model, at about a quarter of the size (4.06 MB vs
14.9 MB).

## How well it works

ESC-50 is a public benchmark of 2,000 labelled five-second clips of everyday sounds, split into 5 equal parts
("folds"). The table uses fold 5: 376 clips in 47 classes, none of them used for calibration or tuning.
Calibration is the sample audio INT8 conversion needs to set each layer's number range: here, 600 windows from
folds 1 to 4.

| Model | Size | Accuracy | Same top sound as FP32 |
|---|---|---|---|
| Float model (FP32), the reference | 14.9 MB | 71.5% | – |
| INT8, plain post-training quantization | 4.06 MB | 27.9% | 34.0% |
| INT8, fix 1 only | 4.06 MB | 31.6% | 35.4% |
| INT8, both fixes (this repo) | 4.06 MB | 70.5% | 86.4% |

These are off-the-shelf scores: YAMNet was never trained on ESC-50. What matters here is that INT8 keeps the
float model's accuracy at a quarter of the size. For a product, the usual next step is a small classifier on top
of YAMNet, trained on the sounds that matter.

What the columns mean:

- **Accuracy:** each ESC-50 class is scored as the best of the AudioSet classes that name the same sound (`dog`
  is Dog, Bark or Bow-wow; the full list is `MAP` in `tools/evaluate.py`), and a clip counts as right when its own
  class scores strictly highest. The 3 ESC-50 classes with no counterpart among YAMNet's 521 are left out. A
  clip's score is the mean over its 0.96 s windows, as in YAMNet's own inference. With 376 clips one point is
  about 4 clips, so a gap of a few clips between two models is within the test's noise.
- **Same top sound as FP32:** how often the INT8 model's single best guess among all 521 classes is the same
  as the float model's (a tie goes to the lower class index).
- The float row is the model as `tools/convert.py` builds it, with fix 1's clip on the logits; without the clip
  it scores the same (71.5%, with the same top sound on every clip).
- The INT8 numbers come from TFLite's reference kernels; the firmware's own outputs are compared with them below.

### What broke, and the two fixes

In short: 8-bit numbers can't cover two places where YAMNet's values span a huge range. Both fixes reshape the
model so they fit. Fix 2 leaves the float model unchanged (up to float32 rounding); fix 1 moves its scores by at
most 0.00034, under half an int8 output step.

1. **The output logits.** Before the final sigmoid YAMNet's logits run from about -680 to +70. Spread over 256
   int8 steps that is a step of about 3, so -1 and +1 become the same number. Every logit is clipped to ±8
   first: sigmoid(±8) already lands on the int8 output's lowest or highest code, so the clip changes nothing the
   INT8 model can output, and the logits get a step of 0.06.
2. **Uneven channels.** From layer 12's pointwise output to layer 14's depthwise one, the widest channel is 24 to
   160 times the median one: in layer 13 the widest reach 22 while the median channel reaches only 0.19 to 0.35,
   2 to 4 int8 steps, so with one int8 scale per tensor most of its values round to 0 or 1 step. The fix uses the
   positive-rescaling identity of cross-layer equalization (Nagel et al., *Data-Free Quantization*, 2019):
   each channel is scaled toward its layer's widest one and the next layer's weights are scaled back by the same
   factor (ReLU doesn't mind positive scaling), so the float model is unchanged but int8 sees evenly sized
   channels. Unlike that paper, the scales come from activation ranges on the calibration clips rather than from
   the weights, and an exponent alpha (like SmoothQuant's migration strength, Xiao et al., 2023, but without its
   weight term) moves each channel only part of the way. The factor lands in the next layer's weights: free when
   that layer is depthwise, with one int8 scale per channel, but in a pointwise or the final dense layer the
   scaled-down weights share one scale with the rest, so they lose precision.

Alpha was picked on fold 4, not on the fold the table reports (fold 4, like folds 1 to 3, also supplies calibration
clips). On fold 4, alpha 0.3 to 0.7 all land within 10 clips of each other: 69.7% at 0.3 and 0.7, 72.1% and 72.3%
at 0.4 and 0.6, 70.7% at 0.5 (the float model gets 72.1%). With no clear winner between 0.4 and 0.6, the default is
the middle, 0.5.

Fix 1 alone barely helps (31.6%): the uneven channels inside the network still erase most of the signal before
it reaches the logits.

Both fixes are in `tools/convert.py`: `--naive` builds the plain version and `--alpha 0` the one with fix 1 only.

## On the ESP32-S3

- **Model:** 4.06 MB in flash, read in place; about 69 M multiply-accumulates per 0.96 s window.
- **Working memory:** 264 KB of internal RAM reserved for TFLite Micro's tensors, so no PSRAM is needed. TFLite
  Micro reports 238 KB used with esp-nn's plain C kernels and 254 KB with its S3-optimized ones, both read in
  the emulator: memory is allocated before the first inference, so the optimized build reports its figure before
  it reaches the instruction QEMU can't run (see below).
- **Firmware image:** 5.35 MB with the model and the six test clips (0.96 MB) embedded; fits 8 MB of flash.
- **Front-end:** `firmware/main/frontend.c`, the same log-mel math as YAMNet's TensorFlow front-end (25 ms Hann
  window, 10 ms hop, 64 mel bands from 125 Hz to 7.5 kHz) in plain C with its own FFT. Its Python twin,
  `tools/frontend.py`, is checked against YAMNet's TensorFlow front-end by `tools/convert.py`.

The firmware classifies six ESC-50 clips embedded at build time (dog, glass breaking, knock on a door, crying baby,
siren, alarm clock) and prints the top 3 sounds for each ("patches" are the 0.96 s windows, taken every 0.48 s).
Output in Espressif's QEMU emulator, with the dump, the arena line and the per-window time removed (the
emulator's time says nothing about the chip's):

```
dog: 5.00 s, 10 patches
  0.94  Animal
  0.93  Dog
  0.93  Domestic animals, pets

glass_breaking: 5.00 s, 10 patches
  0.17  Glass
  0.15  Chink, clink
  0.08  Breaking

door_wood_knock: 5.00 s, 10 patches
  0.18  Knock
  0.13  Door
  0.09  Bouncing

crying_baby: 5.00 s, 10 patches
  0.89  Crying, sobbing
  0.84  Baby cry, infant cry
  0.10  Whimper

siren: 5.00 s, 10 patches
  0.40  Alarm
  0.40  Siren
  0.25  Civil defense siren

clock_alarm: 5.00 s, 10 patches
  0.55  Alarm clock
  0.46  Beep, bleep
  0.38  Alarm

done
```

Scores are averaged over the clip's ten 0.96 s windows, so short sounds like the glass break and the knock score
low even when they rank first. A detector acts on each window as it arrives instead, and there the sound is
clear: the glass break's first window scores 0.94 for Glass, and the knock's first two score 0.84 and 0.86 for
Knock (from the same emulator run). Silence is left out of the top 3, since it's not something a detector
reports.

### Checked against the Python pipeline

Why it matters: the accuracy table was measured on a PC, and this shows the firmware computes the same numbers.

With `CONFIG_YAMNET_DUMP=y` the firmware prints every int8 input it feeds the model and every int8 output it
gets back, and `tools/check_qemu.py` compares them with the Python front-end and with TFLite on the same input.
For the six clips (60 overlapping windows):

- **Front-end:** 8 of the 368,640 model-input values differ from Python, each by 1 step: values within float
  error of a rounding boundary.
- **Model:** 3 of 31,260 output values differ from TFLite's reference kernels, by at most 3/256 (the check allows
  up to 5/256, what one step of difference in a logit can cause). Most outputs sit at the lowest code (scores
  under 0.002), where a difference can't show; 1,145 are above it. Both run the same int8 model; the cause is
  most likely a kernel of TFLite Micro's (esp-nn's or its own) rounding a value differently.

The emulator runs esp-nn's plain C kernels: the build with the S3-optimized ones stops on an illegal instruction
in QEMU. The board build uses the optimized kernels.

## Build it

You need Python 3.12 and ESP-IDF 5.4.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# YAMNet's weights (Apache 2.0) and ESC-50 (CC BY-NC, not redistributed here)
curl -L -o build_py/yamnet.h5 --create-dirs https://storage.googleapis.com/audioset/yamnet.h5
curl -L -o data/esc50.zip --create-dirs https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip -q data/esc50.zip -d data

.venv/bin/python tools/convert.py             # build_py/yamnet_core_{fp32,int8}.tflite
.venv/bin/python tools/convert.py --naive     # build_py/yamnet_core_int8_naive.tflite
.venv/bin/python tools/convert.py --alpha 0   # build_py/yamnet_core_int8_a0.tflite (fix 1 only)
.venv/bin/python tools/evaluate.py build_py/yamnet_core_{fp32,int8_naive,int8_a0,int8}.tflite  # the table
.venv/bin/python tools/make_clips.py          # firmware/main/clips.bin

. ~/esp/esp-idf/export.sh                     # ESP-IDF's environment, wherever it is installed
cd firmware
idf.py build flash monitor                    # an ESP32-S3 with 8 MB of flash or more
```

To see how alpha was picked, build another strength (`.venv/bin/python tools/convert.py --alpha 0.3`) and score
it on fold 4: `FOLD=4 .venv/bin/python tools/evaluate.py build_py/yamnet_core_int8_a0.3.tflite`.

In QEMU instead of a board (Espressif's `qemu-xtensa`: run `idf_tools.py install qemu-xtensa`, then source
`export.sh` again), from `firmware/`:

```sh
idf.py -B build_qemu -D SDKCONFIG=build_qemu/sdkconfig \
  -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.qemu;sdkconfig.dump" build
cd build_qemu && esptool.py --chip esp32s3 merge_bin --fill-flash-size 8MB -o flash.bin @flash_args
qemu-system-xtensa -nographic -machine esp32s3 -drive file=flash.bin,if=mtd,format=raw \
  -global driver=timer.esp32s3.timg,property=wdt_disable,value=true | tee ../qemu.log
# quit QEMU with Ctrl-A X once it prints "done"
cd ../.. && .venv/bin/python tools/check_qemu.py firmware/qemu.log
```

`sdkconfig.qemu` switches esp-nn to its plain C kernels and `sdkconfig.dump` turns on the dump.

## Limits

- Not yet run on a physical board. Everything above was checked in Espressif's emulator with esp-nn's plain C
  kernels; the board build compiles with the S3-optimized kernels, but their outputs and timing aren't measured
  yet, so no per-window latency is given here.
- The six clips are the first fold-5 clip of each category that the model gets right (`tools/make_clips.py`), to
  show the output format; accuracy is what the table measures.
- ESC-50 clips are clean, single-sound recordings. A real room adds noise, reverb and overlapping sounds, so
  test on audio from where the device will live before trusting a threshold.
- The int8 input scale is calibrated on ESC-50's levels. Loud live audio would saturate the input, so recalibrate
  on recordings from the actual microphone.
- Live audio is the next step: an I2S microphone feeding the same front-end frame by frame instead of whole clips.

## Adapting it to your product

- **Your sounds:** a small classifier on top of YAMNet, trained on your recordings.
- **Live input:** an I2S microphone, with per-window detection and thresholds tuned in the real room.
- **Alerts:** whatever the product needs once a sound is detected: Wi-Fi or MQTT, a GPIO, BLE.

## Licenses

- YAMNet's weights and `tools/yamnet_ref/` (YAMNet's reference code and class map, unchanged from
  [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)): Apache 2.0,
  © The TensorFlow Authors.
- ESC-50: CC BY-NC 3.0, by Karol J. Piczak. Download it yourself; it is not in this repo.
- Everything else: MIT.
