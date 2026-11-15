#pragma once
#include <stdint.h>

#define FE_NMEL 64
#define FE_PATCH 96      /* frames per model input */
#define FE_PATCH_HOP 48  /* frames between patches */

void frontend_init(void);

/* Frames frontend_frames() writes for n samples, after YAMNet's padding. */
int frontend_n_frames(int n);

/* Log-mel spectrogram of n 16 kHz samples, quantized to the model input (q = x / scale + zp),
 * frontend_n_frames(n) rows of FE_NMEL into out. Same maths as tools/frontend.py. */
void frontend_frames(const int16_t *pcm, int n, float scale, int zp, int8_t *out);
