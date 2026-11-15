#include "frontend.h"

#include <math.h>

#define SR 16000
#define WIN 400
#define HOP 160
#define NFFT 512
#define NBINS (NFFT / 2 + 1)
/* yamnet_ref.features.pad_waveform: at least one patch, then whole patch hops. */
#define MIN_LEN 15600
#define PATCH_SAMPLES (FE_PATCH_HOP * HOP)

static float hann[WIN];
static float tw_re[NFFT / 2], tw_im[NFFT / 2];
/* Each FFT bin falls in at most two neighbouring mel bands: band0[k] and band0[k] + 1. */
static int8_t band0[NBINS];
static float weight[NBINS][2];

static double hz_to_mel(double f) { return 1127.0 * log1p(f / 700.0); }

void frontend_init(void)
{
    for (int i = 0; i < WIN; i++)
        hann[i] = (float)(0.5 - 0.5 * cos(2 * M_PI * i / WIN));
    for (int k = 0; k < NFFT / 2; k++) {
        tw_re[k] = (float)cos(-2 * M_PI * k / NFFT);
        tw_im[k] = (float)sin(-2 * M_PI * k / NFFT);
    }
    /* tf.signal.linear_to_mel_weight_matrix, in double like tools/frontend.py. */
    double lo = hz_to_mel(125.0), hi = hz_to_mel(7500.0), step = (hi - lo) / (FE_NMEL + 1);
    double edge[FE_NMEL + 2];
    for (int m = 0; m < FE_NMEL + 2; m++)  /* np.linspace */
        edge[m] = m == FE_NMEL + 1 ? hi : lo + m * step;
    for (int k = 0; k < NBINS; k++) {
        band0[k] = -1;
        weight[k][0] = weight[k][1] = 0;
        if (k == 0)
            continue;  /* the DC bin gets no weight */
        double s = hz_to_mel(k * (SR / 2.0) / (NBINS - 1));
        for (int m = 0; m < FE_NMEL; m++) {
            double w = fmin((s - edge[m]) / (edge[m + 1] - edge[m]),
                            (edge[m + 2] - s) / (edge[m + 2] - edge[m + 1]));
            if (w <= 0)
                continue;
            if (band0[k] < 0)
                band0[k] = m;
            weight[k][m - band0[k]] = (float)w;
        }
    }
}

static void fft(float *re, float *im)
{
    for (int i = 1, j = 0; i < NFFT; i++) {
        int bit = NFFT >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= NFFT; len <<= 1) {
        int step = NFFT / len, half = len / 2;
        for (int i = 0; i < NFFT; i += len)
            for (int k = 0; k < half; k++) {
                float wr = tw_re[k * step], wi = tw_im[k * step];
                int a = i + k, b = a + half;
                float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
    }
}

static int padded_len(int n)
{
    int extra = n > MIN_LEN ? n - MIN_LEN : 0;
    return MIN_LEN + (extra + PATCH_SAMPLES - 1) / PATCH_SAMPLES * PATCH_SAMPLES;
}

int frontend_n_frames(int n) { return 1 + (padded_len(n) - WIN) / HOP; }

void frontend_frames(const int16_t *pcm, int n, float scale, int zp, int8_t *out)
{
    static float re[NFFT], im[NFFT];
    int frames = frontend_n_frames(n);
    for (int f = 0; f < frames; f++, out += FE_NMEL) {
        for (int i = 0; i < NFFT; i++) {
            int t = f * HOP + i;
            re[i] = i < WIN && t < n ? pcm[t] / 32768.0f * hann[i] : 0;
            im[i] = 0;
        }
        fft(re, im);
        float mel[FE_NMEL] = {0};
        for (int k = 1; k < NBINS; k++) {
            if (band0[k] < 0)
                continue;
            float mag = sqrtf(re[k] * re[k] + im[k] * im[k]);
            mel[band0[k]] += mag * weight[k][0];
            if (band0[k] + 1 < FE_NMEL)
                mel[band0[k] + 1] += mag * weight[k][1];
        }
        for (int m = 0; m < FE_NMEL; m++) {
            /* Divided in float and rounded half to even, like numpy on a float32 array. */
            long q = lrintf(logf(mel[m] + 0.001f) / scale) + zp;
            out[m] = q < -128 ? -128 : q > 127 ? 127 : q;
        }
    }
}
