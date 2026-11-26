// YAMNet INT8 on the ESP32-S3: classifies the clips embedded by tools/make_clips.py and
// prints the top 3 of the 521 AudioSet classes for each one, Silence aside.
#include <cstdio>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

extern "C" {
#include "frontend.h"
}

extern const uint8_t yamnet_model[];
extern const uint8_t yamnet_clips[];
extern const char class_map[] asm("_binary_yamnet_class_map_csv_start");

namespace {

constexpr int kClasses = 521;
// "Silence" in yamnet_class_map.csv. Left out of the top 3: a short event in a 5 s clip still
// leaves most windows silent, and silence is not what a detector reports.
constexpr int kSilence = 494;
// Sized from arena_used_bytes() on this model, plus a little room.
constexpr size_t kArenaSize = 264 * 1024;

const char *name[kClasses];
int name_len[kClasses];

// yamnet_class_map.csv: "index,mid,display_name"; names with a comma are quoted.
void parse_class_map()
{
    const char *p = strchr(class_map, '\n') + 1;
    for (int i = 0; i < kClasses; i++) {
        p = strchr(strchr(p, ',') + 1, ',') + 1;
        bool quoted = *p == '"';
        name[i] = p + quoted;
        const char *end = quoted ? strchr(p + 1, '"') : strpbrk(p, "\r\n");
        name_len[i] = end - name[i];
        p = strchr(end, '\n') + 1;
    }
}

#if CONFIG_YAMNET_DUMP
void dump(const char *tag, int clip, int patch, const int8_t *v, int n)
{
    printf("%s %d %d ", tag, clip, patch);
    for (int i = 0; i < n; i++)
        printf("%02x", (uint8_t)v[i]);
    printf("\n");
}
#endif

}  // namespace

extern "C" void app_main(void)
{
    frontend_init();
    parse_class_map();

    const tflite::Model *model = tflite::GetModel(yamnet_model);
    static tflite::MicroMutableOpResolver<8> ops;
    ops.AddConv2D();
    ops.AddDepthwiseConv2D();
    ops.AddFullyConnected();
    ops.AddLogistic();
    ops.AddMaximum();
    ops.AddMean();
    ops.AddMinimum();
    ops.AddReshape();
    // The arena goes in fast internal RAM; the 4 MB of weights stay in flash, too big for it.
    auto *arena = static_cast<uint8_t *>(heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena) {
        printf("no %u bytes for the arena (largest internal block %u)\n", (unsigned)kArenaSize,
               (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        return;
    }
    static tflite::MicroInterpreter interp(model, ops, arena, kArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        return;
    }
    TfLiteTensor *in = interp.input(0), *out = interp.output(0);
    printf("arena: %u of %u KB used\n", (unsigned)(interp.arena_used_bytes() / 1024),
           (unsigned)(kArenaSize / 1024));

    const uint8_t *p = yamnet_clips;
    uint32_t count;
    memcpy(&count, p, 4);
    p += 4;
    for (uint32_t c = 0; c < count; c++) {
        const char *label = reinterpret_cast<const char *>(p);
        uint32_t n;
        memcpy(&n, p + 32, 4);
        auto *pcm = reinterpret_cast<const int16_t *>(p + 36);
        p += 36 + 2 * n;

        int frames = frontend_n_frames(n);
        auto *mel = static_cast<int8_t *>(malloc(frames * FE_NMEL));
        if (!mel) {
            printf("no %d bytes for %.32s's features\n", frames * FE_NMEL, label);
            return;
        }
        int64_t fe_t0 = esp_timer_get_time();
        frontend_frames(pcm, n, in->params.scale, in->params.zero_point, mel);
        int64_t fe_us = esp_timer_get_time() - fe_t0;
        int patches = 1 + (frames - FE_PATCH) / FE_PATCH_HOP;

        static float score[kClasses];
        memset(score, 0, sizeof score);
        int64_t us = 0;
        for (int i = 0; i < patches; i++) {
            memcpy(in->data.int8, mel + i * FE_PATCH_HOP * FE_NMEL, FE_PATCH * FE_NMEL);
#if CONFIG_YAMNET_DUMP
            dump("IN", c, i, in->data.int8, FE_PATCH * FE_NMEL);  // Invoke() reuses this buffer
#endif
            int64_t t0 = esp_timer_get_time();
            if (interp.Invoke() != kTfLiteOk) {
                printf("Invoke failed\n");
                return;
            }
            us += esp_timer_get_time() - t0;
#if CONFIG_YAMNET_DUMP
            dump("OUT", c, i, out->data.int8, kClasses);
#endif
            for (int k = 0; k < kClasses; k++)
                score[k] += (out->data.int8[k] - out->params.zero_point) * out->params.scale;
        }
        free(mel);

        printf("\n%.32s: %.2f s, %d patches, %lld ms per patch, front-end %lld ms\n", label, n / 16000.0,
               patches, us / patches / 1000, fe_us / 1000);
        int top[3] = {-1, -1, -1};
        for (int k = 0; k < kClasses; k++)
            for (int j = 0; j < 3 && k != kSilence; j++)
                if (top[j] < 0 || score[k] > score[top[j]]) {
                    memmove(top + j + 1, top + j, (2 - j) * sizeof top[0]);
                    top[j] = k;
                    break;
                }
        for (int k : top)
            printf("  %.2f  %.*s\n", score[k] / patches, name_len[k], name[k]);
    }
    printf("\ndone\n");
}
