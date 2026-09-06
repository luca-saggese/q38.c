#include "q38_directional_steering.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static bool write_direction(const char *path) {
    float *data = calloc(Q38_DIRECTIONAL_STEERING_FLOATS, sizeof(*data));
    if (!data) return false;
    for (uint32_t layer = 0; layer < Q38_DIRECTIONAL_STEERING_LAYERS;
         ++layer)
        data[(size_t)layer * Q38_DIRECTIONAL_STEERING_HIDDEN + layer %
             Q38_DIRECTIONAL_STEERING_HIDDEN] = 1.0f;
    FILE *file = fopen(path, "wb");
    bool ok = false;
    if (file) {
        ok = fwrite(data, sizeof(*data), Q38_DIRECTIONAL_STEERING_FLOATS,
                    file) == Q38_DIRECTIONAL_STEERING_FLOATS;
        if (fclose(file) != 0) ok = false;
    }
    free(data);
    return ok;
}

int main(void) {
    const char *path = "/tmp/q38-directional-steering-test.f32";
    char error[256] = {0};
    check(write_direction(path), "write normalized Q38 direction");

    q38_directional_steering steering;
    q38_directional_steering_init(&steering);
    check(q38_directional_steering_load(&steering, path, -1.0f, 0.5f,
                                         error, sizeof(error)),
          "load 48x2560 Q38 direction");
    check(steering.layers == 48 && steering.hidden_size == 2560 &&
              steering.enabled && steering.fingerprint != 0,
          "loaded steering metadata");

    float values[Q38_DIRECTIONAL_STEERING_HIDDEN] = {0};
    values[0] = 2.0f;
    q38_directional_steering_apply_cpu(values, 1, &steering, 0, 1.0f);
    check(fabsf(values[0]) < 1.0e-6f,
          "direction projection removes positive component");
    values[0] = 2.0f;
    q38_directional_steering_apply_cpu(values, 1, &steering, 0, 0.0f);
    check(fabsf(values[0] - 2.0f) < 1.0e-6f,
          "zero scale leaves values unchanged");

    q38_directional_steering_override override = {
        .set = true, .ffn_scale = -0.5f, .attn_scale = 0.0f,
    };
    float ffn = 0.0f, attn = 0.0f;
    q38_directional_steering_effective(&steering, &override, &ffn, &attn);
    check(ffn == -0.5f && attn == 0.0f, "session override is effective");

    FILE *bad = fopen(path, "wb");
    check(bad && fputc(0, bad) != EOF && fclose(bad) == 0,
          "write invalid-size direction");
    q38_directional_steering_destroy(&steering);
    check(!q38_directional_steering_load(&steering, path, 1.0f, 0.0f,
                                          error, sizeof(error)),
          "reject invalid-size direction");
    remove(path);
    q38_directional_steering_destroy(&steering);
    if (failures) return 1;
    puts("test_q38_directional_steering: all tests passed");
    return 0;
}
