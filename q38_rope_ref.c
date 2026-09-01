#include "q38_rope_ref.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

bool q38_rope_apply_ref(const q38_rope_config *config,
                        const int64_t positions[4], const float *input,
                        float *output, size_t element_count, char *error,
                        size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (!config || !positions || !input || !output || !element_count ||
        config->n_dims == 0 || config->n_dims % 2 != 0 ||
        config->n_dims > element_count || config->sections[0] == 0 ||
        config->sections[0] + config->sections[1] + config->sections[2] +
                config->sections[3] == 0) {
        return fail(error, error_len, "invalid RoPE reference arguments");
    }
    if (config->n_dims / 2 > UINT32_MAX /
                              (config->sections[0] + config->sections[1] +
                               config->sections[2] + config->sections[3])) {
        return fail(error, error_len, "RoPE section geometry overflows");
    }
    memcpy(output, input, element_count * sizeof(*output));
    const uint32_t section_count =
        config->sections[0] + config->sections[1] + config->sections[2] +
        config->sections[3];
    const double theta_scale =
        pow((double)config->theta, -2.0 / (double)config->n_dims);
    double theta[4] = {
        (double)positions[0], (double)positions[1],
        (double)positions[2], (double)positions[3],
    };

    for (uint32_t i = 0; i < config->n_dims; i += 2) {
        const uint32_t sector = (i / 2) % section_count;
        uint32_t section = 3;
        if (config->interleaved) {
            if (sector % 3 == 1 &&
                sector < 3 * config->sections[1]) section = 1;
            else if (sector % 3 == 2 &&
                     sector < 3 * config->sections[2]) section = 2;
            else if (sector % 3 == 0 &&
                     sector < 3 * config->sections[0]) section = 0;
        } else {
            const uint32_t first = config->sections[0];
            const uint32_t second = first + config->sections[1];
            const uint32_t third = second + config->sections[2];
            if (sector < first) section = 0;
            else if (sector < second) section = 1;
            else if (sector < third) section = 2;
        }
        const double angle = theta[section];
        const float cosine = (float)cos(angle);
        const float sine = (float)sin(angle);
        const size_t first = i / 2;
        const size_t second = first + config->n_dims / 2;
        const float x = input[first];
        const float y = input[second];
        output[first] = x * cosine - y * sine;
        output[second] = x * sine + y * cosine;
        for (uint32_t s = 0; s < 4; ++s) theta[s] *= theta_scale;
    }
    return true;
}
