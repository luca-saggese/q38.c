#include "../../q38_gguf.h"
#include "../../q38_weights.h"
#include "gr_reference.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

typedef struct {
    uint32_t layer;
    const char *name;
    const char *capture_path;
    const char *capture_note;
} fixture_spec;

static const fixture_spec FIXTURES[] = {
    {0, "early", "artifacts/post_m8_opt/shared_hidden_layer0.bin",
     "captured runtime hidden; branch 0 source"},
    {23, "middle", "artifacts/post_m8_opt/qsa_layer3_hidden.bin",
     "captured layer-3 runtime hidden replayed for layer-23 shape coverage"},
    {47, "late", "artifacts/post_m8_opt/qsa_layer3_hidden.bin",
     "captured layer-3 runtime hidden replayed for layer-47 shape coverage"},
};

static bool mkdir_one(const char *path) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST) return true;
    fprintf(stderr, "mkdir %s: %s\n", path, strerror(errno));
    return false;
}

static bool write_bytes(const char *path, const void *data, size_t bytes) {
    FILE *file = fopen(path, "wb");
    bool ok = false;
    if (!file) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return false;
    }
    ok = fwrite(data, 1, bytes, file) == bytes && fclose(file) == 0;
    if (!ok) fprintf(stderr, "write %s failed\n", path);
    return ok;
}

static bool read_hidden(const char *path, float *hidden) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "open capture %s: %s\n", path, strerror(errno));
        return false;
    }
    const bool ok = fread(hidden, sizeof(*hidden), Q38_GR_HIDDEN, file) ==
                       Q38_GR_HIDDEN &&
                   fclose(file) == 0;
    if (!ok) fprintf(stderr, "capture %s is not a 2560-float hidden vector\n",
                     path);
    return ok;
}

static float bf16_to_float(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static bool extract_tensor(const q38_gguf *model, const q38_tensor *tensor,
                           float *output, size_t elements) {
    if (!tensor || tensor->type != 30 || tensor->elements != elements) {
        fprintf(stderr, "unexpected GR tensor type or element count\n");
        return false;
    }
    const uint16_t *source = q38_gguf_tensor_data(model, tensor);
    if (!source) {
        fprintf(stderr, "GR tensor payload unavailable\n");
        return false;
    }
    for (size_t i = 0; i < elements; ++i)
        output[i] = bf16_to_float(source[i]);
    return true;
}

static bool write_fixture(const q38_gguf *model, const q38_gr_weights *gr,
                          const fixture_spec *spec, const char *root) {
    const size_t width = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
    const size_t down_size = Q38_GR_RANK * width;
    const size_t up_size = width * Q38_GR_RANK;
    const size_t inject_size = Q38_GR_BRANCHES * width;
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/layer_%u_%s", root, spec->layer, spec->name);
    if (!mkdir_one(dir)) return false;

    float *captured = malloc(Q38_GR_HIDDEN * sizeof(*captured));
    float *residual = calloc(width, sizeof(*residual));
    float *block = malloc(Q38_GR_HIDDEN * sizeof(*block));
    float *hc_norm = malloc(width * sizeof(*hc_norm));
    float *down = malloc(down_size * sizeof(*down));
    float *up = malloc(up_size * sizeof(*up));
    float *inject = malloc(inject_size * sizeof(*inject));
    float *expected_input = calloc(Q38_GR_HIDDEN, sizeof(*expected_input));
    float *expected_updated = calloc(width, sizeof(*expected_updated));
    if (!captured || !residual || !block || !hc_norm || !down || !up ||
        !inject || !expected_input || !expected_updated) {
        fprintf(stderr, "fixture allocation failed for layer %u\n", spec->layer);
        free(captured); free(residual); free(block); free(hc_norm);
        free(down); free(up); free(inject); free(expected_input);
        free(expected_updated);
        return false;
    }
    bool ok = read_hidden(spec->capture_path, captured);
    memcpy(residual, captured, Q38_GR_HIDDEN * sizeof(*captured));
    for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
        residual[Q38_GR_HIDDEN + channel] = captured[channel] * 0.5f;
        residual[2 * Q38_GR_HIDDEN + channel] = captured[channel] * -0.25f;
        residual[3 * Q38_GR_HIDDEN + channel] = captured[channel] * 0.125f;
    }
    memcpy(block, captured, Q38_GR_HIDDEN * sizeof(*block));
    ok = ok && extract_tensor(model, gr->hc_norm, hc_norm, width);
    ok = ok && extract_tensor(model, gr->input_mix_weight_down, down,
                              down_size);
    ok = ok && extract_tensor(model, gr->input_mix_weight_up, up, up_size);
    ok = ok && extract_tensor(model, gr->block_inject_weight, inject,
                              inject_size);
    gr_reference_params params = {hc_norm, down, up, inject};
    ok = ok && gr_reference_collapse(residual, block, &params,
                                     expected_input, expected_updated);
    char path[640];
#define WRITE(name, data, count) \
    do { \
        snprintf(path, sizeof(path), "%s/%s", dir, name); \
        ok = ok && write_bytes(path, data, (count) * sizeof(float)); \
    } while (0)
    WRITE("residual.bin", residual, width);
    WRITE("block_output.bin", block, Q38_GR_HIDDEN);
    WRITE("hc_norm.bin", hc_norm, width);
    WRITE("input_mix_down.bin", down, down_size);
    WRITE("input_mix_up.bin", up, up_size);
    WRITE("block_inject.bin", inject, inject_size);
    WRITE("expected_input.bin", expected_input, Q38_GR_HIDDEN);
    WRITE("expected_updated.bin", expected_updated, width);
#undef WRITE
    snprintf(path, sizeof(path), "%s/metadata.json", dir);
    FILE *metadata = fopen(path, "w");
    if (!metadata) ok = false;
    if (metadata) {
        fprintf(metadata,
                "{\n"
                "  \"format\": \"q38-gr-fixture-v1\",\n"
                "  \"layer\": %u,\n"
                "  \"fixture_class\": \"%s\",\n"
                "  \"gr_family\": \"mlp_hyper_connection\",\n"
                "  \"hidden\": %u,\n"
                "  \"branches\": %u,\n"
                "  \"rank\": %u,\n"
                "  \"weight_dtype\": \"BF16 converted to F32\",\n"
                "  \"input_dtype\": \"F32\",\n"
                "  \"capture_path\": \"%s\",\n"
                "  \"capture_note\": \"%s\",\n"
                "  \"reference0_boundary_capture\": false,\n"
                "  \"branch_completion\": \"branch0 captured; branches 1-3 deterministic scaled replay\",\n"
                "  \"expected_outputs\": \"gr_reference production contract\"\n"
                "}\n",
                spec->layer, spec->name, Q38_GR_HIDDEN, Q38_GR_BRANCHES,
                Q38_GR_RANK, spec->capture_path, spec->capture_note);
        if (fclose(metadata) != 0) ok = false;
    }
    free(captured); free(residual); free(block); free(hc_norm);
    free(down); free(up); free(inject); free(expected_input);
    free(expected_updated);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL OUTPUT_DIR\n", argv[0]);
        return 2;
    }
    char error[256] = {0};
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open model failed: %s\n", error);
        return 1;
    }
    q38_weights weights = {0};
    bool ok = q38_weights_bind_subset(model, 47, &weights, error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "bind GR fixtures failed: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    ok = mkdir_one(argv[2]);
    for (size_t i = 0; ok && i < sizeof(FIXTURES) / sizeof(FIXTURES[0]); ++i) {
        const fixture_spec *spec = &FIXTURES[i];
        ok = write_fixture(model, &weights.layer[spec->layer].mlp_gr, spec,
                           argv[2]);
    }
    q38_weights_release(&weights);
    q38_gguf_close(model);
    return ok ? 0 : 1;
}
