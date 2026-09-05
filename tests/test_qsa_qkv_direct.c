#include "q38_forward.h"
#include "q38_forward_cuda.h"
#include "q38_gguf.h"
#include "q38_weights.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    Q_ROWS = 12288,
    K_ROWS = 512,
    V_ROWS = 512,
    HIDDEN = 2560,
    QSA_LAYER = 3,
};

typedef struct {
    size_t first_mismatch;
    float expected;
    float actual;
    float max_abs;
    float max_rel;
    uint64_t expected_hash;
    uint64_t actual_hash;
} comparison;

static float bf16_to_float(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static uint64_t hash_floats(const float *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    return hash;
}

static bool project_reference(const q38_gguf *model, const q38_tensor *tensor,
                              const float *input, size_t rows, float *output,
                              char *error, size_t error_len) {
    if (!tensor || tensor->type != Q38_FORWARD_BF16 ||
        tensor->ndim != 2 || tensor->dim[0] != rows ||
        tensor->dim[1] != HIDDEN)
        return false;
    const uint16_t *weights = (const uint16_t *)
        q38_gguf_tensor_data(model, tensor);
    if (!weights) {
        if (error && error_len) snprintf(error, error_len,
                                         "QKV tensor payload unavailable");
        return false;
    }
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (size_t col = 0; col < HIDDEN; ++col)
            sum += bf16_to_float(weights[row * HIDDEN + col]) * input[col];
        output[row] = sum;
    }
    return true;
}

static comparison compare_vectors(const float *expected, const float *actual,
                                  size_t count) {
    comparison result = {
        .first_mismatch = SIZE_MAX,
        .expected_hash = hash_floats(expected, count),
        .actual_hash = hash_floats(actual, count),
    };
    for (size_t i = 0; i < count; ++i) {
        const float abs_error = fabsf(actual[i] - expected[i]);
        const float rel_error = abs_error /
            fmaxf(fabsf(expected[i]), 1.0e-12f);
        if (abs_error > result.max_abs) result.max_abs = abs_error;
        if (rel_error > result.max_rel) result.max_rel = rel_error;
        if (result.first_mismatch == SIZE_MAX &&
            memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
            result.first_mismatch = i;
            result.expected = expected[i];
            result.actual = actual[i];
        }
    }
    return result;
}

static size_t tensor_index(const q38_gguf *model, const q38_tensor *tensor) {
    for (uint64_t i = 0; i < model->n_tensors; ++i)
        if (&model->tensors[i] == tensor) return (size_t)i;
    return SIZE_MAX;
}

static void write_tensor_metadata(FILE *out, const q38_gguf *model,
                                  const char *label, const q38_tensor *tensor,
                                  size_t rows) {
    fprintf(out, "\"%s\":{\"id\":%zu,\"name\":\"%.*s\",\"rows\":%zu,"
            "\"cols\":%d,\"qtype\":%u,\"rel_offset\":%" PRIu64
            ",\"abs_offset\":%" PRIu64 ",\"bytes\":%" PRIu64 "}",
            label, tensor_index(model, tensor), tensor->name.len > 200 ? 200 :
            (int)tensor->name.len, tensor->name.ptr, rows, HIDDEN, tensor->type,
            tensor->rel_offset, tensor->abs_offset, tensor->bytes);
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    const char *hidden_path = argc > 2 ? argv[2] :
        "artifacts/post_m8_opt/qsa_layer3_hidden.bin";
    const char *artifact_path = argc > 3 ? argv[3] :
        "artifacts/post_m8_opt/qsa_layer3_qkv_direct.json";
    char error[256] = {0};
    q38_gguf *model = q38_gguf_open(model_path, error, sizeof(error));
    if (!model) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 47, &weights, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    FILE *hidden_file = fopen(hidden_path, "rb");
    float hidden[HIDDEN];
    if (!hidden_file || fread(hidden, sizeof(float), HIDDEN, hidden_file) != HIDDEN) {
        fprintf(stderr, "failed to read QSA hidden fixture\n");
        if (hidden_file) fclose(hidden_file);
        q38_weights_release(&weights);
        q38_gguf_close(model);
        return 1;
    }
    fclose(hidden_file);
    const q38_tensor *q = weights.layer[QSA_LAYER].qsa.q_proj;
    const q38_tensor *k = weights.layer[QSA_LAYER].qsa.k_proj;
    const q38_tensor *v = weights.layer[QSA_LAYER].qsa.v_proj;
    float q_ref[Q_ROWS], k_ref[K_ROWS], v_ref[V_ROWS];
    float q_cuda[Q_ROWS], k_cuda[K_ROWS], v_cuda[V_ROWS];
    if (!project_reference(model, q, hidden, Q_ROWS, q_ref, error,
                           sizeof(error)) ||
        !project_reference(model, k, hidden, K_ROWS, k_ref, error,
                           sizeof(error)) ||
        !project_reference(model, v, hidden, V_ROWS, v_ref, error,
                           sizeof(error))) {
        fprintf(stderr, "reference QKV projection failed\n");
        q38_weights_release(&weights);
        q38_gguf_close(model);
        return 1;
    }
    FILE *reference = fopen("artifacts/post_m8_opt/qsa_layer3_qkv_reference.bin",
                            "wb");
    if (!reference ||
        fwrite(hidden, sizeof(float), HIDDEN, reference) != HIDDEN ||
        fwrite(q_ref, sizeof(float), Q_ROWS, reference) != Q_ROWS ||
        fwrite(k_ref, sizeof(float), K_ROWS, reference) != K_ROWS ||
        fwrite(v_ref, sizeof(float), V_ROWS, reference) != V_ROWS) {
        fprintf(stderr, "failed to save QKV reference fixture\n");
        if (reference) fclose(reference);
        q38_weights_release(&weights);
        q38_gguf_close(model);
        return 1;
    }
    fclose(reference);

    q38_forward_cuda_context *context =
        q38_forward_cuda_context_create(error, sizeof(error));
    if (!context ||
        !q38_forward_cuda_enable_all_non_ple_residency(
            context, model, error, sizeof(error))) {
        fprintf(stderr, "CUDA residency setup failed: %s\n", error);
        q38_forward_cuda_context_destroy(context);
        q38_weights_release(&weights);
        q38_gguf_close(model);
        return 1;
    }
    q38_forward_qsa_timing timing = {0};
    if (!q38_forward_cuda_qsa_qkv_backend(
            model, q, k, v, hidden, 1, q_cuda, k_cuda, v_cuda, &timing,
            context, error, sizeof(error))) {
        fprintf(stderr, "CUDA QKV backend failed: %s\n", error);
        q38_forward_cuda_context_destroy(context);
        q38_weights_release(&weights);
        q38_gguf_close(model);
        return 1;
    }
    q38_forward_cuda_residency_stats initial_stats;
    q38_forward_cuda_get_residency_stats(context, &initial_stats);
    const comparison q_result = compare_vectors(q_ref, q_cuda, Q_ROWS);
    const comparison k_result = compare_vectors(k_ref, k_cuda, K_ROWS);
    const comparison v_result = compare_vectors(v_ref, v_cuda, V_ROWS);
    FILE *out = fopen(artifact_path, "w");
    if (!out) {
        fprintf(stderr, "failed to write QKV artifact\n");
        return 1;
    }
    fprintf(out, "{\"layer\":%d,\"token_count\":1,\"hidden_checksum\":\"%016"
            PRIx64 "\",\"Q_ref_checksum\":\"%016" PRIx64
            "\",\"K_ref_checksum\":\"%016" PRIx64
            "\",\"V_ref_checksum\":\"%016" PRIx64 "\",",
            QSA_LAYER, hash_floats(hidden, HIDDEN), q_result.expected_hash,
            k_result.expected_hash, v_result.expected_hash);
    write_tensor_metadata(out, model, "Wq", q, Q_ROWS);
    fputc(',', out);
    write_tensor_metadata(out, model, "Wk", k, K_ROWS);
    fputc(',', out);
    write_tensor_metadata(out, model, "Wv", v, V_ROWS);
    fprintf(out, ",\"Q_cuda_checksum\":\"%016" PRIx64
            "\",\"K_cuda_checksum\":\"%016" PRIx64
            "\",\"V_cuda_checksum\":\"%016" PRIx64
            "\",\"Q_first_mismatch\":%zu,\"K_first_mismatch\":%zu,"
            "\"V_first_mismatch\":%zu,\"Q_max_abs\":%.9g,\"K_max_abs\":%.9g,"
            "\"V_max_abs\":%.9g,\"Q_max_rel\":%.9g,\"K_max_rel\":%.9g,"
            "\"V_max_rel\":%.9g,\"kernel_launches\":%" PRIu64
            ",\"host_syncs\":%" PRIu64 ",\"H2D_bytes\":%" PRIu64
            ",\"D2H_bytes\":%" PRIu64 ",\"pass\":%s}\n",
            q_result.actual_hash, k_result.actual_hash, v_result.actual_hash,
            q_result.first_mismatch, k_result.first_mismatch,
            v_result.first_mismatch, q_result.max_abs, k_result.max_abs,
            v_result.max_abs, q_result.max_rel, k_result.max_rel,
            v_result.max_rel, timing.kernel_launches, timing.host_syncs,
            timing.h2d_bytes, timing.d2h_bytes,
            q_result.first_mismatch == SIZE_MAX &&
            k_result.first_mismatch == SIZE_MAX &&
            v_result.first_mismatch == SIZE_MAX ? "true" : "false");
    fclose(out);
    const comparison results[] = {q_result, k_result, v_result};
    const char *names[] = {"Q", "K", "V"};
    bool pass = true;
    for (size_t i = 0; i < 3; ++i) {
        printf("%s first mismatch index=%zu expected=%.9g actual=%.9g "
               "max_abs=%.9g max_rel=%.9g ref_hash=%016" PRIx64
               " cuda_hash=%016" PRIx64 "\n", names[i],
               results[i].first_mismatch, results[i].expected,
               results[i].actual, results[i].max_abs, results[i].max_rel,
               results[i].expected_hash, results[i].actual_hash);
        pass &= results[i].first_mismatch == SIZE_MAX;
    }
    if (getenv("Q38_QSA_QKV_DEV_LOOP")) {
        char command[32];
        while (fgets(command, sizeof(command), stdin)) {
            if (!strncmp(command, "QUIT", 4) ||
                !strncmp(command, "SHUTDOWN", 8))
                break;
            if (strncmp(command, "RUN", 3) != 0)
                continue;
            memset(&timing, 0, sizeof(timing));
            if (!q38_forward_cuda_qsa_qkv_backend(
                    model, q, k, v, hidden, 1, q_cuda, k_cuda, v_cuda,
                    &timing, context, error, sizeof(error))) {
                fprintf(stderr, "persistent QKV run failed: %s\n", error);
                pass = false;
                break;
            }
            q38_forward_cuda_residency_stats after_stats;
            q38_forward_cuda_get_residency_stats(context, &after_stats);
            const comparison loop_results[] = {
                compare_vectors(q_ref, q_cuda, Q_ROWS),
                compare_vectors(k_ref, k_cuda, K_ROWS),
                compare_vectors(v_ref, v_cuda, V_ROWS),
            };
            printf("RUN Q=%s K=%s V=%s pointer_stable=%s "
                   "resident_non_ple_bytes=%zu bytes_constant=%s "
                   "weight_upload_after_init=%zu residency_misses=%" PRIu64
                   "\n",
                   loop_results[0].first_mismatch == SIZE_MAX ? "pass" : "fail",
                   loop_results[1].first_mismatch == SIZE_MAX ? "pass" : "fail",
                   loop_results[2].first_mismatch == SIZE_MAX ? "pass" : "fail",
                   initial_stats.persistent_pointer_fingerprint ==
                           after_stats.persistent_pointer_fingerprint ? "true" :
                           "false",
                   after_stats.persistent_resident_bytes,
                   initial_stats.persistent_resident_bytes ==
                           after_stats.persistent_resident_bytes ? "true" :
                           "false",
                   after_stats.matrix_upload_bytes -
                       initial_stats.matrix_upload_bytes,
                   after_stats.resident_misses -
                       initial_stats.resident_misses);
            fflush(stdout);
            pass &= loop_results[0].first_mismatch == SIZE_MAX &&
                    loop_results[1].first_mismatch == SIZE_MAX &&
                    loop_results[2].first_mismatch == SIZE_MAX;
        }
    }
    q38_forward_cuda_context_destroy(context);
    q38_weights_release(&weights);
    q38_gguf_close(model);
    return pass ? 0 : 1;
}
