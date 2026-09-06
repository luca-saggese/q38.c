#include "q38_forward.h"
#include "q38_forward_cuda.h"
#include "q38_gguf.h"
#include "q38_tokenizer.h"
#include "q38_weights.h"

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    QSA_LAYER = 3,
    HIDDEN = 2560,
    Q_ROWS = 12288,
    K_ROWS = 512,
    V_ROWS = 512,
    VOCAB = 248320,
};

typedef struct {
    q38_gguf *model;
    q38_weights weights;
    q38_forward_cuda_context *cuda;
    q38_forward_state state;
    bool state_initialized;
    q38_tokenizer tokenizer_state;
    bool tokenizer_initialized;
    void *qsa_plugin_handle;
    q38_qsa_candidate_fn qsa_plugin;
    char tokenizer[512];
    char qsa_plugin_path[512];
    char qsa_plugin_staged_path[512];
    uint64_t qsa_plugin_generation;
    char fixture_dir[512];
} worker;

typedef struct {
    size_t first_mismatch;
    float expected;
    float actual;
    float max_abs;
    float max_rel;
    uint64_t expected_hash;
    uint64_t actual_hash;
} comparison;

typedef struct {
    bool captured;
    float hidden[HIDDEN];
    char error[256];
} capture_context;

enum {
    MOE_CAPTURE_COUNT = 3,
    MOE_ROUTER_EXPERTS = 512,
    MOE_TOP_K = 10,
    MOE_INTERMEDIATE = 640,
    MOE_QK_K = 256,
};

typedef struct {
    uint32_t layer;
    bool routed_captured;
    bool shared_captured;
    float hidden[HIDDEN];
    float router_logits_pre[MOE_ROUTER_EXPERTS];
    float router_logits_effective[MOE_ROUTER_EXPERTS];
    uint16_t selected_experts[MOE_TOP_K];
    float selected_weights_pre[MOE_TOP_K];
    float selected_weights[MOE_TOP_K];
    float routed[HIDDEN];
    float shared[HIDDEN];
} moe_capture_case;

typedef struct {
    worker *worker;
    moe_capture_case cases[MOE_CAPTURE_COUNT];
} moe_capture_context;

static void residency_telemetry(
    const q38_forward_cuda_telemetry *telemetry, void *user) {
    (void)user;
    if (!telemetry || (!telemetry->non_ple_residency_miss &&
                       !telemetry->ple_file_backed_access))
        return;
    printf("{\"residency_event\":{\"layer\":%u,\"tensor_id\":%u,"
           "\"tensor_name\":\"%s\",\"subsystem\":\"%s\","
           "\"operation\":\"%s\",\"fallback_path\":\"%s\","
           "\"qtype\":%u,\"rows\":%zu,\"cols\":%zu,\"bytes\":%zu,"
           "\"resident_lookup\":\"%s\",\"upload_bytes\":%zu}}\n",
           telemetry->layer, telemetry->tensor_id,
           telemetry->tensor_name ? telemetry->tensor_name : "",
           telemetry->subsystem ? telemetry->subsystem : "unknown",
           telemetry->operation ? telemetry->operation : "unknown",
           telemetry->fallback_path ? telemetry->fallback_path : "none",
           telemetry->qtype, telemetry->rows, telemetry->cols,
           telemetry->bytes,
           telemetry->ple_file_backed_access ? "file_backed" :
               telemetry->resident_hit ? "hit" : "miss",
           telemetry->ple_file_backed_access
               ? telemetry->ple_file_bytes : telemetry->upload_bytes);
    fflush(stdout);
}

static void backend_context_trace(uint32_t layer, const char *logical_stage,
                                  const q38_tensor *tensor, size_t rows,
                                  size_t cols, void *user) {
    (void)tensor;
    (void)rows;
    (void)cols;
    q38_forward_cuda_set_stage_context(
        (q38_forward_cuda_context *)user, layer, logical_stage);
}

static q38_forward_backend_config cuda_backend_config(worker *w) {
    q38_forward_backend_config config = {0};
    config.matvec = q38_forward_cuda_matvec_backend;
    config.matrix = q38_forward_cuda_matrix_backend;
    config.matrix_batch = q38_forward_cuda_matrix_batch_backend;
    config.gr_read = q38_forward_cuda_gr_read_backend;
    config.gr_write = q38_forward_cuda_gr_write_backend;
    config.expert = q38_forward_cuda_expert_backend;
    config.moe_layer = q38_forward_cuda_moe_layer_q2_backend;
    config.qsa_qkv = q38_forward_cuda_qsa_qkv_backend;
    config.user = w->cuda;
    return config;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

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

static size_t argmax(const float *values, size_t count) {
    size_t best = 0;
    for (size_t i = 1; i < count; ++i)
        if (values[i] > values[best]) best = i;
    return best;
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
        if (error && error_len)
            snprintf(error, error_len, "QKV tensor payload unavailable");
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

static size_t tensor_index(const q38_gguf *model, const q38_tensor *tensor) {
    for (uint64_t i = 0; i < model->n_tensors; ++i)
        if (&model->tensors[i] == tensor) return (size_t)i;
    return SIZE_MAX;
}

static bool write_fixture(worker *w, const float *hidden, char *error,
                          size_t error_len) {
    const q38_tensor *q = w->weights.layer[QSA_LAYER].qsa.q_proj;
    const q38_tensor *k = w->weights.layer[QSA_LAYER].qsa.k_proj;
    const q38_tensor *v = w->weights.layer[QSA_LAYER].qsa.v_proj;
    float q_ref[Q_ROWS], k_ref[K_ROWS], v_ref[V_ROWS];
    if (!project_reference(w->model, q, hidden, Q_ROWS, q_ref, error,
                           error_len) ||
        !project_reference(w->model, k, hidden, K_ROWS, k_ref, error,
                           error_len) ||
        !project_reference(w->model, v, hidden, V_ROWS, v_ref, error,
                           error_len))
        return false;
    char path[1024];
    snprintf(path, sizeof(path), "%s/hidden.bin", w->fixture_dir);
    FILE *out = fopen(path, "wb");
    if (!out || fwrite(hidden, sizeof(float), HIDDEN, out) != HIDDEN) {
        if (out) fclose(out);
        snprintf(error, error_len, "failed to write QSA fixture hidden");
        return false;
    }
    fclose(out);
    const struct {
        const char *name;
        const float *data;
        size_t count;
    } vectors[] = {
        {"q_ref.bin", q_ref, Q_ROWS},
        {"k_ref.bin", k_ref, K_ROWS},
        {"v_ref.bin", v_ref, V_ROWS},
    };
    for (size_t i = 0; i < 3; ++i) {
        snprintf(path, sizeof(path), "%s/%s", w->fixture_dir,
                 vectors[i].name);
        out = fopen(path, "wb");
        if (!out || fwrite(vectors[i].data, sizeof(float), vectors[i].count,
                           out) != vectors[i].count) {
            if (out) fclose(out);
            snprintf(error, error_len, "failed to write QSA fixture vector");
            return false;
        }
        fclose(out);
    }
    snprintf(path, sizeof(path), "%s/metadata.json", w->fixture_dir);
    out = fopen(path, "w");
    if (!out) {
        snprintf(error, error_len, "failed to write QSA fixture metadata");
        return false;
    }
    fprintf(out,
            "{\"layer\":%d,\"hidden_elements\":%d,"
            "\"hidden_checksum\":\"%016" PRIx64 "\","
            "\"tensors\":{\"Wq\":{\"id\":%zu,\"name\":\"%.*s\","
            "\"rows\":%d,\"cols\":%d,\"qtype\":%u,\"rel_offset\":%" PRIu64
            ",\"abs_offset\":%" PRIu64 "},"
            "\"Wk\":{\"id\":%zu,\"name\":\"%.*s\",\"rows\":%d,\"cols\":%d,"
            "\"qtype\":%u,\"rel_offset\":%" PRIu64 ",\"abs_offset\":%" PRIu64
            "},\"Wv\":{\"id\":%zu,\"name\":\"%.*s\",\"rows\":%d,\"cols\":%d,"
            "\"qtype\":%u,\"rel_offset\":%" PRIu64 ",\"abs_offset\":%" PRIu64
            "}},\"Q_ref_checksum\":\"%016" PRIx64
            "\",\"K_ref_checksum\":\"%016" PRIx64
            "\",\"V_ref_checksum\":\"%016" PRIx64 "\"}\n",
            QSA_LAYER, HIDDEN, hash_floats(hidden, HIDDEN),
            tensor_index(w->model, q), (int)q->name.len, q->name.ptr,
            Q_ROWS, HIDDEN, q->type, q->rel_offset, q->abs_offset,
            tensor_index(w->model, k), (int)k->name.len, k->name.ptr,
            K_ROWS, HIDDEN, k->type, k->rel_offset, k->abs_offset,
            tensor_index(w->model, v), (int)v->name.len, v->name.ptr,
            V_ROWS, HIDDEN, v->type, v->rel_offset, v->abs_offset,
            hash_floats(q_ref, Q_ROWS), hash_floats(k_ref, K_ROWS),
            hash_floats(v_ref, V_ROWS));
    fclose(out);
    return true;
}

static bool load_fixture(worker *w, float *hidden, float *q_ref,
                         float *k_ref, float *v_ref, char *error,
                         size_t error_len) {
    const struct {
        const char *name;
        float *data;
        size_t count;
    } files[] = {
        {"hidden.bin", hidden, HIDDEN},
        {"q_ref.bin", q_ref, Q_ROWS},
        {"k_ref.bin", k_ref, K_ROWS},
        {"v_ref.bin", v_ref, V_ROWS},
    };
    for (size_t i = 0; i < 4; ++i) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", w->fixture_dir,
                 files[i].name);
        FILE *in = fopen(path, "rb");
        if (!in || fread(files[i].data, sizeof(float), files[i].count, in) !=
                        files[i].count) {
            if (in) fclose(in);
            snprintf(error, error_len, "missing or truncated QSA fixture %s",
                     files[i].name);
            return false;
        }
        fclose(in);
    }
    return true;
}

static bool run_qkv_fixture(worker *w) {
    char error[256] = {0};
    float hidden[HIDDEN], q_ref[Q_ROWS], k_ref[K_ROWS], v_ref[V_ROWS];
    float q_cuda[Q_ROWS], k_cuda[K_ROWS], v_cuda[V_ROWS];
    if (!load_fixture(w, hidden, q_ref, k_ref, v_ref, error, sizeof(error))) {
        fprintf(stderr, "RUN_QKV_FIXTURE error=%s\n", error);
        return false;
    }
    q38_forward_qsa_timing timing = {0};
    if (!q38_forward_cuda_qsa_qkv_backend(
            w->model, w->weights.layer[QSA_LAYER].qsa.q_proj,
            w->weights.layer[QSA_LAYER].qsa.k_proj,
            w->weights.layer[QSA_LAYER].qsa.v_proj, hidden, 1, q_cuda,
            k_cuda, v_cuda, &timing, w->cuda, error, sizeof(error))) {
        fprintf(stderr, "RUN_QKV_FIXTURE error=%s\n", error);
        return false;
    }
    const comparison results[] = {
        compare_vectors(q_ref, q_cuda, Q_ROWS),
        compare_vectors(k_ref, k_cuda, K_ROWS),
        compare_vectors(v_ref, v_cuda, V_ROWS),
    };
    const char *names[] = {"Q", "K", "V"};
    bool pass = true;
    for (size_t i = 0; i < 3; ++i) {
        printf("{\"qkv\":\"%s\",\"first_mismatch\":%zu,"
               "\"expected\":%.9g,\"actual\":%.9g,\"max_abs\":%.9g,"
               "\"max_rel\":%.9g,\"ref_hash\":\"%016" PRIx64
               "\",\"cuda_hash\":\"%016" PRIx64 "\"}\n",
               names[i], results[i].first_mismatch, results[i].expected,
               results[i].actual, results[i].max_abs, results[i].max_rel,
               results[i].expected_hash, results[i].actual_hash);
        pass &= results[i].first_mismatch == SIZE_MAX;
    }
    printf("{\"qkv_fixture_pass\":%s,\"kernel_launches\":%" PRIu64
           ",\"host_syncs\":%" PRIu64 ",\"H2D_bytes\":%" PRIu64
           ",\"D2H_bytes\":%" PRIu64 ",\"qkv_ms\":%.6f}\n",
           pass ? "true" : "false", timing.kernel_launches,
           timing.host_syncs, timing.h2d_bytes, timing.d2h_bytes,
           timing.qkv_projection_ms);
    fflush(stdout);
    return pass;
}

static bool capture_boundary(uint32_t layer, const char *boundary,
                             const float *values, size_t token_count,
                             size_t width, void *user, char *error,
                             size_t error_len) {
    capture_context *capture = (capture_context *)user;
    if (layer != QSA_LAYER || strcmp(boundary, "gdn_qsa_input") != 0)
        return true;
    if (token_count != 1 || width != HIDDEN || !values) {
        snprintf(error, error_len, "invalid QSA capture boundary");
        return false;
    }
    memcpy(capture->hidden, values, sizeof(capture->hidden));
    capture->captured = true;
    snprintf(capture->error, sizeof(capture->error),
             "QSA hidden captured at layer %u", layer);
    return false;
}

static bool capture_hidden(worker *w, uint32_t token) {
    capture_context capture = {0};
    uint32_t tokens[] = {token};
    float *logits = (float *)calloc(VOCAB, sizeof(float));
    char error[256] = {0};
    if (!logits) return false;
    q38_forward_state_reset(&w->state);
    q38_forward_diagnostics diagnostics = {0};
    diagnostics.boundary_trace = capture_boundary;
    diagnostics.trace_user = &capture;
    setenv("Q38_TRACE_ALL_QSA", "1", 1);
    const q38_forward_backend_config backend = cuda_backend_config(w);
    const bool ok = q38_forward_full_with_backend_config(
        w->model, &w->weights, &w->state, tokens, 1, logits, VOCAB,
        &diagnostics, &backend, error, sizeof(error));
    unsetenv("Q38_TRACE_ALL_QSA");
    free(logits);
    if (!capture.captured) {
        fprintf(stderr, "CAPTURE error=%s forward=%s\n",
                error[0] ? error : "boundary not reached",
                ok ? "true" : "false");
        return false;
    }
    if (!write_fixture(w, capture.hidden, error, sizeof(error))) {
        fprintf(stderr, "CAPTURE error=%s\n", error);
        return false;
    }
    printf("{\"capture\":\"qsa_layer3\",\"token\":%u,\"fixture\":\"%s\"}\n",
           token, w->fixture_dir);
    fflush(stdout);
    return true;
}

static moe_capture_case *find_moe_capture_case(moe_capture_context *capture,
                                               uint32_t layer) {
    for (size_t i = 0; i < MOE_CAPTURE_COUNT; ++i)
        if (capture->cases[i].layer == layer)
            return &capture->cases[i];
    return NULL;
}

static bool capture_moe_trace(uint32_t layer, const q38_moe_trace *trace,
                              void *user, char *error, size_t error_len) {
    moe_capture_context *capture = (moe_capture_context *)user;
    moe_capture_case *case_data = find_moe_capture_case(capture, layer);
    if (!case_data || !trace || trace->router_input_count != HIDDEN ||
        trace->router_logits_count != MOE_ROUTER_EXPERTS ||
        trace->selected_count != MOE_TOP_K ||
        trace->routed_output_count != HIDDEN) {
        if (case_data)
            snprintf(error, error_len, "invalid MoE capture trace at layer %u",
                     layer);
        return case_data == NULL;
    }
    memcpy(case_data->hidden, trace->router_input,
           sizeof(case_data->hidden));
    memcpy(case_data->router_logits_pre, trace->router_logits_pre_cast,
           sizeof(case_data->router_logits_pre));
    memcpy(case_data->router_logits_effective,
           trace->router_logits_effective,
           sizeof(case_data->router_logits_effective));
    memcpy(case_data->selected_experts, trace->selected_experts,
           sizeof(case_data->selected_experts));
    memcpy(case_data->selected_weights_pre,
           trace->selected_weights_pre_cast,
           sizeof(case_data->selected_weights_pre));
    memcpy(case_data->selected_weights, trace->selected_weights_effective,
           sizeof(case_data->selected_weights));
    memcpy(case_data->routed, trace->routed_output, sizeof(case_data->routed));
    case_data->routed_captured = true;
    return true;
}

static bool capture_moe_boundary(uint32_t layer, const char *boundary,
                                 const float *values, size_t token_count,
                                 size_t width, void *user, char *error,
                                 size_t error_len) {
    moe_capture_context *capture = (moe_capture_context *)user;
    moe_capture_case *case_data = find_moe_capture_case(capture, layer);
    if (!case_data || strcmp(boundary, "shared_expert") != 0)
        return true;
    if (!values || token_count != 1 || width != HIDDEN) {
        snprintf(error, error_len, "invalid shared MoE capture at layer %u",
                 layer);
        return false;
    }
    memcpy(case_data->shared, values, sizeof(case_data->shared));
    case_data->shared_captured = true;
    return true;
}

static bool write_binary_atomic(const char *path, const void *data,
                                size_t bytes, char *error, size_t error_len) {
    char temporary[1200];
    snprintf(temporary, sizeof(temporary), "%s.tmp.%ld", path, (long)getpid());
    FILE *out = fopen(temporary, "wb");
    if (!out || fwrite(data, 1, bytes, out) != bytes || fclose(out) != 0) {
        if (out) fclose(out);
        (void)remove(temporary);
        snprintf(error, error_len, "failed to write %s", path);
        return false;
    }
    if (rename(temporary, path) != 0) {
        (void)remove(temporary);
        snprintf(error, error_len, "failed to publish %s", path);
        return false;
    }
    return true;
}

static bool make_directory(const char *path, char *error, size_t error_len) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST)
        return true;
    snprintf(error, error_len, "failed to create fixture directory %s", path);
    return false;
}

static bool write_moe_tensor(const char *directory, const char *name,
                             const q38_gguf *model, const q38_tensor *tensor,
                             char *error, size_t error_len) {
    const void *data = q38_gguf_tensor_data(model, tensor);
    if (!data || !tensor) {
        snprintf(error, error_len, "missing MoE tensor payload %s", name);
        return false;
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", directory, name);
    return write_binary_atomic(path, data, (size_t)tensor->bytes, error,
                               error_len);
}

static bool write_moe_selected_tensor(
    const char *directory, const char *name, const q38_gguf *model,
    const q38_tensor *tensor, const uint16_t *experts, size_t rows_per_expert,
    char *error, size_t error_len) {
    const unsigned char *data = (const unsigned char *)
        q38_gguf_tensor_data(model, tensor);
    if (!data || !tensor || !experts || tensor->ndim != 3 ||
        tensor->dim[0] != MOE_ROUTER_EXPERTS ||
        tensor->dim[1] != rows_per_expert ||
        tensor->bytes % tensor->dim[0] != 0) {
        snprintf(error, error_len, "invalid selected MoE tensor %s", name);
        return false;
    }
    const size_t expert_bytes = (size_t)(tensor->bytes / tensor->dim[0]);
    const size_t total_bytes = MOE_TOP_K * expert_bytes;
    unsigned char *selected = (unsigned char *)malloc(total_bytes);
    if (!selected) {
        snprintf(error, error_len, "selected MoE tensor allocation failed");
        return false;
    }
    for (size_t k = 0; k < MOE_TOP_K; ++k) {
        if (experts[k] >= MOE_ROUTER_EXPERTS) {
            free(selected);
            snprintf(error, error_len, "invalid selected expert ID");
            return false;
        }
        memcpy(selected + k * expert_bytes,
               data + (size_t)experts[k] * expert_bytes, expert_bytes);
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", directory, name);
    const bool ok = write_binary_atomic(path, selected, total_bytes, error,
                                        error_len);
    free(selected);
    return ok;
}

static bool write_moe_fixture(worker *w, const moe_capture_case *case_data,
                              const char *directory, uint32_t token,
                              char *error, size_t error_len) {
    const q38_layer_weights *layer = &w->weights.layer[case_data->layer];
    const q38_tensor *gate_up = layer->experts.bank[0].gate_up;
    const q38_tensor *down = layer->experts.bank[0].down;
    if (!layer->router || !layer->shared_gate_proj || !layer->shared_up_proj ||
        !layer->shared_down_proj || !layer->shared_expert_gate || !gate_up ||
        !down || gate_up->type != 10 || down->type != 10 ||
        layer->router->type != Q38_FORWARD_BF16 ||
        layer->shared_gate_proj->type != Q38_FORWARD_BF16 ||
        layer->shared_up_proj->type != Q38_FORWARD_BF16 ||
        layer->shared_down_proj->type != Q38_FORWARD_BF16 ||
        layer->shared_expert_gate->type != Q38_FORWARD_BF16) {
        snprintf(error, error_len, "layer %u MoE fixture tensors incomplete",
                 case_data->layer);
        return false;
    }
    if (!case_data->routed_captured || !case_data->shared_captured) {
        snprintf(error, error_len, "layer %u MoE trace incomplete",
                 case_data->layer);
        return false;
    }
    if (!make_directory(directory, error, error_len))
        return false;

    float final_output[HIDDEN];
    for (size_t i = 0; i < HIDDEN; ++i)
        final_output[i] = case_data->routed[i] + case_data->shared[i];

    char path[1200];
    snprintf(path, sizeof(path), "%s/hidden.f32", directory);
    if (!write_binary_atomic(path, case_data->hidden, sizeof(case_data->hidden),
                             error, error_len))
        return false;
    snprintf(path, sizeof(path), "%s/router_logits_pre.f32", directory);
    if (!write_binary_atomic(path, case_data->router_logits_pre,
                             sizeof(case_data->router_logits_pre), error,
                             error_len))
        return false;
    snprintf(path, sizeof(path), "%s/router_logits_effective.f32", directory);
    if (!write_binary_atomic(path, case_data->router_logits_effective,
                             sizeof(case_data->router_logits_effective), error,
                             error_len))
        return false;
    snprintf(path, sizeof(path), "%s/selected_experts.u16", directory);
    if (!write_binary_atomic(path, case_data->selected_experts,
                             sizeof(case_data->selected_experts), error,
                             error_len))
        return false;
    snprintf(path, sizeof(path), "%s/selected_weights_pre.f32", directory);
    if (!write_binary_atomic(path, case_data->selected_weights_pre,
                             sizeof(case_data->selected_weights_pre), error,
                             error_len))
        return false;
    snprintf(path, sizeof(path), "%s/selected_weights.f32", directory);
    if (!write_binary_atomic(path, case_data->selected_weights,
                             sizeof(case_data->selected_weights), error,
                             error_len))
        return false;
    snprintf(path, sizeof(path), "%s/expected_routed.f32", directory);
    if (!write_binary_atomic(path, case_data->routed, sizeof(case_data->routed),
                             error, error_len))
        return false;
    snprintf(path, sizeof(path), "%s/expected_shared.f32", directory);
    if (!write_binary_atomic(path, case_data->shared, sizeof(case_data->shared),
                             error, error_len))
        return false;
    snprintf(path, sizeof(path), "%s/expected.f32", directory);
    if (!write_binary_atomic(path, final_output, sizeof(final_output), error,
                             error_len))
        return false;
    if (!write_moe_tensor(directory, "router.bf16", w->model, layer->router,
                           error, error_len) ||
        !write_moe_selected_tensor(directory, "selected_gate_up.q2_k",
                                    w->model, gate_up, case_data->selected_experts,
                                    1280, error, error_len) ||
        !write_moe_selected_tensor(directory, "selected_down.q2_k", w->model,
                                    down, case_data->selected_experts, 640,
                                    error, error_len) ||
        !write_moe_tensor(directory, "shared_gate.bf16", w->model,
                           layer->shared_gate_proj, error, error_len) ||
        !write_moe_tensor(directory, "shared_up.bf16", w->model,
                           layer->shared_up_proj, error, error_len) ||
        !write_moe_tensor(directory, "shared_down.bf16", w->model,
                           layer->shared_down_proj, error, error_len) ||
        !write_moe_tensor(directory, "shared_gate_weight.bf16", w->model,
                           layer->shared_expert_gate, error, error_len))
        return false;

    snprintf(path, sizeof(path), "%s/metadata.json", directory);
    char metadata[4096];
    const int length = snprintf(
        metadata, sizeof(metadata),
        "{\"real_capture\":true,\"token\":%u,\"layer\":%u,"
        "\"hidden_elements\":%d,\"router_shape\":[%u,%d],"
        "\"selected_count\":%d,\"gate_up_shape\":[%d,%d,%d],"
        "\"down_storage_shape\":[%d,%d,%d],"
        "\"source_model\":\"%s\",\"router_tensor\":\"%.*s\","
        "\"gate_up_tensor\":\"%.*s\",\"down_tensor\":\"%.*s\","
        "\"selected_experts\":[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u],"
        "\"tensor_offsets\":{\"router\":%" PRIu64
        ",\"gate_up\":%" PRIu64 ",\"down\":%" PRIu64 "}}\n",
        token, case_data->layer, HIDDEN, MOE_ROUTER_EXPERTS, HIDDEN, MOE_TOP_K,
        MOE_TOP_K, 1280, HIDDEN / MOE_QK_K, MOE_TOP_K, MOE_INTERMEDIATE,
        HIDDEN / MOE_QK_K, w->model ? "q38_runtime_q2" : "unknown",
        (int)layer->router->name.len, layer->router->name.ptr,
        (int)gate_up->name.len, gate_up->name.ptr,
        (int)down->name.len, down->name.ptr,
        case_data->selected_experts[0], case_data->selected_experts[1],
        case_data->selected_experts[2], case_data->selected_experts[3],
        case_data->selected_experts[4], case_data->selected_experts[5],
        case_data->selected_experts[6], case_data->selected_experts[7],
        case_data->selected_experts[8], case_data->selected_experts[9],
        layer->router->rel_offset, gate_up->rel_offset, down->rel_offset);
    if (length < 0 || (size_t)length >= sizeof(metadata)) {
        snprintf(error, error_len, "MoE fixture metadata is too large");
        return false;
    }
    return write_binary_atomic(path, metadata, (size_t)length, error,
                               error_len);
}

static bool capture_moe_fixtures(worker *w, uint32_t token) {
    static const uint32_t layers[MOE_CAPTURE_COUNT] = {0, 23, 47};
    static const char *directories[MOE_CAPTURE_COUNT] = {
        "tests/fixtures/moe/early",
        "tests/fixtures/moe/middle",
        "tests/fixtures/moe/late",
    };
    moe_capture_context capture = {.worker = w};
    for (size_t i = 0; i < MOE_CAPTURE_COUNT; ++i)
        capture.cases[i].layer = layers[i];

    float *logits = (float *)calloc(VOCAB, sizeof(float));
    char error[256] = {0};
    if (!logits) {
        fprintf(stderr, "CAPTURE_MOE error=logit allocation failed\n");
        return false;
    }
    q38_forward_state_reset(&w->state);
    q38_forward_diagnostics diagnostics = {0};
    diagnostics.moe_trace = capture_moe_trace;
    diagnostics.boundary_trace = capture_moe_boundary;
    diagnostics.trace_user = &capture;
    diagnostics.backend_context = backend_context_trace;
    diagnostics.backend_context_user = w->cuda;
    const q38_forward_backend_config backend = cuda_backend_config(w);
    setenv("Q38_TRACE_ALL_QSA", "1", 1);
    const bool ok = q38_forward_full_with_backend_config(
        w->model, &w->weights, &w->state, &token, 1, logits, VOCAB,
        &diagnostics, &backend, error, sizeof(error));
    unsetenv("Q38_TRACE_ALL_QSA");
    free(logits);
    if (!ok) {
        fprintf(stderr, "CAPTURE_MOE error=%s\n",
                error[0] ? error : "forward failed");
        return false;
    }
    for (size_t i = 0; i < MOE_CAPTURE_COUNT; ++i) {
        if (!write_moe_fixture(w, &capture.cases[i], directories[i], token,
                               error, sizeof(error))) {
            fprintf(stderr, "CAPTURE_MOE error=%s\n", error);
            return false;
        }
        printf("{\"capture_moe\":{\"layer\":%u,\"fixture\":\"%s\"}}\n",
               capture.cases[i].layer, directories[i]);
    }
    fflush(stdout);
    return true;
}

static void status(worker *w) {
    q38_forward_cuda_residency_stats stats;
    q38_forward_cuda_get_residency_stats(w->cuda, &stats);
    printf("{\"status\":{\"model_loaded\":true,\"cuda_initialized\":true,"
           "\"model_mmap_identity\":\"%016" PRIxPTR
           "\",\"model_bindings_identity\":\"%016" PRIxPTR
           "\",\"state_identity\":\"%016" PRIxPTR
           "\",\"tokenizer_identity\":\"%016" PRIxPTR
           "\",\"all_non_ple_resident\":%s,\"resident_non_ple_bytes\":%zu,"
           "\"resident_non_ple_tensors\":%" PRIu64
           ",\"resident_pointer_fingerprint\":\"%016" PRIx64
           "\",\"cuda_context_identity\":\"%016" PRIx64
           "\",\"cuda_stream_identity\":\"%016" PRIx64
           "\",\"workspace_pointer_fingerprint\":\"%016" PRIx64
           "\",\"cuda_allocations\":%" PRIu64
           ",\"weight_upload_bytes\":%zu,\"residency_misses\":%" PRIu64
           ",\"non_ple_residency_miss\":%" PRIu64
           ",\"non_ple_upload_bytes\":%zu"
           ",\"ple_file_backed_accesses\":%" PRIu64
           ",\"ple_file_bytes\":%zu"
           ",\"persistent_ple_entries\":%" PRIu64
           ",\"tokenizer\":\"%s\"}}\n",
           (uintptr_t)w->model->map, (uintptr_t)&w->weights,
           (uintptr_t)&w->state, (uintptr_t)&w->tokenizer_state,
           stats.all_non_ple_resident ? "true" : "false",
           stats.persistent_resident_bytes, stats.persistent_resident_tensors,
           stats.persistent_pointer_fingerprint,
           stats.cuda_context_identity, stats.cuda_stream_identity,
           stats.workspace_pointer_fingerprint, stats.cuda_allocations,
           stats.matrix_upload_bytes, stats.resident_misses,
           stats.non_ple_residency_miss,
           stats.non_ple_upload_bytes_per_token,
           stats.ple_file_backed_accesses, stats.ple_file_bytes,
           stats.persistent_ple_entries, w->tokenizer);
    fflush(stdout);
}

static bool unload_qsa_plugin(worker *w) {
    q38_forward_cuda_set_qsa_candidate(w->cuda, NULL);
    w->qsa_plugin = NULL;
    w->qsa_plugin_path[0] = '\0';
    if (w->qsa_plugin_staged_path[0]) {
        (void)remove(w->qsa_plugin_staged_path);
        w->qsa_plugin_staged_path[0] = '\0';
    }
    puts("{\"qsa_plugin\":null,\"module_retained\":true}");
    fflush(stdout);
    return true;
}

static bool stage_qsa_plugin(worker *w, const char *source,
                             char *staged, size_t staged_len) {
    snprintf(staged, staged_len, "/tmp/q38_qsa_candidate_%ld_%" PRIu64 ".so",
             (long)getpid(), ++w->qsa_plugin_generation);
    FILE *in = fopen(source, "rb");
    FILE *out = in ? fopen(staged, "wb") : NULL;
    if (!in || !out) {
        if (in) fclose(in);
        if (out) fclose(out);
        (void)remove(staged);
        fprintf(stderr, "LOAD_QSA_PLUGIN error=failed to stage %s\n", source);
        return false;
    }
    unsigned char buffer[16384];
    size_t count;
    bool ok = true;
    while ((count = fread(buffer, 1, sizeof(buffer), in)) != 0) {
        if (fwrite(buffer, 1, count, out) != count) {
            ok = false;
            break;
        }
    }
    if (ferror(in)) ok = false;
    fclose(in);
    fclose(out);
    if (!ok) {
        (void)remove(staged);
        fprintf(stderr, "LOAD_QSA_PLUGIN error=failed to copy %s\n", source);
        return false;
    }
    return true;
}

static bool load_qsa_plugin(worker *w, const char *path) {
    if (!path || !path[0]) {
        fprintf(stderr, "LOAD_QSA_PLUGIN error=missing path\n");
        return false;
    }
    if (w->qsa_plugin && !unload_qsa_plugin(w))
        return false;
    char staged_path[sizeof(w->qsa_plugin_staged_path)];
    if (!stage_qsa_plugin(w, path, staged_path, sizeof(staged_path)))
        return false;
    void *handle = dlopen(staged_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "LOAD_QSA_PLUGIN error=%s\n", dlerror());
        (void)remove(staged_path);
        return false;
    }
    dlerror();
    q38_qsa_candidate_abi_fn abi = NULL;
    *(void **)(&abi) = dlsym(handle, Q38_QSA_CANDIDATE_ABI_SYMBOL);
    const char *lookup_error = dlerror();
    if (lookup_error || !abi || abi() != Q38_QSA_CANDIDATE_ABI_VERSION) {
        fprintf(stderr, "LOAD_QSA_PLUGIN error=%s\n",
                lookup_error ? lookup_error : "incompatible QSA candidate ABI");
        dlclose(handle);
        (void)remove(staged_path);
        return false;
    }
    q38_qsa_candidate_fn candidate = NULL;
    dlerror();
    *(void **)(&candidate) = dlsym(handle, Q38_QSA_CANDIDATE_SYMBOL);
    lookup_error = dlerror();
    if (lookup_error || !candidate) {
        fprintf(stderr, "LOAD_QSA_PLUGIN error=%s\n",
                lookup_error ? lookup_error : "missing QSA candidate symbol");
        dlclose(handle);
        (void)remove(staged_path);
        return false;
    }
    w->qsa_plugin_handle = handle;
    w->qsa_plugin = candidate;
    snprintf(w->qsa_plugin_path, sizeof(w->qsa_plugin_path), "%s", path);
    snprintf(w->qsa_plugin_staged_path,
             sizeof(w->qsa_plugin_staged_path), "%s", staged_path);
    q38_forward_cuda_set_qsa_candidate(w->cuda, candidate);
    printf("{\"qsa_plugin\":\"%s\",\"abi\":%d}\n", w->qsa_plugin_path,
           abi());
    fflush(stdout);
    return true;
}

static bool run_forward_tokens(worker *w, const uint32_t *tokens,
                               size_t token_count) {
    float *logits = (float *)calloc(token_count * VOCAB, sizeof(float));
    char error[256] = {0};
    if (!logits) return false;
    q38_forward_state_reset(&w->state);
    q38_forward_diagnostics diagnostics = {0};
    q38_forward_qsa_timing qsa_timing = {0};
    diagnostics.backend_context = backend_context_trace;
    diagnostics.trace_user = w->cuda;
    diagnostics.qsa_timing = &qsa_timing;
    diagnostics.qsa_qkv_backend = q38_forward_cuda_qsa_qkv_backend;
    diagnostics.qsa_qkv_backend_user = w->cuda;
    const double started = now_ms();
    const q38_forward_backend_config backend = cuda_backend_config(w);
    const bool ok = q38_forward_full_with_backend_config(
        w->model, &w->weights, &w->state, tokens, token_count, logits, VOCAB,
        &diagnostics, &backend, error, sizeof(error));
    if (!ok)
        fprintf(stderr, "RUN_FORWARD error=%s\n", error);
    else {
        q38_ple_scheduler_stats ple = {0};
        const bool have_ple =
            q38_forward_state_get_ple_prefetch_stats(&w->state, &ple);
        const double decoder_available_ms =
            have_ple && ple.consume_ms > ple.start_ms
                ? ple.consume_ms - ple.start_ms : 0.0;
        printf("{\"run_forward\":{\"token\":%u,\"argmax\":%zu,"
               "\"logits_hash\":\"%016" PRIx64
               "\",\"wall_ms\":%.6f,\"qsa_total_ms\":%.6f,"
               "\"qkv_ms\":%.6f,\"indexer_ms\":%.6f,\"score_ms\":%.6f,"
               "\"topk_ms\":%.6f,\"kv_gather_ms\":%.6f,"
               "\"attention_ms\":%.6f,\"state_update_ms\":%.6f,"
               "\"qsa_allocations\":%" PRIu64
               ",\"qsa_kernel_launches\":%" PRIu64
               ",\"qsa_host_syncs\":%" PRIu64
               ",\"qsa_h2d_bytes\":%" PRIu64
               ",\"qsa_d2h_bytes\":%" PRIu64
               ",\"qsa_residency_misses\":%" PRIu64
               ",\"ple_logical_accesses\":%" PRIu64
               ",\"ple_unique_rows\":%" PRIu64
               ",\"ple_unique_physical_blocks\":%" PRIu64
               ",\"ple_file_read_ops\":%" PRIu64
               ",\"ple_logical_bytes\":%" PRIu64
               ",\"ple_physical_bytes\":%" PRIu64
               ",\"ple_cache_hits\":%" PRIu64
               ",\"ple_cache_misses\":%" PRIu64
               ",\"ple_start_ms\":%.6f,\"ple_ready_ms\":%.6f"
               ",\"ple_consume_ms\":%.6f,\"ple_elapsed_ms\":%.6f"
               ",\"ple_overlap_ms\":%.6f"
               ",\"ple_wait_at_injection_ms\":%.6f"
               ",\"decoder_compute_available_for_overlap_ms\":%.6f}}\n",
               tokens[token_count - 1],
               argmax(logits + (token_count - 1) * VOCAB, VOCAB),
               hash_floats(logits + (token_count - 1) * VOCAB, VOCAB),
               now_ms() - started, qsa_timing.total_ms,
               qsa_timing.qkv_projection_ms,
               qsa_timing.indexer_compression_ms, qsa_timing.score_ms,
               qsa_timing.exact_top_k_ms, qsa_timing.selected_kv_gather_ms,
               qsa_timing.attention_ms, qsa_timing.state_update_ms,
               qsa_timing.allocations, qsa_timing.kernel_launches,
               qsa_timing.host_syncs, qsa_timing.h2d_bytes,
               qsa_timing.d2h_bytes, qsa_timing.residency_misses,
               have_ple ? ple.logical_accesses : 0,
               have_ple ? ple.unique_rows : 0,
               have_ple ? ple.unique_physical_blocks : 0,
               have_ple ? ple.file_read_ops : 0,
               have_ple ? ple.logical_bytes : 0,
               have_ple ? ple.physical_bytes : 0,
               have_ple ? ple.cache_hits : 0,
               have_ple ? ple.cache_misses : 0,
               have_ple ? ple.start_ms : 0.0,
               have_ple ? ple.ready_ms : 0.0,
               have_ple ? ple.consume_ms : 0.0,
               have_ple ? ple.elapsed_ms : 0.0,
               have_ple ? ple.overlap_ms : 0.0,
               have_ple ? ple.wait_ms : 0.0,
               decoder_available_ms);
    }
    free(logits);
    fflush(stdout);
    return ok;
}

static bool run_forward(worker *w, uint32_t token) {
    return run_forward_tokens(w, &token, 1);
}

static void bench_qsa(worker *w, unsigned count) {
    double total = 0.0;
    bool pass = true;
    for (unsigned i = 0; i < count; ++i) {
        const double started = now_ms();
        pass &= run_qkv_fixture(w);
        total += now_ms() - started;
    }
    printf("{\"bench_qsa\":{\"runs\":%u,\"avg_fixture_ms\":%.6f,"
           "\"pass\":%s}}\n", count, count ? total / count : 0.0,
           pass ? "true" : "false");
    fflush(stdout);
}

static bool parse_u32(const char *line, const char *key, uint32_t *value) {
    const char *found = strstr(line, key);
    if (!found) return false;
    char *end = NULL;
    unsigned long parsed = strtoul(found + strlen(key), &end, 10);
    if (end == found + strlen(key) || parsed > UINT32_MAX) return false;
    *value = (uint32_t)parsed;
    return true;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *tokenizer = "";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc)
            tokenizer = argv[++i];
    }
    if (!model_path) {
        fprintf(stderr, "usage: q38_dev_worker --model PATH "
                        "[--tokenizer PATH]\n");
        return 2;
    }
    worker w;
    memset(&w, 0, sizeof(w));
    snprintf(w.tokenizer, sizeof(w.tokenizer), "%s", tokenizer);
    snprintf(w.fixture_dir, sizeof(w.fixture_dir),
             "artifacts/post_m8_opt/qsa_layer3_fixture");
    (void)mkdir(w.fixture_dir, 0755);
    char error[256] = {0};
    w.model = q38_gguf_open(model_path, error, sizeof(error));
    if (!w.model ||
        !q38_weights_bind_subset(w.model, 47, &w.weights, error,
                                 sizeof(error)) ||
        !(w.cuda = q38_forward_cuda_context_create(error, sizeof(error))) ||
        !q38_forward_cuda_enable_all_non_ple_residency(
            w.cuda, w.model, error, sizeof(error)) ||
        !q38_forward_state_init(&w.state, &w.weights, 248044, error,
                                sizeof(error))) {
        fprintf(stderr, "worker startup failed: %s\n",
                error[0] ? error : "unknown error");
        q38_forward_cuda_context_destroy(w.cuda);
        q38_weights_release(&w.weights);
        q38_gguf_close(w.model);
        return 1;
    }
    /* Keep detailed row-level PLE telemetry out of performance runs. */
    q38_forward_cuda_set_telemetry_observer(w.cuda, NULL, NULL);
    if (tokenizer[0]) {
        if (!q38_tokenizer_init(&w.tokenizer_state, tokenizer, NULL, error,
                                sizeof(error))) {
            fprintf(stderr, "tokenizer startup failed: %s\n", error);
            q38_forward_state_destroy(&w.state);
            q38_forward_cuda_context_destroy(w.cuda);
            q38_weights_release(&w.weights);
            q38_gguf_close(w.model);
            return 1;
        }
        w.tokenizer_initialized = true;
    }
    w.state_initialized = true;
    printf("{\"ready\":true,\"commands\":[\"RESET\",\"LOAD_QSA_PLUGIN\","
           "\"UNLOAD_QSA_PLUGIN\",\"RUN_QKV_FIXTURE\",\"RUN_QKV\","
           "\"RUN_FORWARD\",\"BENCH_QSA\",\"CAPTURE\",\"CAPTURE_QSA\","
           "\"CAPTURE_MOE\","
           "\"STATUS\",\"QUIT\"]}\n");
    status(&w);
    char line[1024];
    while (fgets(line, sizeof(line), stdin)) {
        if (!strncmp(line, "QUIT", 4) || !strncmp(line, "SHUTDOWN", 8))
            break;
        if (!strncmp(line, "RESET", 5)) {
            q38_forward_state_reset(&w.state);
            puts("{\"reset\":true}");
            fflush(stdout);
        } else if (!strncmp(line, "LOAD_QSA_PLUGIN", 15)) {
            const char *path = strstr(line, "path=");
            if (!path) {
                fprintf(stderr, "LOAD_QSA_PLUGIN requires path=...\n");
                continue;
            }
            char plugin_path[512];
            snprintf(plugin_path, sizeof(plugin_path), "%s", path + 5);
            plugin_path[strcspn(plugin_path, "\r\n")] = '\0';
            load_qsa_plugin(&w, plugin_path);
        } else if (!strncmp(line, "UNLOAD_QSA_PLUGIN", 17)) {
            unload_qsa_plugin(&w);
        } else if (!strncmp(line, "STATUS", 6)) {
            status(&w);
        } else if (!strncmp(line, "CAPTURE_MOE", 11)) {
            uint32_t token = 9419;
            parse_u32(line, "token=", &token);
            capture_moe_fixtures(&w, token);
        } else if (!strncmp(line, "CAPTURE_QSA", 11) ||
                   !strncmp(line, "CAPTURE", 7)) {
            uint32_t token = 9419;
            uint32_t layer = QSA_LAYER;
            parse_u32(line, "token=", &token);
            if (parse_u32(line, "layer=", &layer) && layer != QSA_LAYER)
                fprintf(stderr, "CAPTURE supports only layer=%d\n", QSA_LAYER);
            else
                capture_hidden(&w, token);
        } else if (!strncmp(line, "RUN_QKV", 7)) {
            uint32_t layer = QSA_LAYER;
            if (parse_u32(line, "layer=", &layer) && layer != QSA_LAYER)
                fprintf(stderr, "RUN_QKV_FIXTURE supports only layer=%d\n",
                        QSA_LAYER);
            else
                run_qkv_fixture(&w);
        } else if (!strncmp(line, "BENCH_QSA", 9)) {
            uint32_t count = 20;
            parse_u32(line, "runs=", &count);
            bench_qsa(&w, count);
        } else if (!strncmp(line, "RUN_FORWARD", 11)) {
            const char *prompt = strstr(line, "prompt=");
            if (prompt && w.tokenizer_initialized) {
                q38_token_batch batch = {0};
                char *text = strdup(prompt + 7);
                if (!text) {
                    fprintf(stderr, "RUN_FORWARD prompt allocation failed\n");
                    continue;
                }
                text[strcspn(text, "\r\n")] = '\0';
                if (!q38_tokenizer_encode(&w.tokenizer_state, text, false,
                                          &batch, error, sizeof(error))) {
                    fprintf(stderr, "RUN_FORWARD tokenization error=%s\n",
                            error);
                    free(text);
                    continue;
                }
                free(text);
                run_forward_tokens(&w, batch.tokens, batch.token_count);
                q38_token_batch_free(&batch);
                continue;
            }
            if (prompt) {
                fprintf(stderr, "RUN_FORWARD prompt requires --tokenizer\n");
                continue;
            }
            uint32_t token = 9419;
            parse_u32(line, "token=", &token);
            run_forward(&w, token);
        } else {
            fprintf(stderr, "unknown command: %s", line);
        }
    }
    if (w.qsa_plugin_handle)
        unload_qsa_plugin(&w);
    if (w.state_initialized)
        q38_forward_state_destroy(&w.state);
    if (w.tokenizer_initialized)
        q38_tokenizer_destroy(&w.tokenizer_state);
    q38_forward_cuda_context_destroy(w.cuda);
    q38_weights_release(&w.weights);
    q38_gguf_close(w.model);
    return 0;
}
