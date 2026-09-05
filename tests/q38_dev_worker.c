#include "q38_forward.h"
#include "q38_forward_cuda.h"
#include "q38_gguf.h"
#include "q38_tokenizer.h"
#include "q38_weights.h"

#include <dlfcn.h>
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
    const bool ok = q38_forward_full_with_matrix_moe_layer_backend(
        w->model, &w->weights, &w->state, tokens, 1, logits, VOCAB,
        &diagnostics, q38_forward_cuda_matvec_backend,
        q38_forward_cuda_matrix_backend, q38_forward_cuda_expert_backend,
        q38_forward_cuda_moe_layer_q2_backend, w->cuda, error,
        sizeof(error));
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
           ",\"persistent_ple_entries\":%" PRIu64 ",\"tokenizer\":\"%s\"}}\n",
           (uintptr_t)w->model->map, (uintptr_t)&w->weights,
           (uintptr_t)&w->state, (uintptr_t)&w->tokenizer_state,
           stats.all_non_ple_resident ? "true" : "false",
           stats.persistent_resident_bytes, stats.persistent_resident_tensors,
           stats.persistent_pointer_fingerprint,
           stats.cuda_context_identity, stats.cuda_stream_identity,
           stats.workspace_pointer_fingerprint, stats.cuda_allocations,
           stats.matrix_upload_bytes, stats.resident_misses,
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
    diagnostics.qsa_timing = &qsa_timing;
    diagnostics.qsa_qkv_backend = q38_forward_cuda_qsa_qkv_backend;
    diagnostics.qsa_qkv_backend_user = w->cuda;
    const bool ok = q38_forward_full_with_matrix_moe_layer_backend(
        w->model, &w->weights, &w->state, tokens, token_count, logits, VOCAB,
        &diagnostics, q38_forward_cuda_matvec_backend,
        q38_forward_cuda_matrix_backend, q38_forward_cuda_expert_backend,
        q38_forward_cuda_moe_layer_q2_backend, w->cuda, error,
        sizeof(error));
    if (!ok)
        fprintf(stderr, "RUN_FORWARD error=%s\n", error);
    else
        printf("{\"run_forward\":{\"token\":%u,\"argmax\":%zu,"
               "\"logits_hash\":\"%016" PRIx64
               "\",\"qkv_ms\":%.6f}}\n", tokens[token_count - 1],
               argmax(logits + (token_count - 1) * VOCAB, VOCAB),
               hash_floats(logits + (token_count - 1) * VOCAB, VOCAB),
               qsa_timing.qkv_projection_ms);
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
