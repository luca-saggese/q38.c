#include "q38_gdn.h"
#include "q38_gdn_ref.h"
#include "q38_gr.h"
#include "q38_gr_ref.h"
#include "q38_cuda_primitives.h"
#include "q38_oracle.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kTokens = 2;
constexpr size_t kHidden = Q38_GR_HIDDEN;
constexpr size_t kBranches = Q38_GR_BRANCHES;
constexpr size_t kGrWidth = kBranches * kHidden;
constexpr size_t kQkvRows = Q38_GDN_QKV_CHANNELS;
constexpr size_t kZRows = Q38_GDN_VALUE_CHANNELS;
constexpr size_t kGateRows = Q38_GDN_VALUE_HEADS;
constexpr size_t kOutRows = kHidden;
constexpr size_t kOutCols = kZRows;
constexpr size_t kHeads = Q38_GDN_VALUE_HEADS;
constexpr size_t kHeadDim = Q38_GDN_HEAD_DIM;
constexpr size_t kStateElements = kHeads * kHeadDim * kHeadDim;

struct Check {
    const char *name;
    size_t elements;
    float max_abs;
    bool pass;
};

static std::vector<Check> checks;

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static float softplus(float x) {
    return x > 20.0f ? x : std::log1pf(std::exp(x));
}

static void record_compare(const char *name, const std::vector<float> &expected,
                           const std::vector<float> &actual, float tolerance) {
    q38_oracle_metrics metrics;
    q38_oracle_compare(expected.data(), actual.data(), expected.size(), 1e-7f,
                       &metrics);
    const bool pass = metrics.max_abs <= tolerance;
    checks.push_back({name, expected.size(), metrics.max_abs, pass});
    if (!pass) {
        std::fprintf(stderr,
                     "%s mismatch: max_abs=%g tolerance=%g at %zu\n", name,
                     metrics.max_abs, tolerance, expected.size());
    }
}

static void record_compare_raw(const char *name, const float *expected,
                               const float *actual, size_t elements,
                               float tolerance) {
    std::vector<float> e(expected, expected + elements);
    std::vector<float> a(actual, actual + elements);
    record_compare(name, e, a, tolerance);
}

static void project_ref(const std::vector<float> &weights, size_t rows,
                        size_t cols, const std::vector<float> &input,
                        size_t tokens, std::vector<float> &output) {
    output.assign(tokens * rows, 0.0f);
    for (size_t t = 0; t < tokens; t++)
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                output[t * rows + r] +=
                    weights[r * cols + c] * input[t * cols + c];
}

static void conv_ref(const std::vector<float> &input, size_t tokens,
                     size_t channels, const std::vector<float> &kernel,
                     size_t kernel_size, std::vector<float> &history,
                     std::vector<float> &output) {
    const size_t history_tokens = kernel_size - 1u;
    output.assign(tokens * channels, 0.0f);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (size_t tap = 0; tap < kernel_size; tap++) {
                const size_t source = history_tokens + t -
                                      (kernel_size - 1u - tap);
                const float sample = source < history_tokens
                    ? history[source * channels + c]
                    : input[(source - history_tokens) * channels + c];
                sum += kernel[tap * channels + c] * sample;
            }
            output[t * channels + c] = sum;
        }
    }
    std::vector<float> combined(history_tokens * channels + input.size());
    std::memcpy(combined.data(), history.data(),
                history.size() * sizeof(float));
    std::memcpy(combined.data() + history.size(), input.data(),
                input.size() * sizeof(float));
    for (size_t h = 0; h < history_tokens; h++)
        std::memcpy(history.data() + h * channels,
                    combined.data() + (tokens + h) * channels,
                    channels * sizeof(float));
}

static void split_qkv_ref(const std::vector<float> &mixed, size_t tokens,
                          std::vector<float> &q, std::vector<float> &k,
                          std::vector<float> &v) {
    q.assign(tokens * Q38_GDN_KEY_CHANNELS, 0.0f);
    k.assign(tokens * Q38_GDN_KEY_CHANNELS, 0.0f);
    v.assign(tokens * Q38_GDN_VALUE_CHANNELS, 0.0f);
    for (size_t t = 0; t < tokens; t++) {
        const float *source = mixed.data() + t * Q38_GDN_QKV_CHANNELS;
        std::memcpy(q.data() + t * Q38_GDN_KEY_CHANNELS, source,
                    Q38_GDN_KEY_CHANNELS * sizeof(float));
        std::memcpy(k.data() + t * Q38_GDN_KEY_CHANNELS,
                    source + Q38_GDN_KEY_CHANNELS,
                    Q38_GDN_KEY_CHANNELS * sizeof(float));
        std::memcpy(v.data() + t * Q38_GDN_VALUE_CHANNELS,
                    source + 2u * Q38_GDN_KEY_CHANNELS,
                    Q38_GDN_VALUE_CHANNELS * sizeof(float));
    }
}

static void repeat_key_ref(const std::vector<float> &key, size_t tokens,
                           std::vector<float> &value) {
    value.assign(tokens * Q38_GDN_VALUE_CHANNELS, 0.0f);
    for (size_t t = 0; t < tokens; t++)
        for (size_t h = 0; h < kHeads; h++)
            std::memcpy(value.data() + t * Q38_GDN_VALUE_CHANNELS +
                            h * kHeadDim,
                        key.data() + t * Q38_GDN_KEY_CHANNELS +
                            (h / 3u) * kHeadDim,
                        kHeadDim * sizeof(float));
}

static void normalize_heads_ref(std::vector<float> &values, size_t tokens) {
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < kHeads; h++) {
            float sum = 0.0f;
            float *head = values.data() + t * kHeads * kHeadDim + h * kHeadDim;
            for (size_t d = 0; d < kHeadDim; d++) sum += head[d] * head[d];
            const float inv = 1.0f / std::sqrt(sum / kHeadDim + 1e-6f);
            for (size_t d = 0; d < kHeadDim; d++) head[d] *= inv;
        }
    }
}

static void gates_ref(const std::vector<float> &a, const std::vector<float> &b,
                      const std::vector<float> &dt,
                      const std::vector<float> &a_log, size_t tokens,
                      std::vector<float> &decay, std::vector<float> &beta) {
    decay.resize(tokens * kGateRows);
    beta.resize(tokens * kGateRows);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < kGateRows; h++) {
            const float x = a[t * kGateRows + h] + dt[h];
            beta[t * kGateRows + h] = sigmoid(b[t * kGateRows + h]);
            decay[t * kGateRows + h] =
                std::exp(-std::exp(a_log[h]) * softplus(x));
        }
    }
}

static void gated_norm_ref(const std::vector<float> &recurrent,
                           const std::vector<float> &z,
                           const std::vector<float> &norm, size_t tokens,
                           std::vector<float> &output) {
    output.resize(tokens * kZRows);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t h = 0; h < kHeads; h++) {
            const float *source =
                recurrent.data() + t * kZRows + h * kHeadDim;
            float *destination = output.data() + t * kZRows + h * kHeadDim;
            float sum = 0.0f;
            for (size_t d = 0; d < kHeadDim; d++) sum += source[d] * source[d];
            const float inv = 1.0f / std::sqrt(sum / kHeadDim + 1e-6f);
            for (size_t d = 0; d < kHeadDim; d++)
                destination[d] = source[d] * inv * norm[d] *
                                 sigmoid(z[t * kZRows + h * kHeadDim + d]);
        }
    }
}

static void make_sparse_matrix(std::vector<float> &matrix, size_t rows,
                               size_t cols, float scale, size_t salt) {
    matrix.assign(rows * cols, 0.0f);
    for (size_t row = 0; row < rows; row++) {
        const size_t col = (row * 17u + salt * 13u + 5u) % cols;
        matrix[row * cols + col] =
            scale * (static_cast<float>((row + salt) % 11u) - 5.0f);
    }
}

__global__ static void gate_transform_kernel(const float *a, const float *b,
                                             const float *dt,
                                             const float *a_log, size_t tokens,
                                             float *decay, float *beta) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = tokens * Q38_GDN_VALUE_HEADS;
    if (i >= total) return;
    const size_t head = i % Q38_GDN_VALUE_HEADS;
    const float x = a[i] + dt[head];
    beta[i] = 1.0f / (1.0f + expf(-b[i]));
    const float positive = x > 20.0f ? x : log1pf(expf(x));
    decay[i] = expf(-expf(a_log[head]) * positive);
}

__global__ static void normalize_repeated_kernel(const float *input,
                                                 size_t tokens, float *output) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = tokens * Q38_GDN_VALUE_HEADS;
    if (i >= total) return;
    const float *source = input + i * Q38_GDN_HEAD_DIM;
    float *destination = output + i * Q38_GDN_HEAD_DIM;
    float sum = 0.0f;
    for (size_t d = 0; d < Q38_GDN_HEAD_DIM; d++)
        sum += source[d] * source[d];
    const float inv = rsqrtf(sum / Q38_GDN_HEAD_DIM + 1e-6f);
    for (size_t d = 0; d < Q38_GDN_HEAD_DIM; d++) destination[d] = source[d] * inv;
}

__global__ static void apply_gate_kernel(const float *normalized,
                                         const float *z, size_t tokens,
                                         float *output) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = tokens * Q38_GDN_VALUE_CHANNELS;
    if (i >= total)
        return;
    output[i] = normalized[i] / (1.0f + expf(-z[i]));
}

static bool write_artifact(const char *path) {
    if (!path) return true;
    FILE *file = std::fopen(path, "w");
    if (!file) {
        std::perror(path);
        return false;
    }
    std::fprintf(file,
                 "{\n"
                 "  \"gate\": \"M3-C09\",\n"
                 "  \"status\": \"pass\",\n"
                 "  \"layer\": 0,\n"
                 "  \"layer_kind\": \"linear_attention\",\n"
                 "  \"fixture\": {\"kind\": \"deterministic_sparse_f32\", "
                 "\"tokens\": %zu, \"model_weights_loaded\": false},\n"
                 "  \"scope\": \"layer-0-shaped GR+GDN probe; not full model "
                 "correctness\",\n"
                 "  \"logical_shapes\": {\n"
                 "    \"gr_residual\": [4, 2560],\n"
                 "    \"gdn_input\": [\"token\", 2560],\n"
                 "    \"qkv_mixed\": [\"token\", 10240],\n"
                 "    \"z\": [\"token\", 48, 128],\n"
                 "    \"q_k_v\": [\"token\", 48, 128],\n"
                 "    \"state\": [1, 48, 128, 128],\n"
                 "    \"block_output\": [\"token\", 2560]\n"
                 "  },\n"
                 "  \"physical_layout\": {\"matrices\": "
                 "\"logical output-by-input row-major\", \"activations\": "
                 "\"token-major\", \"state\": \"contiguous [sequence, "
                 "value_head, row, column] row-major\"},\n"
                 "  \"frozen_order\": [\"GR read\", \"qkv/z/a/b projection\", "
                 "\"beta sigmoid and decay transform\", \"causal conv4\", "
                 "\"SiLU\", \"QKV split\", \"16-to-48 repeat\", \"Q/K L2 "
                 "normalization\", \"FP32 recurrence\", \"RMSNorm\", "
                 "\"sigmoid(z) gate\", \"out_proj\", \"GR write\"],\n"
                 "  \"intermediates\": {\n",
                 kTokens);
    for (size_t i = 0; i < checks.size(); i++) {
        std::fprintf(file, "    \"%s\": {\"elements\": %zu, "
                           "\"max_abs\": %.9g, \"pass\": %s}%s\n",
                     checks[i].name, checks[i].elements, checks[i].max_abs,
                     checks[i].pass ? "true" : "false",
                     i + 1 == checks.size() ? "" : ",");
    }
    std::fprintf(file, "  }\n}\n");
    return std::fclose(file) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    int devices = 0;
    cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "M3-C09: CUDA device unavailable: %s\n",
                     cudaGetErrorString(status));
        return 2;
    }

    std::vector<float> gamma(kGrWidth, 1.0f);
    std::vector<float> gr_down, gr_up, gr_inject;
    make_sparse_matrix(gr_down, Q38_GR_RANK, kGrWidth, 0.002f, 1);
    make_sparse_matrix(gr_up, kGrWidth, Q38_GR_RANK, 0.003f, 2);
    make_sparse_matrix(gr_inject, kBranches, kGrWidth, 0.001f, 3);

    std::vector<float> qkv_weight, z_weight, a_weight, b_weight, out_weight;
    make_sparse_matrix(qkv_weight, kQkvRows, kHidden, 0.001f, 4);
    make_sparse_matrix(z_weight, kZRows, kHidden, 0.001f, 5);
    make_sparse_matrix(a_weight, kGateRows, kHidden, 0.0005f, 6);
    make_sparse_matrix(b_weight, kGateRows, kHidden, 0.0007f, 7);
    make_sparse_matrix(out_weight, kOutRows, kOutCols, 0.001f, 8);
    std::vector<float> conv_kernel(Q38_GDN_CONV_KERNEL * kQkvRows);
    for (size_t tap = 0; tap < Q38_GDN_CONV_KERNEL; tap++)
        for (size_t c = 0; c < kQkvRows; c++)
            conv_kernel[tap * kQkvRows + c] =
                0.02f * (static_cast<float>((tap + c) % 5u) - 2.0f);
    std::vector<float> dt(kGateRows), a_log(kGateRows), norm(kHeadDim, 1.0f);
    for (size_t h = 0; h < kGateRows; h++) {
        dt[h] = -0.1f + 0.01f * static_cast<float>(h % 7u);
        a_log[h] = -1.0f + 0.005f * static_cast<float>(h % 9u);
    }

    std::vector<float> residual(kTokens * kGrWidth);
    for (size_t t = 0; t < kTokens; t++)
        for (size_t i = 0; i < kGrWidth; i++)
            residual[t * kGrWidth + i] =
                (static_cast<float>((i * 7u + t * 11u) % 29u) - 14.0f) /
                31.0f;

    q38_gr_ref_params gr_params = {
        gamma.data(), gr_down.data(), gr_up.data(), gr_inject.data()};
    std::vector<float> zero_block(kTokens * kHidden, 0.0f);
    std::vector<float> gdn_input_ref(kTokens * kHidden);
    std::vector<float> unused_updated(kGrWidth);
    for (size_t t = 0; t < kTokens; t++) {
        q38_gr_collapse(residual.data() + t * kGrWidth, zero_block.data() +
                            t * kHidden, &gr_params,
                        gdn_input_ref.data() + t * kHidden,
                        unused_updated.data());
    }

    std::vector<float *> device;
    auto alloc = [&](size_t elements, const char *what) -> float * {
        float *ptr = nullptr;
        if (!cuda_ok(cudaMalloc(&ptr, elements * sizeof(float)), what))
            return nullptr;
        device.push_back(ptr);
        return ptr;
    };
    auto copy_to_device = [&](float *dst, const std::vector<float> &src,
                              const char *what) -> bool {
        return cuda_ok(cudaMemcpy(dst, src.data(), src.size() * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       what);
    };
    auto copy_from_device = [&](std::vector<float> &dst, const float *src,
                                const char *what) -> bool {
        return cuda_ok(cudaMemcpy(dst.data(), src, dst.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       what);
    };

    float *d_gamma = alloc(gamma.size(), "alloc gamma");
    float *d_gr_down = alloc(gr_down.size(), "alloc GR down");
    float *d_gr_up = alloc(gr_up.size(), "alloc GR up");
    float *d_gr_inject = alloc(gr_inject.size(), "alloc GR inject");
    float *d_qkv_weight = alloc(qkv_weight.size(), "alloc QKV weight");
    float *d_z_weight = alloc(z_weight.size(), "alloc Z weight");
    float *d_a_weight = alloc(a_weight.size(), "alloc A weight");
    float *d_b_weight = alloc(b_weight.size(), "alloc B weight");
    float *d_out_weight = alloc(out_weight.size(), "alloc output weight");
    float *d_conv_kernel = alloc(conv_kernel.size(), "alloc conv kernel");
    float *d_dt = alloc(dt.size(), "alloc dt");
    float *d_a_log = alloc(a_log.size(), "alloc A_log");
    float *d_norm = alloc(norm.size(), "alloc norm");
    float *d_residual = alloc(kGrWidth, "alloc residual");
    float *d_zero_block = alloc(kHidden, "alloc zero block");
    float *d_gr_input = alloc(kHidden, "alloc GR input");
    float *d_gr_updated = alloc(kGrWidth, "alloc GR updated");
    float *d_qkv_input = alloc(gdn_input_ref.size(), "alloc GDN input");
    float *d_qkv = alloc(kTokens * kQkvRows, "alloc QKV");
    float *d_z = alloc(kTokens * kZRows, "alloc Z");
    float *d_a = alloc(kTokens * kGateRows, "alloc A");
    float *d_b = alloc(kTokens * kGateRows, "alloc B");
    float *d_decay = alloc(kTokens * kGateRows, "alloc decay");
    float *d_beta = alloc(kTokens * kGateRows, "alloc beta");
    float *d_conv_history = alloc((Q38_GDN_CONV_KERNEL - 1u) * kQkvRows,
                                  "alloc conv history");
    float *d_conv = alloc(kTokens * kQkvRows, "alloc conv output");
    float *d_silu = alloc(kTokens * kQkvRows, "alloc SiLU");
    float *d_q16 = alloc(kTokens * Q38_GDN_KEY_CHANNELS, "alloc Q16");
    float *d_k16 = alloc(kTokens * Q38_GDN_KEY_CHANNELS, "alloc K16");
    float *d_v = alloc(kTokens * kZRows, "alloc V");
    float *d_q48 = alloc(kTokens * kZRows, "alloc Q48");
    float *d_k48 = alloc(kTokens * kZRows, "alloc K48");
    float *d_q = alloc(kTokens * kZRows, "alloc Q normalized");
    float *d_k = alloc(kTokens * kZRows, "alloc K normalized");
    float *d_state = alloc(kStateElements, "alloc state");
    float *d_recurrent = alloc(kTokens * kZRows, "alloc recurrence output");
    float *d_normed = alloc(kTokens * kZRows, "alloc normed output");
    float *d_gated = alloc(kTokens * kZRows, "alloc gated output");
    float *d_block = alloc(kTokens * kHidden, "alloc block output");
    bool ok = d_block != nullptr;
    if (ok) {
        ok = copy_to_device(d_gamma, gamma, "copy gamma") &&
             copy_to_device(d_gr_down, gr_down, "copy GR down") &&
             copy_to_device(d_gr_up, gr_up, "copy GR up") &&
             copy_to_device(d_gr_inject, gr_inject, "copy GR inject") &&
             copy_to_device(d_qkv_weight, qkv_weight, "copy QKV weight") &&
             copy_to_device(d_z_weight, z_weight, "copy Z weight") &&
             copy_to_device(d_a_weight, a_weight, "copy A weight") &&
             copy_to_device(d_b_weight, b_weight, "copy B weight") &&
             copy_to_device(d_out_weight, out_weight, "copy output weight") &&
             copy_to_device(d_conv_kernel, conv_kernel, "copy conv kernel") &&
             copy_to_device(d_dt, dt, "copy dt") &&
             copy_to_device(d_a_log, a_log, "copy A_log") &&
             copy_to_device(d_norm, norm, "copy norm") &&
             copy_to_device(d_zero_block, std::vector<float>(kHidden, 0.0f),
                            "copy zero block");
    }

    std::vector<float> gdn_input_actual(kTokens * kHidden);
    char error[256] = {};
    for (size_t t = 0; ok && t < kTokens; t++) {
        ok = cuda_ok(cudaMemcpy(d_residual, residual.data() + t * kGrWidth,
                                kGrWidth * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy residual") &&
             q38_cuda_gr_collapse(
                 d_residual, d_gamma, d_gr_down, d_gr_up, d_gr_inject,
                 d_zero_block, d_gr_input, d_gr_updated, nullptr, error,
                 sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "GR synchronize");
        if (ok) {
            std::vector<float> one(kHidden);
            ok = cuda_ok(cudaMemcpy(one.data(), d_gr_input,
                                    kHidden * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy GR input");
            if (ok) {
                std::memcpy(gdn_input_actual.data() + t * kHidden, one.data(),
                            kHidden * sizeof(float));
                record_compare_raw(t == 0 ? "gr_input_t0" : "gr_input_t1",
                                   gdn_input_ref.data() + t * kHidden,
                                   one.data(), kHidden, 5e-5f);
            }
        }
    }
    if (!checks.empty() && !checks.back().pass) ok = false;

    std::vector<float> qkv_ref, z_ref, a_ref, b_ref;
    if (ok) {
        project_ref(qkv_weight, kQkvRows, kHidden, gdn_input_ref, kTokens,
                    qkv_ref);
        project_ref(z_weight, kZRows, kHidden, gdn_input_ref, kTokens, z_ref);
        project_ref(a_weight, kGateRows, kHidden, gdn_input_ref, kTokens, a_ref);
        project_ref(b_weight, kGateRows, kHidden, gdn_input_ref, kTokens, b_ref);
        ok = copy_to_device(d_qkv_input, gdn_input_actual, "copy GDN input") &&
             q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_qkv_weight, kQkvRows,
                                  kHidden, d_qkv_input, kTokens, d_qkv,
                                  nullptr, error, sizeof(error)) &&
             q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_z_weight, kZRows,
                                  kHidden, d_qkv_input, kTokens, d_z, nullptr,
                                  error, sizeof(error)) &&
             q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_a_weight, kGateRows,
                                  kHidden, d_qkv_input, kTokens, d_a, nullptr,
                                  error, sizeof(error)) &&
             q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_b_weight, kGateRows,
                                  kHidden, d_qkv_input, kTokens, d_b, nullptr,
                                  error, sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "projection synchronize");
        std::vector<float> actual(qkv_ref.size());
        if (ok) ok = copy_from_device(actual, d_qkv, "copy QKV");
        if (ok) record_compare("qkv_projection", qkv_ref, actual, 2e-6f);
        actual.resize(z_ref.size());
        if (ok) ok = copy_from_device(actual, d_z, "copy Z");
        if (ok) record_compare("z_projection", z_ref, actual, 2e-6f);
        actual.resize(a_ref.size());
        if (ok) ok = copy_from_device(actual, d_a, "copy A");
        if (ok) record_compare("a_projection", a_ref, actual, 2e-6f);
        actual.resize(b_ref.size());
        if (ok) ok = copy_from_device(actual, d_b, "copy B");
        if (ok) record_compare("b_projection", b_ref, actual, 2e-6f);
        if (!checks.empty() && !checks.back().pass) ok = false;
    }

    std::vector<float> decay_ref, beta_ref;
    if (ok) {
        gates_ref(a_ref, b_ref, dt, a_log, kTokens, decay_ref, beta_ref);
        gate_transform_kernel<<<(unsigned)((kTokens * kGateRows + 255u) / 256u),
                                256>>>(d_a, d_b, d_dt, d_a_log, kTokens,
                                       d_decay, d_beta);
        ok = cuda_ok(cudaGetLastError(), "gate transform launch") &&
             cuda_ok(cudaDeviceSynchronize(), "gate transform synchronize");
        std::vector<float> actual(decay_ref.size());
        if (ok) ok = copy_from_device(actual, d_decay, "copy decay");
        if (ok) record_compare("decay", decay_ref, actual, 2e-6f);
        actual.resize(beta_ref.size());
        if (ok) ok = copy_from_device(actual, d_beta, "copy beta");
        if (ok) record_compare("beta", beta_ref, actual, 2e-6f);
    }

    std::vector<float> conv_history_ref((Q38_GDN_CONV_KERNEL - 1u) * kQkvRows,
                                        0.0f);
    std::vector<float> conv_ref_output, silu_ref_output;
    if (ok) {
        conv_ref(qkv_ref, kTokens, kQkvRows, conv_kernel, Q38_GDN_CONV_KERNEL,
                 conv_history_ref, conv_ref_output);
        silu_ref_output.resize(conv_ref_output.size());
        q38_oracle_silu(conv_ref_output.data(), silu_ref_output.data(),
                        silu_ref_output.size());
        ok = cuda_ok(cudaMemset(d_conv_history, 0,
                                (Q38_GDN_CONV_KERNEL - 1u) * kQkvRows *
                                    sizeof(float)),
                     "clear conv history") &&
             q38_cuda_gdn_conv(Q38_GDN_WEIGHT_F32, d_conv_kernel, d_qkv,
                               kTokens, kQkvRows, Q38_GDN_CONV_KERNEL,
                               d_conv_history, d_conv, nullptr, error,
                               sizeof(error)) &&
             q38_cuda_silu(d_conv, d_silu, conv_ref_output.size(), nullptr,
                           error, sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "convolution synchronize");
        std::vector<float> actual(conv_ref_output.size());
        if (ok) ok = copy_from_device(actual, d_conv, "copy convolution");
        if (ok) record_compare("conv_raw", conv_ref_output, actual, 2e-6f);
        actual.resize(silu_ref_output.size());
        if (ok) ok = copy_from_device(actual, d_silu, "copy SiLU");
        if (ok) record_compare("conv_silu", silu_ref_output, actual, 2e-6f);
        std::vector<float> actual_history(conv_history_ref.size());
        if (ok)
            ok = copy_from_device(actual_history, d_conv_history,
                                  "copy conv history");
        if (ok)
            record_compare("conv_history", conv_history_ref, actual_history,
                           2e-6f);
    }

    std::vector<float> q16_ref, k16_ref, v_ref, q48_ref, k48_ref;
    if (ok) {
        split_qkv_ref(silu_ref_output, kTokens, q16_ref, k16_ref, v_ref);
        repeat_key_ref(q16_ref, kTokens, q48_ref);
        repeat_key_ref(k16_ref, kTokens, k48_ref);
        normalize_heads_ref(q48_ref, kTokens);
        normalize_heads_ref(k48_ref, kTokens);
        ok = q38_cuda_gdn_split_qkv(d_silu, kTokens, d_q16, d_k16, d_v,
                                    nullptr, error, sizeof(error)) &&
             q38_cuda_gdn_repeat_key_heads(d_q16, kTokens, d_q48, nullptr,
                                           error, sizeof(error)) &&
             q38_cuda_gdn_repeat_key_heads(d_k16, kTokens, d_k48, nullptr,
                                           error, sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "QKV layout synchronize");
        normalize_repeated_kernel<<<
            (unsigned)((kTokens * kHeads + 255u) / 256u), 256>>>(
            d_q48, kTokens, d_q);
        normalize_repeated_kernel<<<
            (unsigned)((kTokens * kHeads + 255u) / 256u), 256>>>(
            d_k48, kTokens, d_k);
        ok = ok && cuda_ok(cudaGetLastError(), "Q/K normalization launch") &&
             cuda_ok(cudaDeviceSynchronize(), "Q/K normalization synchronize");
        std::vector<float> actual(q48_ref.size());
        if (ok) ok = copy_from_device(actual, d_q, "copy normalized Q");
        if (ok) record_compare("q_normalized", q48_ref, actual, 3e-6f);
        if (ok) ok = copy_from_device(actual, d_k, "copy normalized K");
        if (ok) record_compare("k_normalized", k48_ref, actual, 3e-6f);
        actual.resize(v_ref.size());
        if (ok) ok = copy_from_device(actual, d_v, "copy V");
        if (ok) record_compare("v_split", v_ref, actual, 2e-6f);
    }

    std::vector<float> recurrent_ref(kTokens * kZRows);
    std::vector<float> state_ref(kStateElements, 0.0f);
    if (ok) {
        ok = q38_cuda_gdn_recurrence_reset(d_state, nullptr, error,
                                           sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "state reset synchronize");
        if (ok) {
            std::vector<float> reset_actual(state_ref.size());
            ok = copy_from_device(reset_actual, d_state, "copy reset state");
            if (ok)
                record_compare("state_reset", state_ref, reset_actual, 0.0f);
        }
        if (ok)
            ok = q38_cuda_gdn_recurrence(
                     d_state, kTokens, d_q, d_k, d_v, d_decay, d_beta,
                     1.0f / std::sqrt(128.0f), d_recurrent, nullptr, error,
                     sizeof(error)) &&
                 cuda_ok(cudaDeviceSynchronize(), "recurrence synchronize");
        q38_gdn_ref_reset(state_ref.data(), 1);
        if (ok)
            ok = q38_gdn_ref_run(
                state_ref.data(), 1, kTokens, q48_ref.data(), k48_ref.data(),
                v_ref.data(), decay_ref.data(), beta_ref.data(),
                1.0f / std::sqrt(128.0f), recurrent_ref.data());
        std::vector<float> actual(recurrent_ref.size());
        if (ok) ok = copy_from_device(actual, d_recurrent, "copy recurrence");
        if (ok) record_compare("recurrence_output", recurrent_ref, actual,
                               4e-5f);
        actual.resize(state_ref.size());
        if (ok) ok = copy_from_device(actual, d_state, "copy recurrence state");
        if (ok) record_compare("recurrence_state", state_ref, actual, 4e-5f);
    }

    std::vector<float> gated_ref;
    if (ok) {
        gated_norm_ref(recurrent_ref, z_ref, norm, kTokens, gated_ref);
        for (size_t t = 0; t < kTokens; t++) {
            for (size_t h = 0; h < kHeads; h++) {
                if (!q38_cuda_rms_norm(
                        d_recurrent + t * kZRows + h * kHeadDim,
                        d_norm, d_normed + t * kZRows + h * kHeadDim,
                        kHeadDim, 1e-6f, nullptr, error, sizeof(error))) {
                    ok = false;
                    break;
                }
            }
        }
        apply_gate_kernel<<<(unsigned)((kTokens * kZRows + 255u) / 256u),
                            256>>>(d_normed, d_z, kTokens, d_gated);
        ok = ok && cuda_ok(cudaGetLastError(), "output gate launch") &&
             cuda_ok(cudaDeviceSynchronize(), "output gate synchronize");
        std::vector<float> actual(gated_ref.size());
        if (ok) ok = copy_from_device(actual, d_gated, "copy gated output");
        if (ok) record_compare("gated_norm_output", gated_ref, actual, 5e-5f);
    }

    std::vector<float> block_ref, block_actual;
    if (ok) {
        project_ref(out_weight, kOutRows, kOutCols, gated_ref, kTokens,
                    block_ref);
        ok = q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_out_weight, kOutRows,
                                  kOutCols, d_gated, kTokens, d_block, nullptr,
                                  error, sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "output projection synchronize");
        block_actual.resize(block_ref.size());
        if (ok) ok = copy_from_device(block_actual, d_block, "copy block output");
        if (ok)
            record_compare("out_projection", block_ref, block_actual, 3e-5f);
    }

    if (ok) {
        std::vector<float> final_updated_ref(kTokens * kGrWidth);
        std::vector<float> final_updated_actual(kTokens * kGrWidth);
        for (size_t t = 0; t < kTokens; t++) {
            q38_gr_write(residual.data() + t * kGrWidth,
                         block_ref.data() + t * kHidden, &gr_params,
                         final_updated_ref.data() + t * kGrWidth);
            ok = cuda_ok(cudaMemcpy(d_residual, residual.data() + t * kGrWidth,
                                    kGrWidth * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "copy final residual") &&
                 cuda_ok(cudaMemcpy(d_zero_block, block_actual.data() + t * kHidden,
                                    kHidden * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "copy block output") &&
                 q38_cuda_gr_collapse(
                     d_residual, d_gamma, d_gr_down, d_gr_up, d_gr_inject,
                     d_zero_block, d_gr_input, d_gr_updated, nullptr, error,
                     sizeof(error)) &&
                 cuda_ok(cudaDeviceSynchronize(), "final GR synchronize") &&
                 cuda_ok(cudaMemcpy(final_updated_actual.data() + t * kGrWidth,
                                    d_gr_updated, kGrWidth * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy final GR");
        }
        if (ok)
            record_compare("gr_updated", final_updated_ref,
                           final_updated_actual, 5e-5f);
    }

    bool all_checks_pass = ok;
    for (const Check &check : checks) all_checks_pass = all_checks_pass && check.pass;
    if (all_checks_pass && !write_artifact(argc == 2 ? argv[1] : nullptr))
        all_checks_pass = false;
    for (float *ptr : device) cudaFree(ptr);
    if (!all_checks_pass) {
        std::fprintf(stderr, "test_m3_forward_probe: failed\n");
        return 1;
    }
    std::puts("test_m3_forward_probe: layer-0-shaped GR + GDN intermediates match scalar reference");
    return 0;
}
