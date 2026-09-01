#include "q38_gdn.h"
#include "q38_gdn_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kLayers = 3;
constexpr size_t kTokens = 2;
constexpr size_t kHidden = Q38_GDN_INPUT_DIM;
constexpr size_t kQkvChannels = Q38_GDN_QKV_CHANNELS;
constexpr size_t kValueChannels = Q38_GDN_VALUE_CHANNELS;
constexpr size_t kHistoryTokens = Q38_GDN_CONV_KERNEL - 1u;
constexpr size_t kStateElements =
    Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM;

struct Metric {
    const char *name;
    size_t elements;
    float max_abs;
    float tolerance;
    bool pass;
};

struct LayerVectors {
    std::vector<float> qkv;
    std::vector<float> conv_silu;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> decay;
    std::vector<float> beta;
    std::vector<float> recurrent;
    std::vector<float> output;
    std::vector<float> next_input;
    std::vector<float> state;
    std::vector<float> history;
};

static std::vector<Metric> metrics;

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

static float max_abs_difference(const std::vector<float> &expected,
                               const std::vector<float> &actual) {
    float maximum = 0.0f;
    for (size_t i = 0; i < expected.size(); i++)
        maximum = std::fmax(maximum, std::fabs(expected[i] - actual[i]));
    return maximum;
}

static bool record_compare(const char *name, const std::vector<float> &expected,
                           const std::vector<float> &actual, float tolerance) {
    if (expected.size() != actual.size()) {
        std::fprintf(stderr, "%s size mismatch: expected=%zu actual=%zu\n", name,
                     expected.size(), actual.size());
        metrics.push_back({name, actual.size(), INFINITY, tolerance, false});
        return false;
    }
    const float maximum = max_abs_difference(expected, actual);
    const bool pass = maximum <= tolerance;
    metrics.push_back(
        {name, expected.size(), maximum, tolerance, pass});
    if (!pass)
        std::fprintf(stderr, "%s mismatch: max_abs=%g tolerance=%g\n", name,
                     maximum, tolerance);
    return pass;
}

static void make_sparse_matrix(std::vector<float> &matrix, size_t rows,
                               size_t cols, float scale, size_t salt) {
    matrix.assign(rows * cols, 0.0f);
    for (size_t row = 0; row < rows; row++) {
        const size_t col = (row * 17u + salt * 13u + 5u) % cols;
        const float code =
            static_cast<float>((row * 7u + salt * 3u) % 13u) - 6.0f;
        matrix[row * cols + col] = scale * code;
    }
}

static void project_ref(const std::vector<float> &weights, size_t rows,
                        size_t cols, const std::vector<float> &input,
                        size_t tokens, std::vector<float> &output) {
    output.assign(tokens * rows, 0.0f);
    for (size_t token = 0; token < tokens; token++)
        for (size_t row = 0; row < rows; row++)
            for (size_t col = 0; col < cols; col++)
                output[token * rows + row] +=
                    weights[row * cols + col] * input[token * cols + col];
}

static void conv_ref(const std::vector<float> &input, size_t tokens,
                     size_t channels, const std::vector<float> &kernel,
                     size_t kernel_size, std::vector<float> &history,
                     std::vector<float> &output) {
    const size_t history_tokens = kernel_size - 1u;
    output.assign(tokens * channels, 0.0f);
    for (size_t token = 0; token < tokens; token++) {
        for (size_t channel = 0; channel < channels; channel++) {
            float sum = 0.0f;
            for (size_t tap = 0; tap < kernel_size; tap++) {
                const size_t source = history_tokens + token -
                                      (kernel_size - 1u - tap);
                const float sample =
                    source < history_tokens
                        ? history[source * channels + channel]
                        : input[(source - history_tokens) * channels + channel];
                sum += kernel[tap * channels + channel] * sample;
            }
            output[token * channels + channel] = sum;
        }
    }
    std::vector<float> combined(history.size() + input.size());
    std::memcpy(combined.data(), history.data(),
                history.size() * sizeof(float));
    std::memcpy(combined.data() + history.size(), input.data(),
                input.size() * sizeof(float));
    for (size_t tail = 0; tail < history_tokens; tail++)
        std::memcpy(history.data() + tail * channels,
                    combined.data() + (tokens + tail) * channels,
                    channels * sizeof(float));
}

static void silu_ref(std::vector<float> &values) {
    for (float &value : values) value *= 1.0f / (1.0f + std::exp(-value));
}

static void split_qkv_ref(const std::vector<float> &mixed,
                          std::vector<float> &q16, std::vector<float> &k16,
                          std::vector<float> &v) {
    q16.assign(kTokens * Q38_GDN_KEY_CHANNELS, 0.0f);
    k16.assign(kTokens * Q38_GDN_KEY_CHANNELS, 0.0f);
    v.assign(kTokens * kValueChannels, 0.0f);
    for (size_t token = 0; token < kTokens; token++) {
        const float *source = mixed.data() + token * kQkvChannels;
        std::memcpy(q16.data() + token * Q38_GDN_KEY_CHANNELS, source,
                    Q38_GDN_KEY_CHANNELS * sizeof(float));
        std::memcpy(k16.data() + token * Q38_GDN_KEY_CHANNELS,
                    source + Q38_GDN_KEY_CHANNELS,
                    Q38_GDN_KEY_CHANNELS * sizeof(float));
        std::memcpy(v.data() + token * kValueChannels,
                    source + 2u * Q38_GDN_KEY_CHANNELS,
                    kValueChannels * sizeof(float));
    }
}

static void make_gates(size_t layer, std::vector<float> &decay,
                       std::vector<float> &beta) {
    decay.resize(kTokens * Q38_GDN_VALUE_HEADS);
    beta.resize(kTokens * Q38_GDN_VALUE_HEADS);
    for (size_t token = 0; token < kTokens; token++) {
        for (size_t head = 0; head < Q38_GDN_VALUE_HEADS; head++) {
            decay[token * Q38_GDN_VALUE_HEADS + head] =
                0.76f + 0.01f * static_cast<float>(
                                  (layer * 5u + token + head) % 9u);
            beta[token * Q38_GDN_VALUE_HEADS + head] =
                0.22f + 0.03f * static_cast<float>(
                                  (layer * 3u + token + head * 2u) % 8u);
        }
    }
}

static bool write_artifact(const char *path, bool all_pass) {
    if (!path) return true;
    FILE *file = std::fopen(path, "w");
    if (!file) {
        std::perror(path);
        return false;
    }
    std::fprintf(
        file,
        "{\n"
        "  \"gate\": \"M3-C10\",\n"
        "  \"status\": \"%s\",\n"
        "  \"layers\": 3,\n"
        "  \"tokens_per_layer\": 2,\n"
        "  \"fixture\": {\"kind\": \"deterministic_sparse_f32\", "
        "\"model_weights_loaded\": false},\n"
        "  \"scope\": \"three consecutive reference/simple GDN layers; "
        "not full model correctness\",\n"
        "  \"logical_layout\": {\"input\": [\"token\", 2560], "
        "\"qkv\": [\"token\", 10240], \"q_k_v\": [\"token\", 48, 128], "
        "\"state\": [1, 48, 128, 128], \"conv_history\": [1, 3, 10240], "
        "\"layer_output\": [\"token\", 2560]},\n"
        "  \"state_regions\": \"independent contiguous FP32 region per "
        "layer\",\n"
        "  \"conv_history_regions\": \"independent [3,10240] region per "
        "layer\",\n"
        "  \"frozen_order\": [\"projection\", \"causal conv4\", \"SiLU\", "
        "\"QKV split\", \"16-to-48 repeat\", \"FP32 recurrence\", "
        "\"out projection\", \"residual to next layer\"],\n"
        "  \"no_gb10_optimization\": true,\n"
        "  \"checks\": {\n",
        all_pass ? "pass" : "fail");
    for (size_t i = 0; i < metrics.size(); i++) {
        const Metric &metric = metrics[i];
        std::fprintf(file,
                     "    \"%s\": {\"elements\": %zu, \"max_abs\": %.9g, "
                     "\"tolerance\": %.9g, \"pass\": %s}%s\n",
                     metric.name, metric.elements, metric.max_abs,
                     metric.tolerance, metric.pass ? "true" : "false",
                     i + 1 == metrics.size() ? "" : ",");
    }
    std::fprintf(file, "  }\n}\n");
    return std::fclose(file) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    int devices = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&devices);
    if (device_status != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "M3-C10: CUDA device unavailable: %s\n",
                     cudaGetErrorString(device_status));
        return 2;
    }

    std::vector<float> qkv_weight, out_weight, conv_kernel;
    make_sparse_matrix(qkv_weight, kQkvChannels, kHidden, 0.012f, 31);
    make_sparse_matrix(out_weight, kHidden, kValueChannels, 0.018f, 47);
    conv_kernel.resize(Q38_GDN_CONV_KERNEL * kQkvChannels);
    for (size_t tap = 0; tap < Q38_GDN_CONV_KERNEL; tap++)
        for (size_t channel = 0; channel < kQkvChannels; channel++)
            conv_kernel[tap * kQkvChannels + channel] =
                0.08f * (static_cast<float>((tap * 11u + channel) % 7u) -
                          3.0f);

    std::vector<float> initial_input(kTokens * kHidden);
    for (size_t token = 0; token < kTokens; token++)
        for (size_t channel = 0; channel < kHidden; channel++)
            initial_input[token * kHidden + channel] =
                (static_cast<float>((channel * 19u + token * 23u) % 41u) -
                 20.0f) /
                37.0f;

    std::vector<LayerVectors> reference(kLayers);
    std::vector<float> current_input = initial_input;
    std::vector<std::vector<float>> reference_history(
        kLayers, std::vector<float>(kHistoryTokens * kQkvChannels, 0.0f));
    for (size_t layer = 0; layer < kLayers; layer++) {
        LayerVectors &result = reference[layer];
        project_ref(qkv_weight, kQkvChannels, kHidden, current_input, kTokens,
                    result.qkv);
        conv_ref(result.qkv, kTokens, kQkvChannels, conv_kernel,
                 Q38_GDN_CONV_KERNEL, reference_history[layer],
                 result.conv_silu);
        silu_ref(result.conv_silu);

        std::vector<float> q16, k16;
        split_qkv_ref(result.conv_silu, q16, k16, result.v);
        result.q.assign(kTokens * kValueChannels, 0.0f);
        result.k.assign(kTokens * kValueChannels, 0.0f);
        for (size_t token = 0; token < kTokens; token++) {
            q38_gdn_ref_repeat_key_heads(
                q16.data() + token * Q38_GDN_KEY_CHANNELS,
                result.q.data() + token * kValueChannels);
            q38_gdn_ref_repeat_key_heads(
                k16.data() + token * Q38_GDN_KEY_CHANNELS,
                result.k.data() + token * kValueChannels);
        }
        make_gates(layer, result.decay, result.beta);
        result.state.assign(kStateElements, 0.0f);
        result.recurrent.assign(kTokens * kValueChannels, 0.0f);
        if (!q38_gdn_ref_run(
                result.state.data(), 1, kTokens, result.q.data(), result.k.data(),
                result.v.data(), result.decay.data(), result.beta.data(), 0.75f,
                result.recurrent.data())) {
            std::fprintf(stderr, "M3-C10: scalar reference recurrence failed\n");
            return 1;
        }
        project_ref(out_weight, kHidden, kValueChannels, result.recurrent,
                    kTokens, result.output);
        result.next_input.resize(current_input.size());
        for (size_t i = 0; i < current_input.size(); i++)
            result.next_input[i] = current_input[i] + result.output[i];
        result.history = reference_history[layer];
        current_input = result.next_input;
    }

    std::vector<float *> allocations;
    auto alloc = [&](size_t elements, const char *what) -> float * {
        float *pointer = nullptr;
        if (!cuda_ok(cudaMalloc(&pointer, elements * sizeof(float)), what))
            return nullptr;
        allocations.push_back(pointer);
        return pointer;
    };

    float *d_qkv_weight = alloc(qkv_weight.size(), "alloc QKV weight");
    float *d_out_weight = alloc(out_weight.size(), "alloc output weight");
    float *d_conv_kernel = alloc(conv_kernel.size(), "alloc convolution kernel");
    float *d_input = alloc(kTokens * kHidden, "alloc layer input");
    float *d_qkv = alloc(kTokens * kQkvChannels, "alloc projected QKV");
    float *d_conv = alloc(kTokens * kQkvChannels, "alloc convolution output");
    float *d_q16 = alloc(kTokens * Q38_GDN_KEY_CHANNELS, "alloc Q16");
    float *d_k16 = alloc(kTokens * Q38_GDN_KEY_CHANNELS, "alloc K16");
    float *d_v = alloc(kTokens * kValueChannels, "alloc V");
    float *d_q = alloc(kTokens * kValueChannels, "alloc repeated Q");
    float *d_k = alloc(kTokens * kValueChannels, "alloc repeated K");
    float *d_decay =
        alloc(kTokens * Q38_GDN_VALUE_HEADS, "alloc decay gates");
    float *d_beta = alloc(kTokens * Q38_GDN_VALUE_HEADS, "alloc beta gates");
    float *d_recurrent = alloc(kTokens * kValueChannels, "alloc recurrence output");
    float *d_output = alloc(kTokens * kHidden, "alloc layer output");
    std::vector<float *> d_states(kLayers, nullptr);
    std::vector<float *> d_histories(kLayers, nullptr);
    for (size_t layer = 0; layer < kLayers; layer++) {
        d_states[layer] = alloc(kStateElements, "alloc layer state");
        d_histories[layer] =
            alloc(kHistoryTokens * kQkvChannels, "alloc layer history");
    }

    bool ok = d_output != nullptr;
    char error[256] = {};
    if (ok)
        ok = cuda_ok(cudaMemcpy(d_qkv_weight, qkv_weight.data(),
                                qkv_weight.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy QKV weight") &&
             cuda_ok(cudaMemcpy(d_out_weight, out_weight.data(),
                                out_weight.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy output weight") &&
             cuda_ok(cudaMemcpy(d_conv_kernel, conv_kernel.data(),
                                conv_kernel.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy convolution kernel");

    auto reset_regions = [&]() -> bool {
        bool reset_ok = true;
        for (size_t layer = 0; layer < kLayers; layer++) {
            reset_ok =
                q38_cuda_gdn_recurrence_reset(d_states[layer], nullptr, error,
                                              sizeof(error)) &&
                cuda_ok(cudaMemset(d_histories[layer], 0,
                                   kHistoryTokens * kQkvChannels *
                                       sizeof(float)),
                        "clear convolution history") &&
                reset_ok;
            if (!reset_ok) {
                std::fprintf(stderr, "M3-C10 reset layer %zu failed: %s\n", layer,
                             error);
                return false;
            }
        }
        return cuda_ok(cudaDeviceSynchronize(), "reset synchronize");
    };

    if (ok) ok = reset_regions();
    std::vector<float> current_actual = initial_input;
    std::vector<std::vector<float>> first_recurrent(kLayers);
    std::vector<std::vector<float>> first_states(kLayers);
    std::vector<std::vector<float>> first_histories(kLayers);
    std::vector<std::vector<float>> first_outputs(kLayers);
    for (size_t layer = 0; ok && layer < kLayers; layer++) {
        const LayerVectors &expected = reference[layer];
        ok = cuda_ok(cudaMemcpy(d_input, current_actual.data(),
                                current_actual.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy layer input") &&
             q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_qkv_weight,
                                  kQkvChannels, kHidden, d_input, kTokens,
                                  d_qkv, nullptr, error, sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "projection synchronize");
        std::vector<float> actual(expected.qkv.size());
        if (ok) ok = cuda_ok(cudaMemcpy(actual.data(), d_qkv,
                                        actual.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "copy projected QKV");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.qkv_projection"
                                             : (layer == 1) ? "layer1.qkv_projection"
                                                            : "layer2.qkv_projection",
                               expected.qkv, actual, 2e-5f) &&
                 q38_cuda_gdn_conv_silu(
                     Q38_GDN_WEIGHT_F32, d_conv_kernel, d_qkv, kTokens,
                     kQkvChannels, Q38_GDN_CONV_KERNEL, d_histories[layer],
                     d_conv, nullptr, error, sizeof(error)) &&
                 cuda_ok(cudaDeviceSynchronize(), "convolution synchronize");
        actual.resize(expected.conv_silu.size());
        if (ok) ok = cuda_ok(cudaMemcpy(actual.data(), d_conv,
                                        actual.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "copy convolution output");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.conv_silu"
                                             : (layer == 1) ? "layer1.conv_silu"
                                                            : "layer2.conv_silu",
                               expected.conv_silu, actual, 2e-5f) &&
                 q38_cuda_gdn_split_qkv(d_conv, kTokens, d_q16, d_k16, d_v,
                                        nullptr, error, sizeof(error)) &&
                 q38_cuda_gdn_repeat_key_heads(d_q16, kTokens, d_q, nullptr,
                                               error, sizeof(error)) &&
                 q38_cuda_gdn_repeat_key_heads(d_k16, kTokens, d_k, nullptr,
                                               error, sizeof(error)) &&
                 cuda_ok(cudaDeviceSynchronize(), "QKV layout synchronize");
        actual.resize(expected.q.size());
        if (ok) ok = cuda_ok(cudaMemcpy(actual.data(), d_q,
                                        actual.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "copy repeated Q");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.q_repeat"
                                             : (layer == 1) ? "layer1.q_repeat"
                                                            : "layer2.q_repeat",
                               expected.q, actual, 2e-5f);
        actual.resize(expected.k.size());
        if (ok) ok = cuda_ok(cudaMemcpy(actual.data(), d_k,
                                        actual.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "copy repeated K");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.k_repeat"
                                             : (layer == 1) ? "layer1.k_repeat"
                                                            : "layer2.k_repeat",
                               expected.k, actual, 2e-5f);
        actual.resize(expected.v.size());
        if (ok) ok = cuda_ok(cudaMemcpy(actual.data(), d_v,
                                        actual.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "copy split V");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.v_split"
                                             : (layer == 1) ? "layer1.v_split"
                                                            : "layer2.v_split",
                               expected.v, actual, 2e-5f);
        if (ok)
            ok = cuda_ok(cudaMemcpy(d_decay, expected.decay.data(),
                                    expected.decay.size() * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "copy decay") &&
                 cuda_ok(cudaMemcpy(d_beta, expected.beta.data(),
                                    expected.beta.size() * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "copy beta") &&
                 q38_cuda_gdn_recurrence(
                     d_states[layer], kTokens, d_q, d_k, d_v, d_decay, d_beta,
                     0.75f, d_recurrent, nullptr, error, sizeof(error)) &&
                 cuda_ok(cudaDeviceSynchronize(), "recurrence synchronize");
        actual.resize(expected.recurrent.size());
        if (ok) ok = cuda_ok(cudaMemcpy(actual.data(), d_recurrent,
                                        actual.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "copy recurrence output");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.recurrent_output"
                                             : (layer == 1) ? "layer1.recurrent_output"
                                                            : "layer2.recurrent_output",
                               expected.recurrent, actual, 5e-5f);
        if (ok) first_recurrent[layer] = actual;
        std::vector<float> state_actual(expected.state.size());
        if (ok)
            ok = cuda_ok(cudaMemcpy(state_actual.data(), d_states[layer],
                                    state_actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy layer state");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.state"
                                             : (layer == 1) ? "layer1.state"
                                                            : "layer2.state",
                               expected.state, state_actual, 5e-5f);
        if (ok) first_states[layer] = state_actual;
        std::vector<float> history_actual(expected.history.size());
        if (ok)
            ok = cuda_ok(cudaMemcpy(history_actual.data(), d_histories[layer],
                                    history_actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy layer history");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.conv_history"
                                             : (layer == 1) ? "layer1.conv_history"
                                                            : "layer2.conv_history",
                               expected.history, history_actual, 2e-5f);
        if (ok) first_histories[layer] = history_actual;
        if (ok)
            ok = q38_cuda_gdn_project(
                     Q38_GDN_WEIGHT_F32, d_out_weight, kHidden, kValueChannels,
                     d_recurrent, kTokens, d_output, nullptr, error,
                     sizeof(error)) &&
                 cuda_ok(cudaDeviceSynchronize(), "output projection synchronize");
        actual.resize(expected.output.size());
        if (ok) ok = cuda_ok(cudaMemcpy(actual.data(), d_output,
                                        actual.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "copy layer output");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.out_projection"
                                             : (layer == 1) ? "layer1.out_projection"
                                                            : "layer2.out_projection",
                               expected.output, actual, 5e-5f);
        if (ok) first_outputs[layer] = actual;
        if (ok) {
            for (size_t i = 0; i < current_actual.size(); i++)
                current_actual[i] = current_actual[i] + actual[i];
            ok = record_compare((layer == 0) ? "layer0.next_input"
                                             : (layer == 1) ? "layer1.next_input"
                                                            : "layer2.next_input",
                               expected.next_input, current_actual, 5e-5f);
        }
    }

    if (ok) {
        ok = reset_regions();
        current_actual = initial_input;
    }
    for (size_t layer = 0; ok && layer < kLayers; layer++) {
        const LayerVectors &expected = reference[layer];
        ok = cuda_ok(cudaMemcpy(d_input, current_actual.data(),
                                current_actual.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy reset layer input") &&
             q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_qkv_weight,
                                  kQkvChannels, kHidden, d_input, kTokens,
                                  d_qkv, nullptr, error, sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "reset projection synchronize") &&
             q38_cuda_gdn_conv_silu(
                 Q38_GDN_WEIGHT_F32, d_conv_kernel, d_qkv, kTokens, kQkvChannels,
                 Q38_GDN_CONV_KERNEL, d_histories[layer], d_conv, nullptr,
                 error, sizeof(error)) &&
             q38_cuda_gdn_split_qkv(d_conv, kTokens, d_q16, d_k16, d_v, nullptr,
                                    error, sizeof(error)) &&
             q38_cuda_gdn_repeat_key_heads(d_q16, kTokens, d_q, nullptr, error,
                                           sizeof(error)) &&
             q38_cuda_gdn_repeat_key_heads(d_k16, kTokens, d_k, nullptr, error,
                                           sizeof(error)) &&
             cuda_ok(cudaMemcpy(d_decay, expected.decay.data(),
                                expected.decay.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy reset decay") &&
             cuda_ok(cudaMemcpy(d_beta, expected.beta.data(),
                                expected.beta.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy reset beta") &&
             q38_cuda_gdn_recurrence(
                 d_states[layer], kTokens, d_q, d_k, d_v, d_decay, d_beta, 0.75f,
                 d_recurrent, nullptr, error, sizeof(error)) &&
             q38_cuda_gdn_project(Q38_GDN_WEIGHT_F32, d_out_weight, kHidden,
                                  kValueChannels, d_recurrent, kTokens, d_output,
                                  nullptr, error, sizeof(error)) &&
             cuda_ok(cudaDeviceSynchronize(), "reset layer synchronize");
        std::vector<float> actual(expected.recurrent.size());
        if (ok)
            ok = cuda_ok(cudaMemcpy(actual.data(), d_recurrent,
                                    actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy reset recurrence");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.reset_recurrent_output"
                                             : (layer == 1)
                                                   ? "layer1.reset_recurrent_output"
                                                   : "layer2.reset_recurrent_output",
                               first_recurrent[layer], actual, 0.0f);
        std::vector<float> state_actual(expected.state.size());
        if (ok)
            ok = cuda_ok(cudaMemcpy(state_actual.data(), d_states[layer],
                                    state_actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy reset state");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.reset_state"
                                             : (layer == 1) ? "layer1.reset_state"
                                                            : "layer2.reset_state",
                               first_states[layer], state_actual, 0.0f);
        std::vector<float> history_actual(expected.history.size());
        if (ok)
            ok = cuda_ok(cudaMemcpy(history_actual.data(), d_histories[layer],
                                    history_actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy reset history");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.reset_conv_history"
                                             : (layer == 1)
                                                   ? "layer1.reset_conv_history"
                                                   : "layer2.reset_conv_history",
                               first_histories[layer], history_actual, 0.0f);
        std::vector<float> output_actual(expected.output.size());
        if (ok)
            ok = cuda_ok(cudaMemcpy(output_actual.data(), d_output,
                                    output_actual.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy reset output");
        if (ok)
            ok = record_compare((layer == 0) ? "layer0.reset_out_projection"
                                             : (layer == 1)
                                                   ? "layer1.reset_out_projection"
                                                   : "layer2.reset_out_projection",
                               first_outputs[layer], output_actual, 0.0f);
        if (ok)
            for (size_t i = 0; i < current_actual.size(); i++)
                current_actual[i] += output_actual[i];
    }

    bool all_pass = ok;
    for (const Metric &metric : metrics) all_pass = all_pass && metric.pass;
    if (all_pass && !write_artifact(argc == 2 ? argv[1] : nullptr, all_pass))
        all_pass = false;
    for (float *pointer : allocations) cudaFree(pointer);
    if (!all_pass) {
        std::fprintf(stderr, "test_m3_multigdn: failed\n");
        return 1;
    }
    std::puts("test_m3_multigdn: three consecutive GDN layers match scalar references and reset deterministically");
    return 0;
}
