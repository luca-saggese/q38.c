#include "q38_gdn.h"
#include "q38_cuda_primitives.h"
#include "q38_state.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kTokens = 2;
constexpr size_t kInputDim = Q38_GDN_INPUT_DIM;
constexpr size_t kQkvRows = Q38_GDN_QKV_CHANNELS;
constexpr size_t kValueChannels = Q38_GDN_VALUE_CHANNELS;
constexpr size_t kStateElements =
    Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM;
constexpr size_t kSamples = 3;

struct StageTiming {
    const char *name;
    float samples[kSamples];
    size_t launch_count;
};

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

template <typename Function>
static bool time_stage(const char *name, size_t launch_count, Function function,
                       cudaStream_t stream, char *error, size_t error_len,
                       StageTiming *timing) {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "cudaEventCreate start") ||
        !cuda_ok(cudaEventCreate(&end), "cudaEventCreate end")) {
        if (start) cudaEventDestroy(start);
        if (end) cudaEventDestroy(end);
        return false;
    }
    timing->name = name;
    timing->launch_count = launch_count;
    bool ok = true;
    for (size_t sample = 0; sample < kSamples; sample++) {
        ok = ok && cuda_ok(cudaEventRecord(start, stream), "record stage start");
        if (ok) ok = function();
        ok = ok && cuda_ok(cudaEventRecord(end, stream), "record stage end");
        ok = ok && cuda_ok(cudaEventSynchronize(end), "synchronize stage end");
        ok = ok && cuda_ok(cudaEventElapsedTime(&timing->samples[sample],
                                                start, end),
                           "elapsed stage event");
        if (!ok) break;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    if (!ok && error && error[0])
        std::fprintf(stderr, "%s failed: %s\n", name, error);
    return ok;
}

static void make_input(std::vector<float> &input) {
    input.resize(kTokens * kInputDim);
    for (size_t token = 0; token < kTokens; token++)
        for (size_t column = 0; column < kInputDim; column++) {
            const size_t code = (token * 29u + column * 7u + 3u) % 101u;
            input[token * kInputDim + column] =
                (static_cast<float>(code) - 50.0f) * 0.0005f;
        }
}

static void make_sparse_weight(std::vector<float> &weight, size_t rows,
                               size_t cols, size_t salt) {
    weight.assign(rows * cols, 0.0f);
    for (size_t row = 0; row < rows; row++) {
        const size_t column = (row * 17u + salt * 13u + 5u) % cols;
        weight[row * cols + column] =
            (static_cast<float>((row + salt) % 11u) - 5.0f) * 0.001f;
    }
}

static void make_kernel(std::vector<float> &kernel) {
    kernel.resize(Q38_GDN_CONV_KERNEL * kQkvRows);
    for (size_t tap = 0; tap < Q38_GDN_CONV_KERNEL; tap++)
        for (size_t channel = 0; channel < kQkvRows; channel++) {
            const size_t code = (tap * 19u + channel * 7u + 3u) % 9u;
            kernel[tap * kQkvRows + channel] =
                (static_cast<float>(code) - 4.0f) * 0.025f;
        }
}

static void make_gates(std::vector<float> &decay, std::vector<float> &beta) {
    decay.resize(kTokens * Q38_GDN_VALUE_HEADS);
    beta.resize(kTokens * Q38_GDN_VALUE_HEADS);
    for (size_t token = 0; token < kTokens; token++)
        for (size_t head = 0; head < Q38_GDN_VALUE_HEADS; head++) {
            decay[token * Q38_GDN_VALUE_HEADS + head] =
                0.82f + 0.01f * static_cast<float>((token + head) % 7u);
            beta[token * Q38_GDN_VALUE_HEADS + head] =
                0.35f + 0.025f * static_cast<float>((head * 3u + token) % 6u);
        }
}

static bool all_finite(const std::vector<float> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

static float average(const StageTiming &timing) {
    float total = 0.0f;
    for (float sample : timing.samples) total += sample;
    return total / static_cast<float>(kSamples);
}

static float minimum(const StageTiming &timing) {
    float result = timing.samples[0];
    for (size_t i = 1; i < kSamples; i++)
        result = std::min(result, timing.samples[i]);
    return result;
}

static float maximum(const StageTiming &timing) {
    float result = timing.samples[0];
    for (size_t i = 1; i < kSamples; i++)
        result = std::max(result, timing.samples[i]);
    return result;
}

static bool write_artifact(const char *path, const cudaDeviceProp &properties,
                           const q38_session_state &layout,
                           const StageTiming *timings) {
    FILE *file = std::fopen(path, "w");
    if (!file) {
        std::perror(path);
        return false;
    }
    const q38_state_memory &memory = layout.memory;
    std::fprintf(file,
                 "{\n"
                 "  \"gate\": \"M3-C12\",\n"
                 "  \"status\": \"pass\",\n"
                 "  \"tool_scope\": {\n"
                 "    \"cuda_events\": true,\n"
                 "    \"nsight_systems\": \"not_collected\",\n"
                 "    \"launch_accounting\": \"static call-site accounting for "
                 "the reference/simple path\",\n"
                 "    \"optimization\": \"none; no fusion or GB10 tiling\"\n"
                 "  },\n"
                 "  \"device\": {\"name\": \"%s\", \"compute_capability\": "
                 "\"%d.%d\", \"stream\": \"single non-default CUDA stream\"},\n",
                 properties.name, properties.major, properties.minor);
    std::fprintf(file,
                 "  \"workload\": {\"tokens\": %zu, \"weight_type\": \"F32\", "
                 "\"projection\": {\"input\": [\"token\", %u], \"qkv\": "
                 "[\"token\", %u]}, \"conv\": {\"input\": [\"token\", %u], "
                 "\"kernel\": [%u, %u], \"history\": [\"token\", %u, %u]}, "
                 "\"recurrence\": {\"q\": [\"token\", %u, %u], \"k\": "
                 "[\"token\", %u, %u], \"v\": [\"token\", %u, %u], \"state\": "
                 "[\"gdn_slot\", 1, %u, %u, %u]}, \"output\": {\"input\": [\"token\", %u], "
                 "\"output\": [\"token\", %u]}},\n"
                 "  \"layouts\": {\"logical\": \"projection matrices are "
                 "output-by-input; activations are token-major; state is "
                 "[gdn_slot,sequence,value_head,row,column]\", \"physical\": "
                 "\"contiguous F32 row-major regions; no transposition, "
                 "packing, fusion, or tiling\"},\n",
                 kTokens, static_cast<unsigned>(kInputDim),
                 static_cast<unsigned>(kQkvRows),
                 static_cast<unsigned>(kQkvRows),
                 static_cast<unsigned>(Q38_GDN_CONV_KERNEL),
                 static_cast<unsigned>(kQkvRows),
                 static_cast<unsigned>(Q38_GDN_CONV_KERNEL - 1u),
                 static_cast<unsigned>(kQkvRows),
                 static_cast<unsigned>(Q38_GDN_VALUE_HEADS),
                 static_cast<unsigned>(Q38_GDN_HEAD_DIM),
                 static_cast<unsigned>(Q38_GDN_VALUE_HEADS),
                 static_cast<unsigned>(Q38_GDN_HEAD_DIM),
                 static_cast<unsigned>(Q38_GDN_VALUE_HEADS),
                 static_cast<unsigned>(Q38_GDN_HEAD_DIM),
                 static_cast<unsigned>(Q38_GDN_VALUE_HEADS),
                 static_cast<unsigned>(Q38_GDN_HEAD_DIM),
                 static_cast<unsigned>(Q38_GDN_HEAD_DIM),
                 static_cast<unsigned>(kValueChannels),
                 static_cast<unsigned>(kInputDim));
    std::fprintf(
        file,
        "  \"state_bytes\": {\"source\": \"q38_state\", "
        "\"recurrent_descriptor\": {\"sequence_count\": %u, "
        "\"value_heads\": %u, \"head_dim\": %u, \"slot_count\": %u, "
        "\"elements_per_slot\": %llu, \"bytes_per_slot\": %llu, "
        "\"elements\": %llu, \"bytes\": %llu}, \"conv_history_descriptor\": "
        "{\"sequence_count\": %u, \"channels\": %u, \"kernel\": %u, "
        "\"history_tokens\": %u, \"slot_count\": %u, "
        "\"elements_per_slot\": %llu, \"bytes_per_slot\": %llu, "
        "\"elements\": %llu, \"bytes\": %llu}, "
        "\"gr_workspace_bytes\": %llu, \"persistent_recurrent_state_bytes\": "
        "%llu, \"conv_history_bytes\": %llu, \"workspace_bytes\": %llu, "
        "\"persistent_bytes\": %llu, \"activation_bytes\": %llu, "
        "\"allocation_bytes\": %llu},\n"
        "  \"stages\": [\n",
        layout.recurrent.sequence_count, layout.recurrent.value_heads,
        layout.recurrent.head_dim,
        layout.recurrent.slot_count,
        static_cast<unsigned long long>(layout.recurrent.elements_per_slot),
        static_cast<unsigned long long>(layout.recurrent.bytes_per_slot),
        static_cast<unsigned long long>(layout.recurrent.elements),
        static_cast<unsigned long long>(layout.recurrent.bytes),
        layout.conv_history.sequence_count, layout.conv_history.channels,
        layout.conv_history.kernel, layout.conv_history.history_tokens,
        layout.conv_history.slot_count,
        static_cast<unsigned long long>(layout.conv_history.elements_per_slot),
        static_cast<unsigned long long>(layout.conv_history.bytes_per_slot),
        static_cast<unsigned long long>(layout.conv_history.elements),
        static_cast<unsigned long long>(layout.conv_history.bytes),
        static_cast<unsigned long long>(memory.gr_workspace_bytes),
        static_cast<unsigned long long>(memory.persistent_recurrent_state_bytes),
        static_cast<unsigned long long>(memory.conv_history_bytes),
        static_cast<unsigned long long>(memory.workspace_bytes),
        static_cast<unsigned long long>(memory.persistent_bytes),
        static_cast<unsigned long long>(memory.activation_bytes),
        static_cast<unsigned long long>(memory.allocation_bytes));

    for (size_t i = 0; i < 4; i++) {
        const StageTiming &timing = timings[i];
        std::fprintf(
            file,
            "    {\"name\": \"%s\", \"kernel_launches\": %zu, "
            "\"cuda_event_ms\": {\"samples\": [%g, %g, %g], \"min\": %g, "
            "\"mean\": %g, \"max\": %g}, \"scope\": \"single stage event "
            "around existing API calls\"}%s\n",
            timing.name, timing.launch_count, timing.samples[0],
            timing.samples[1], timing.samples[2], minimum(timing),
            average(timing), maximum(timing), i == 3 ? "" : ",");
    }
    std::fprintf(file, "  ],\n"
                        "  \"kernel_launches\": [\n"
                        "    {\"stage\": \"projection\", \"name\": "
                        "\"gdn_dense_project_kernel\", \"count\": 1},\n"
                        "    {\"stage\": \"conv\", \"name\": "
                        "\"gdn_conv_kernel\", \"count\": 1},\n"
                        "    {\"stage\": \"conv\", \"name\": "
                        "\"gdn_conv_history_tail_kernel\", \"count\": 1},\n"
                        "    {\"stage\": \"conv\", \"name\": \"silu_kernel\", "
                        "\"count\": 1},\n"
                        "    {\"stage\": \"recurrence\", \"name\": "
                        "\"gdn_split_qkv_kernel\", \"count\": 1},\n"
                        "    {\"stage\": \"recurrence\", \"name\": "
                        "\"gdn_repeat_key_kernel\", \"count\": 2},\n"
                        "    {\"stage\": \"recurrence\", \"name\": "
                        "\"gdn_recurrence_kernel\", \"count\": 1},\n"
                        "    {\"stage\": \"output\", \"name\": "
                        "\"gdn_dense_project_kernel\", \"count\": 1}\n"
                        "  ],\n"
                        "  \"checks\": [\"CUDA launch/runtime errors\", "
                        "\"finite output\", \"q38_state descriptor validation\", "
                        "\"deterministic fixture\"]\n"
                        "}\n");
    return std::fclose(file) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    const char *artifact = argc == 2 ? argv[1] : "artifacts/m3/cuda_profile_baseline.json";
    int devices = 0;
    if (!cuda_ok(cudaGetDeviceCount(&devices), "cudaGetDeviceCount") ||
        devices == 0) {
        std::fprintf(stderr, "M3-C12: CUDA device unavailable\n");
        return 2;
    }
    cudaDeviceProp properties;
    if (!cuda_ok(cudaGetDeviceProperties(&properties, 0),
                 "cudaGetDeviceProperties"))
        return 1;

    q38_session_state layout;
    char error[256] = {};
    if (!q38_session_state_init(&layout, 0, error, sizeof(error)) ||
        !q38_session_state_validate(&layout, error, sizeof(error))) {
        std::fprintf(stderr, "M3-C12: q38_state layout failed: %s\n", error);
        return 1;
    }

    std::vector<float> input;
    std::vector<float> qkv_weight;
    std::vector<float> output_weight;
    std::vector<float> kernel;
    std::vector<float> decay;
    std::vector<float> beta;
    make_input(input);
    make_sparse_weight(qkv_weight, kQkvRows, kInputDim, 1);
    make_sparse_weight(output_weight, kInputDim, kValueChannels, 7);
    make_kernel(kernel);
    make_gates(decay, beta);

    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;
    std::vector<float *> device;
    auto allocate = [&](size_t elements, const char *what) -> float * {
        float *pointer = nullptr;
        if (!cuda_ok(cudaMalloc(&pointer, elements * sizeof(float)), what))
            return nullptr;
        device.push_back(pointer);
        return pointer;
    };
    auto copy = [&](float *destination, const std::vector<float> &source,
                    const char *what) {
        return cuda_ok(cudaMemcpyAsync(destination, source.data(),
                                       source.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       what);
    };

    float *d_input = allocate(input.size(), "cudaMalloc input");
    float *d_qkv_weight = allocate(qkv_weight.size(), "cudaMalloc qkv weight");
    float *d_output_weight =
        allocate(output_weight.size(), "cudaMalloc output weight");
    float *d_kernel = allocate(kernel.size(), "cudaMalloc conv kernel");
    float *d_decay = allocate(decay.size(), "cudaMalloc decay");
    float *d_beta = allocate(beta.size(), "cudaMalloc beta");
    float *d_qkv = allocate(kTokens * kQkvRows, "cudaMalloc qkv");
    float *d_conv = allocate(kTokens * kQkvRows, "cudaMalloc conv");
    float *d_silu = allocate(kTokens * kQkvRows, "cudaMalloc silu");
    float *d_q16 = allocate(kTokens * Q38_GDN_KEY_CHANNELS, "cudaMalloc q16");
    float *d_k16 = allocate(kTokens * Q38_GDN_KEY_CHANNELS, "cudaMalloc k16");
    float *d_v = allocate(kTokens * kValueChannels, "cudaMalloc v");
    float *d_q = allocate(kTokens * kValueChannels, "cudaMalloc q");
    float *d_k = allocate(kTokens * kValueChannels, "cudaMalloc k");
    float *d_state = allocate(layout.recurrent.elements, "cudaMalloc state");
    float *d_conv_history =
        allocate(layout.conv_history.elements, "cudaMalloc conv history");
    float *d_gr_workspace =
        allocate(layout.gr_workspace.elements, "cudaMalloc GR workspace");
    float *d_recurrent =
        allocate(kTokens * kValueChannels, "cudaMalloc recurrence output");
    float *d_output = allocate(kTokens * kInputDim, "cudaMalloc output");
    bool ok = d_output != nullptr;

    if (ok)
        ok = copy(d_input, input, "copy input") &&
             copy(d_qkv_weight, qkv_weight, "copy qkv weight") &&
             copy(d_output_weight, output_weight, "copy output weight") &&
             copy(d_kernel, kernel, "copy conv kernel") &&
             copy(d_decay, decay, "copy decay") && copy(d_beta, beta, "copy beta") &&
             cuda_ok(cudaStreamSynchronize(stream), "upload synchronize");

    StageTiming timings[4] = {};
    if (ok) {
        for (size_t sample = 0; sample < kSamples && ok; sample++) {
            ok = cuda_ok(cudaMemsetAsync(d_state, 0, layout.recurrent.bytes,
                                         stream),
                         "reset all GDN state slots") &&
                 cuda_ok(cudaMemsetAsync(
                             d_conv_history, 0, layout.conv_history.bytes,
                             stream),
                         "reset conv history") &&
                 cuda_ok(cudaMemsetAsync(d_gr_workspace, 0,
                                         layout.gr_workspace.bytes, stream),
                         "reset GR workspace") &&
                 cuda_ok(cudaStreamSynchronize(stream), "reset synchronize");
            if (!ok) break;

            if (sample == 0) {
                ok = time_stage(
                    "projection", 1,
                    [&]() {
                        return q38_cuda_gdn_project(
                            Q38_GDN_WEIGHT_F32, d_qkv_weight, kQkvRows,
                            kInputDim, d_input, kTokens, d_qkv, stream, error,
                            sizeof(error));
                    },
                    stream, error, sizeof(error), &timings[0]);
                if (!ok) break;
                ok = time_stage(
                    "conv", 3,
                    [&]() {
                        return q38_cuda_gdn_conv(
                                   Q38_GDN_WEIGHT_F32, d_kernel, d_qkv, kTokens,
                                   kQkvRows, Q38_GDN_CONV_KERNEL,
                                   d_conv_history, d_conv, stream, error,
                                   sizeof(error)) &&
                               q38_cuda_silu(d_conv, d_silu,
                                             kTokens * kQkvRows, stream, error,
                                             sizeof(error));
                    },
                    stream, error, sizeof(error), &timings[1]);
                if (!ok) break;
                ok = time_stage(
                    "recurrence", 4,
                    [&]() {
                        return q38_cuda_gdn_split_qkv(
                                   d_silu, kTokens, d_q16, d_k16, d_v, stream,
                                   error, sizeof(error)) &&
                               q38_cuda_gdn_repeat_key_heads(
                                   d_q16, kTokens, d_q, stream, error,
                                   sizeof(error)) &&
                               q38_cuda_gdn_repeat_key_heads(
                                   d_k16, kTokens, d_k, stream, error,
                                   sizeof(error)) &&
                               q38_cuda_gdn_recurrence(
                                   d_state, kTokens, d_q, d_k, d_v, d_decay,
                                   d_beta, 1.0f / std::sqrt(128.0f),
                                   d_recurrent, stream, error, sizeof(error));
                    },
                    stream, error, sizeof(error), &timings[2]);
                if (!ok) break;
                ok = time_stage(
                    "output", 1,
                    [&]() {
                        return q38_cuda_gdn_project(
                            Q38_GDN_WEIGHT_F32, d_output_weight, kInputDim,
                            kValueChannels, d_recurrent, kTokens, d_output,
                            stream, error, sizeof(error));
                    },
                    stream, error, sizeof(error), &timings[3]);
            } else {
                /*
                 * Repeat the same sequence for a deterministic warm-cache
                 * check.  Only the first measured pass is emitted so event
                 * samples remain independent of host-side scheduling.
                 */
                ok = q38_cuda_gdn_project(
                         Q38_GDN_WEIGHT_F32, d_qkv_weight, kQkvRows, kInputDim,
                         d_input, kTokens, d_qkv, stream, error, sizeof(error)) &&
                     q38_cuda_gdn_conv(
                         Q38_GDN_WEIGHT_F32, d_kernel, d_qkv, kTokens, kQkvRows,
                         Q38_GDN_CONV_KERNEL, d_conv_history, d_conv, stream,
                         error, sizeof(error)) &&
                     q38_cuda_silu(d_conv, d_silu, kTokens * kQkvRows, stream,
                                   error, sizeof(error)) &&
                     q38_cuda_gdn_split_qkv(d_silu, kTokens, d_q16, d_k16, d_v,
                                            stream, error, sizeof(error)) &&
                     q38_cuda_gdn_repeat_key_heads(
                         d_q16, kTokens, d_q, stream, error, sizeof(error)) &&
                     q38_cuda_gdn_repeat_key_heads(
                         d_k16, kTokens, d_k, stream, error, sizeof(error)) &&
                     q38_cuda_gdn_recurrence(
                         d_state, kTokens, d_q, d_k, d_v, d_decay, d_beta,
                         1.0f / std::sqrt(128.0f), d_recurrent, stream, error,
                         sizeof(error)) &&
                     q38_cuda_gdn_project(
                         Q38_GDN_WEIGHT_F32, d_output_weight, kInputDim,
                         kValueChannels, d_recurrent, kTokens, d_output, stream,
                         error, sizeof(error)) &&
                     cuda_ok(cudaStreamSynchronize(stream), "warmup synchronize");
            }
        }
    }

    if (ok) {
        std::vector<float> output(kTokens * kInputDim);
        std::vector<float> state(kStateElements);
        ok = cuda_ok(cudaMemcpy(output.data(), d_output,
                                output.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy output") &&
             cuda_ok(cudaMemcpy(state.data(), d_state,
                                state.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy state") &&
             all_finite(output) && all_finite(state);
        if (!ok) std::fprintf(stderr, "M3-C12: non-finite output or state\n");
    }
    if (ok) ok = write_artifact(artifact, properties, layout, timings);

    for (float *pointer : device) cudaFree(pointer);
    cudaStreamDestroy(stream);
    if (!ok) return 1;
    std::printf("test_m3_cuda_profile_baseline: wrote %s\n", artifact);
    return 0;
}
