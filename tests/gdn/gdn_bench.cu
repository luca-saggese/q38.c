#include "gdn_reference.h"

#include "../../q38_cuda_primitives.h"
#include "../../q38_gdn.h"
#include "../../q38_gdn_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr size_t kWarmup = 100;
constexpr size_t kSamples = 1000;
constexpr size_t kHidden = Q38_TEST_GDN_HIDDEN;
constexpr size_t kQkv = Q38_TEST_GDN_QKV;
constexpr size_t kZ = Q38_TEST_GDN_Z;
constexpr size_t kHeads = Q38_TEST_GDN_HEADS;
constexpr size_t kDim = Q38_TEST_GDN_DIM;
constexpr size_t kStateElements = kHeads * kDim * kDim;
constexpr size_t kHistoryElements = Q38_TEST_GDN_HISTORY * kQkv;
constexpr double kOutputMaxAbsTolerance = 5.0e-3;
constexpr double kOutputMaxRelTolerance = 1.0e-2;
constexpr double kStateMaxAbsTolerance = 1.0e-5;
constexpr double kHistoryMaxAbsTolerance = 1.0e-4;

struct Weight {
    uint32_t type = 0;
    size_t rows = 0;
    size_t cols = 0;
    std::vector<unsigned char> raw;
};

struct Fixture {
    uint32_t layer = 0;
    std::vector<float> hidden;
    std::vector<float> history;
    std::vector<float> state;
    std::vector<float> expected;
    std::vector<float> next_history;
    std::vector<float> next_state;
    Weight qkv, z, a, b, conv, a_log, dt_bias, norm, out_proj;
};

struct Device {
    cudaStream_t stream = nullptr;
    std::vector<void *> weights;
    float *input = nullptr;
    float *output = nullptr;
    float *qkv = nullptr;
    float *z = nullptr;
    float *a = nullptr;
    float *b = nullptr;
    float *conv = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *decay = nullptr;
    float *beta = nullptr;
    float *recurrent = nullptr;
    float *gated = nullptr;
    float *history = nullptr;
    float *state = nullptr;
};

enum class BenchMode {
    Production,
    C1,
    C2,
    C3,
};

struct Sample {
    double total_us = 0.0;
    double qkv_us = 0.0;
    double z_us = 0.0;
    double a_us = 0.0;
    double b_us = 0.0;
    double conv_us = 0.0;
    double fused_us = 0.0;
    double gate_us = 0.0;
    double recurrence_us = 0.0;
    double post_gate_us = 0.0;
    double output_us = 0.0;
    double dispatch_us = 0.0;
    double sync_us = 0.0;
    double memcpy_us = 0.0;
    uint64_t launches = 0;
    uint64_t syncs = 0;
    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t state_reset_h2d_bytes = 0;
};

struct Compare {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
    size_t nonfinite = 0;
};

struct RunResult {
    Sample sample;
    std::vector<float> output;
    std::vector<float> next_history;
    std::vector<float> next_state;
};

static double now_us() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) * 1.0e6 +
           static_cast<double>(ts.tv_nsec) / 1.0e3;
}

static bool cuda_ok(cudaError_t status, const char *what, std::string *error) {
    if (status == cudaSuccess) return true;
    *error = std::string(what) + ": " + cudaGetErrorString(status);
    return false;
}

static bool read_file(const std::string &path, void *data, size_t bytes,
                      std::string *error) {
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) {
        *error = "missing fixture file: " + path;
        return false;
    }
    const size_t got = std::fread(data, 1, bytes, file);
    const bool closed = std::fclose(file) == 0;
    if (got != bytes || !closed) {
        *error = "short or unreadable fixture file: " + path;
        return false;
    }
    return true;
}

static bool read_text(const std::string &path, std::string *text,
                      std::string *error) {
    std::ifstream file(path);
    if (!file) {
        *error = "missing metadata: " + path;
        return false;
    }
    *text = std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    return true;
}

static bool metadata_type(const std::string &metadata, const char *name,
                          uint32_t *type) {
    const std::string needle =
        std::string("\"") + name + "\":{\"type\":";
    const size_t position = metadata.find(needle);
    if (position == std::string::npos) return false;
    char *end = nullptr;
    const unsigned long value =
        std::strtoul(metadata.c_str() + position + needle.size(), &end, 10);
    if (end == metadata.c_str() + position + needle.size() ||
        value > UINT32_MAX)
        return false;
    *type = static_cast<uint32_t>(value);
    return true;
}

static bool load_weight(const std::string &dir, const char *name, size_t rows,
                        size_t cols, uint32_t type, Weight *weight,
                        std::string *error) {
    const std::string path = dir + "/" + name;
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) {
        *error = "missing weight: " + path;
        return false;
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        *error = "failed to size weight: " + path;
        return false;
    }
    const long size = std::ftell(file);
    if (size < 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        std::fclose(file);
        *error = "failed to seek weight: " + path;
        return false;
    }
    weight->type = type;
    weight->rows = rows;
    weight->cols = cols;
    weight->raw.resize(static_cast<size_t>(size));
    const size_t got = std::fread(weight->raw.data(), 1, weight->raw.size(),
                                  file);
    const bool closed = std::fclose(file) == 0;
    if (got != weight->raw.size() || !closed) {
        *error = "short weight: " + path;
        return false;
    }
    return true;
}

static bool load_fixture(const std::string &dir, Fixture *fixture,
                         std::string *error) {
    std::string metadata;
    if (!read_text(dir + "/metadata.json", &metadata, error))
        return false;
    const size_t layer_key = metadata.find("\"layer\":");
    if (layer_key == std::string::npos) {
        *error = "GDN metadata has no layer";
        return false;
    }
    fixture->layer = static_cast<uint32_t>(
        std::strtoul(metadata.c_str() + layer_key + 8, nullptr, 10));
    const char *names[] = {
        "in_proj_qkv.bin", "in_proj_z.bin", "in_proj_a.bin", "in_proj_b.bin",
        "conv1d.bin", "A_log.bin", "dt_bias.bin", "norm.bin", "out_proj.bin",
    };
    uint32_t types[9] = {};
    for (size_t i = 0; i < 9; ++i)
        if (!metadata_type(metadata, names[i], &types[i])) {
            *error = std::string("missing tensor type in metadata: ") + names[i];
            return false;
        }
    fixture->hidden.resize(kHidden);
    fixture->history.resize(kHistoryElements);
    fixture->state.resize(kStateElements);
    fixture->expected.resize(kHidden);
    fixture->next_history.resize(kHistoryElements);
    fixture->next_state.resize(kStateElements);
    return read_file(dir + "/hidden.f32", fixture->hidden.data(),
                     fixture->hidden.size() * sizeof(float), error) &&
           read_file(dir + "/conv_history.f32", fixture->history.data(),
                     fixture->history.size() * sizeof(float), error) &&
           read_file(dir + "/recurrent_state.f32", fixture->state.data(),
                     fixture->state.size() * sizeof(float), error) &&
           read_file(dir + "/expected.f32", fixture->expected.data(),
                     fixture->expected.size() * sizeof(float), error) &&
           read_file(dir + "/next_conv_history.f32",
                     fixture->next_history.data(),
                     fixture->next_history.size() * sizeof(float), error) &&
           read_file(dir + "/next_recurrent_state.f32",
                     fixture->next_state.data(),
                     fixture->next_state.size() * sizeof(float), error) &&
           load_weight(dir, names[0], kQkv, kHidden, types[0], &fixture->qkv,
                       error) &&
           load_weight(dir, names[1], kZ, kHidden, types[1], &fixture->z,
                       error) &&
           load_weight(dir, names[2], kHeads, kHidden, types[2], &fixture->a,
                       error) &&
           load_weight(dir, names[3], kHeads, kHidden, types[3], &fixture->b,
                       error) &&
           load_weight(dir, names[4], kQkv, Q38_TEST_GDN_KERNEL, types[4],
                       &fixture->conv, error) &&
           load_weight(dir, names[5], kHeads, 1, types[5], &fixture->a_log,
                       error) &&
           load_weight(dir, names[6], kHeads, 1, types[6], &fixture->dt_bias,
                       error) &&
           load_weight(dir, names[7], kDim, 1, types[7], &fixture->norm,
                       error) &&
           load_weight(dir, names[8], kHidden, kZ, types[8], &fixture->out_proj,
                       error);
}

static float bf16_to_float(uint16_t bits) {
    uint32_t raw = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

static float host_weight_scalar(const Weight &weight, size_t index) {
    if (weight.type == 30) {
        uint16_t bits;
        std::memcpy(&bits, weight.raw.data() + index * sizeof(bits),
                    sizeof(bits));
        return bf16_to_float(bits);
    }
    if (weight.type == 0) {
        float value;
        std::memcpy(&value, weight.raw.data() + index * sizeof(value),
                    sizeof(value));
        return value;
    }
    if (weight.type == 8) {
        const unsigned char *block = weight.raw.data() +
                                     (index / 32u) * 34u;
        uint16_t bits;
        std::memcpy(&bits, block, sizeof(bits));
        return q38_half_to_float(bits) *
               static_cast<float>(
                   reinterpret_cast<const int8_t *>(block + 2)[index % 32u]);
    }
    return NAN;
}

static bool alloc_device(const Fixture &fixture, Device *device,
                         std::string *error) {
    if (!cuda_ok(cudaStreamCreate(&device->stream), "cudaStreamCreate",
                 error))
        return false;
    const Weight *weights[] = {
        &fixture.qkv, &fixture.z, &fixture.a, &fixture.b, &fixture.conv,
        &fixture.a_log, &fixture.dt_bias, &fixture.norm, &fixture.out_proj,
    };
    for (const Weight *weight : weights) {
        void *ptr = nullptr;
        if (!cuda_ok(cudaMalloc(&ptr, weight->raw.size()), "cudaMalloc weight",
                     error))
            return false;
        device->weights.push_back(ptr);
    }
    const auto alloc = [&](float **ptr, size_t elements,
                           const char *name) {
        return cuda_ok(cudaMalloc(ptr, elements * sizeof(float)), name, error);
    };
    return alloc(&device->input, std::max(kHidden, kZ),
                 "cudaMalloc input") &&
           alloc(&device->output, kQkv, "cudaMalloc output") &&
           alloc(&device->qkv, kQkv, "cudaMalloc qkv") &&
           alloc(&device->z, kZ, "cudaMalloc z") &&
           alloc(&device->a, kHeads, "cudaMalloc a") &&
           alloc(&device->b, kHeads, "cudaMalloc b") &&
           alloc(&device->conv, kQkv, "cudaMalloc conv") &&
           alloc(&device->q, kHeads * kDim, "cudaMalloc q") &&
           alloc(&device->k, kHeads * kDim, "cudaMalloc k") &&
           alloc(&device->v, kHeads * kDim, "cudaMalloc v") &&
           alloc(&device->decay, kHeads, "cudaMalloc decay") &&
           alloc(&device->beta, kHeads, "cudaMalloc beta") &&
           alloc(&device->recurrent, kHeads * kDim, "cudaMalloc recurrent") &&
           alloc(&device->gated, kZ, "cudaMalloc gated") &&
           alloc(&device->history, kHistoryElements, "cudaMalloc history") &&
           alloc(&device->state, kStateElements, "cudaMalloc state");
}

static void free_device(Device *device) {
    for (void *ptr : device->weights) cudaFree(ptr);
    cudaFree(device->input);
    cudaFree(device->output);
    cudaFree(device->qkv);
    cudaFree(device->z);
    cudaFree(device->a);
    cudaFree(device->b);
    cudaFree(device->conv);
    cudaFree(device->q);
    cudaFree(device->k);
    cudaFree(device->v);
    cudaFree(device->decay);
    cudaFree(device->beta);
    cudaFree(device->recurrent);
    cudaFree(device->gated);
    cudaFree(device->history);
    cudaFree(device->state);
    if (device->stream) cudaStreamDestroy(device->stream);
}

static bool upload_weights(const Fixture &fixture, Device *device,
                           std::string *error) {
    const Weight *weights[] = {
        &fixture.qkv, &fixture.z, &fixture.a, &fixture.b, &fixture.conv,
        &fixture.a_log, &fixture.dt_bias, &fixture.norm, &fixture.out_proj,
    };
    for (size_t i = 0; i < 9; ++i)
        if (!cuda_ok(cudaMemcpyAsync(device->weights[i], weights[i]->raw.data(),
                                     weights[i]->raw.size(),
                                     cudaMemcpyHostToDevice, device->stream),
                     "weight upload", error))
            return false;
    return cuda_ok(cudaStreamSynchronize(device->stream), "weight sync", error);
}

static uint32_t cuda_weight_type(uint32_t type) {
    return type == 10 ? Q38_QUANT_Q2_K : type;
}

static bool project(const Weight &weight, void *device_weight,
                    const float *input, size_t input_elements, float *output,
                    size_t output_elements, bool candidate, Device *device,
                    Sample *sample, double *stage_us, std::string *error) {
    const double started = now_us();
    const double h2d_started = now_us();
    if (!cuda_ok(cudaMemcpyAsync(device->input, input,
                                 input_elements * sizeof(float),
                                 cudaMemcpyHostToDevice, device->stream),
                 "projection input upload", error))
        return false;
    sample->memcpy_us += now_us() - h2d_started;
    sample->h2d_bytes += input_elements * sizeof(float);

    const double dispatch_started = now_us();
    if (candidate) {
        char cuda_error[256] = {};
        if (!q38_cuda_gdn_project(
                cuda_weight_type(weight.type), device_weight, weight.rows,
                weight.cols, device->input, 1, device->output, device->stream,
                cuda_error, sizeof(cuda_error))) {
            *error = cuda_error;
            return false;
        }
    } else if (!q38_cuda_matrix_batch_generic(
                   weight.type, device_weight, device->input, 1, weight.rows,
                   weight.cols, device->output, device->stream, nullptr, 0)) {
        *error = "generic matrix batch launch failed";
        return false;
    }
    sample->dispatch_us += now_us() - dispatch_started;
    ++sample->launches;

    const double d2h_started = now_us();
    if (!cuda_ok(cudaMemcpyAsync(output, device->output,
                                 output_elements * sizeof(float),
                                 cudaMemcpyDeviceToHost, device->stream),
                 "projection output download", error))
        return false;
    sample->memcpy_us += now_us() - d2h_started;
    sample->d2h_bytes += output_elements * sizeof(float);

    const double sync_started = now_us();
    if (!cuda_ok(cudaStreamSynchronize(device->stream), "projection sync",
                 error))
        return false;
    sample->sync_us += now_us() - sync_started;
    ++sample->syncs;
    *stage_us = now_us() - started;
    return true;
}

static void conv_and_silu(const Fixture &fixture, const float *qkv,
                          const float *history, float *conv,
                          float *next_history) {
    for (size_t channel = 0; channel < kQkv; ++channel) {
        float sum = 0.0f;
        for (size_t tap = 0; tap < Q38_TEST_GDN_KERNEL; ++tap) {
            const size_t source = tap;
            const float sample = source < Q38_TEST_GDN_HISTORY
                ? history[source * kQkv + channel]
                : qkv[(source - Q38_TEST_GDN_HISTORY) * kQkv + channel];
            sum += host_weight_scalar(fixture.conv,
                                      channel * Q38_TEST_GDN_KERNEL + tap) *
                   sample;
        }
        conv[channel] = sum / (1.0f + std::exp(-sum));
    }
    for (size_t tail = 0; tail < Q38_TEST_GDN_HISTORY; ++tail) {
        const size_t source = 1u + tail;
        if (source < Q38_TEST_GDN_HISTORY)
            std::memcpy(next_history + tail * kQkv,
                        history + source * kQkv, kQkv * sizeof(float));
        else
            std::memcpy(next_history + tail * kQkv,
                        qkv + (source - Q38_TEST_GDN_HISTORY) * kQkv,
                        kQkv * sizeof(float));
    }
}

static void prepare_recurrence(const Fixture &fixture, const float *conv,
                               const float *a, const float *b, float *q,
                               float *k, float *v, float *decay,
                               float *beta) {
    for (size_t head = 0; head < kHeads; ++head) {
        const size_t key_head = head / 3u;
        for (size_t d = 0; d < kDim; ++d) {
            q[head * kDim + d] =
                conv[key_head * kDim + d];
            k[head * kDim + d] =
                conv[Q38_TEST_GDN_KEY_HEADS * kDim + key_head * kDim + d];
            v[head * kDim + d] =
                conv[2u * Q38_TEST_GDN_KEY_HEADS * kDim + head * kDim + d];
        }
        float q_norm = 0.0f, k_norm = 0.0f;
        for (size_t d = 0; d < kDim; ++d) {
            q_norm += q[head * kDim + d] * q[head * kDim + d];
            k_norm += k[head * kDim + d] * k[head * kDim + d];
        }
        const float q_scale = 1.0f / std::sqrt(q_norm + 1.0e-6f);
        const float k_scale = 1.0f / std::sqrt(k_norm + 1.0e-6f);
        for (size_t d = 0; d < kDim; ++d) {
            q[head * kDim + d] *= q_scale;
            k[head * kDim + d] *= k_scale;
        }
        const float av = a[head] + host_weight_scalar(fixture.dt_bias, head);
        decay[head] =
            std::exp(-std::exp(host_weight_scalar(fixture.a_log, head)) *
                     std::log1pf(std::exp(av)));
        beta[head] = 1.0f / (1.0f + std::exp(-b[head]));
    }
}

static void post_recurrence(const Fixture &fixture, const float *z,
                            float *recurrent, float *gated) {
    float norm[kDim];
    for (size_t d = 0; d < kDim; ++d)
        norm[d] = host_weight_scalar(fixture.norm, d);
    for (size_t head = 0; head < kHeads; ++head) {
        double sum = 0.0;
        float *head_values = recurrent + head * kDim;
        for (size_t d = 0; d < kDim; ++d)
            sum += static_cast<double>(head_values[d]) * head_values[d];
        const float scale =
            1.0f / std::sqrt(static_cast<float>(sum / kDim) + 1.0e-6f);
        for (size_t d = 0; d < kDim; ++d)
            gated[head * kDim + d] =
                head_values[d] * scale * norm[d] /
                (1.0f + std::exp(-z[head * kDim + d]));
    }
}

static bool run_once(const Fixture &fixture, Device *device, bool candidate,
                     RunResult *result, std::string *error) {
    result->sample = {};
    result->output.resize(kHidden);
    result->next_history.resize(kHistoryElements);
    result->next_state.resize(kStateElements);
    std::vector<float> qkv(kQkv), z(kZ), a(kHeads), b(kHeads);
    std::vector<float> conv(kQkv), q(kHeads * kDim), k(kHeads * kDim);
    std::vector<float> v(kHeads * kDim), decay(kHeads), beta(kHeads);
    std::vector<float> recurrent(kHeads * kDim), gated(kZ);
    std::vector<float> state = fixture.state;
    const double started = now_us();
    double stage = 0.0;
    if (!project(fixture.qkv, device->weights[0], fixture.hidden.data(),
                 kHidden, qkv.data(), kQkv, candidate, device, &result->sample,
                 &stage, error))
        return false;
    result->sample.qkv_us = stage;
    if (!project(fixture.z, device->weights[1], fixture.hidden.data(), kHidden,
                 z.data(), kZ, candidate, device, &result->sample, &stage,
                 error))
        return false;
    result->sample.z_us = stage;
    if (!project(fixture.a, device->weights[2], fixture.hidden.data(), kHidden,
                 a.data(), kHeads, candidate, device, &result->sample, &stage,
                 error))
        return false;
    result->sample.a_us = stage;
    if (!project(fixture.b, device->weights[3], fixture.hidden.data(), kHidden,
                 b.data(), kHeads, candidate, device, &result->sample, &stage,
                 error))
        return false;
    result->sample.b_us = stage;

    const double conv_started = now_us();
    conv_and_silu(fixture, qkv.data(), fixture.history.data(), conv.data(),
                  result->next_history.data());
    result->sample.conv_us = now_us() - conv_started;

    const double gate_started = now_us();
    prepare_recurrence(fixture, conv.data(), a.data(), b.data(), q.data(),
                       k.data(), v.data(), decay.data(), beta.data());
    result->sample.gate_us = now_us() - gate_started;

    const double recurrence_started = now_us();
    if (!q38_gdn_ref_run(
            state.data(), 1, 1, q.data(), k.data(), v.data(), decay.data(),
            beta.data(), 1.0f / std::sqrt(128.0f), recurrent.data())) {
        *error = "CPU recurrence failed";
        return false;
    }
    result->next_state = state;
    result->sample.recurrence_us = now_us() - recurrence_started;

    const double post_started = now_us();
    post_recurrence(fixture, z.data(), recurrent.data(), gated.data());
    result->sample.post_gate_us = now_us() - post_started;

    if (!project(fixture.out_proj, device->weights[8], gated.data(), kZ,
                 result->output.data(), kHidden, candidate, device,
                 &result->sample, &stage, error))
        return false;
    result->sample.output_us = stage;
    result->sample.total_us = now_us() - started;
    return true;
}

enum {
    C2_EVENT_TOTAL = 0,
    C2_EVENT_H2D = 1,
    C2_EVENT_QKV = 2,
    C2_EVENT_Z = 3,
    C2_EVENT_A = 4,
    C2_EVENT_B = 5,
    C2_EVENT_CONV = 6,
    C2_EVENT_PREPARE = 7,
    C2_EVENT_RECURRENCE = 8,
    C2_EVENT_POST = 9,
    C2_EVENT_OUTPUT = 10,
    C2_EVENT_D2H = 11,
    C2_EVENT_COUNT = 12,
};

static bool run_once_device(const Fixture &fixture, Device *device,
                            bool fused, bool capture_state,
                            RunResult *result, std::string *error) {
    result->sample = {};
    result->output.resize(kHidden);
    if (capture_state) {
        result->next_history.resize(kHistoryElements);
        result->next_state.resize(kStateElements);
    }

    const size_t history_bytes = kHistoryElements * sizeof(float);
    const size_t state_bytes = kStateElements * sizeof(float);
    if (!cuda_ok(cudaMemcpyAsync(device->history, fixture.history.data(),
                                 history_bytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "C2 history reset", error) ||
        !cuda_ok(cudaMemcpyAsync(device->state, fixture.state.data(),
                                 state_bytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "C2 state reset", error) ||
        !cuda_ok(cudaStreamSynchronize(device->stream), "C2 reset sync",
                 error))
        return false;
    result->sample.state_reset_h2d_bytes =
        static_cast<uint64_t>(history_bytes + state_bytes);

    cudaEvent_t starts[C2_EVENT_COUNT] = {};
    cudaEvent_t stops[C2_EVENT_COUNT] = {};
    for (size_t i = 0; i < C2_EVENT_COUNT; ++i) {
        if (!cuda_ok(cudaEventCreate(&starts[i]), "C2 event create", error) ||
            !cuda_ok(cudaEventCreate(&stops[i]), "C2 event create", error)) {
            for (size_t j = 0; j <= i; ++j) {
                if (starts[j]) cudaEventDestroy(starts[j]);
                if (stops[j]) cudaEventDestroy(stops[j]);
            }
            return false;
        }
    }
    const auto cleanup_events = [&]() {
        for (size_t i = 0; i < C2_EVENT_COUNT; ++i) {
            cudaEventDestroy(starts[i]);
            cudaEventDestroy(stops[i]);
        }
    };
    const auto mark_start = [&](size_t index) {
        return cuda_ok(cudaEventRecord(starts[index], device->stream),
                       "C2 event record", error);
    };
    const auto mark_stop = [&](size_t index) {
        return cuda_ok(cudaEventRecord(stops[index], device->stream),
                       "C2 event record", error);
    };
    const auto elapsed = [&](size_t index, double *microseconds) {
        float milliseconds = 0.0f;
        if (!cuda_ok(cudaEventElapsedTime(&milliseconds, starts[index],
                                          stops[index]),
                     "C2 event elapsed", error))
            return false;
        *microseconds = static_cast<double>(milliseconds) * 1000.0;
        return true;
    };

    const double started = now_us();
    double dispatch_us = 0.0;
    if (!mark_start(C2_EVENT_TOTAL) || !mark_start(C2_EVENT_H2D)) {
        cleanup_events();
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(device->input, fixture.hidden.data(),
                                 kHidden * sizeof(float),
                                 cudaMemcpyHostToDevice, device->stream),
                 "C2 hidden upload", error) ||
        !mark_stop(C2_EVENT_H2D)) {
        cleanup_events();
        return false;
    }
    result->sample.h2d_bytes = kHidden * sizeof(float);

    const auto launch_projection = [&](size_t event_index, const Weight &weight,
                                       void *device_weight, float *output) {
        if (!mark_start(event_index)) return false;
        const double dispatch_started = now_us();
        char cuda_error[256] = {};
        const bool ok = q38_cuda_gdn_project(
            cuda_weight_type(weight.type), device_weight, weight.rows,
            weight.cols, device->input, 1, output, device->stream, cuda_error,
            sizeof(cuda_error));
        dispatch_us += now_us() - dispatch_started;
        if (!ok) {
            *error = cuda_error;
            return false;
        }
        ++result->sample.launches;
        return mark_stop(event_index);
    };
    if (!launch_projection(C2_EVENT_QKV, fixture.qkv, device->weights[0],
                           device->qkv) ||
        !launch_projection(C2_EVENT_Z, fixture.z, device->weights[1],
                           device->z) ||
        !launch_projection(C2_EVENT_A, fixture.a, device->weights[2],
                           device->a) ||
        !launch_projection(C2_EVENT_B, fixture.b, device->weights[3],
                           device->b))
        {
            cleanup_events();
            return false;
        }

    if (!mark_start(C2_EVENT_CONV)) {
        cleanup_events();
        return false;
    }
    if (fused) {
        const double dispatch_started = now_us();
        char cuda_error[256] = {};
        if (!q38_cuda_gdn_fused_recurrent(
                device->qkv, device->z, device->a, device->b,
                cuda_weight_type(fixture.conv.type), device->weights[4],
                cuda_weight_type(fixture.a_log.type), device->weights[5],
                device->weights[6], cuda_weight_type(fixture.norm.type),
                device->weights[7], device->state, device->history,
                device->gated, device->stream, cuda_error,
                sizeof(cuda_error)) ||
            !q38_cuda_gdn_history_update(
                device->qkv, 1, kQkv, Q38_TEST_GDN_KERNEL, device->history,
                device->stream, cuda_error, sizeof(cuda_error))) {
            *error = cuda_error;
            cleanup_events();
            return false;
        }
        dispatch_us += now_us() - dispatch_started;
        result->sample.launches += 2;
    } else {
        const double dispatch_started = now_us();
        char cuda_error[256] = {};
        if (!q38_cuda_gdn_conv_silu_fused_channel_major(
                cuda_weight_type(fixture.conv.type), device->weights[4],
                device->qkv, 1, kQkv, Q38_TEST_GDN_KERNEL, device->history,
                device->conv, device->stream, cuda_error, sizeof(cuda_error))) {
            *error = cuda_error;
            cleanup_events();
            return false;
        }
        dispatch_us += now_us() - dispatch_started;
        ++result->sample.launches;
        if (!mark_stop(C2_EVENT_CONV) || !mark_start(C2_EVENT_PREPARE)) {
            cleanup_events();
            return false;
        }
        {
            const double dispatch_started = now_us();
            if (!q38_cuda_gdn_prepare_recurrence(
                    device->conv, device->a, device->b,
                    cuda_weight_type(fixture.a_log.type), device->weights[5],
                    device->weights[6], 1, device->q, device->k, device->v,
                    device->decay, device->beta, device->stream, cuda_error,
                    sizeof(cuda_error))) {
                *error = cuda_error;
                cleanup_events();
                return false;
            }
            dispatch_us += now_us() - dispatch_started;
            ++result->sample.launches;
        }
        if (!mark_stop(C2_EVENT_PREPARE) || !mark_start(C2_EVENT_RECURRENCE)) {
            cleanup_events();
            return false;
        }
        {
            const double dispatch_started = now_us();
            if (!q38_cuda_gdn_recurrence(
                    device->state, 1, device->q, device->k, device->v,
                    device->decay, device->beta, 1.0f / std::sqrt(128.0f),
                    device->recurrent, device->stream, cuda_error,
                    sizeof(cuda_error))) {
                *error = cuda_error;
                cleanup_events();
                return false;
            }
            dispatch_us += now_us() - dispatch_started;
            ++result->sample.launches;
        }
        if (!mark_stop(C2_EVENT_RECURRENCE) || !mark_start(C2_EVENT_POST)) {
            cleanup_events();
            return false;
        }
        {
            const double dispatch_started = now_us();
            if (!q38_cuda_gdn_post_recurrence(
                    device->recurrent, device->z,
                    cuda_weight_type(fixture.norm.type), device->weights[7],
                    1, device->gated, device->stream, cuda_error,
                    sizeof(cuda_error))) {
                *error = cuda_error;
                cleanup_events();
                return false;
            }
            dispatch_us += now_us() - dispatch_started;
            ++result->sample.launches;
        }
    }
    if (!fused && !mark_stop(C2_EVENT_POST)) {
        cleanup_events();
        return false;
    }
    if (fused) {
        if (!mark_stop(C2_EVENT_CONV)) {
            cleanup_events();
            return false;
        }
    }
    if (!mark_start(C2_EVENT_OUTPUT)) {
        cleanup_events();
        return false;
    }
    {
        const double dispatch_started = now_us();
        char cuda_error[256] = {};
        if (!q38_cuda_gdn_project(
                cuda_weight_type(fixture.out_proj.type), device->weights[8],
                fixture.out_proj.rows, fixture.out_proj.cols, device->gated, 1,
                device->output, device->stream, cuda_error, sizeof(cuda_error))) {
            *error = cuda_error;
            cleanup_events();
            return false;
        }
        dispatch_us += now_us() - dispatch_started;
        ++result->sample.launches;
    }
    if (!mark_stop(C2_EVENT_OUTPUT) || !mark_start(C2_EVENT_D2H) ||
        !cuda_ok(cudaMemcpyAsync(result->output.data(), device->output,
                                 kHidden * sizeof(float),
                                 cudaMemcpyDeviceToHost, device->stream),
                 "C2 output download", error) ||
        !mark_stop(C2_EVENT_D2H) || !mark_stop(C2_EVENT_TOTAL)) {
        cleanup_events();
        return false;
    }
    const double sync_started = now_us();
    if (!cuda_ok(cudaStreamSynchronize(device->stream), "C2 final sync",
                 error)) {
        cleanup_events();
        return false;
    }
    result->sample.sync_us = now_us() - sync_started;
    result->sample.syncs = 1;
    result->sample.dispatch_us = dispatch_us;
    if (!elapsed(C2_EVENT_TOTAL, &result->sample.total_us) ||
        !elapsed(C2_EVENT_QKV, &result->sample.qkv_us) ||
        !elapsed(C2_EVENT_Z, &result->sample.z_us) ||
        !elapsed(C2_EVENT_A, &result->sample.a_us) ||
        !elapsed(C2_EVENT_B, &result->sample.b_us) ||
        !elapsed(C2_EVENT_OUTPUT, &result->sample.output_us)) {
        cleanup_events();
        return false;
    }
    if (fused) {
        if (!elapsed(C2_EVENT_CONV, &result->sample.fused_us)) {
            cleanup_events();
            return false;
        }
        result->sample.conv_us = result->sample.fused_us;
    } else if (!elapsed(C2_EVENT_CONV, &result->sample.conv_us) ||
               !elapsed(C2_EVENT_PREPARE, &result->sample.gate_us) ||
               !elapsed(C2_EVENT_RECURRENCE, &result->sample.recurrence_us) ||
               !elapsed(C2_EVENT_POST, &result->sample.post_gate_us)) {
        cleanup_events();
        return false;
    }
    double h2d_us = 0.0;
    double d2h_us = 0.0;
    if (!elapsed(C2_EVENT_H2D, &h2d_us) ||
        !elapsed(C2_EVENT_D2H, &d2h_us)) {
        cleanup_events();
        return false;
    }
    result->sample.memcpy_us = h2d_us + d2h_us;
    result->sample.d2h_bytes = kHidden * sizeof(float);
    result->sample.total_us = std::max(result->sample.total_us,
                                       now_us() - started);

    if (capture_state) {
        if (!cuda_ok(cudaMemcpy(result->next_history.data(), device->history,
                                history_bytes, cudaMemcpyDeviceToHost),
                     "C2 history download", error) ||
            !cuda_ok(cudaMemcpy(result->next_state.data(), device->state,
                                state_bytes, cudaMemcpyDeviceToHost),
                     "C2 state download", error)) {
            cleanup_events();
            return false;
        }
    }
    cleanup_events();
    return true;
}

static Compare compare(const std::vector<float> &expected,
                       const std::vector<float> &actual) {
    Compare result;
    double squared = 0.0;
    const size_t count = std::min(expected.size(), actual.size());
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) {
            ++result.nonfinite;
            continue;
        }
        const double abs = std::fabs(static_cast<double>(actual[i]) -
                                     static_cast<double>(expected[i]));
        const double rel = abs / std::max(std::fabs((double)expected[i]), 1e-12);
        result.max_abs = std::max(result.max_abs, abs);
        result.max_rel = std::max(result.max_rel, rel);
        squared += abs * abs;
    }
    result.rmse = std::sqrt(squared / static_cast<double>(count));
    return result;
}

static double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

static double median(const std::vector<double> &values) {
    return percentile(values, 0.5);
}

static void print_metric(std::ostream &output, const char *name,
                         double median_us, double p95_us) {
    output << "\"" << name << "\":{\"median_us\":" << median_us
           << ",\"p95_us\":" << p95_us << "}";
}

static bool benchmark_fixture(const std::string &dir, BenchMode mode,
                              std::string *json, std::string *error) {
    const bool candidate = mode != BenchMode::Production;
    const bool c2 = mode == BenchMode::C2;
    const bool c3 = mode == BenchMode::C3;
    Fixture fixture;
    if (!load_fixture(dir, &fixture, error)) return false;
    Device device;
    if (!alloc_device(fixture, &device, error) ||
        !upload_weights(fixture, &device, error)) {
        free_device(&device);
        return false;
    }

    q38_test_gdn_weight refs[] = {
        {fixture.qkv.type, fixture.qkv.rows, fixture.qkv.cols,
         fixture.qkv.raw.data()},
        {fixture.z.type, fixture.z.rows, fixture.z.cols, fixture.z.raw.data()},
        {fixture.a.type, fixture.a.rows, fixture.a.cols, fixture.a.raw.data()},
        {fixture.b.type, fixture.b.rows, fixture.b.cols, fixture.b.raw.data()},
        {fixture.conv.type, fixture.conv.rows, fixture.conv.cols,
         fixture.conv.raw.data()},
        {fixture.a_log.type, fixture.a_log.rows, fixture.a_log.cols,
         fixture.a_log.raw.data()},
        {fixture.dt_bias.type, fixture.dt_bias.rows, fixture.dt_bias.cols,
         fixture.dt_bias.raw.data()},
        {fixture.norm.type, fixture.norm.rows, fixture.norm.cols,
         fixture.norm.raw.data()},
        {fixture.out_proj.type, fixture.out_proj.rows, fixture.out_proj.cols,
         fixture.out_proj.raw.data()},
    };
    std::vector<float> oracle_output(kHidden), oracle_history(kHistoryElements),
        oracle_state(kStateElements);
    char reference_error[256] = {};
    if (!q38_test_gdn_reference(
            fixture.hidden.data(), fixture.history.data(), fixture.state.data(),
            &refs[0], &refs[1], &refs[2], &refs[3], &refs[4], &refs[5],
            &refs[6], &refs[7], &refs[8], oracle_output.data(),
            oracle_history.data(), oracle_state.data(), reference_error,
            sizeof(reference_error))) {
        if (reference_error[0]) *error = reference_error;
        free_device(&device);
        return false;
    }
    RunResult check;
    if ((c2 || c3)
            ? !run_once_device(fixture, &device, c3, true, &check, error)
            : !run_once(fixture, &device, candidate, &check, error)) {
        free_device(&device);
        return false;
    }
    const Compare output_check = compare(fixture.expected, check.output);
    const Compare oracle_check = compare(oracle_output, check.output);
    const Compare history_check = compare(fixture.next_history,
                                          check.next_history);
    const Compare state_check = compare(fixture.next_state, check.next_state);
    const bool correctness_ok =
        output_check.nonfinite == 0 && oracle_check.nonfinite == 0 &&
        history_check.nonfinite == 0 && state_check.nonfinite == 0 &&
        output_check.max_abs <= kOutputMaxAbsTolerance &&
        output_check.max_rel <= kOutputMaxRelTolerance &&
        oracle_check.max_abs <= kOutputMaxAbsTolerance &&
        oracle_check.max_rel <= kOutputMaxRelTolerance &&
        history_check.max_abs <= kHistoryMaxAbsTolerance &&
        state_check.max_abs <= kStateMaxAbsTolerance;
    if (!correctness_ok) {
        std::ostringstream detail;
        detail << "GDN correctness tolerance failed for layer "
               << fixture.layer << ": captured_abs=" << output_check.max_abs
               << ", captured_rel=" << output_check.max_rel
               << ", oracle_abs=" << oracle_check.max_abs
               << ", oracle_rel=" << oracle_check.max_rel
               << ", history_abs=" << history_check.max_abs
               << ", state_abs=" << state_check.max_abs
               << ", nonfinite="
               << output_check.nonfinite + oracle_check.nonfinite +
                      history_check.nonfinite + state_check.nonfinite;
        *error = detail.str();
        free_device(&device);
        return false;
    }

    for (size_t i = 0; i < kWarmup; ++i) {
        RunResult warmup;
        if ((c2 || c3)
                ? !run_once_device(fixture, &device, c3, false, &warmup,
                                   error)
                : !run_once(fixture, &device, candidate, &warmup, error)) {
            free_device(&device);
            return false;
        }
    }
    std::vector<Sample> samples;
    samples.reserve(kSamples);
    for (size_t i = 0; i < kSamples; ++i) {
        RunResult measured;
        if ((c2 || c3)
                ? !run_once_device(fixture, &device, c3, false, &measured,
                                   error)
                : !run_once(fixture, &device, candidate, &measured, error)) {
            free_device(&device);
            return false;
        }
        samples.push_back(std::move(measured.sample));
    }
    std::vector<double> total, qkv, z, a, b, conv, fused, gate, recurrence,
        post, out;
    std::vector<double> dispatch, sync, memcpy;
    for (const Sample &sample : samples) {
        total.push_back(sample.total_us);
        qkv.push_back(sample.qkv_us);
        z.push_back(sample.z_us);
        a.push_back(sample.a_us);
        b.push_back(sample.b_us);
        conv.push_back(sample.conv_us);
        fused.push_back(sample.fused_us);
        gate.push_back(sample.gate_us);
        recurrence.push_back(sample.recurrence_us);
        post.push_back(sample.post_gate_us);
        out.push_back(sample.output_us);
        dispatch.push_back(sample.dispatch_us);
        sync.push_back(sample.sync_us);
        memcpy.push_back(sample.memcpy_us);
    }
    const Sample &accounting = samples.front();
    double stage_sum = median(qkv) + median(z) + median(a) + median(b) +
                       (c3 ? 0.0 : median(conv)) + median(gate) +
                       median(recurrence) + median(post) + median(out);
    if (c2 || c3) stage_sum += median(memcpy);
    if (c3) {
        stage_sum += median(fused);
    }
    const double total_median = median(total);
    const double accounting_ratio = total_median > 0.0
        ? stage_sum / total_median : 0.0;

    std::ostringstream output;
    output << std::fixed << std::setprecision(3);
    output << "{\"layer\":" << fixture.layer
           << ",\"mode\":\""
           << (mode == BenchMode::C3 ? "gdn_c3" :
               mode == BenchMode::C2 ? "gdn_c2" :
               candidate ? "gdn_c1" : "production")
           << "\",\"warmup\":" << kWarmup << ",\"samples\":" << kSamples
           << ",\"total\":";
    output << "{\"median_us\":" << total_median
           << ",\"p95_us\":" << percentile(total, 0.95) << "},\"stages\":{";
    print_metric(output, "qkv_projection", median(qkv), percentile(qkv, 0.95));
    output << ",";
    print_metric(output, "z_projection", median(z), percentile(z, 0.95));
    output << ",";
    print_metric(output, "a_projection", median(a), percentile(a, 0.95));
    output << ",";
    print_metric(output, "b_projection", median(b), percentile(b, 0.95));
    output << ",";
    if (c3) {
        print_metric(output, "fused_recurrent_block", median(fused),
                     percentile(fused, 0.95));
        output << ",";
    } else {
        print_metric(output, "conv_update_silu", median(conv),
                     percentile(conv, 0.95));
        output << ",";
    }
    print_metric(output, "gate_activation", median(gate), percentile(gate, 0.95));
    output << ",";
    print_metric(output, "recurrent_update", median(recurrence),
                 percentile(recurrence, 0.95));
    output << ",";
    print_metric(output, "post_recurrence_gate", median(post),
                 percentile(post, 0.95));
    output << ",";
    print_metric(output, "output_projection", median(out),
                 percentile(out, 0.95));
    output << "},\"accounting_ratio\":" << accounting_ratio
           << ",\"launches\":" << accounting.launches
           << ",\"syncs\":" << accounting.syncs
           << ",\"h2d_bytes\":" << accounting.h2d_bytes
           << ",\"d2h_bytes\":" << accounting.d2h_bytes
           << ",\"d2d_bytes\":0,\"weight_bytes_read\":"
           << fixture.qkv.raw.size() + fixture.z.raw.size() +
                  fixture.a.raw.size() + fixture.b.raw.size() +
                  fixture.conv.raw.size() + fixture.a_log.raw.size() +
                  fixture.dt_bias.raw.size() + fixture.norm.raw.size() +
                  fixture.out_proj.raw.size()
           << ",\"dispatch_us\":" << median(dispatch)
           << ",\"sync_wait_us\":" << median(sync)
           << ",\"memcpy_us\":" << median(memcpy)
           << ",\"state_reset_h2d_bytes\":" << accounting.state_reset_h2d_bytes
           << ",\"correctness\":{\"captured_output_max_abs\":"
           << output_check.max_abs << ",\"captured_output_max_rel\":"
           << output_check.max_rel << ",\"oracle_output_max_abs\":"
           << oracle_check.max_abs << ",\"oracle_output_max_rel\":"
           << oracle_check.max_rel << ",\"oracle_output_rmse\":"
           << oracle_check.rmse << ",\"next_history_max_abs\":"
           << history_check.max_abs << ",\"next_state_max_abs\":"
           << state_check.max_abs << ",\"nan_inf\":"
           << output_check.nonfinite + oracle_check.nonfinite +
                  history_check.nonfinite + state_check.nonfinite
           << ",\"passed\":" << (correctness_ok ? "true" : "false")
           << ",\"tolerances\":{\"output_max_abs\":"
           << kOutputMaxAbsTolerance << ",\"output_max_rel\":"
           << kOutputMaxRelTolerance << ",\"next_history_max_abs\":"
           << kHistoryMaxAbsTolerance << ",\"next_state_max_abs\":"
           << kStateMaxAbsTolerance
           << "}}}";
    *json = output.str();
    free_device(&device);
    return accounting_ratio >= 0.97;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
                     "usage: gdn_bench FIXTURE_ROOT ARTIFACT "
                     "[gdn_c1|gdn_c2|gdn_c3]\n");
        return 2;
    }
    BenchMode mode = BenchMode::Production;
    if (argc == 4 && std::strcmp(argv[3], "gdn_c1") == 0)
        mode = BenchMode::C1;
    else if (argc == 4 && std::strcmp(argv[3], "gdn_c2") == 0)
        mode = BenchMode::C2;
    else if (argc == 4 && std::strcmp(argv[3], "gdn_c3") == 0)
        mode = BenchMode::C3;
    else if (argc == 4) {
        std::fprintf(stderr, "gdn_bench: unknown mode %s\n", argv[3]);
        return 2;
    }
    const bool candidate = mode != BenchMode::Production;
    const char *mode_name = mode == BenchMode::C3 ? "gdn_c3" :
                            mode == BenchMode::C2 ? "gdn_c2" :
                            candidate ? "gdn_c1" : "production";
    const char *names[] = {"early", "middle", "late"};
    std::string results[3];
    std::string error;
    for (size_t i = 0; i < 3; ++i) {
        if (!benchmark_fixture(std::string(argv[1]) + "/" + names[i], mode,
                                &results[i], &error)) {
            std::fprintf(stderr, "gdn_bench: %s\n", error.c_str());
            return 1;
        }
        std::printf("%s\n", results[i].c_str());
    }
    std::ofstream artifact(argv[2]);
    if (!artifact) {
        std::fprintf(stderr, "gdn_bench: cannot write %s\n", argv[2]);
        return 1;
    }
    artifact << "{\"format\":\"q38-gdn-subsystem-v1\",\"status\":\""
             << (mode == BenchMode::C3 ? "S3-C3" :
                 mode == BenchMode::C2 ? "S3-C2" :
                 candidate ? "S3-C1" : "S3-BASELINE")
             << "\",\"mode\":\"" << mode_name
             << "\",\"fixtures\":[";
    for (size_t i = 0; i < 3; ++i)
        artifact << (i ? "," : "") << results[i];
    artifact << "]}\n";
    return 0;
}
