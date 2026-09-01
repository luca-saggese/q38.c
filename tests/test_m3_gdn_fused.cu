#include "q38_gdn.h"
#include "q38_gdn_ref.h"
#include "q38_cuda_primitives.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kMaxTokens = 10;
constexpr size_t kQkvChannels = Q38_GDN_QKV_CHANNELS;
constexpr size_t kKeyChannels = Q38_GDN_KEY_CHANNELS;
constexpr size_t kValueChannels = Q38_GDN_VALUE_CHANNELS;
constexpr size_t kHistoryTokens = Q38_GDN_CONV_KERNEL - 1u;
constexpr size_t kStateElements =
    Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM;
constexpr float kVectorTolerance = 3e-6f;
constexpr float kRecurrenceTolerance = 3e-5f;

enum Stage {
    kRawConv,
    kSilu,
    kQ16,
    kK16,
    kV,
    kQ,
    kK,
    kRecurrence,
    kState,
    kHistory,
    kStageCount
};

const char *const kStageNames[kStageCount] = {
    "raw_conv", "silu", "q16", "k16", "v", "q", "k", "recurrence",
    "state", "conv_history"};

struct Check {
    float max_abs = 0.0f;
    float tolerance = 0.0f;
    bool pass = true;
};

struct CaseReport {
    size_t chunk_size = 0;
    size_t boundaries = 0;
    std::array<Check, kStageCount> unfused_scalar;
    std::array<Check, kStageCount> fused_unfused;
    std::array<Check, kStageCount> fused_scalar;
    bool pass = true;
};

struct DeviceBuffers {
    float *input = nullptr;
    float *kernel = nullptr;
    float *decay = nullptr;
    float *beta = nullptr;
    float *raw = nullptr;
    float *unfused_silu = nullptr;
    float *fused_silu = nullptr;
    float *unfused_q16 = nullptr;
    float *unfused_k16 = nullptr;
    float *unfused_v = nullptr;
    float *unfused_q = nullptr;
    float *unfused_k = nullptr;
    float *fused_q16 = nullptr;
    float *fused_k16 = nullptr;
    float *fused_v = nullptr;
    float *fused_q = nullptr;
    float *fused_k = nullptr;
    float *unfused_output = nullptr;
    float *fused_output = nullptr;
    float *unfused_state = nullptr;
    float *fused_state = nullptr;
    float *unfused_history = nullptr;
    float *fused_history = nullptr;
};

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

static float max_abs(const std::vector<float> &expected,
                     const std::vector<float> &actual) {
    float result = 0.0f;
    for (size_t i = 0; i < expected.size(); i++)
        result = std::fmax(result, std::fabs(expected[i] - actual[i]));
    return result;
}

static bool record_check(Check &check, const std::vector<float> &expected,
                         const std::vector<float> &actual, float tolerance,
                         const char *label) {
    const float difference = max_abs(expected, actual);
    check.max_abs = std::fmax(check.max_abs, difference);
    check.tolerance = tolerance;
    if (difference > tolerance) {
        check.pass = false;
        std::fprintf(stderr, "M3-C13 %s mismatch: max_abs=%g tolerance=%g\n",
                     label, difference, tolerance);
    }
    return difference <= tolerance;
}

static void make_fixture(std::vector<float> &input, std::vector<float> &kernel,
                         std::vector<float> &decay,
                         std::vector<float> &beta) {
    input.resize(kMaxTokens * kQkvChannels);
    kernel.resize(Q38_GDN_CONV_KERNEL * kQkvChannels);
    decay.resize(kMaxTokens * Q38_GDN_VALUE_HEADS);
    beta.resize(kMaxTokens * Q38_GDN_VALUE_HEADS);
    for (size_t token = 0; token < kMaxTokens; token++) {
        for (size_t channel = 0; channel < kQkvChannels; channel++) {
            const size_t code = (token * 37u + channel * 13u + 11u) % 101u;
            input[token * kQkvChannels + channel] =
                (static_cast<float>(code) - 50.0f) * 0.0005f;
        }
        for (size_t head = 0; head < Q38_GDN_VALUE_HEADS; head++) {
            decay[token * Q38_GDN_VALUE_HEADS + head] =
                0.77f + 0.0125f * static_cast<float>((token + head) % 8u);
            beta[token * Q38_GDN_VALUE_HEADS + head] =
                0.21f + 0.0275f * static_cast<float>((token * 3u + head) % 7u);
        }
    }
    for (size_t tap = 0; tap < Q38_GDN_CONV_KERNEL; tap++)
        for (size_t channel = 0; channel < kQkvChannels; channel++) {
            const size_t code = (tap * 19u + channel * 7u + 3u) % 9u;
            kernel[tap * kQkvChannels + channel] =
                (static_cast<float>(code) - 4.0f) * 0.025f;
        }
}

static void conv_ref_chunk(const float *input, size_t tokens,
                           const std::vector<float> &kernel,
                           std::vector<float> &history,
                           std::vector<float> &output) {
    output.assign(tokens * kQkvChannels, 0.0f);
    for (size_t token = 0; token < tokens; token++)
        for (size_t channel = 0; channel < kQkvChannels; channel++) {
            float sum = 0.0f;
            for (size_t tap = 0; tap < Q38_GDN_CONV_KERNEL; tap++) {
                const size_t source =
                    kHistoryTokens + token -
                    (Q38_GDN_CONV_KERNEL - 1u - tap);
                const float sample =
                    source < kHistoryTokens
                        ? history[source * kQkvChannels + channel]
                        : input[(source - kHistoryTokens) * kQkvChannels +
                                channel];
                sum += kernel[tap * kQkvChannels + channel] * sample;
            }
            output[token * kQkvChannels + channel] = sum;
        }
    for (size_t tail = 0; tail < kHistoryTokens; tail++) {
        const size_t source = tokens + tail;
        for (size_t channel = 0; channel < kQkvChannels; channel++)
            history[tail * kQkvChannels + channel] =
                source < kHistoryTokens
                    ? history[source * kQkvChannels + channel]
                    : input[(source - kHistoryTokens) * kQkvChannels + channel];
    }
}

static void split_ref(const std::vector<float> &mixed, std::vector<float> &q16,
                      std::vector<float> &k16, std::vector<float> &v,
                      std::vector<float> &q, std::vector<float> &k) {
    const size_t tokens = mixed.size() / kQkvChannels;
    q16.assign(tokens * kKeyChannels, 0.0f);
    k16.assign(tokens * kKeyChannels, 0.0f);
    v.assign(tokens * kValueChannels, 0.0f);
    q.assign(tokens * kValueChannels, 0.0f);
    k.assign(tokens * kValueChannels, 0.0f);
    for (size_t token = 0; token < tokens; token++) {
        const float *source = mixed.data() + token * kQkvChannels;
        std::memcpy(q16.data() + token * kKeyChannels, source,
                    kKeyChannels * sizeof(float));
        std::memcpy(k16.data() + token * kKeyChannels,
                    source + kKeyChannels, kKeyChannels * sizeof(float));
        std::memcpy(v.data() + token * kValueChannels,
                    source + 2u * kKeyChannels,
                    kValueChannels * sizeof(float));
        q38_gdn_ref_repeat_key_heads(
            q16.data() + token * kKeyChannels,
            q.data() + token * kValueChannels);
        q38_gdn_ref_repeat_key_heads(
            k16.data() + token * kKeyChannels,
            k.data() + token * kValueChannels);
    }
}

static bool copy_device(float *device, std::vector<float> &host,
                        const char *what) {
    return cuda_ok(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   what);
}

static bool run_unfused(const DeviceBuffers &d, size_t offset, size_t tokens,
                        cudaStream_t stream, char *error, size_t error_len) {
    return q38_cuda_gdn_conv(
               Q38_GDN_WEIGHT_F32, d.kernel, d.input + offset * kQkvChannels,
               tokens, kQkvChannels, Q38_GDN_CONV_KERNEL, d.unfused_history,
               d.raw + offset * kQkvChannels, stream, error, error_len) &&
           q38_cuda_silu(d.raw + offset * kQkvChannels,
                         d.unfused_silu + offset * kQkvChannels,
                         tokens * kQkvChannels, stream, error, error_len) &&
           q38_cuda_gdn_split_qkv(
               d.unfused_silu + offset * kQkvChannels, tokens,
               d.unfused_q16 + offset * kKeyChannels,
               d.unfused_k16 + offset * kKeyChannels,
               d.unfused_v + offset * kValueChannels, stream, error,
               error_len) &&
           q38_cuda_gdn_repeat_key_heads(
               d.unfused_q16 + offset * kKeyChannels, tokens,
               d.unfused_q + offset * kValueChannels, stream, error,
               error_len) &&
           q38_cuda_gdn_repeat_key_heads(
               d.unfused_k16 + offset * kKeyChannels, tokens,
               d.unfused_k + offset * kValueChannels, stream, error,
               error_len) &&
           q38_cuda_gdn_recurrence(
               d.unfused_state, tokens,
               d.unfused_q + offset * kValueChannels,
               d.unfused_k + offset * kValueChannels,
               d.unfused_v + offset * kValueChannels,
               d.decay + offset * Q38_GDN_VALUE_HEADS,
               d.beta + offset * Q38_GDN_VALUE_HEADS, 0.75f,
               d.unfused_output + offset * kValueChannels, stream, error,
               error_len);
}

static bool run_fused(const DeviceBuffers &d, size_t offset, size_t tokens,
                      cudaStream_t stream, char *error, size_t error_len) {
    return q38_cuda_gdn_conv_silu_fused(
               Q38_GDN_WEIGHT_F32, d.kernel, d.input + offset * kQkvChannels,
               tokens, kQkvChannels, Q38_GDN_CONV_KERNEL, d.fused_history,
               d.fused_silu + offset * kQkvChannels, stream, error, error_len) &&
           q38_cuda_gdn_split_qkv(
               d.fused_silu + offset * kQkvChannels, tokens,
               d.fused_q16 + offset * kKeyChannels,
               d.fused_k16 + offset * kKeyChannels,
               d.fused_v + offset * kValueChannels, stream, error, error_len) &&
           q38_cuda_gdn_repeat_key_heads(
               d.fused_q16 + offset * kKeyChannels, tokens,
               d.fused_q + offset * kValueChannels, stream, error, error_len) &&
           q38_cuda_gdn_repeat_key_heads(
               d.fused_k16 + offset * kKeyChannels, tokens,
               d.fused_k + offset * kValueChannels, stream, error, error_len) &&
           q38_cuda_gdn_recurrence(
               d.fused_state, tokens, d.fused_q + offset * kValueChannels,
               d.fused_k + offset * kValueChannels,
               d.fused_v + offset * kValueChannels,
               d.decay + offset * Q38_GDN_VALUE_HEADS,
               d.beta + offset * Q38_GDN_VALUE_HEADS, 0.75f,
               d.fused_output + offset * kValueChannels, stream, error,
               error_len);
}

static void reset_check(CaseReport &report) {
    report.pass = true;
    for (size_t i = 0; i < kStageCount; i++) {
        report.unfused_scalar[i] = Check{};
        report.fused_unfused[i] = Check{};
        report.fused_scalar[i] = Check{};
    }
}

static bool run_case(size_t chunk_size, const std::vector<float> &input,
                     const std::vector<float> &kernel,
                     const std::vector<float> &decay,
                     const std::vector<float> &beta, const DeviceBuffers &d,
                     cudaStream_t stream, CaseReport &report) {
    report.chunk_size = chunk_size;
    report.boundaries = 0;
    reset_check(report);
    char error[256] = {};
    const size_t history_elements = kHistoryTokens * kQkvChannels;
    if (!cuda_ok(cudaMemsetAsync(d.unfused_state, 0,
                                 kStateElements * sizeof(float), stream),
                 "reset unfused state") ||
        !cuda_ok(cudaMemsetAsync(d.fused_state, 0,
                                 kStateElements * sizeof(float), stream),
                 "reset fused state") ||
        !cuda_ok(cudaMemsetAsync(d.unfused_history, 0,
                                 history_elements * sizeof(float), stream),
                 "reset unfused history") ||
        !cuda_ok(cudaMemsetAsync(d.fused_history, 0,
                                 history_elements * sizeof(float), stream),
                 "reset fused history") ||
        !cuda_ok(cudaStreamSynchronize(stream), "reset case synchronize"))
        return false;

    std::vector<float> ref_state(kStateElements, 0.0f);
    std::vector<float> ref_history(history_elements, 0.0f);
    std::vector<float> unfused_state(kStateElements);
    std::vector<float> fused_state(kStateElements);
    std::vector<float> unfused_history(history_elements);
    std::vector<float> fused_history(history_elements);
    for (size_t offset = 0; offset < kMaxTokens; offset += chunk_size) {
        const size_t tokens = std::min(chunk_size, kMaxTokens - offset);
        std::vector<float> ref_silu;
        conv_ref_chunk(input.data() + offset * kQkvChannels, tokens, kernel,
                       ref_history, ref_silu);
        std::vector<float> ref_raw = ref_silu;
        for (float &value : ref_silu)
            value /= 1.0f + std::exp(-value);
        std::vector<float> ref_q16, ref_k16, ref_v, ref_q, ref_k;
        split_ref(ref_silu, ref_q16, ref_k16, ref_v, ref_q, ref_k);
        std::vector<float> ref_output(tokens * kValueChannels, 0.0f);
        if (!q38_gdn_ref_run(
                ref_state.data(), 1, tokens, ref_q.data(), ref_k.data(),
                ref_v.data(), decay.data() + offset * Q38_GDN_VALUE_HEADS,
                beta.data() + offset * Q38_GDN_VALUE_HEADS, 0.75f,
                ref_output.data())) {
            std::fprintf(stderr, "M3-C13 scalar reference failed at chunk %zu\n",
                         offset / chunk_size);
            return false;
        }
        const bool launched =
            run_unfused(d, offset, tokens, stream, error, sizeof(error)) &&
            run_fused(d, offset, tokens, stream, error, sizeof(error)) &&
            cuda_ok(cudaStreamSynchronize(stream), "chunk synchronize");
        if (!launched) {
            std::fprintf(stderr, "M3-C13 chunk size %zu failed: %s\n",
                         chunk_size, error);
            return false;
        }

        std::vector<float> actual_raw(tokens * kQkvChannels);
        std::vector<float> actual_unfused_silu(tokens * kQkvChannels);
        std::vector<float> actual_fused_silu(tokens * kQkvChannels);
        std::vector<float> actual_unfused_q16(tokens * kKeyChannels);
        std::vector<float> actual_unfused_k16(tokens * kKeyChannels);
        std::vector<float> actual_unfused_v(tokens * kValueChannels);
        std::vector<float> actual_unfused_q(tokens * kValueChannels);
        std::vector<float> actual_unfused_k(tokens * kValueChannels);
        std::vector<float> actual_fused_q16(tokens * kKeyChannels);
        std::vector<float> actual_fused_k16(tokens * kKeyChannels);
        std::vector<float> actual_fused_v(tokens * kValueChannels);
        std::vector<float> actual_fused_q(tokens * kValueChannels);
        std::vector<float> actual_fused_k(tokens * kValueChannels);
        std::vector<float> actual_unfused_output(tokens * kValueChannels);
        std::vector<float> actual_fused_output(tokens * kValueChannels);
        bool copied =
            cuda_ok(cudaMemcpy(actual_raw.data(),
                               d.raw + offset * kQkvChannels,
                               actual_raw.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy raw convolution") &&
            cuda_ok(cudaMemcpy(actual_unfused_silu.data(),
                               d.unfused_silu + offset * kQkvChannels,
                               actual_unfused_silu.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy unfused SiLU") &&
            cuda_ok(cudaMemcpy(actual_fused_silu.data(),
                               d.fused_silu + offset * kQkvChannels,
                               actual_fused_silu.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy fused SiLU") &&
            cuda_ok(cudaMemcpy(actual_unfused_q16.data(),
                               d.unfused_q16 + offset * kKeyChannels,
                               actual_unfused_q16.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy unfused Q16") &&
            cuda_ok(cudaMemcpy(actual_unfused_k16.data(),
                               d.unfused_k16 + offset * kKeyChannels,
                               actual_unfused_k16.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy unfused K16") &&
            cuda_ok(cudaMemcpy(actual_unfused_v.data(),
                               d.unfused_v + offset * kValueChannels,
                               actual_unfused_v.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy unfused V") &&
            cuda_ok(cudaMemcpy(actual_unfused_q.data(),
                               d.unfused_q + offset * kValueChannels,
                               actual_unfused_q.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy unfused Q") &&
            cuda_ok(cudaMemcpy(actual_unfused_k.data(),
                               d.unfused_k + offset * kValueChannels,
                               actual_unfused_k.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy unfused K") &&
            cuda_ok(cudaMemcpy(actual_fused_q16.data(),
                               d.fused_q16 + offset * kKeyChannels,
                               actual_fused_q16.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy fused Q16") &&
            cuda_ok(cudaMemcpy(actual_fused_k16.data(),
                               d.fused_k16 + offset * kKeyChannels,
                               actual_fused_k16.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy fused K16") &&
            cuda_ok(cudaMemcpy(actual_fused_v.data(),
                               d.fused_v + offset * kValueChannels,
                               actual_fused_v.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy fused V") &&
            cuda_ok(cudaMemcpy(actual_fused_q.data(),
                               d.fused_q + offset * kValueChannels,
                               actual_fused_q.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy fused Q") &&
            cuda_ok(cudaMemcpy(actual_fused_k.data(),
                               d.fused_k + offset * kValueChannels,
                               actual_fused_k.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy fused K") &&
            cuda_ok(cudaMemcpy(actual_unfused_output.data(),
                               d.unfused_output + offset * kValueChannels,
                               actual_unfused_output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy unfused output") &&
            cuda_ok(cudaMemcpy(actual_fused_output.data(),
                               d.fused_output + offset * kValueChannels,
                               actual_fused_output.size() * sizeof(float),
                               cudaMemcpyDeviceToHost),
                    "copy fused output") &&
            copy_device(d.unfused_state, unfused_state, "copy unfused state") &&
            copy_device(d.fused_state, fused_state, "copy fused state") &&
            copy_device(d.unfused_history, unfused_history,
                        "copy unfused history") &&
            copy_device(d.fused_history, fused_history, "copy fused history");
        if (!copied) return false;

        const float vector_tolerance = kVectorTolerance;
        const float recurrence_tolerance = kRecurrenceTolerance;
        bool pass = true;
        pass = record_check(report.unfused_scalar[kRawConv], ref_raw,
                            actual_raw, vector_tolerance,
                            "unfused raw convolution vs scalar") &&
               pass;
        /*
         * The raw convolution is intentionally retained only by the unfused
         * path; fused output is the mathematically identical SiLU result.
         */
        pass = record_check(report.unfused_scalar[kSilu], ref_silu,
                            actual_unfused_silu, vector_tolerance,
                            "unfused SiLU vs scalar") &&
               pass;
        pass = record_check(report.fused_unfused[kSilu], actual_unfused_silu,
                            actual_fused_silu, vector_tolerance,
                            "fused SiLU vs unfused") &&
               pass;
        pass = record_check(report.fused_scalar[kSilu], ref_silu,
                            actual_fused_silu, vector_tolerance,
                            "fused SiLU vs scalar") &&
               pass;

        const std::array<std::vector<float>, 5> scalar_intermediates = {
            ref_q16, ref_k16, ref_v, ref_q, ref_k};
        const std::array<std::vector<float>, 5> unfused_intermediates = {
            actual_unfused_q16, actual_unfused_k16, actual_unfused_v,
            actual_unfused_q, actual_unfused_k};
        const std::array<std::vector<float>, 5> fused_intermediates = {
            actual_fused_q16, actual_fused_k16, actual_fused_v, actual_fused_q,
            actual_fused_k};
        const std::array<Stage, 5> intermediate_stages = {
            kQ16, kK16, kV, kQ, kK};
        for (size_t i = 0; i < intermediate_stages.size(); i++) {
            const Stage stage = intermediate_stages[i];
            pass = record_check(report.unfused_scalar[stage],
                                scalar_intermediates[i],
                                unfused_intermediates[i], vector_tolerance,
                                "unfused intermediate vs scalar") &&
                   pass;
            pass = record_check(report.fused_scalar[stage],
                                scalar_intermediates[i],
                                fused_intermediates[i], vector_tolerance,
                                "fused intermediate vs scalar") &&
                   pass;
        }
        for (size_t i = 0; i < intermediate_stages.size(); i++)
            pass = record_check(
                       report.fused_unfused[intermediate_stages[i]],
                       unfused_intermediates[i], fused_intermediates[i],
                       vector_tolerance, "fused intermediate vs unfused") &&
                   pass;

        pass = record_check(report.unfused_scalar[kRecurrence], ref_output,
                            actual_unfused_output, recurrence_tolerance,
                            "unfused recurrence output vs scalar") &&
               pass;
        pass = record_check(report.unfused_scalar[kState], ref_state,
                            unfused_state, recurrence_tolerance,
                            "unfused state vs scalar") &&
               pass;
        pass = record_check(report.unfused_scalar[kHistory], ref_history,
                            unfused_history, vector_tolerance,
                            "unfused history vs scalar") &&
               pass;
        pass = record_check(report.fused_unfused[kRecurrence],
                            actual_unfused_output, actual_fused_output,
                            recurrence_tolerance,
                            "fused recurrence output vs unfused") &&
               pass;
        pass = record_check(report.fused_scalar[kRecurrence], ref_output,
                            actual_fused_output, recurrence_tolerance,
                            "fused recurrence output vs scalar") &&
               pass;
        pass = record_check(report.fused_unfused[kState], unfused_state,
                            fused_state, recurrence_tolerance,
                            "fused state vs unfused") &&
               pass;
        pass = record_check(report.fused_scalar[kState], ref_state, fused_state,
                            recurrence_tolerance, "fused state vs scalar") &&
               pass;
        pass = record_check(report.fused_unfused[kHistory], unfused_history,
                            fused_history, vector_tolerance,
                            "fused history vs unfused") &&
               pass;
        pass = record_check(report.fused_scalar[kHistory], ref_history,
                            fused_history, vector_tolerance,
                            "fused history vs scalar") &&
               pass;
        report.boundaries++;
        report.pass = report.pass && pass;
    }
    return report.pass;
}

static bool time_path(const DeviceBuffers &d, bool fused, cudaStream_t stream,
                      std::array<float, 3> &samples) {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "create timing start") ||
        !cuda_ok(cudaEventCreate(&end), "create timing end")) {
        if (start) cudaEventDestroy(start);
        if (end) cudaEventDestroy(end);
        return false;
    }
    char error[256] = {};
    bool ok = true;
    for (size_t sample = 0; sample < samples.size(); sample++) {
        ok = cuda_ok(cudaMemsetAsync(
                         fused ? d.fused_history : d.unfused_history, 0,
                         kHistoryTokens * kQkvChannels * sizeof(float), stream),
                     "reset timing history") &&
             cuda_ok(cudaEventRecord(start, stream), "record timing start");
        if (ok) {
            if (fused)
                ok = q38_cuda_gdn_conv_silu_fused(
                    Q38_GDN_WEIGHT_F32, d.kernel, d.input, 5, kQkvChannels,
                    Q38_GDN_CONV_KERNEL, d.fused_history, d.fused_silu, stream,
                    error, sizeof(error));
            else
                ok = q38_cuda_gdn_conv(
                         Q38_GDN_WEIGHT_F32, d.kernel, d.input, 5,
                         kQkvChannels, Q38_GDN_CONV_KERNEL, d.unfused_history,
                         d.raw, stream, error, sizeof(error)) &&
                     q38_cuda_silu(d.raw, d.unfused_silu,
                                   5 * kQkvChannels, stream, error,
                                   sizeof(error));
        }
        ok = ok && cuda_ok(cudaEventRecord(end, stream), "record timing end") &&
             cuda_ok(cudaEventSynchronize(end), "synchronize timing end") &&
             cuda_ok(cudaEventElapsedTime(&samples[sample], start, end),
                     "elapsed timing");
        if (!ok) {
            std::fprintf(stderr, "M3-C13 timing path failed: %s\n", error);
            break;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ok;
}

static bool write_check(FILE *file, const Check &check) {
    return std::fprintf(file, "{\"max_abs\":%.9g,\"tolerance\":%.9g,"
                               "\"pass\":%s}",
                        check.max_abs, check.tolerance,
                        check.pass ? "true" : "false") >= 0;
}

static bool write_artifact(const char *path, const cudaDeviceProp &properties,
                           const std::vector<CaseReport> &reports,
                           const std::array<float, 3> &unfused_ms,
                           const std::array<float, 3> &fused_ms, bool pass) {
    FILE *file = std::fopen(path, "w");
    if (!file) {
        std::perror(path);
        return false;
    }
    std::fprintf(
        file,
        "{\n"
        "  \"gate\":\"M3-C13\",\n"
        "  \"status\":\"%s\",\n"
        "  \"device\":{\"name\":\"%s\",\"compute_capability\":\"%d.%d\"},\n"
        "  \"fusion\":{\"operation\":\"causal convolution + SiLU + history "
        "tail\",\"kernel\":\"gdn_conv_silu_fused_kernel\",\"history_update\":"
        "\"one channel per block; __syncthreads before thread-zero ordered "
        "tail update\",\"logical_state_unchanged\":true,\"physical_layout\":"
        "\"contiguous logical row-major; no GB10 tiling or packing\"},\n"
        "  \"launch_accounting\":{\"unfused\":{\"conv\":1,\"history\":1,"
        "\"silu\":1,\"total\":3},\"fused\":{\"conv_silu_history\":1,"
        "\"total\":1}},\n"
        "  \"cuda_event_ms\":{\"tokens\":5,\"unfused_samples\":[%g,%g,%g],"
        "\"fused_samples\":[%g,%g,%g],\"interpretation\":\"diagnostic "
        "only; launch reduction is not a speedup claim\"},\n"
        "  \"chunk_cases\":[\n",
        pass ? "pass" : "fail", properties.name, properties.major,
        properties.minor, unfused_ms[0], unfused_ms[1], unfused_ms[2],
        fused_ms[0], fused_ms[1], fused_ms[2]);
    for (size_t case_index = 0; case_index < reports.size(); case_index++) {
        const CaseReport &report = reports[case_index];
        std::fprintf(file, "    {\"chunk_size\":%zu,\"boundaries\":%zu,"
                           "\"status\":\"%s\",\"unfused_vs_scalar\":{",
                     report.chunk_size, report.boundaries,
                     report.pass ? "pass" : "fail");
        for (size_t stage = 0; stage < kStageCount; stage++) {
            std::fprintf(file, "%s\"%s\":", stage ? "," : "",
                         kStageNames[stage]);
            write_check(file, report.unfused_scalar[stage]);
        }
        std::fprintf(file, "},\"fused_vs_unfused\":{");
        for (size_t stage = 0; stage < kStageCount; stage++) {
            std::fprintf(file, "%s\"%s\":", stage ? "," : "",
                         kStageNames[stage]);
            if (stage == kRawConv)
                std::fprintf(file, "{\"not_applicable\":true}");
            else
                write_check(file, report.fused_unfused[stage]);
        }
        std::fprintf(file, "},\"fused_vs_scalar\":{");
        for (size_t stage = 0; stage < kStageCount; stage++) {
            std::fprintf(file, "%s\"%s\":", stage ? "," : "",
                         kStageNames[stage]);
            if (stage == kRawConv)
                std::fprintf(file, "{\"not_applicable\":true}");
            else
                write_check(file, report.fused_scalar[stage]);
        }
        std::fprintf(file, "}}%s\n", case_index + 1 == reports.size() ? "" : ",");
    }
    std::fprintf(file,
                 "  ],\n"
                 "  \"scope\":\"F32 deterministic GDN parity only; no "
                 "quantized weights, model upload, or GB10 physical "
                 "optimization\",\n"
                 "  \"checks\":[\"chunk sizes 1,2,4,5\","
                 "\"C07 unfused convolution/SiLU\","
                 "\"C08 recurrence\", \"scalar recurrence and history\","
                 "\"output/state/history/intermediate parity\","
                 "\"CUDA launch/runtime errors\"]\n"
                 "}\n");
    return std::fclose(file) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    const char *artifact =
        argc == 2 ? argv[1] : "artifacts/m3/cuda_profile_fused.json";
    int devices = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&devices);
    if (!cuda_ok(device_status, "cudaGetDeviceCount") || devices == 0) {
        std::fprintf(stderr, "M3-C13: CUDA device unavailable\n");
        return 2;
    }
    cudaDeviceProp properties;
    if (!cuda_ok(cudaGetDeviceProperties(&properties, 0),
                 "cudaGetDeviceProperties"))
        return 1;

    std::vector<float> input, kernel, decay, beta;
    make_fixture(input, kernel, decay, beta);
    DeviceBuffers d;
    std::vector<float *> allocations;
    auto allocate = [&](float **pointer, size_t elements,
                        const char *what) -> bool {
        if (!cuda_ok(cudaMalloc(pointer, elements * sizeof(float)), what))
            return false;
        allocations.push_back(*pointer);
        return true;
    };
    bool ok =
        allocate(&d.input, input.size(), "allocate input") &&
        allocate(&d.kernel, kernel.size(), "allocate kernel") &&
        allocate(&d.decay, decay.size(), "allocate decay") &&
        allocate(&d.beta, beta.size(), "allocate beta") &&
        allocate(&d.raw, kMaxTokens * kQkvChannels, "allocate raw") &&
        allocate(&d.unfused_silu, kMaxTokens * kQkvChannels,
                 "allocate unfused SiLU") &&
        allocate(&d.fused_silu, kMaxTokens * kQkvChannels,
                 "allocate fused SiLU") &&
        allocate(&d.unfused_q16, kMaxTokens * kKeyChannels,
                 "allocate unfused Q16") &&
        allocate(&d.unfused_k16, kMaxTokens * kKeyChannels,
                 "allocate unfused K16") &&
        allocate(&d.unfused_v, kMaxTokens * kValueChannels,
                 "allocate unfused V") &&
        allocate(&d.unfused_q, kMaxTokens * kValueChannels,
                 "allocate unfused Q") &&
        allocate(&d.unfused_k, kMaxTokens * kValueChannels,
                 "allocate unfused K") &&
        allocate(&d.fused_q16, kMaxTokens * kKeyChannels,
                 "allocate fused Q16") &&
        allocate(&d.fused_k16, kMaxTokens * kKeyChannels,
                 "allocate fused K16") &&
        allocate(&d.fused_v, kMaxTokens * kValueChannels, "allocate fused V") &&
        allocate(&d.fused_q, kMaxTokens * kValueChannels, "allocate fused Q") &&
        allocate(&d.fused_k, kMaxTokens * kValueChannels, "allocate fused K") &&
        allocate(&d.unfused_output, kMaxTokens * kValueChannels,
                 "allocate unfused output") &&
        allocate(&d.fused_output, kMaxTokens * kValueChannels,
                 "allocate fused output") &&
        allocate(&d.unfused_state, kStateElements, "allocate unfused state") &&
        allocate(&d.fused_state, kStateElements, "allocate fused state") &&
        allocate(&d.unfused_history, kHistoryTokens * kQkvChannels,
                 "allocate unfused history") &&
        allocate(&d.fused_history, kHistoryTokens * kQkvChannels,
                 "allocate fused history");
    if (!ok) {
        for (float *pointer : allocations) cudaFree(pointer);
        return 1;
    }

    cudaStream_t stream = nullptr;
    ok = cuda_ok(cudaStreamCreate(&stream), "create stream") &&
         cuda_ok(cudaMemcpy(d.input, input.data(), input.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "copy input") &&
         cuda_ok(cudaMemcpy(d.kernel, kernel.data(),
                            kernel.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "copy kernel") &&
         cuda_ok(cudaMemcpy(d.decay, decay.data(),
                            decay.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "copy decay") &&
         cuda_ok(cudaMemcpy(d.beta, beta.data(), beta.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "copy beta");
    if (!ok) {
        if (stream) cudaStreamDestroy(stream);
        for (float *pointer : allocations) cudaFree(pointer);
        return 1;
    }

    std::vector<CaseReport> reports;
    const size_t chunk_cases[] = {1, 2, 4, 5};
    for (size_t chunk_size : chunk_cases) {
        CaseReport report;
        if (!run_case(chunk_size, input, kernel, decay, beta, d, stream,
                      report))
            ok = false;
        reports.push_back(report);
    }

    std::array<float, 3> unfused_ms = {};
    std::array<float, 3> fused_ms = {};
    ok = time_path(d, false, stream, unfused_ms) &&
         time_path(d, true, stream, fused_ms) && ok;
    for (const CaseReport &report : reports) ok = report.pass && ok;
    if (ok) ok = write_artifact(artifact, properties, reports, unfused_ms,
                                fused_ms, true);

    cudaStreamDestroy(stream);
    for (float *pointer : allocations) cudaFree(pointer);
    if (!ok) return 1;
    std::printf("test_m3_gdn_fused: chunk sizes 1, 2, 4, and 5 passed; wrote %s\n",
                artifact);
    return 0;
}
