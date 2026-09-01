#include "q38_gdn.h"
#include "q38_gdn_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kTokens = 17;
constexpr size_t kQkvChannels = Q38_GDN_QKV_CHANNELS;
constexpr size_t kKeyChannels = Q38_GDN_KEY_CHANNELS;
constexpr size_t kValueChannels = Q38_GDN_VALUE_CHANNELS;
constexpr size_t kHistoryTokens = Q38_GDN_CONV_KERNEL - 1u;
constexpr size_t kStateElements =
    Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM;
constexpr float kOutputTolerance = 5e-5f;
constexpr float kStateTolerance = 5e-5f;
constexpr float kHistoryTolerance = 2e-5f;

struct Check {
    float max_abs = 0.0f;
    float tolerance = 0.0f;
    bool pass = true;
};

struct CaseStats {
    std::string name;
    std::vector<size_t> parts;
    size_t boundaries = 0;
    Check output;
    Check state;
    Check history;
    Check final_output;
    Check final_state;
    Check final_history;
};

struct RunResult {
    std::vector<float> output;
    std::vector<float> state;
    std::vector<float> history;
    bool pass = true;
};

struct DeviceBuffers {
    float *input = nullptr;
    float *kernel = nullptr;
    float *conv = nullptr;
    float *q16 = nullptr;
    float *k16 = nullptr;
    float *v = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *decay = nullptr;
    float *beta = nullptr;
    float *recurrent = nullptr;
    float *state = nullptr;
    float *history = nullptr;
};

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
}

static float max_abs(const float *expected, const float *actual, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; i++)
        result = std::fmax(result, std::fabs(expected[i] - actual[i]));
    return result;
}

static bool update_check(Check &check, const float *expected,
                         const float *actual, size_t count, float tolerance,
                         const char *label) {
    const float difference = max_abs(expected, actual, count);
    check.max_abs = std::fmax(check.max_abs, difference);
    check.tolerance = tolerance;
    if (difference > tolerance) {
        check.pass = false;
        std::fprintf(stderr, "M3-C11 %s mismatch: max_abs=%g tolerance=%g\n",
                     label, difference, tolerance);
    }
    return difference <= tolerance;
}

static void make_fixture(std::vector<float> &input, std::vector<float> &kernel,
                         std::vector<float> &decay,
                         std::vector<float> &beta) {
    input.resize(kTokens * kQkvChannels);
    kernel.resize(Q38_GDN_CONV_KERNEL * kQkvChannels);
    decay.resize(kTokens * Q38_GDN_VALUE_HEADS);
    beta.resize(kTokens * Q38_GDN_VALUE_HEADS);
    for (size_t token = 0; token < kTokens; token++) {
        for (size_t channel = 0; channel < kQkvChannels; channel++) {
            const size_t code =
                (token * 37u + channel * 13u + 11u) % 101u;
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
    for (size_t token = 0; token < tokens; token++) {
        for (size_t channel = 0; channel < kQkvChannels; channel++) {
            float sum = 0.0f;
            for (size_t tap = 0; tap < Q38_GDN_CONV_KERNEL; tap++) {
                const size_t source = kHistoryTokens + token -
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
    }
    for (size_t tail = 0; tail < kHistoryTokens; tail++) {
        const size_t source = tokens + tail;
        for (size_t channel = 0; channel < kQkvChannels; channel++)
            history[tail * kQkvChannels + channel] =
                source < kHistoryTokens
                    ? history[source * kQkvChannels + channel]
                    : input[(source - kHistoryTokens) * kQkvChannels +
                            channel];
    }
}

static void silu(std::vector<float> &values) {
    for (float &value : values)
        value *= 1.0f / (1.0f + std::exp(-value));
}

static void split_repeat_ref(const std::vector<float> &mixed,
                             std::vector<float> &q, std::vector<float> &k,
                             std::vector<float> &v) {
    const size_t tokens = mixed.size() / kQkvChannels;
    q.assign(tokens * kValueChannels, 0.0f);
    k.assign(tokens * kValueChannels, 0.0f);
    v.assign(tokens * kValueChannels, 0.0f);
    for (size_t token = 0; token < tokens; token++) {
        const float *source = mixed.data() + token * kQkvChannels;
        const float *q16 = source;
        const float *k16 = source + kKeyChannels;
        const float *v_source = source + 2u * kKeyChannels;
        std::memcpy(v.data() + token * kValueChannels, v_source,
                    kValueChannels * sizeof(float));
        q38_gdn_ref_repeat_key_heads(q16,
                                      q.data() + token * kValueChannels);
        q38_gdn_ref_repeat_key_heads(k16,
                                      k.data() + token * kValueChannels);
    }
}

static bool reset_device(const DeviceBuffers &device, cudaStream_t stream,
                         char *error, size_t error_len) {
    return q38_cuda_gdn_recurrence_reset(device.state, stream, error,
                                          error_len) &&
           cuda_ok(cudaMemsetAsync(
                       device.history, 0,
                       kHistoryTokens * kQkvChannels * sizeof(float), stream),
                   "clear convolution history") &&
           cuda_ok(cudaStreamSynchronize(stream), "reset synchronize");
}

static bool copy_device(const float *device, float *host, size_t elements,
                        const char *what) {
    return cuda_ok(cudaMemcpy(host, device, elements * sizeof(float),
                               cudaMemcpyDeviceToHost),
                   what);
}

static bool run_partition(const std::vector<size_t> &parts,
                          const std::vector<float> &input,
                          const std::vector<float> &kernel,
                          const std::vector<float> &decay,
                          const std::vector<float> &beta,
                          const DeviceBuffers &device, cudaStream_t stream,
                          CaseStats &stats, RunResult &result) {
    char error[256] = {};
    if (!reset_device(device, stream, error, sizeof(error))) {
        std::fprintf(stderr, "M3-C11 %s reset failed: %s\n", stats.name.c_str(),
                     error);
        return false;
    }
    result.output.assign(kTokens * kValueChannels, 0.0f);
    result.state.assign(kStateElements, 0.0f);
    result.history.assign(kHistoryTokens * kQkvChannels, 0.0f);
    std::vector<float> ref_full_output(kTokens * kValueChannels, 0.0f);
    std::vector<float> ref_state(kStateElements, 0.0f);
    std::vector<float> ref_history(kHistoryTokens * kQkvChannels, 0.0f);
    size_t offset = 0;
    bool ok = true;
    for (size_t part_index = 0; part_index < parts.size(); part_index++) {
        const size_t tokens = parts[part_index];
        if (!tokens || offset + tokens > kTokens) {
            std::fprintf(stderr, "M3-C11 %s has invalid partition\n",
                         stats.name.c_str());
            return false;
        }

        std::vector<float> ref_conv;
        conv_ref_chunk(input.data() + offset * kQkvChannels, tokens, kernel,
                       ref_history, ref_conv);
        silu(ref_conv);
        std::vector<float> ref_q, ref_k, ref_v;
        split_repeat_ref(ref_conv, ref_q, ref_k, ref_v);
        std::vector<float> ref_output(tokens * kValueChannels, 0.0f);
        if (!q38_gdn_ref_run(
                ref_state.data(), 1, tokens, ref_q.data(), ref_k.data(),
                ref_v.data(), decay.data() + offset * Q38_GDN_VALUE_HEADS,
                beta.data() + offset * Q38_GDN_VALUE_HEADS, 0.75f,
                ref_output.data())) {
            std::fprintf(stderr, "M3-C11 %s scalar reference failed\n",
                         stats.name.c_str());
            return false;
        }

        const bool chunk_ok = q38_cuda_gdn_conv_silu(
                 Q38_GDN_WEIGHT_F32, device.kernel,
                 device.input + offset * kQkvChannels, tokens, kQkvChannels,
                 Q38_GDN_CONV_KERNEL, device.history,
                 device.conv + offset * kQkvChannels, stream, error,
                 sizeof(error)) &&
             q38_cuda_gdn_split_qkv(
                 device.conv + offset * kQkvChannels, tokens,
                 device.q16 + offset * kKeyChannels,
                 device.k16 + offset * kKeyChannels,
                 device.v + offset * kValueChannels, stream, error,
                 sizeof(error)) &&
             q38_cuda_gdn_repeat_key_heads(
                 device.q16 + offset * kKeyChannels, tokens,
                 device.q + offset * kValueChannels, stream, error,
                 sizeof(error)) &&
             q38_cuda_gdn_repeat_key_heads(
                 device.k16 + offset * kKeyChannels, tokens,
                 device.k + offset * kValueChannels, stream, error,
                 sizeof(error)) &&
             q38_cuda_gdn_recurrence(
                 device.state, tokens, device.q + offset * kValueChannels,
                 device.k + offset * kValueChannels,
                 device.v + offset * kValueChannels,
                 device.decay + offset * Q38_GDN_VALUE_HEADS,
                 device.beta + offset * Q38_GDN_VALUE_HEADS, 0.75f,
                 device.recurrent + offset * kValueChannels, stream, error,
                 sizeof(error)) &&
             cuda_ok(cudaStreamSynchronize(stream), "chunk synchronize");
        ok = ok && chunk_ok;
        if (!chunk_ok) {
            std::fprintf(stderr, "M3-C11 %s chunk %zu failed: %s\n",
                         stats.name.c_str(), part_index, error);
            return false;
        }

        std::vector<float> actual_output(tokens * kValueChannels);
        std::vector<float> actual_state(kStateElements);
        std::vector<float> actual_history(kHistoryTokens * kQkvChannels);
        const bool copy_ok =
            copy_device(device.recurrent + offset * kValueChannels,
                        actual_output.data(), actual_output.size(),
                        "copy chunk output") &&
            copy_device(device.state, actual_state.data(), actual_state.size(),
                        "copy chunk state") &&
            copy_device(device.history, actual_history.data(),
                        actual_history.size(), "copy chunk history");
        ok = ok && copy_ok;
        if (!copy_ok) return false;
        stats.boundaries++;
        const bool boundary_ok =
            update_check(stats.output, ref_output.data(), actual_output.data(),
                         actual_output.size(), kOutputTolerance,
                         (stats.name + " boundary output").c_str()) &&
            update_check(stats.state, ref_state.data(), actual_state.data(),
                         actual_state.size(), kStateTolerance,
                         (stats.name + " boundary state").c_str()) &&
            update_check(stats.history, ref_history.data(),
                         actual_history.data(), actual_history.size(),
                         kHistoryTolerance,
                         (stats.name + " boundary history").c_str());
        ok = ok && boundary_ok;
        std::memcpy(ref_full_output.data() + offset * kValueChannels,
                    ref_output.data(), ref_output.size() * sizeof(float));
        std::memcpy(result.output.data() + offset * kValueChannels,
                    actual_output.data(), actual_output.size() * sizeof(float));
        offset += tokens;
    }
    if (offset != kTokens) {
        std::fprintf(stderr, "M3-C11 %s does not cover all tokens\n",
                     stats.name.c_str());
        return false;
    }
    std::vector<float> actual_state(kStateElements);
    std::vector<float> actual_history(kHistoryTokens * kQkvChannels);
    if (!copy_device(device.state, actual_state.data(), actual_state.size(),
                     "copy final state") ||
        !copy_device(device.history, actual_history.data(),
                     actual_history.size(), "copy final history"))
        return false;
    result.state = actual_state;
    result.history = actual_history;
    const bool final_ok =
        update_check(stats.final_output, ref_full_output.data(),
                     result.output.data(), result.output.size(),
                     kOutputTolerance,
                     (stats.name + " final output").c_str()) &&
        update_check(stats.final_state, ref_state.data(), actual_state.data(),
                     actual_state.size(), kStateTolerance,
                     (stats.name + " final state").c_str()) &&
        update_check(stats.final_history, ref_history.data(),
                     actual_history.data(), actual_history.size(),
                     kHistoryTolerance,
                     (stats.name + " final history").c_str());
    result.pass = ok && final_ok;
    return result.pass;
}

static std::vector<size_t> random_partition(uint32_t seed) {
    std::vector<size_t> result;
    size_t remaining = kTokens;
    while (remaining) {
        seed = seed * 1664525u + 1013904223u;
        size_t part = 1u + (seed % 7u);
        if (part > remaining) part = remaining;
        result.push_back(part);
        remaining -= part;
    }
    return result;
}

static bool compare_result(const RunResult &expected, const RunResult &actual,
                          CaseStats &stats, const char *label) {
    return update_check(stats.final_output, expected.output.data(),
                        actual.output.data(), expected.output.size(),
                        kOutputTolerance,
                        (stats.name + " " + label + " output").c_str()) &&
           update_check(stats.final_state, expected.state.data(),
                        actual.state.data(), expected.state.size(),
                        kStateTolerance,
                        (stats.name + " " + label + " state").c_str()) &&
           update_check(stats.final_history, expected.history.data(),
                        actual.history.data(), expected.history.size(),
                        kHistoryTolerance,
                        (stats.name + " " + label + " history").c_str());
}

static void write_check(FILE *file, const Check &check) {
    std::fprintf(file,
                 "{\"max_abs\":%.9g,\"tolerance\":%.9g,\"pass\":%s}",
                 check.max_abs, check.tolerance, check.pass ? "true" : "false");
}

static bool write_artifact(const char *path, const std::vector<CaseStats> &stats,
                           const Check &reset_output, const Check &reset_state,
                           const Check &reset_history, bool all_pass) {
    FILE *file = std::fopen(path, "w");
    if (!file) {
        std::perror(path);
        return false;
    }
    std::fprintf(file,
                 "{\n"
                 "  \"gate\":\"M3-C11\",\n"
                 "  \"status\":\"%s\",\n"
                 "  \"tokens\":%zu,\n"
                 "  \"dtype\":\"F32\",\n"
                 "  \"single_sequence\":true,\n"
                 "  \"logical_state\":[1,48,128,128],\n"
                 "  \"conv_history\":[1,3,10240],\n"
                 "  \"fixture\":\"deterministic F32 reference/simple GDN stream\",\n"
                 "  \"partition_cases\":[\n",
                 all_pass ? "pass" : "fail", kTokens);
    for (size_t i = 0; i < stats.size(); i++) {
        const CaseStats &item = stats[i];
        std::fprintf(file, "    {\"name\":\"%s\",\"parts\":[", item.name.c_str());
        for (size_t p = 0; p < item.parts.size(); p++)
            std::fprintf(file, "%s%zu", p ? "," : "", item.parts[p]);
        std::fprintf(file, "],\"boundaries\":%zu,\"boundary_checks\":{"
                           "\"output\":",
                     item.boundaries);
        write_check(file, item.output);
        std::fprintf(file, ",\"state\":");
        write_check(file, item.state);
        std::fprintf(file, ",\"conv_history\":");
        write_check(file, item.history);
        std::fprintf(file, "},\"final_checks\":{\"output\":");
        write_check(file, item.final_output);
        std::fprintf(file, ",\"state\":");
        write_check(file, item.final_state);
        std::fprintf(file, ",\"conv_history\":");
        write_check(file, item.final_history);
        std::fprintf(file, "}}%s\n", i + 1 == stats.size() ? "" : ",");
    }
    std::fprintf(file,
                 "  ],\n"
                 "  \"reset_determinism\":{\"output\":");
    write_check(file, reset_output);
    std::fprintf(file, ",\"state\":");
    write_check(file, reset_state);
    std::fprintf(file, ",\"conv_history\":");
    write_check(file, reset_history);
    std::fprintf(file,
                 "},\n"
                 "  \"reference\":\"q38_gdn_ref; CUDA C07 conv_silu/split/repeat "
                 "and C08 recurrence APIs only\",\n"
                 "  \"no_gb10_optimization\":true\n"
                 "}\n");
    return std::fclose(file) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    int devices = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&devices);
    if (device_status != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "M3-C11: CUDA device unavailable: %s\n",
                     cudaGetErrorString(device_status));
        return 2;
    }

    std::vector<float> input, kernel, decay, beta;
    make_fixture(input, kernel, decay, beta);
    DeviceBuffers device;
    auto allocate = [](float **pointer, size_t elements,
                       const char *what) -> bool {
        return cuda_ok(cudaMalloc(pointer, elements * sizeof(float)), what);
    };
    bool ok =
        allocate(&device.input, input.size(), "alloc input") &&
        allocate(&device.kernel, kernel.size(), "alloc kernel") &&
        allocate(&device.conv, input.size(), "alloc convolution output") &&
        allocate(&device.q16, kTokens * kKeyChannels, "alloc Q16") &&
        allocate(&device.k16, kTokens * kKeyChannels, "alloc K16") &&
        allocate(&device.v, kTokens * kValueChannels, "alloc V") &&
        allocate(&device.q, kTokens * kValueChannels, "alloc repeated Q") &&
        allocate(&device.k, kTokens * kValueChannels, "alloc repeated K") &&
        allocate(&device.decay, decay.size(), "alloc decay") &&
        allocate(&device.beta, beta.size(), "alloc beta") &&
        allocate(&device.recurrent, kTokens * kValueChannels,
                 "alloc recurrence output") &&
        allocate(&device.state, kStateElements, "alloc recurrent state") &&
        allocate(&device.history, kHistoryTokens * kQkvChannels,
                 "alloc convolution history");
    cudaStream_t stream = nullptr;
    if (ok) ok = cuda_ok(cudaStreamCreate(&stream), "create stream");
    if (ok)
        ok = cuda_ok(cudaMemcpy(device.input, input.data(),
                                input.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy input") &&
             cuda_ok(cudaMemcpy(device.kernel, kernel.data(),
                                kernel.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy kernel") &&
             cuda_ok(cudaMemcpy(device.decay, decay.data(),
                                decay.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy decay") &&
             cuda_ok(cudaMemcpy(device.beta, beta.data(),
                                beta.size() * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "copy beta");

    std::vector<CaseStats> cases;
    std::vector<size_t> one_chunk = {kTokens};
    std::vector<size_t> one_token(kTokens, 1u);
    std::vector<size_t> small = {2u, 4u, 5u, 6u};
    std::vector<size_t> mixed = {1u, 3u, 4u, 5u, 4u};
    cases.push_back({"one_chunk", one_chunk});
    cases.push_back({"one_token", one_token});
    cases.push_back({"small_boundaries", small});
    cases.push_back({"mixed_boundaries", mixed});
    for (uint32_t seed = 0x11u; seed <= 0x44u; seed += 0x11u)
        cases.push_back({"random_" + std::to_string(seed),
                         random_partition(seed)});

    std::vector<RunResult> results(cases.size());
    if (ok) {
        for (size_t i = 0; i < cases.size() && ok; i++)
            ok = run_partition(cases[i].parts, input, kernel, decay, beta,
                               device, stream, cases[i], results[i]);
    }
    if (ok) {
        for (size_t i = 1; i < cases.size(); i++)
            ok = compare_result(results[0], results[i], cases[i],
                                "vs one_chunk") &&
                 ok;
    }

    Check reset_output, reset_state, reset_history;
    if (ok) {
        RunResult reset_first, reset_second;
        CaseStats reset_first_stats{"reset_first", mixed};
        CaseStats reset_second_stats{"reset_second", mixed};
        ok = run_partition(mixed, input, kernel, decay, beta, device, stream,
                           reset_first_stats, reset_first) &&
             run_partition(mixed, input, kernel, decay, beta, device, stream,
                           reset_second_stats, reset_second);
        if (ok) {
            ok = update_check(reset_output, reset_first.output.data(),
                              reset_second.output.data(), reset_first.output.size(),
                              0.0f, "reset output") &&
                 update_check(reset_state, reset_first.state.data(),
                              reset_second.state.data(), reset_first.state.size(),
                              0.0f, "reset state") &&
                 update_check(reset_history, reset_first.history.data(),
                              reset_second.history.data(),
                              reset_first.history.size(), 0.0f,
                              "reset history") &&
                 ok;
        }
    }

    bool all_pass = ok;
    for (const CaseStats &item : cases)
        all_pass = all_pass && item.output.pass && item.state.pass &&
                   item.history.pass && item.final_output.pass &&
                   item.final_state.pass && item.final_history.pass;
    all_pass = all_pass && reset_output.pass && reset_state.pass &&
               reset_history.pass;
    if (all_pass && argc == 2)
        all_pass = write_artifact(argv[1], cases, reset_output, reset_state,
                                  reset_history, all_pass);

    cudaFree(device.input);
    cudaFree(device.kernel);
    cudaFree(device.conv);
    cudaFree(device.q16);
    cudaFree(device.k16);
    cudaFree(device.v);
    cudaFree(device.q);
    cudaFree(device.k);
    cudaFree(device.decay);
    cudaFree(device.beta);
    cudaFree(device.recurrent);
    cudaFree(device.state);
    cudaFree(device.history);
    if (stream) cudaStreamDestroy(stream);
    if (!all_pass) {
        std::fprintf(stderr, "test_m3_chunk_invariance: failed\n");
        return 1;
    }
    std::puts("test_m3_chunk_invariance: one-chunk, boundary, random, and reset invariance passed");
    return 0;
}
