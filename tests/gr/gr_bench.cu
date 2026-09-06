#include "../../q38_gr.h"
#include "gr_reference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kWidth = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
constexpr size_t kDownSize = Q38_GR_RANK * kWidth;
constexpr size_t kUpSize = kWidth * Q38_GR_RANK;
constexpr size_t kInjectSize = Q38_GR_BRANCHES * kWidth;
constexpr size_t kWarmups = 100;
constexpr size_t kSamples = 1000;
constexpr size_t kDecompositionSamples = 16;

struct Case {
    const char *name;
    uint32_t layer;
};

constexpr Case kCases[] = {
    {"early", 0},
    {"middle", 23},
    {"late", 47},
};

struct Fixture {
    std::vector<float> residual;
    std::vector<float> block;
    std::vector<float> hc_norm;
    std::vector<float> down;
    std::vector<float> up;
    std::vector<float> inject;
    std::vector<float> expected_input;
    std::vector<float> expected_updated;
};

struct Device {
    float *residual = nullptr;
    float *gamma = nullptr;
    float *down = nullptr;
    float *up = nullptr;
    float *inject = nullptr;
    float *block = nullptr;
    float *input = nullptr;
    float *updated = nullptr;
};

struct Stats {
    double median = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static bool read_file(const std::string &path, float *data, size_t count) {
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "open %s failed\n", path.c_str());
        return false;
    }
    const bool ok = std::fread(data, sizeof(*data), count, file) == count &&
                    std::fclose(file) == 0;
    if (!ok) std::fprintf(stderr, "read %s failed\n", path.c_str());
    return ok;
}

static bool load_fixture(const std::string &root, const Case &spec,
                         Fixture &fixture) {
    char dir[256];
    std::snprintf(dir, sizeof(dir), "%s/layer_%u_%s", root.c_str(),
                  spec.layer, spec.name);
    fixture.residual.resize(kWidth);
    fixture.block.resize(Q38_GR_HIDDEN);
    fixture.hc_norm.resize(kWidth);
    fixture.down.resize(kDownSize);
    fixture.up.resize(kUpSize);
    fixture.inject.resize(kInjectSize);
    fixture.expected_input.resize(Q38_GR_HIDDEN);
    fixture.expected_updated.resize(kWidth);
    const char *names[] = {
        "residual.bin", "block_output.bin", "hc_norm.bin",
        "input_mix_down.bin", "input_mix_up.bin", "block_inject.bin",
        "expected_input.bin", "expected_updated.bin",
    };
    float *destinations[] = {
        fixture.residual.data(), fixture.block.data(), fixture.hc_norm.data(),
        fixture.down.data(), fixture.up.data(), fixture.inject.data(),
        fixture.expected_input.data(), fixture.expected_updated.data(),
    };
    const size_t counts[] = {
        kWidth, Q38_GR_HIDDEN, kWidth, kDownSize, kUpSize, kInjectSize,
        Q38_GR_HIDDEN, kWidth,
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        std::string path = std::string(dir) + "/" + names[i];
        if (!read_file(path, destinations[i], counts[i])) return false;
    }
    return true;
}

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
}

static bool alloc(Device &device) {
    return cuda_ok(cudaMalloc(&device.residual, kWidth * sizeof(float)),
                   "alloc residual") &&
           cuda_ok(cudaMalloc(&device.gamma, kWidth * sizeof(float)),
                   "alloc gamma") &&
           cuda_ok(cudaMalloc(&device.down, kDownSize * sizeof(float)),
                   "alloc down") &&
           cuda_ok(cudaMalloc(&device.up, kUpSize * sizeof(float)),
                   "alloc up") &&
           cuda_ok(cudaMalloc(&device.inject, kInjectSize * sizeof(float)),
                   "alloc inject") &&
           cuda_ok(cudaMalloc(&device.block, Q38_GR_HIDDEN * sizeof(float)),
                   "alloc block") &&
           cuda_ok(cudaMalloc(&device.input, Q38_GR_HIDDEN * sizeof(float)),
                   "alloc input") &&
           cuda_ok(cudaMalloc(&device.updated, kWidth * sizeof(float)),
                   "alloc updated");
}

static void release(Device &device) {
    cudaFree(device.residual);
    cudaFree(device.gamma);
    cudaFree(device.down);
    cudaFree(device.up);
    cudaFree(device.inject);
    cudaFree(device.block);
    cudaFree(device.input);
    cudaFree(device.updated);
    device = {};
}

static bool upload(const Fixture &fixture, Device &device) {
    std::vector<float> effective_gamma(kWidth);
    for (size_t i = 0; i < kWidth; ++i)
        effective_gamma[i] = 1.0f + fixture.hc_norm[i];
    return cuda_ok(cudaMemcpy(device.residual, fixture.residual.data(),
                              kWidth * sizeof(float), cudaMemcpyHostToDevice),
                   "upload residual") &&
           cuda_ok(cudaMemcpy(device.gamma, effective_gamma.data(),
                              kWidth * sizeof(float), cudaMemcpyHostToDevice),
                   "upload effective gamma") &&
           cuda_ok(cudaMemcpy(device.down, fixture.down.data(),
                              kDownSize * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload down") &&
           cuda_ok(cudaMemcpy(device.up, fixture.up.data(),
                              kUpSize * sizeof(float), cudaMemcpyHostToDevice),
                   "upload up") &&
           cuda_ok(cudaMemcpy(device.inject, fixture.inject.data(),
                              kInjectSize * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload inject") &&
           cuda_ok(cudaMemcpy(device.block, fixture.block.data(),
                              Q38_GR_HIDDEN * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload block");
}

static bool run_once(const Device &device) {
    char error[256] = {};
    return q38_cuda_gr_collapse(
        device.residual, device.gamma, device.down, device.up, device.inject,
        device.block, device.input, device.updated, nullptr, error,
        sizeof(error)) || (std::fprintf(stderr, "GR: %s\n", error), false);
}

static bool run_timed(const Device &device, q38_cuda_gr_timing *timing) {
    char error[256] = {};
    return q38_cuda_gr_collapse_timed(
               device.residual, device.gamma, device.down, device.up,
               device.inject, device.block, device.input, device.updated,
               nullptr, timing, error, sizeof(error)) ||
           (std::fprintf(stderr, "GR timed: %s\n", error), false);
}

static Stats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    Stats stats;
    stats.min = values.front();
    stats.max = values.back();
    stats.median = values[values.size() / 2];
    stats.p95 = values[(values.size() * 95) / 100];
    return stats;
}

static void compare(const float *actual, const float *expected, size_t count,
                    double *max_abs, double *max_rel, double *rmse,
                    size_t *nonfinite) {
    double sum_sq = 0.0;
    *max_abs = 0.0;
    *max_rel = 0.0;
    *nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) ++*nonfinite;
        const double a = actual[i], e = expected[i];
        const double abs_diff = std::fabs(a - e);
        const double rel_diff = abs_diff / std::max(1.0, std::fabs(e));
        *max_abs = std::max(*max_abs, abs_diff);
        *max_rel = std::max(*max_rel, rel_diff);
        sum_sq += abs_diff * abs_diff;
    }
    *rmse = std::sqrt(sum_sq / (double)count);
}

static bool copy_outputs(const Device &device, std::vector<float> &input,
                         std::vector<float> &updated) {
    return cuda_ok(cudaMemcpy(input.data(), device.input,
                              Q38_GR_HIDDEN * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy GR input") &&
           cuda_ok(cudaMemcpy(updated.data(), device.updated,
                              kWidth * sizeof(float), cudaMemcpyDeviceToHost),
                   "copy GR updated");
}

static void print_stats(const char *name, const Stats &kernel,
                        const Stats &host, const Stats stage[5],
                        double read_bytes, double launches, double syncs) {
    const double gbps = read_bytes / (kernel.median * 1e3);
    std::printf(
        "{\"fixture\":\"%s\",\"kernel_only_us\":{\"median\":%.6f,"
        "\"p95\":%.6f,\"min\":%.6f,\"max\":%.6f},"
        "\"end_to_end_us\":{\"median\":%.6f,\"p95\":%.6f,"
        "\"min\":%.6f,\"max\":%.6f},\"kernel_launches\":%.0f,"
        "\"explicit_host_syncs\":%.0f,\"h2d_bytes\":0,\"d2h_bytes\":0,"
        "\"d2d_bytes\":0,\"logical_bytes_read\":%.0f,"
        "\"effective_read_gbps\":%.6f,\"decomposition_us\":{"
        "\"normalize\":%.6f,\"down_projection\":%.6f,"
        "\"up_projection\":%.6f,\"branch_merge\":%.6f,"
        "\"injection\":%.6f}}\n",
        name, kernel.median, kernel.p95, kernel.min, kernel.max, host.median,
        host.p95, host.min, host.max, launches, syncs, read_bytes, gbps,
        stage[0].median, stage[1].median, stage[2].median, stage[3].median,
        stage[4].median);
}

static bool run_case(const std::string &root, const Case &spec,
                     FILE *artifact, bool *all_correct) {
    Fixture fixture;
    if (!load_fixture(root, spec, fixture)) return false;
    gr_reference_params params = {
        fixture.hc_norm.data(), fixture.down.data(), fixture.up.data(),
        fixture.inject.data()};
    std::vector<float> reference_input(Q38_GR_HIDDEN);
    std::vector<float> reference_updated(kWidth);
    if (!gr_reference_collapse(fixture.residual.data(), fixture.block.data(),
                               &params, reference_input.data(),
                               reference_updated.data()))
        return false;
    double input_abs, input_rel, input_rmse;
    double updated_abs, updated_rel, updated_rmse;
    size_t input_nonfinite, updated_nonfinite;
    compare(reference_input.data(), fixture.expected_input.data(),
            Q38_GR_HIDDEN, &input_abs, &input_rel, &input_rmse,
            &input_nonfinite);
    compare(reference_updated.data(), fixture.expected_updated.data(), kWidth,
            &updated_abs, &updated_rel, &updated_rmse, &updated_nonfinite);

    Device device;
    if (!alloc(device) || !upload(fixture, device)) {
        release(device);
        return false;
    }
    for (size_t i = 0; i < kWarmups; ++i)
        if (!run_once(device)) {
            release(device);
            return false;
        }
    if (!cuda_ok(cudaDeviceSynchronize(), "GR warmup synchronize")) {
        release(device);
        return false;
    }

    std::vector<double> stage_samples[5];
    for (auto &samples : stage_samples)
        samples.reserve(kDecompositionSamples);
    for (size_t i = 0; i < kDecompositionSamples; ++i) {
        q38_cuda_gr_timing timing = {};
        if (!run_timed(device, &timing)) {
            release(device);
            return false;
        }
        stage_samples[0].push_back((double)timing.normalize_ms * 1000.0);
        stage_samples[1].push_back(
            (double)timing.down_projection_ms * 1000.0);
        stage_samples[2].push_back((double)timing.up_projection_ms * 1000.0);
        stage_samples[3].push_back((double)timing.branch_merge_ms * 1000.0);
        stage_samples[4].push_back((double)timing.injection_ms * 1000.0);
    }
    Stats stage[5];
    for (size_t i = 0; i < 5; ++i)
        stage[i] = summarize(stage_samples[i]);

    std::vector<double> kernel_times;
    std::vector<double> host_times;
    kernel_times.reserve(kSamples);
    host_times.reserve(kSamples);
    cudaEvent_t start = nullptr, stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "create start event") ||
        !cuda_ok(cudaEventCreate(&stop), "create stop event")) {
        release(device);
        return false;
    }
    for (size_t i = 0; i < kSamples; ++i) {
        const auto host_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, nullptr);
        const bool ok = run_once(device);
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);
        const auto host_stop = std::chrono::steady_clock::now();
        if (!ok) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            release(device);
            return false;
        }
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        kernel_times.push_back((double)elapsed_ms * 1000.0);
        host_times.push_back(
            std::chrono::duration<double, std::micro>(host_stop - host_start)
                .count());
    }
    std::vector<float> actual_input(Q38_GR_HIDDEN);
    std::vector<float> actual_updated(kWidth);
    const bool copied = copy_outputs(device, actual_input, actual_updated);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    release(device);
    if (!copied) return false;

    double cuda_input_abs, cuda_input_rel, cuda_input_rmse;
    double cuda_updated_abs, cuda_updated_rel, cuda_updated_rmse;
    size_t cuda_input_nonfinite, cuda_updated_nonfinite;
    compare(actual_input.data(), fixture.expected_input.data(),
            Q38_GR_HIDDEN, &cuda_input_abs, &cuda_input_rel, &cuda_input_rmse,
            &cuda_input_nonfinite);
    compare(actual_updated.data(), fixture.expected_updated.data(), kWidth,
            &cuda_updated_abs, &cuda_updated_rel, &cuda_updated_rmse,
            &cuda_updated_nonfinite);
    const bool correct = input_abs <= 1e-4 && updated_abs <= 1e-4 &&
                         cuda_input_abs <= 2e-4 && cuda_updated_abs <= 2e-4 &&
                         cuda_input_nonfinite == 0 && cuda_updated_nonfinite == 0;
    *all_correct = *all_correct && correct;

    const double logical_bytes =
        (3.0 * kWidth + kDownSize + kUpSize + kInjectSize +
         2.0 * Q38_GR_RANK + 3.0 * kWidth + Q38_GR_HIDDEN) * sizeof(float);
    const Stats kernel = summarize(kernel_times);
    const Stats host = summarize(host_times);
    print_stats(spec.name, kernel, host, stage, logical_bytes, 5.0, 0.0);
    const double stage_total = stage[0].median + stage[1].median +
                               stage[2].median + stage[3].median +
                               stage[4].median;
    if (artifact) {
        std::fprintf(
            artifact,
            "    {\"fixture\":\"%s\",\"layer\":%u,\"correctness\":"
            "{\"cpu_input_max_abs\":%.9g,\"cpu_updated_max_abs\":%.9g,"
            "\"cuda_input_max_abs\":%.9g,\"cuda_updated_max_abs\":%.9g,"
            "\"cpu_input_rmse\":%.9g,\"cpu_updated_rmse\":%.9g,"
            "\"cuda_input_rmse\":%.9g,\"cuda_updated_rmse\":%.9g,"
            "\"nan_inf\":%s},"
            "\"kernel_only_us\":{\"median\":%.9g,\"p95\":%.9g,"
            "\"min\":%.9g,\"max\":%.9g},\"end_to_end_us\":"
            "{\"median\":%.9g,\"p95\":%.9g,\"min\":%.9g,\"max\":%.9g},"
            "\"kernel_launches\":5,\"explicit_host_syncs\":0,"
            "\"h2d_bytes\":0,\"d2h_bytes\":0,\"d2d_bytes\":0,"
            "\"logical_bytes_read\":%.0f,\"effective_read_gbps\":%.9g,"
            "\"decomposition_us\":{\"normalize\":{\"median\":%.9g,"
            "\"p95\":%.9g,\"share\":%.9g},\"down_projection\":"
            "{\"median\":%.9g,\"p95\":%.9g,\"share\":%.9g},"
            "\"up_projection\":{\"median\":%.9g,\"p95\":%.9g,"
            "\"share\":%.9g},\"branch_merge\":{\"median\":%.9g,"
            "\"p95\":%.9g,\"share\":%.9g},\"injection\":"
            "{\"median\":%.9g,\"p95\":%.9g,\"share\":%.9g}}}",
            spec.name, spec.layer, input_abs, updated_abs, cuda_input_abs,
            cuda_updated_abs, input_rmse, updated_rmse, cuda_input_rmse,
            cuda_updated_rmse,
            (input_nonfinite || updated_nonfinite || cuda_input_nonfinite ||
             cuda_updated_nonfinite) ? "true" : "false",
            kernel.median, kernel.p95, kernel.min, kernel.max, host.median,
            host.p95, host.min, host.max, logical_bytes,
            logical_bytes / (kernel.median * 1e3),
            stage[0].median, stage[0].p95, stage[0].median / stage_total,
            stage[1].median, stage[1].p95, stage[1].median / stage_total,
            stage[2].median, stage[2].p95, stage[2].median / stage_total,
            stage[3].median, stage[3].p95, stage[3].median / stage_total,
            stage[4].median, stage[4].p95, stage[4].median / stage_total);
    }
    return correct;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s FIXTURE_ROOT [OUTPUT_JSON]\n", argv[0]);
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "no CUDA device available\n");
        return 2;
    }
    FILE *artifact = nullptr;
    if (argc == 3) {
        artifact = std::fopen(argv[2], "w");
        if (!artifact) {
            std::fprintf(stderr, "open output %s failed\n", argv[2]);
            return 1;
        }
        std::fprintf(
            artifact,
            "{\n  \"format\":\"q38-gr-reference-v1\",\n"
            "  \"contract\":\"tests/gr/GR_CONTRACT.md\",\n"
            "  \"optimized\":false,\n  \"warmup_iterations\":%zu,\n"
            "  \"measured_iterations\":%zu,\n  \"fixtures\":[\n",
            kWarmups, kSamples);
    }
    bool all_correct = true;
    bool ok = true;
    for (size_t i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
        if (artifact && i) std::fputs(",\n", artifact);
        ok = run_case(argv[1], kCases[i], artifact, &all_correct) && ok;
    }
    if (artifact) {
        std::fprintf(artifact, "\n  ],\n  \"all_correct\":%s\n}\n",
                     all_correct ? "true" : "false");
        if (std::fclose(artifact) != 0) ok = false;
    }
    return ok && all_correct ? 0 : 1;
}
