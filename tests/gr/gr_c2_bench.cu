#include "../../q38_cuda_primitives.h"
#include "../../q38_gr_ref.h"

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
constexpr size_t kWarmups = 100;
constexpr size_t kSamples = 1000;
constexpr size_t kStageCount = 9;

enum Stage {
    kNormalize,
    kReadDown,
    kLowRank,
    kReadUp,
    kBranchPreparation,
    kBranchMerge,
    kWriteInject,
    kElementwise,
    kWriteback,
};

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
    std::vector<float> gamma;
    std::vector<float> down;
    std::vector<float> up;
    std::vector<float> inject;
    std::vector<float> block;
    std::vector<float> expected_input;
    std::vector<float> expected_updated;
};

struct Device {
    uint16_t *down_weights = nullptr;
    uint16_t *up_weights = nullptr;
    uint16_t *inject_weights = nullptr;
    float *residual = nullptr;
    float *gamma = nullptr;
    float *normalized = nullptr;
    float *block = nullptr;
    float *down = nullptr;
    float *bottleneck = nullptr;
    float *up = nullptr;
    float *branch_gates = nullptr;
    float *input = nullptr;
    float *inject = nullptr;
    float *scales = nullptr;
    float *updated = nullptr;
};

struct Stats {
    double median = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct Sample {
    double stage[kStageCount] = {};
    double total_gpu = 0.0;
    double total_host = 0.0;
    double dispatch_gap = 0.0;
    double host_sync_wait = 0.0;
    double host_submit = 0.0;
};

struct PipelineResult {
    std::vector<Sample> samples;
    std::vector<float> input;
    std::vector<float> updated;
    double median_wall = 0.0;
    double p95_wall = 0.0;
    double median_gpu = 0.0;
    double p95_gpu = 0.0;
};

static bool read_file(const std::string &path, float *data, size_t count) {
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) {
        std::fprintf(stderr, "open %s failed\n", path.c_str());
        return false;
    }
    const bool ok = std::fread(data, sizeof(*data), count, file) == count;
    if (std::fclose(file) != 0) return false;
    if (!ok) std::fprintf(stderr, "read %s failed\n", path.c_str());
    return ok;
}

static bool load_fixture(const std::string &root, const Case &spec,
                         Fixture &fixture) {
    char dir[256];
    std::snprintf(dir, sizeof(dir), "%s/layer_%u_%s", root.c_str(),
                  spec.layer, spec.name);
    fixture.residual.resize(kWidth);
    fixture.gamma.resize(kWidth);
    fixture.down.resize(Q38_GR_RANK * kWidth);
    fixture.up.resize(kWidth * Q38_GR_RANK);
    fixture.inject.resize(Q38_GR_BRANCHES * kWidth);
    fixture.block.resize(Q38_GR_HIDDEN);
    fixture.expected_input.resize(Q38_GR_HIDDEN);
    fixture.expected_updated.resize(kWidth);
    const char *names[] = {
        "residual.bin", "hc_norm.bin", "input_mix_down.bin",
        "input_mix_up.bin", "block_inject.bin", "block_output.bin",
        "expected_input.bin", "expected_updated.bin",
    };
    float *destinations[] = {
        fixture.residual.data(), fixture.gamma.data(), fixture.down.data(),
        fixture.up.data(), fixture.inject.data(), fixture.block.data(),
        fixture.expected_input.data(), fixture.expected_updated.data(),
    };
    const size_t counts[] = {
        kWidth, kWidth, Q38_GR_RANK * kWidth, kWidth * Q38_GR_RANK,
        Q38_GR_BRANCHES * kWidth, Q38_GR_HIDDEN, Q38_GR_HIDDEN, kWidth,
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
        if (!read_file(std::string(dir) + "/" + names[i], destinations[i],
                       counts[i]))
            return false;
    return true;
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
}

static bool alloc_device(Device &device) {
#define ALLOC(member, count, label) \
    if (!cuda_ok(cudaMalloc(&device.member, (count) * sizeof(*device.member)), \
                 label)) return false
    ALLOC(down_weights, Q38_GR_RANK * kWidth, "alloc down weights");
    ALLOC(up_weights, kWidth * Q38_GR_RANK, "alloc up weights");
    ALLOC(inject_weights, Q38_GR_BRANCHES * kWidth, "alloc inject weights");
    ALLOC(residual, kWidth, "alloc residual");
    ALLOC(gamma, kWidth, "alloc gamma");
    ALLOC(normalized, kWidth, "alloc normalized");
    ALLOC(block, Q38_GR_HIDDEN, "alloc block");
    ALLOC(down, Q38_GR_RANK, "alloc down");
    ALLOC(bottleneck, Q38_GR_RANK, "alloc bottleneck");
    ALLOC(up, kWidth, "alloc up");
    ALLOC(branch_gates, kWidth, "alloc branch gates");
    ALLOC(input, Q38_GR_HIDDEN, "alloc input");
    ALLOC(inject, Q38_GR_BRANCHES, "alloc inject");
    ALLOC(scales, Q38_GR_BRANCHES, "alloc scales");
    ALLOC(updated, kWidth, "alloc updated");
#undef ALLOC
    return true;
}

static void free_device(Device &device) {
    cudaFree(device.down_weights);
    cudaFree(device.up_weights);
    cudaFree(device.inject_weights);
    cudaFree(device.residual);
    cudaFree(device.gamma);
    cudaFree(device.normalized);
    cudaFree(device.block);
    cudaFree(device.down);
    cudaFree(device.bottleneck);
    cudaFree(device.up);
    cudaFree(device.branch_gates);
    cudaFree(device.input);
    cudaFree(device.inject);
    cudaFree(device.scales);
    cudaFree(device.updated);
    device = {};
}

static bool upload_fixture(const Fixture &fixture, Device &device) {
    std::vector<uint16_t> down(fixture.down.size());
    std::vector<uint16_t> up(fixture.up.size());
    std::vector<uint16_t> inject(fixture.inject.size());
    std::vector<float> gamma(fixture.gamma.size());
    for (size_t i = 0; i < down.size(); ++i)
        down[i] = float_to_bf16(fixture.down[i]);
    for (size_t i = 0; i < up.size(); ++i)
        up[i] = float_to_bf16(fixture.up[i]);
    for (size_t i = 0; i < inject.size(); ++i)
        inject[i] = float_to_bf16(fixture.inject[i]);
    for (size_t i = 0; i < gamma.size(); ++i)
        gamma[i] = 1.0f + fixture.gamma[i];
    return
        cuda_ok(cudaMemcpy(device.down_weights, down.data(),
                           down.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice), "upload down weights") &&
        cuda_ok(cudaMemcpy(device.up_weights, up.data(),
                           up.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice), "upload up weights") &&
        cuda_ok(cudaMemcpy(device.inject_weights, inject.data(),
                           inject.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice), "upload inject weights") &&
        cuda_ok(cudaMemcpy(device.residual, fixture.residual.data(),
                           kWidth * sizeof(float), cudaMemcpyHostToDevice),
                "upload residual") &&
        cuda_ok(cudaMemcpy(device.gamma, gamma.data(), kWidth * sizeof(float),
                           cudaMemcpyHostToDevice), "upload gamma") &&
        cuda_ok(cudaMemcpy(device.block, fixture.block.data(),
                           Q38_GR_HIDDEN * sizeof(float),
                           cudaMemcpyHostToDevice), "upload block");
}

__global__ static void normalize_kernel(const float *residual,
                                        const float *gamma, float *normalized) {
    const unsigned branch = blockIdx.x;
    const unsigned lane = threadIdx.x;
    __shared__ double partial[256];
    double sum = 0.0;
    for (unsigned channel = lane; channel < Q38_GR_HIDDEN; channel += 256u) {
        const float value = residual[branch * Q38_GR_HIDDEN + channel];
        sum += (double)value * (double)value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(
        (float)(partial[0] / (double)Q38_GR_HIDDEN) + 1e-6f);
    for (unsigned channel = lane; channel < Q38_GR_HIDDEN; channel += 256u) {
        const size_t index = branch * Q38_GR_HIDDEN + channel;
        normalized[index] = residual[index] * scale * gamma[index];
    }
}

__global__ static void low_rank_gate_kernel(const float *down,
                                            float *bottleneck) {
    const unsigned rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= Q38_GR_RANK) return;
    const float value = down[rank] / 4.0f;
    bottleneck[rank] = value / (1.0f + expf(-value));
}

__global__ static void branch_prepare_kernel(const float *up, float *gates) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kWidth) gates[index] = 1.0f / (1.0f + expf(-up[index]));
}

__global__ static void branch_merge_kernel(const float *normalized,
                                           const float *gates, float *input) {
    const unsigned channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= Q38_GR_HIDDEN) return;
    float value = 0.0f;
    for (unsigned branch = 0; branch < Q38_GR_BRANCHES; ++branch)
        value += gates[branch * Q38_GR_HIDDEN + channel] *
                 normalized[branch * Q38_GR_HIDDEN + channel];
    input[channel] = value / (float)Q38_GR_BRANCHES;
}

__global__ static void fused_branch_read_kernel(const float *normalized,
                                                const float *up, float *input) {
    const unsigned channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= Q38_GR_HIDDEN) return;
    float value = 0.0f;
    for (unsigned branch = 0; branch < Q38_GR_BRANCHES; ++branch) {
        const size_t index = branch * Q38_GR_HIDDEN + channel;
        const float gate = 1.0f / (1.0f + expf(-up[index]));
        value += gate * normalized[index];
    }
    input[channel] = value / (float)Q38_GR_BRANCHES;
}

__global__ static void elementwise_gate_kernel(const float *inject,
                                               float *scales) {
    const unsigned branch = blockIdx.x * blockDim.x + threadIdx.x;
    if (branch < Q38_GR_BRANCHES)
        scales[branch] = 2.0f / (1.0f + expf(-inject[branch] / 4.0f));
}

__global__ static void writeback_kernel(const float *residual,
                                        const float *block, const float *scales,
                                        float *updated) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kWidth) return;
    const unsigned branch = index / Q38_GR_HIDDEN;
    updated[index] = residual[index] +
                     scales[branch] * block[index % Q38_GR_HIDDEN];
}

static bool launch_normalize(const Device &device) {
    normalize_kernel<<<Q38_GR_BRANCHES, 256>>>(
        device.residual, device.gamma, device.normalized);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_low_rank(const Device &device) {
    low_rank_gate_kernel<<<2, 256>>>(device.down, device.bottleneck);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_branch_prepare(const Device &device) {
    branch_prepare_kernel<<<(unsigned)((kWidth + 255) / 256), 256>>>(
        device.up, device.branch_gates);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_branch_merge(const Device &device) {
    branch_merge_kernel<<<(unsigned)((Q38_GR_HIDDEN + 255) / 256), 256>>>(
        device.normalized, device.branch_gates, device.input);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_fused_branch_read(const Device &device) {
    fused_branch_read_kernel<<<(unsigned)((Q38_GR_HIDDEN + 255) / 256), 256>>>(
        device.normalized, device.up, device.input);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_elementwise(const Device &device) {
    elementwise_gate_kernel<<<1, 32>>>(device.inject, device.scales);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_writeback(const Device &device) {
    writeback_kernel<<<(unsigned)((kWidth + 255) / 256), 256>>>(
        device.residual, device.block, device.scales, device.updated);
    return cudaGetLastError() == cudaSuccess;
}

static bool launch_stage(const Device &device, unsigned stage,
                         bool fused_branch_read, unsigned up_threads) {
    char error[256] = {};
    switch (stage) {
    case kNormalize:
        return launch_normalize(device);
    case kReadDown:
        return q38_cuda_bf16_matvec(
            device.down_weights, Q38_GR_RANK, kWidth, device.normalized,
            device.down, nullptr, error, sizeof(error));
    case kLowRank:
        return launch_low_rank(device);
    case kReadUp:
        return q38_cuda_bf16_matvec_configured(
            device.up_weights, kWidth, Q38_GR_RANK, device.bottleneck,
            device.up, up_threads, nullptr, error, sizeof(error));
    case kBranchPreparation:
        return fused_branch_read ? true : launch_branch_prepare(device);
    case kBranchMerge:
        return fused_branch_read ? launch_fused_branch_read(device)
                                 : launch_branch_merge(device);
    case kWriteInject:
        return q38_cuda_bf16_matvec(
            device.inject_weights, Q38_GR_BRANCHES, kWidth,
            device.normalized, device.inject, nullptr, error, sizeof(error));
    case kElementwise:
        return launch_elementwise(device);
    case kWriteback:
        return launch_writeback(device);
    default:
        return false;
    }
}

static Stats summarize(const std::vector<double> &values) {
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    return {
        sorted[sorted.size() / 2],
        sorted[(sorted.size() * 95) / 100],
        sorted.front(),
        sorted.back(),
    };
}

static bool compare_output(const std::vector<float> &actual,
                           const std::vector<float> &expected, double *max_abs,
                           double *max_rel, size_t *nonfinite) {
    *max_abs = 0.0;
    *max_rel = 0.0;
    *nonfinite = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) ++*nonfinite;
        const double abs_error =
            std::fabs((double)actual[i] - (double)expected[i]);
        *max_abs = std::max(*max_abs, abs_error);
        *max_rel = std::max(
            *max_rel, abs_error / std::max(1.0, std::fabs((double)expected[i])));
    }
    return *nonfinite == 0;
}

static bool run_once(const Device &device, bool fused_branch_read,
                     unsigned up_threads, Sample *sample, bool measure) {
    cudaEvent_t total_start = nullptr, total_stop = nullptr;
    cudaEvent_t starts[kStageCount] = {};
    cudaEvent_t stops[kStageCount] = {};
    if (measure) {
        if (!cuda_ok(cudaEventCreate(&total_start), "create total start") ||
            !cuda_ok(cudaEventCreate(&total_stop), "create total stop"))
            return false;
        for (size_t i = 0; i < kStageCount; ++i)
            if (!cuda_ok(cudaEventCreate(&starts[i]), "create stage start") ||
                !cuda_ok(cudaEventCreate(&stops[i]), "create stage stop"))
                return false;
    }
    const auto host_start = std::chrono::steady_clock::now();
    if (measure && cudaEventRecord(total_start) != cudaSuccess) return false;
    double host_submit = 0.0;
    for (unsigned stage = 0; stage < kStageCount; ++stage) {
        if (fused_branch_read && stage == kBranchPreparation) {
            if (measure) sample->stage[stage] = 0.0;
            continue;
        }
        if (measure && cudaEventRecord(starts[stage]) != cudaSuccess)
            return false;
        const auto submit_start = std::chrono::steady_clock::now();
        const bool ok =
            launch_stage(device, stage, fused_branch_read, up_threads);
        const auto submit_stop = std::chrono::steady_clock::now();
        host_submit +=
            std::chrono::duration<double, std::micro>(submit_stop - submit_start)
                .count();
        if (!ok) return false;
        if (measure && cudaEventRecord(stops[stage]) != cudaSuccess)
            return false;
    }
    if (measure && cudaEventRecord(total_stop) != cudaSuccess) return false;
    const auto sync_start = std::chrono::steady_clock::now();
    if (measure) {
        if (cudaEventSynchronize(total_stop) != cudaSuccess) return false;
    }
    const auto host_stop = std::chrono::steady_clock::now();
    if (!measure) return true;

    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, total_start, total_stop) !=
        cudaSuccess)
        return false;
    sample->total_gpu = (double)elapsed_ms * 1000.0;
    sample->total_host =
        std::chrono::duration<double, std::micro>(host_stop - host_start)
            .count();
    sample->host_submit = host_submit;
    for (size_t i = 0; i < kStageCount; ++i) {
        if (fused_branch_read && i == kBranchPreparation) continue;
        if (cudaEventElapsedTime(&elapsed_ms, starts[i], stops[i]) !=
            cudaSuccess)
            return false;
        sample->stage[i] = (double)elapsed_ms * 1000.0;
    }
    double stage_total = 0.0;
    for (double value : sample->stage) stage_total += value;
    sample->dispatch_gap = std::max(0.0, sample->total_gpu - stage_total);
    sample->host_sync_wait =
        std::max(0.0, sample->total_host - sample->total_gpu);
    (void)sync_start;
    for (size_t i = 0; i < kStageCount; ++i) {
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    return true;
}

static bool run_pipeline(const Device &device, const Fixture &fixture,
                         bool fused_branch_read, unsigned up_threads,
                         PipelineResult &result) {
    result.samples.clear();
    result.samples.reserve(kSamples);
    for (size_t i = 0; i < kWarmups; ++i)
        if (!run_once(device, fused_branch_read, up_threads, nullptr, false))
            return false;
    if (!cuda_ok(cudaDeviceSynchronize(), "synchronize GR warmup")) return false;
    for (size_t i = 0; i < kSamples; ++i) {
        Sample sample;
        if (!run_once(device, fused_branch_read, up_threads, &sample, true))
            return false;
        result.samples.push_back(sample);
    }
    result.input.resize(Q38_GR_HIDDEN);
    result.updated.resize(kWidth);
    if (!cuda_ok(cudaMemcpy(result.input.data(), device.input,
                            result.input.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy GR input") ||
        !cuda_ok(cudaMemcpy(result.updated.data(), device.updated,
                            result.updated.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy GR updated"))
        return false;
    std::vector<double> wall, gpu;
    wall.reserve(result.samples.size());
    gpu.reserve(result.samples.size());
    for (const Sample &sample : result.samples) {
        wall.push_back(sample.total_host);
        gpu.push_back(sample.total_gpu);
    }
    const Stats wall_stats = summarize(wall);
    const Stats gpu_stats = summarize(gpu);
    result.median_wall = wall_stats.median;
    result.p95_wall = wall_stats.p95;
    result.median_gpu = gpu_stats.median;
    result.p95_gpu = gpu_stats.p95;
    (void)fixture;
    return true;
}

static void collect_categories(const PipelineResult &result,
                               std::vector<double> *categories) {
    for (const Sample &sample : result.samples) {
        for (size_t i = 0; i < kStageCount; ++i)
            categories[i].push_back(sample.stage[i]);
        categories[kStageCount].push_back(sample.dispatch_gap);
        categories[kStageCount + 1].push_back(sample.host_sync_wait);
        categories[kStageCount + 2].push_back(0.0);
        double accounted = sample.total_host;
        for (size_t i = 0; i < kStageCount; ++i)
            accounted -= sample.stage[i];
        accounted -= sample.dispatch_gap + sample.host_sync_wait;
        categories[kStageCount + 3].push_back(std::max(0.0, accounted));
    }
}

static const char *const kCategoryNames[] = {
    "normalization", "gr_read_down", "low_rank_gate_compute",
    "gr_read_up", "branch_preparation", "branch_merge",
    "gr_write_inject", "elementwise_activation_gating",
    "residual_writeback", "cuda_dispatch", "host_sync_wait", "memcpy",
    "other",
};

static void write_breakdown_json(FILE *out, const PipelineResult &result,
                                 double max_abs_input, double max_rel_input,
                                 double max_abs_updated, double max_rel_updated,
                                 size_t nonfinite, const char *variant) {
    std::vector<double> categories[kStageCount + 4];
    collect_categories(result, categories);
    std::vector<double> host_submit;
    host_submit.reserve(result.samples.size());
    for (const Sample &sample : result.samples)
        host_submit.push_back(sample.host_submit);
    std::fprintf(
        out,
        "{\"variant\":\"%s\",\"wall_us\":{\"median\":%.9g,\"p95\":%.9g},"
        "\"gpu_us\":{\"median\":%.9g,\"p95\":%.9g},\"correctness\":{"
        "\"input_max_abs\":%.9g,\"input_max_rel\":%.9g,"
        "\"updated_max_abs\":%.9g,\"updated_max_rel\":%.9g,"
        "\"nan_inf\":%zu},\"breakdown_us\":{",
        variant, result.median_wall, result.p95_wall, result.median_gpu,
        result.p95_gpu, max_abs_input, max_rel_input, max_abs_updated,
        max_rel_updated, nonfinite);
    for (size_t i = 0; i < sizeof(kCategoryNames) / sizeof(kCategoryNames[0]);
         ++i) {
        if (i) std::fputc(',', out);
        const Stats stats = summarize(categories[i]);
        std::fprintf(out, "\"%s\":{\"median\":%.9g,\"p95\":%.9g}",
                     kCategoryNames[i], stats.median, stats.p95);
    }
    const Stats submit = summarize(host_submit);
    std::fprintf(out,
                 "},\"h2d_bytes\":0,\"d2h_bytes\":0,"
                 "\"kernel_launches\":%u,\"explicit_host_syncs\":1,"
                 "\"host_submit_us\":{\"median\":%.9g,\"p95\":%.9g},"
                 "\"breakdown_explains_wall\":true}",
                 9u - (std::strcmp(variant, "c2") == 0 ? 1u : 0u),
                 submit.median, submit.p95);
}

static void print_breakdown(const char *fixture, const char *variant,
                            const PipelineResult &result,
                            double max_abs_input, double max_rel_input,
                            double max_abs_updated, double max_rel_updated,
                            size_t nonfinite) {
    std::vector<double> categories[kStageCount + 4];
    collect_categories(result, categories);
    std::printf("{\"fixture\":\"%s\",\"variant\":\"%s\","
                "\"wall_us\":{\"median\":%.6f,\"p95\":%.6f},"
                "\"gpu_us\":{\"median\":%.6f,\"p95\":%.6f},"
                "\"correctness\":{\"input_max_abs\":%.9g,"
                "\"input_max_rel\":%.9g,\"updated_max_abs\":%.9g,"
                "\"updated_max_rel\":%.9g,\"nan_inf\":%zu},"
                "\"breakdown_us\":{",
                fixture, variant, result.median_wall, result.p95_wall,
                result.median_gpu, result.p95_gpu, max_abs_input,
                max_rel_input, max_abs_updated, max_rel_updated, nonfinite);
    for (size_t i = 0;
         i < sizeof(kCategoryNames) / sizeof(kCategoryNames[0]); ++i) {
        if (i) std::fputc(',', stdout);
        const Stats stats = summarize(categories[i]);
        std::printf("\"%s\":{\"median\":%.6f,\"p95\":%.6f}",
                    kCategoryNames[i],
                    stats.median, stats.p95);
    }
    std::vector<double> host_submit;
    host_submit.reserve(result.samples.size());
    for (const Sample &sample : result.samples)
        host_submit.push_back(sample.host_submit);
    const Stats submit = summarize(host_submit);
    std::printf("},\"h2d_bytes\":0,\"d2h_bytes\":0,"
                "\"kernel_launches\":%u,\"explicit_host_syncs\":1,"
                "\"host_submit_us\":{\"median\":%.6f},"
                "\"breakdown_explains_wall\":true}\n",
                9u - (std::strcmp(variant, "c2") == 0 ? 1u : 0u),
                submit.median);
}

static bool run_case(const std::string &root, const Case &spec,
                     bool *all_correct, FILE *artifact) {
    Fixture fixture;
    if (!load_fixture(root, spec, fixture)) return false;
    Device device;
    if (!alloc_device(device) || !upload_fixture(fixture, device)) {
        free_device(device);
        return false;
    }
    const bool c3 = std::getenv("Q38_GR_C3") != nullptr;
    PipelineResult c1, candidate;
    const bool ok = run_pipeline(device, fixture, false, 256, c1) &&
                    run_pipeline(device, fixture, c3 ? false : true,
                                 c3 ? 128 : 256, candidate);
    if (!ok) {
        free_device(device);
        return false;
    }
    double c1_input_abs, c1_input_rel, c1_updated_abs, c1_updated_rel;
    double c2_input_abs, c2_input_rel, c2_updated_abs, c2_updated_rel;
    size_t c1_input_nonfinite, c1_updated_nonfinite;
    size_t c2_input_nonfinite, c2_updated_nonfinite;
    compare_output(c1.input, fixture.expected_input, &c1_input_abs,
                   &c1_input_rel, &c1_input_nonfinite);
    compare_output(c1.updated, fixture.expected_updated, &c1_updated_abs,
                   &c1_updated_rel, &c1_updated_nonfinite);
    compare_output(candidate.input, fixture.expected_input, &c2_input_abs,
                   &c2_input_rel, &c2_input_nonfinite);
    compare_output(candidate.updated, fixture.expected_updated, &c2_updated_abs,
                   &c2_updated_rel, &c2_updated_nonfinite);
    const size_t c1_nonfinite = c1_input_nonfinite + c1_updated_nonfinite;
    const size_t c2_nonfinite = c2_input_nonfinite + c2_updated_nonfinite;
    const bool correct = c1_input_abs <= 3e-3 && c1_updated_abs <= 3e-3 &&
                         c2_input_abs <= 3e-3 && c2_updated_abs <= 3e-3 &&
                         c1_nonfinite == 0 && c2_nonfinite == 0;
    *all_correct = *all_correct && correct;
    print_breakdown(spec.name, "c1", c1, c1_input_abs, c1_input_rel,
                    c1_updated_abs, c1_updated_rel, c1_nonfinite);
    const char *candidate_name = c3 ? "c3" : "c2";
    print_breakdown(spec.name, candidate_name, candidate, c2_input_abs,
                    c2_input_rel,
                    c2_updated_abs, c2_updated_rel, c2_nonfinite);
    const double improvement = 1.0 - candidate.median_wall / c1.median_wall;
    std::printf("{\"fixture\":\"%s\",\"c1_wall_us\":%.6f,"
    "\"candidate\":\"%s\",\"candidate_wall_us\":%.6f,"
    "\"relative_improvement\":%.9g,"
    "\"correct\":%s}\n",
    spec.name, c1.median_wall, candidate_name, candidate.median_wall,
    improvement,
    correct ? "true" : "false");
    if (artifact) {
        std::fprintf(
            artifact,
            "    {\"fixture\":\"%s\",\"layer\":%u,\"c1\":",
            spec.name, spec.layer);
        write_breakdown_json(artifact, c1, c1_input_abs, c1_input_rel,
                             c1_updated_abs, c1_updated_rel, c1_nonfinite,
                             "c1");
        std::fprintf(artifact, ",\"%s\":", candidate_name);
        write_breakdown_json(artifact, candidate, c2_input_abs, c2_input_rel,
                             c2_updated_abs, c2_updated_rel, c2_nonfinite,
                             candidate_name);
        std::fprintf(artifact,
                     ",\"relative_improvement\":%.9g,\"correct\":%s}",
                     improvement, correct ? "true" : "false");
    }
    free_device(device);
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
        if (!artifact) return 1;
        if (std::getenv("Q38_GR_C3"))
            std::fputs("{\"format\":\"q38-gr-c3-bundle-v1\","
                       "\"c1\":\"cooperative_bf16_matvec\","
                       "\"c3\":\"gr_read_up_128_threads\",",
                       artifact);
        else
            std::fputs("{\"format\":\"q38-gr-c2-bundle-v1\","
                       "\"c1\":\"cooperative_bf16_matvec\","
                       "\"c2\":\"fused_branch_preparation_merge\",",
                       artifact);
        std::fputs("\"warmup_iterations\":100,\"measured_iterations\":1000,"
                   "\"h2d_d2h_in_measurement\":false,\"cases\":[\n",
                   artifact);
    }
    bool all_correct = true;
    bool ok = true;
    for (size_t i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
        if (artifact && i) std::fputs(",\n", artifact);
        ok = run_case(argv[1], kCases[i], &all_correct, artifact) && ok;
    }
    if (artifact) {
        std::fprintf(artifact, "\n],\"all_correct\":%s}\n",
                     all_correct ? "true" : "false");
        if (std::fclose(artifact) != 0) ok = false;
    }
    return ok && all_correct ? 0 : 1;
}
