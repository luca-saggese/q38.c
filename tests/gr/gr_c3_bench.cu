#include "../../q38_cuda_primitives.h"
#include "../../q38_gr_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kWidth = Q38_GR_BRANCHES * Q38_GR_HIDDEN;
constexpr size_t kWarmups = 100;
constexpr size_t kSamples = 1000;

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
};

struct Stats {
    double median;
    double p95;
    double min;
    double max;
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
    return read_file(std::string(dir) + "/residual.bin", fixture.residual.data(),
                     fixture.residual.size()) &&
           read_file(std::string(dir) + "/hc_norm.bin", fixture.gamma.data(),
                     fixture.gamma.size()) &&
           read_file(std::string(dir) + "/input_mix_down.bin",
                     fixture.down.data(), fixture.down.size()) &&
           read_file(std::string(dir) + "/input_mix_up.bin", fixture.up.data(),
                     fixture.up.size());
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_float(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
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

static void make_inputs(const Fixture &fixture, std::vector<uint16_t> &weights,
                        std::vector<float> &bottleneck,
                        std::vector<float> &expected) {
    std::vector<float> normalized(kWidth);
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; ++branch) {
        double sum = 0.0;
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
            const float value =
                fixture.residual[branch * Q38_GR_HIDDEN + channel];
            sum += (double)value * (double)value;
        }
        const float scale =
            1.0f / std::sqrt((float)(sum / Q38_GR_HIDDEN) + 1e-6f);
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
            const size_t index = branch * Q38_GR_HIDDEN + channel;
            normalized[index] = fixture.residual[index] * scale *
                                (1.0f + fixture.gamma[index]);
        }
    }
    bottleneck.resize(Q38_GR_RANK);
    for (size_t rank = 0; rank < Q38_GR_RANK; ++rank) {
        float value = 0.0f;
        for (size_t i = 0; i < kWidth; ++i)
            value += fixture.down[rank * kWidth + i] * normalized[i];
        value /= 4.0f;
        bottleneck[rank] = value / (1.0f + std::exp(-value));
    }
    weights.resize(fixture.up.size());
    expected.resize(kWidth);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = float_to_bf16(fixture.up[i]);
    for (size_t row = 0; row < kWidth; ++row) {
        float value = 0.0f;
        for (size_t rank = 0; rank < Q38_GR_RANK; ++rank)
            value += bf16_to_float(weights[row * Q38_GR_RANK + rank]) *
                     bottleneck[rank];
        expected[row] = value;
    }
}

static bool run_geometry(const std::vector<uint16_t> &weights,
                         const std::vector<float> &bottleneck,
                         unsigned threads, std::vector<double> &samples,
                         std::vector<float> &actual) {
    uint16_t *device_weights = nullptr;
    float *device_input = nullptr;
    float *device_output = nullptr;
    if (!cuda_ok(cudaMalloc(&device_weights, weights.size() * sizeof(uint16_t)),
                 "alloc up weights") ||
        !cuda_ok(cudaMalloc(&device_input, bottleneck.size() * sizeof(float)),
                 "alloc bottleneck") ||
        !cuda_ok(cudaMalloc(&device_output, kWidth * sizeof(float)),
                 "alloc up output"))
        return false;
    if (!cuda_ok(cudaMemcpy(device_weights, weights.data(),
                            weights.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice), "upload up weights") ||
        !cuda_ok(cudaMemcpy(device_input, bottleneck.data(),
                            bottleneck.size() * sizeof(float),
                            cudaMemcpyHostToDevice), "upload bottleneck")) {
        cudaFree(device_weights);
        cudaFree(device_input);
        cudaFree(device_output);
        return false;
    }
    char error[256] = {};
    for (size_t i = 0; i < kWarmups; ++i)
        if (!q38_cuda_bf16_matvec_configured(
                device_weights, kWidth, Q38_GR_RANK, device_input,
                device_output, threads, nullptr, error, sizeof(error))) {
            std::fprintf(stderr, "GR-C3 warmup: %s\n", error);
            return false;
        }
    if (!cuda_ok(cudaDeviceSynchronize(), "synchronize GR-C3 warmup"))
        return false;
    cudaEvent_t start = nullptr, stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "create GR-C3 start") ||
        !cuda_ok(cudaEventCreate(&stop), "create GR-C3 stop"))
        return false;
    samples.reserve(kSamples);
    for (size_t i = 0; i < kSamples; ++i) {
        cudaEventRecord(start);
        if (!q38_cuda_bf16_matvec_configured(
                device_weights, kWidth, Q38_GR_RANK, device_input,
                device_output, threads, nullptr, error, sizeof(error))) {
            std::fprintf(stderr, "GR-C3 benchmark: %s\n", error);
            return false;
        }
        cudaEventRecord(stop);
        if (cudaEventSynchronize(stop) != cudaSuccess) return false;
        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess)
            return false;
        samples.push_back((double)elapsed_ms * 1000.0);
    }
    actual.resize(kWidth);
    if (!cuda_ok(cudaMemcpy(actual.data(), device_output,
                            actual.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy GR-C3 output"))
        return false;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_weights);
    cudaFree(device_input);
    cudaFree(device_output);
    return true;
}

static bool compare(const std::vector<float> &actual,
                    const std::vector<float> &expected, double *max_abs,
                    double *max_rel, size_t *nonfinite) {
    *max_abs = 0.0;
    *max_rel = 0.0;
    *nonfinite = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) ++*nonfinite;
        const double error =
            std::fabs((double)actual[i] - (double)expected[i]);
        *max_abs = std::max(*max_abs, error);
        *max_rel = std::max(
            *max_rel, error / std::max(1.0, std::fabs((double)expected[i])));
    }
    return *nonfinite == 0;
}

static bool run_case(const std::string &root, const Case &spec,
                     bool *all_correct, FILE *artifact) {
    Fixture fixture;
    if (!load_fixture(root, spec, fixture)) return false;
    std::vector<uint16_t> weights;
    std::vector<float> bottleneck, expected;
    make_inputs(fixture, weights, bottleneck, expected);
    const unsigned geometries[] = {128, 256, 512};
    std::vector<double> times[3];
    std::vector<float> actual[3];
    for (size_t i = 0; i < 3; ++i)
        if (!run_geometry(weights, bottleneck, geometries[i], times[i],
                          actual[i]))
            return false;
    double max_abs[3], max_rel[3];
    size_t nonfinite[3];
    for (size_t i = 0; i < 3; ++i)
        compare(actual[i], expected, &max_abs[i], &max_rel[i],
                &nonfinite[i]);
    const Stats stats[3] = {
        summarize(times[0]), summarize(times[1]), summarize(times[2])};
    const bool correct = max_abs[0] <= 3e-3 && max_abs[1] <= 3e-3 &&
                         max_abs[2] <= 3e-3 && nonfinite[0] == 0 &&
                         nonfinite[1] == 0 && nonfinite[2] == 0;
    *all_correct = *all_correct && correct;
    std::printf(
        "{\"fixture\":\"%s\",\"geometries\":{\"128\":{\"median_us\":%.6f,"
        "\"p95_us\":%.6f,\"min_us\":%.6f,\"max_us\":%.6f,"
        "\"max_abs\":%.9g,\"max_rel\":%.9g},\"256\":{\"median_us\":%.6f,"
        "\"p95_us\":%.6f,\"min_us\":%.6f,\"max_us\":%.6f,"
        "\"max_abs\":%.9g,\"max_rel\":%.9g},\"512\":{\"median_us\":%.6f,"
        "\"p95_us\":%.6f,\"min_us\":%.6f,\"max_us\":%.6f,"
        "\"max_abs\":%.9g,\"max_rel\":%.9g}},"
        "\"improvement_vs_256\":{\"128\":%.9g,\"512\":%.9g},"
        "\"correct\":%s}\n",
        spec.name, stats[0].median, stats[0].p95, stats[0].min, stats[0].max,
        max_abs[0], max_rel[0], stats[1].median, stats[1].p95, stats[1].min,
        stats[1].max, max_abs[1], max_rel[1], stats[2].median, stats[2].p95,
        stats[2].min, stats[2].max, max_abs[2], max_rel[2],
        1.0 - stats[0].median / stats[1].median,
        1.0 - stats[2].median / stats[1].median, correct ? "true" : "false");
    if (artifact) {
        std::fprintf(
            artifact,
            "    {\"fixture\":\"%s\",\"layer\":%u,\"geometries\":{"
            "\"128\":{\"median_us\":%.9g,\"p95_us\":%.9g,\"min_us\":%.9g,"
            "\"max_us\":%.9g,\"max_abs\":%.9g,\"max_rel\":%.9g},"
            "\"256\":{\"median_us\":%.9g,\"p95_us\":%.9g,\"min_us\":%.9g,"
            "\"max_us\":%.9g,\"max_abs\":%.9g,\"max_rel\":%.9g},"
            "\"512\":{\"median_us\":%.9g,\"p95_us\":%.9g,\"min_us\":%.9g,"
            "\"max_us\":%.9g,\"max_abs\":%.9g,\"max_rel\":%.9g}},"
            "\"improvement_vs_256\":{\"128\":%.9g,\"512\":%.9g},"
            "\"correct\":%s}",
            spec.name, spec.layer, stats[0].median, stats[0].p95, stats[0].min,
            stats[0].max, max_abs[0], max_rel[0], stats[1].median, stats[1].p95,
            stats[1].min, stats[1].max, max_abs[1], max_rel[1],
            stats[2].median, stats[2].p95, stats[2].min, stats[2].max,
            max_abs[2], max_rel[2],
            1.0 - stats[0].median / stats[1].median,
            1.0 - stats[2].median / stats[1].median,
            correct ? "true" : "false");
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
        if (!artifact) return 1;
        std::fputs("{\"format\":\"q38-gr-c3-up-geometry-v1\","
                   "\"stage\":\"gr_read_up\",\"baseline_threads\":256,"
                   "\"warmup_iterations\":100,\"measured_iterations\":1000,"
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
