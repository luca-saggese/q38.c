#include "../../q38_cuda_primitives.h"
#include "../../q38_gr.h"

#include <cuda_runtime.h>

#include <algorithm>
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
};

struct Stats {
    double median;
    double p95;
    double min;
    double max;
};

struct Projection {
    const char *name;
    size_t rows;
    size_t cols;
    std::vector<uint16_t> weights;
    std::vector<float> input;
    std::vector<float> expected;
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
    const char *names[] = {
        "residual.bin", "hc_norm.bin", "input_mix_down.bin",
        "input_mix_up.bin", "block_inject.bin",
    };
    float *destinations[] = {
        fixture.residual.data(), fixture.gamma.data(), fixture.down.data(),
        fixture.up.data(), fixture.inject.data(),
    };
    const size_t counts[] = {
        kWidth, kWidth, Q38_GR_RANK * kWidth, kWidth * Q38_GR_RANK,
        Q38_GR_BRANCHES * kWidth,
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (!read_file(std::string(dir) + "/" + names[i], destinations[i],
                       counts[i]))
            return false;
    }
    return true;
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

static void normalize(const Fixture &fixture, std::vector<float> &output) {
    output.resize(kWidth);
    for (size_t branch = 0; branch < Q38_GR_BRANCHES; ++branch) {
        double sum = 0.0;
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
            const float value =
                fixture.residual[branch * Q38_GR_HIDDEN + channel];
            sum += (double)value * value;
        }
        const float scale =
            1.0f / std::sqrt((float)(sum / Q38_GR_HIDDEN) + 1e-6f);
        for (size_t channel = 0; channel < Q38_GR_HIDDEN; ++channel) {
            const size_t index = branch * Q38_GR_HIDDEN + channel;
            output[index] = fixture.residual[index] * scale *
                            (1.0f + fixture.gamma[index]);
        }
    }
}

static void make_projection(const char *name, size_t rows, size_t cols,
                            const std::vector<float> &weights,
                            const std::vector<float> &input,
                            Projection &projection) {
    projection.name = name;
    projection.rows = rows;
    projection.cols = cols;
    projection.input = input;
    projection.weights.resize(rows * cols);
    projection.expected.resize(rows);
    for (size_t i = 0; i < rows * cols; ++i)
        projection.weights[i] = float_to_bf16(weights[i]);
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (size_t column = 0; column < cols; ++column)
            sum += bf16_to_float(projection.weights[row * cols + column]) *
                   input[column];
        projection.expected[row] = sum;
    }
}

static bool cuda_ok(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
}

static Stats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return {
        values[values.size() / 2],
        values[(values.size() * 95) / 100],
        values.front(),
        values.back(),
    };
}

static double max_abs_diff(const std::vector<float> &actual,
                           const std::vector<float> &expected) {
    double max_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i)
        max_abs = std::max(max_abs,
                           std::fabs((double)actual[i] - expected[i]));
    return max_abs;
}

static bool run_projection(const Projection &projection, FILE *artifact,
                           bool *all_correct) {
    uint16_t *device_weights = nullptr;
    float *device_input = nullptr;
    float *device_candidate = nullptr;
    float *device_generic = nullptr;
    std::vector<double> generic_times;
    std::vector<double> candidate_times;
    std::vector<float> generic(projection.expected.size());
    std::vector<float> candidate(projection.expected.size());
    cudaEvent_t start = nullptr, stop = nullptr;
    if (!cuda_ok(cudaMalloc(&device_weights,
                            projection.weights.size() * sizeof(uint16_t)),
                 "alloc projection weights") ||
        !cuda_ok(cudaMalloc(&device_input,
                            projection.input.size() * sizeof(float)),
                 "alloc projection input") ||
        !cuda_ok(cudaMalloc(&device_candidate,
                            projection.expected.size() * sizeof(float)),
                 "alloc candidate output") ||
        !cuda_ok(cudaMalloc(&device_generic,
                            projection.expected.size() * sizeof(float)),
                 "alloc generic output"))
        goto fail;
    if (!cuda_ok(cudaMemcpy(device_weights, projection.weights.data(),
                            projection.weights.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice),
                 "upload projection weights") ||
        !cuda_ok(cudaMemcpy(device_input, projection.input.data(),
                            projection.input.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "upload projection input"))
        goto fail;

    for (size_t i = 0; i < kWarmups; ++i) {
        char error[256] = {};
        if (!q38_cuda_matrix_batch_generic(
                30, device_weights, device_input, 1, projection.rows,
                projection.cols, device_generic, nullptr, error,
                sizeof(error)) ||
            !q38_cuda_bf16_matvec(
                device_weights, projection.rows, projection.cols,
                device_input, device_candidate, nullptr, error,
                sizeof(error))) {
            std::fprintf(stderr, "%s warmup: %s\n", projection.name, error);
            goto fail;
        }
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "projection warmup synchronize"))
        goto fail;

    generic_times.reserve(kSamples);
    candidate_times.reserve(kSamples);
    if (!cuda_ok(cudaEventCreate(&start), "create projection start event") ||
        !cuda_ok(cudaEventCreate(&stop), "create projection stop event"))
        goto fail;
    for (size_t i = 0; i < kSamples; ++i) {
        char error[256] = {};
        cudaEventRecord(start, nullptr);
        const bool generic_ok = q38_cuda_matrix_batch_generic(
            30, device_weights, device_input, 1, projection.rows,
            projection.cols, device_generic, nullptr, error, sizeof(error));
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);
        if (!generic_ok) {
            std::fprintf(stderr, "%s generic: %s\n", projection.name, error);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            goto fail;
        }
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        generic_times.push_back((double)elapsed_ms * 1000.0);

        cudaEventRecord(start, nullptr);
        const bool candidate_ok = q38_cuda_bf16_matvec(
            device_weights, projection.rows, projection.cols, device_input,
            device_candidate, nullptr, error, sizeof(error));
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);
        if (!candidate_ok) {
            std::fprintf(stderr, "%s candidate: %s\n", projection.name,
                         error);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            goto fail;
        }
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        candidate_times.push_back((double)elapsed_ms * 1000.0);
    }
    if (!cuda_ok(cudaMemcpy(generic.data(), device_generic,
                            generic.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "download generic projection") ||
        !cuda_ok(cudaMemcpy(candidate.data(), device_candidate,
                            candidate.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "download candidate projection")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        goto fail;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    {
        const double generic_abs = max_abs_diff(generic, projection.expected);
        const double candidate_abs =
            max_abs_diff(candidate, projection.expected);
        const bool correct = generic_abs <= 1e-3 && candidate_abs <= 1e-3;
        *all_correct = *all_correct && correct;
        const Stats generic_stats = summarize(generic_times);
        const Stats candidate_stats = summarize(candidate_times);
        const double improvement =
            1.0 - candidate_stats.median / generic_stats.median;
        std::printf(
            "{\"projection\":\"%s\",\"generic_median_us\":%.6f,"
            "\"candidate_median_us\":%.6f,\"improvement\":%.6f,"
            "\"generic_p95_us\":%.6f,\"candidate_p95_us\":%.6f,"
            "\"generic_max_abs\":%.9g,\"candidate_max_abs\":%.9g,"
            "\"correct\":%s}\n",
            projection.name, generic_stats.median, candidate_stats.median,
            improvement, generic_stats.p95, candidate_stats.p95, generic_abs,
            candidate_abs, correct ? "true" : "false");
        if (artifact)
            std::fprintf(
                artifact,
                "{\"projection\":\"%s\",\"rows\":%zu,\"cols\":%zu,"
                "\"generic\":{\"median_us\":%.9g,\"p95_us\":%.9g,"
                "\"min_us\":%.9g,\"max_us\":%.9g},\"candidate\":"
                "{\"median_us\":%.9g,\"p95_us\":%.9g,\"min_us\":%.9g,"
                "\"max_us\":%.9g},\"relative_improvement\":%.9g,"
                "\"generic_max_abs\":%.9g,\"candidate_max_abs\":%.9g,"
                "\"correct\":%s}",
                projection.name, projection.rows, projection.cols,
                generic_stats.median, generic_stats.p95, generic_stats.min,
                generic_stats.max, candidate_stats.median,
                candidate_stats.p95, candidate_stats.min, candidate_stats.max,
                improvement, generic_abs, candidate_abs,
                correct ? "true" : "false");
    }
    cudaFree(device_weights);
    cudaFree(device_input);
    cudaFree(device_candidate);
    cudaFree(device_generic);
    return true;

fail:
    cudaFree(device_weights);
    cudaFree(device_input);
    cudaFree(device_candidate);
    cudaFree(device_generic);
    return false;
}

static bool run_case(const std::string &root, const Case &spec, FILE *artifact,
                     bool *all_correct) {
    Fixture fixture;
    if (!load_fixture(root, spec, fixture)) return false;
    std::vector<float> normalized;
    normalize(fixture, normalized);
    std::vector<float> bottleneck(Q38_GR_RANK);
    for (size_t rank = 0; rank < Q38_GR_RANK; ++rank) {
        float value = 0.0f;
        for (size_t i = 0; i < kWidth; ++i)
            value += fixture.down[rank * kWidth + i] * normalized[i];
        value /= (float)Q38_GR_HC_COUNT;
        bottleneck[rank] = value / (1.0f + std::exp(-value));
    }
    Projection projections[3];
    make_projection("down", Q38_GR_RANK, kWidth, fixture.down, normalized,
                    projections[0]);
    make_projection("up", kWidth, Q38_GR_RANK, fixture.up, bottleneck,
                    projections[1]);
    make_projection("inject", Q38_GR_BRANCHES, kWidth, fixture.inject,
                    normalized, projections[2]);
    if (artifact) std::fprintf(artifact, "{\"fixture\":\"%s\",\"layer\":%u,"
                               "\"projections\":[", spec.name, spec.layer);
    for (size_t i = 0; i < 3; ++i) {
        if (i && artifact) std::fputc(',', artifact);
        if (!run_projection(projections[i], artifact, all_correct))
            return false;
    }
    if (artifact) std::fputs("]}", artifact);
    return true;
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
        std::fputs("{\"format\":\"q38-gr-c1-projection-v1\","
                   "\"candidate\":\"bf16_matvec\",\"cases\":[",
                   artifact);
    }
    bool all_correct = true;
    bool ok = true;
    for (size_t i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
        if (i && artifact) std::fputc(',', artifact);
        ok = run_case(argv[1], kCases[i], artifact, &all_correct) && ok;
    }
    if (artifact) {
        std::fprintf(artifact, "],\"all_correct\":%s}\n",
                     all_correct ? "true" : "false");
        std::fclose(artifact);
    }
    return ok && all_correct ? 0 : 1;
}
