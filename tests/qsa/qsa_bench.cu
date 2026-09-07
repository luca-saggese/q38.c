#include "qsa_reference.h"

#include "../../q38_cuda_primitives.h"
#include "../../q38_qsa_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <inttypes.h>
#include <string>
#include <vector>

namespace {

constexpr size_t kWarmups = 2;
constexpr size_t kSamples = 12;
constexpr size_t kHidden = 2560;
constexpr size_t kQRows = 12288;
constexpr size_t kKvRows = 512;
constexpr size_t kIndexRows = 640;
constexpr size_t kIndexStateRows = 128;
constexpr size_t kAttention = 6144;
constexpr size_t kOutput = 2560;
constexpr size_t kSelectedStride = 2048;

struct Weight {
    uint32_t type = 0;
    size_t rows = 0;
    size_t cols = 0;
    std::vector<unsigned char> raw;
};

struct Fixture {
    uint32_t layer = 0;
    size_t state_count = 0;
    std::vector<float> hidden;
    std::vector<float> q_projection;
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<float> index_projection;
    std::vector<float> attention;
    std::vector<float> expected_output;
    std::vector<uint32_t> selected;
    std::vector<float> state_main_k;
    std::vector<float> state_main_v;
    std::vector<float> state_index_k;
    Weight q_proj, k_proj, v_proj, o_proj, index_qk_proj;
    Weight q_norm, k_norm, index_q_norm, index_k_norm;
};

struct Device {
    cudaStream_t stream = nullptr;
    uint16_t *q_proj = nullptr;
    uint16_t *k_proj = nullptr;
    uint16_t *v_proj = nullptr;
    uint16_t *o_proj = nullptr;
    uint16_t *index_qk_proj = nullptr;
    uint16_t *q_norm = nullptr;
    uint16_t *k_norm = nullptr;
    uint16_t *index_q_norm = nullptr;
    uint16_t *index_k_norm = nullptr;
    float *input = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *output_input = nullptr;
    float *output = nullptr;
    q38_qsa_cuda_chain_state chain_state{};
    q38_qsa_cuda_chain_workspace chain_workspace{};
};

struct Stats {
    double median = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct Compare {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
    size_t nonfinite = 0;
};

static constexpr double kQsaMaxAbsTolerance = 3.0e-3;

struct Sample {
    double total_us = 0.0;
    double qkv_us = 0.0;
    double index_us = 0.0;
    double state_us = 0.0;
    double attention_us = 0.0;
    double output_us = 0.0;
    double dispatch_us = 0.0;
    double sync_wait_us = 0.0;
    double memcpy_us = 0.0;
    uint64_t launches = 0;
    uint64_t syncs = 0;
    uint64_t h2d_bytes = 0;
    uint64_t d2h_bytes = 0;
    uint64_t d2d_bytes = 0;
};

struct Run {
    Sample sample;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> index;
    std::vector<float> output;
    std::vector<float> state_main_k;
    std::vector<float> state_main_v;
    std::vector<float> state_index_k;
    std::vector<float> attention;
    std::vector<uint32_t> selected;
};

enum class Mode {
    Baseline,
    C1,
    ChainC1,
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

static bool metadata_u64(const std::string &metadata, const char *key,
                         uint64_t *value) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t position = metadata.find(needle);
    if (position == std::string::npos) return false;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(
        metadata.c_str() + position + needle.size(), &end, 10);
    if (end == metadata.c_str() + position + needle.size()) return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

static bool metadata_type(const std::string &metadata, const char *name,
                          uint32_t *type) {
    const std::string needle =
        std::string("\"") + name + "\":{\"qtype\":";
    const size_t position = metadata.find(needle);
    if (position == std::string::npos) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(
        metadata.c_str() + position + needle.size(), &end, 10);
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

static q38_test_qsa_weight view(const Weight &weight) {
    return {weight.type, weight.rows, weight.cols, weight.raw.data(),
            weight.raw.size()};
}

static bool load_fixture(const std::string &dir, Fixture *fixture,
                         std::string *error) {
    std::string metadata;
    if (!read_text(dir + "/metadata.json", &metadata, error))
        return false;
    uint64_t value = 0;
    if (!metadata_u64(metadata, "layer", &value)) {
        *error = "QSA metadata has no layer";
        return false;
    }
    fixture->layer = static_cast<uint32_t>(value);
    if (!metadata_u64(metadata, "main_k_count", &value)) {
        *error = "QSA metadata has no state count";
        return false;
    }
    fixture->state_count = static_cast<size_t>(value);
    fixture->hidden.resize(kHidden);
    fixture->q_projection.resize(kQRows);
    fixture->keys.resize(kKvRows);
    fixture->values.resize(kKvRows);
    fixture->index_projection.resize(kIndexRows);
    fixture->attention.resize(kAttention);
    fixture->expected_output.resize(kOutput);
    fixture->selected.resize(kSelectedStride);
    fixture->state_main_k.resize(fixture->state_count * kKvRows);
    fixture->state_main_v.resize(fixture->state_count * kKvRows);
    fixture->state_index_k.resize(fixture->state_count * kIndexStateRows);
    if (!read_file(dir + "/hidden.f32", fixture->hidden.data(),
                   fixture->hidden.size() * sizeof(float), error) ||
        !read_file(dir + "/q_projection.f32", fixture->q_projection.data(),
                   fixture->q_projection.size() * sizeof(float), error) ||
        !read_file(dir + "/keys.f32", fixture->keys.data(),
                   fixture->keys.size() * sizeof(float), error) ||
        !read_file(dir + "/values.f32", fixture->values.data(),
                   fixture->values.size() * sizeof(float), error) ||
        !read_file(dir + "/index_projection.f32",
                   fixture->index_projection.data(),
                   fixture->index_projection.size() * sizeof(float), error) ||
        !read_file(dir + "/attention.f32", fixture->attention.data(),
                   fixture->attention.size() * sizeof(float), error) ||
        !read_file(dir + "/expected_output.f32",
                   fixture->expected_output.data(),
                   fixture->expected_output.size() * sizeof(float), error) ||
        !read_file(dir + "/state_main_k.f32", fixture->state_main_k.data(),
                   fixture->state_main_k.size() * sizeof(float), error) ||
        !read_file(dir + "/state_main_v.f32", fixture->state_main_v.data(),
                   fixture->state_main_v.size() * sizeof(float), error) ||
        !read_file(dir + "/state_index_k.f32", fixture->state_index_k.data(),
                   fixture->state_index_k.size() * sizeof(float), error))
        return false;
    FILE *ids = std::fopen((dir + "/selected_ids.u32").c_str(), "rb");
    if (!ids) {
        *error = "missing selected IDs";
        return false;
    }
    if (std::fseek(ids, 0, SEEK_END) != 0) {
        std::fclose(ids);
        *error = "failed to size selected IDs";
        return false;
    }
    const long ids_size = std::ftell(ids);
    if (ids_size < 0 || std::fseek(ids, 0, SEEK_SET) != 0 ||
        ids_size % (long)sizeof(uint32_t) != 0 ||
        static_cast<size_t>(ids_size) > fixture->selected.size() *
            sizeof(uint32_t)) {
        std::fclose(ids);
        *error = "invalid selected IDs";
        return false;
    }
    fixture->selected.resize(static_cast<size_t>(ids_size) / sizeof(uint32_t));
    if (std::fread(fixture->selected.data(), 1, static_cast<size_t>(ids_size),
                   ids) != static_cast<size_t>(ids_size) ||
        std::fclose(ids) != 0) {
        *error = "failed to read selected IDs";
        return false;
    }

    const char *names[] = {
        "q_proj.bin", "k_proj.bin", "v_proj.bin", "o_proj.bin",
        "index_qk_proj.bin", "q_norm.bin", "k_norm.bin",
        "index_q_norm.bin", "index_k_norm.bin",
    };
    const size_t rows[] = {
        kQRows, kKvRows, kKvRows, kOutput, kIndexRows, kKvRows / 2,
        kKvRows / 2, kIndexStateRows, kIndexStateRows,
    };
    const size_t cols[] = {
        kHidden, kHidden, kHidden, kAttention, kHidden, 1, 1, 1, 1,
    };
    uint32_t types[9] = {};
    for (size_t i = 0; i < 9; ++i)
        if (!metadata_type(metadata, names[i], &types[i])) {
            *error = std::string("missing QSA tensor type: ") + names[i];
            return false;
        }
    Weight *weights[] = {
        &fixture->q_proj, &fixture->k_proj, &fixture->v_proj,
        &fixture->o_proj, &fixture->index_qk_proj, &fixture->q_norm,
        &fixture->k_norm, &fixture->index_q_norm, &fixture->index_k_norm,
    };
    for (size_t i = 0; i < 9; ++i)
        if (!load_weight(dir, names[i], rows[i], cols[i], types[i], weights[i],
                         error))
            return false;
    return true;
}

static bool alloc_device(const Fixture &fixture, Device *device,
                         std::string *error) {
    if (!cuda_ok(cudaStreamCreate(&device->stream), "cudaStreamCreate",
                 error))
        return false;
    const auto alloc = [&](void **ptr, size_t bytes, const char *what) {
        return cuda_ok(cudaMalloc(ptr, bytes), what, error);
    };
    return alloc((void **)&device->q_proj, fixture.q_proj.raw.size(),
                 "q projection allocation") &&
           alloc((void **)&device->k_proj, fixture.k_proj.raw.size(),
                 "k projection allocation") &&
           alloc((void **)&device->v_proj, fixture.v_proj.raw.size(),
                 "v projection allocation") &&
           alloc((void **)&device->o_proj, fixture.o_proj.raw.size(),
                 "output projection allocation") &&
           alloc((void **)&device->index_qk_proj,
                 fixture.index_qk_proj.raw.size(),
                 "index projection allocation") &&
           alloc((void **)&device->q_norm, fixture.q_norm.raw.size(),
                 "q norm allocation") &&
           alloc((void **)&device->k_norm, fixture.k_norm.raw.size(),
                 "k norm allocation") &&
           alloc((void **)&device->index_q_norm, fixture.index_q_norm.raw.size(),
                 "index q norm allocation") &&
           alloc((void **)&device->index_k_norm, fixture.index_k_norm.raw.size(),
                 "index k norm allocation") &&
           alloc((void **)&device->input, kHidden * sizeof(float),
                 "input allocation") &&
           alloc((void **)&device->q, kQRows * sizeof(float),
                 "q output allocation") &&
           alloc((void **)&device->k, kKvRows * sizeof(float),
                 "k output allocation") &&
           alloc((void **)&device->v, kKvRows * sizeof(float),
                 "v output allocation") &&
           alloc((void **)&device->output_input, kAttention * sizeof(float),
                 "output input allocation") &&
           alloc((void **)&device->output, kOutput * sizeof(float),
                 "output allocation") &&
           alloc((void **)&device->chain_workspace.index, kIndexRows * sizeof(float),
                 "chain index workspace allocation") &&
           alloc((void **)&device->chain_workspace.index_q,
                 kIndexStateRows * sizeof(float),
                 "chain index query workspace allocation") &&
           alloc((void **)&device->chain_workspace.raw_index,
                 kIndexStateRows * sizeof(float),
                 "chain raw index workspace allocation") &&
           alloc((void **)&device->chain_workspace.attention,
                 kAttention * sizeof(float), "chain attention workspace allocation") &&
           alloc((void **)&device->chain_workspace.selected_k,
                 kSelectedStride * 2 * 256 * sizeof(float),
                 "chain selected key workspace allocation") &&
           alloc((void **)&device->chain_workspace.selected_v,
                 kSelectedStride * 2 * 256 * sizeof(float),
                 "chain selected value workspace allocation") &&
           alloc((void **)&device->chain_workspace.selected,
                 kSelectedStride * sizeof(uint32_t),
                 "chain selected ID workspace allocation");
}

static void free_device(Device *device) {
    cudaFree(device->q_proj);
    cudaFree(device->k_proj);
    cudaFree(device->v_proj);
    cudaFree(device->o_proj);
    cudaFree(device->index_qk_proj);
    cudaFree(device->q_norm);
    cudaFree(device->k_norm);
    cudaFree(device->index_q_norm);
    cudaFree(device->index_k_norm);
    cudaFree(device->input);
    cudaFree(device->q);
    cudaFree(device->k);
    cudaFree(device->v);
    cudaFree(device->output_input);
    cudaFree(device->output);
    q38_qsa_cuda_chain_release(&device->chain_state);
    cudaFree(device->chain_workspace.index);
    cudaFree(device->chain_workspace.index_q);
    cudaFree(device->chain_workspace.raw_index);
    cudaFree(device->chain_workspace.attention);
    cudaFree(device->chain_workspace.selected_k);
    cudaFree(device->chain_workspace.selected_v);
    cudaFree(device->chain_workspace.selected);
    if (device->stream) cudaStreamDestroy(device->stream);
    *device = {};
}

static bool upload_weights(const Fixture &fixture, Device *device,
                           std::string *error) {
    return cuda_ok(cudaMemcpyAsync(device->q_proj, fixture.q_proj.raw.data(),
                                   fixture.q_proj.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "q projection upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->k_proj, fixture.k_proj.raw.data(),
                                   fixture.k_proj.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "k projection upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->v_proj, fixture.v_proj.raw.data(),
                                   fixture.v_proj.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "v projection upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->o_proj, fixture.o_proj.raw.data(),
                                   fixture.o_proj.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "output projection upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->index_qk_proj,
                                   fixture.index_qk_proj.raw.data(),
                                   fixture.index_qk_proj.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "index projection upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->q_norm, fixture.q_norm.raw.data(),
                                   fixture.q_norm.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "q norm upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->k_norm, fixture.k_norm.raw.data(),
                                   fixture.k_norm.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "k norm upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->index_q_norm,
                                   fixture.index_q_norm.raw.data(),
                                   fixture.index_q_norm.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "index q norm upload", error) &&
           cuda_ok(cudaMemcpyAsync(device->index_k_norm,
                                   fixture.index_k_norm.raw.data(),
                                   fixture.index_k_norm.raw.size(),
                                   cudaMemcpyHostToDevice, device->stream),
                   "index k norm upload", error) &&
           cuda_ok(cudaStreamSynchronize(device->stream), "weight sync",
                   error);
}

static Compare compare(const float *actual, const float *expected,
                       size_t count) {
    Compare result;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) ++result.nonfinite;
        const double difference =
            std::abs(static_cast<double>(actual[i]) - expected[i]);
        const double scale = std::max(std::abs(static_cast<double>(expected[i])),
                                      1.0e-12);
        result.max_abs = std::max(result.max_abs, difference);
        result.max_rel = std::max(result.max_rel, difference / scale);
        sum += difference * difference;
    }
    result.rmse = std::sqrt(sum / static_cast<double>(count));
    return result;
}

static bool compare_passes(const Compare &result) {
    // Relative error is ill-conditioned for values near zero; the contract
    // reports it but uses the absolute error gate for this BF16 path.
    return result.nonfinite == 0 &&
           result.max_abs <= kQsaMaxAbsTolerance;
}

static Stats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    Stats result;
    result.min = values.front();
    result.max = values.back();
    result.median = values[values.size() / 2];
    result.p95 = values[(values.size() * 95) / 100];
    return result;
}

static bool run_once(const Fixture &fixture, Device *device, Mode mode,
                     Run *run, std::string *error) {
    run->sample = {};
    run->q.assign(kQRows, 0.0f);
    run->k.assign(kKvRows, 0.0f);
    run->v.assign(kKvRows, 0.0f);
    run->index.assign(kIndexRows, 0.0f);
    run->output.assign(kOutput, 0.0f);
    run->state_main_k.assign(kKvRows, 0.0f);
    run->state_main_v.assign(kKvRows, 0.0f);
    run->state_index_k.assign(kIndexStateRows, 0.0f);
    run->attention.assign(kAttention, 0.0f);
    run->selected.assign(kSelectedStride, 0);
    if (mode == Mode::ChainC1) {
        q38_qsa_cuda_chain_reset(&device->chain_state);
        if (!q38_qsa_cuda_chain_reserve(
                &device->chain_state, 1, device->stream, error->data(),
                error->size()))
            return false;
        const double started = now_us();
        if (!cuda_ok(cudaMemcpyAsync(
                         device->input, fixture.hidden.data(),
                         kHidden * sizeof(float), cudaMemcpyHostToDevice,
                         device->stream),
                     "QSA chain input upload", error))
            return false;
        char cuda_error[256] = {};
        if (!q38_qsa_cuda_chain_decode(
                device->q_proj, device->k_proj, device->v_proj,
                device->index_qk_proj, device->o_proj, device->q_norm,
                device->k_norm, device->index_q_norm, device->index_k_norm,
                device->input, device->output, 0, &device->chain_state,
                &device->chain_workspace, device->stream, cuda_error,
                sizeof(cuda_error))) {
            *error = cuda_error[0] ? cuda_error : "QSA chain launch failed";
            return false;
        }
        if (!cuda_ok(cudaMemcpyAsync(
                         run->output.data(), device->output,
                         kOutput * sizeof(float), cudaMemcpyDeviceToHost,
                         device->stream),
                     "QSA chain output download", error) ||
            !cuda_ok(cudaStreamSynchronize(device->stream),
                     "QSA chain synchronize", error))
            return false;
        run->sample.total_us = now_us() - started;
        run->selected.resize(fixture.selected.size());
        if (!run->selected.empty() &&
            !cuda_ok(cudaMemcpy(
                         run->selected.data(), device->chain_workspace.selected,
                         run->selected.size() * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
                     "QSA chain selected-ID validation", error))
            return false;
        run->sample.qkv_us = run->sample.total_us;
        run->sample.launches = 12;
        run->sample.syncs = 1;
        run->sample.h2d_bytes = kHidden * sizeof(float);
        run->sample.d2h_bytes = kOutput * sizeof(float);
        return true;
    }
    std::vector<float> q(kQRows), k(kKvRows), v(kKvRows), index(kIndexRows);
    const double total_started = now_us();
    const double qkv_started = now_us();
    char cuda_error[256] = {};
    const double qkv_input_copy_started = now_us();
    if (!cuda_ok(cudaMemcpyAsync(
                     device->input, fixture.hidden.data(),
                     kHidden * sizeof(float), cudaMemcpyHostToDevice,
                     device->stream),
                 "QSA QKV input upload", error))
        return false;
    run->sample.memcpy_us += now_us() - qkv_input_copy_started;
    const double qkv_dispatch_started = now_us();
    if (!q38_qsa_cuda_project_device(
            device->q_proj, kQRows, device->k_proj, kKvRows, device->v_proj,
            kKvRows, kHidden, device->input, 1, device->q, device->k,
            device->v, device->stream, cuda_error, sizeof(cuda_error))) {
        *error = cuda_error[0] ? cuda_error : "QSA CUDA QKV failed";
        return false;
    }
    run->sample.dispatch_us += now_us() - qkv_dispatch_started;
    const double qkv_output_copy_started = now_us();
    if (!cuda_ok(cudaMemcpyAsync(
                     q.data(), device->q, q.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, device->stream),
                 "q copy", error) ||
        !cuda_ok(cudaMemcpyAsync(
                     k.data(), device->k, k.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, device->stream),
                 "k copy", error) ||
        !cuda_ok(cudaMemcpyAsync(
                     v.data(), device->v, v.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, device->stream),
                 "v copy", error))
        return false;
    run->sample.memcpy_us += now_us() - qkv_output_copy_started;
    const double qkv_sync_started = now_us();
    if (!cuda_ok(cudaStreamSynchronize(device->stream), "QSA QKV sync",
                 error))
        return false;
    run->q = q;
    run->k = k;
    run->v = v;
    run->sample.sync_wait_us += now_us() - qkv_sync_started;
    run->sample.qkv_us = now_us() - qkv_started;
    run->sample.launches += 3;
    run->sample.syncs += 1;
    run->sample.h2d_bytes += kHidden * sizeof(float);
    run->sample.d2h_bytes += (kQRows + 2 * kKvRows) * sizeof(float);

    const q38_test_qsa_weight index_weight = view(fixture.index_qk_proj);
    const q38_test_qsa_weight q_norm = view(fixture.q_norm);
    const q38_test_qsa_weight k_norm = view(fixture.k_norm);
    const q38_test_qsa_weight index_q_norm = view(fixture.index_q_norm);
    const q38_test_qsa_weight index_k_norm = view(fixture.index_k_norm);
    const q38_test_qsa_weight o_proj = view(fixture.o_proj);
    const double index_started = now_us();
    if (!q38_test_qsa_project(&index_weight, fixture.hidden.data(), index.data(),
                              error->data(), error->size()))
        return false;
    run->index = index;
    run->sample.index_us = now_us() - index_started;

    const double state_started = now_us();
    if (!q38_test_qsa_build_state(
            q.data(), k.data(), v.data(), index.data(), &k_norm,
            run->state_main_k.data(), run->state_main_v.data(),
            run->state_index_k.data(), error->data(), error->size()))
        return false;
    run->sample.state_us = now_us() - state_started;

    const double attention_started = now_us();
    size_t selected_count = 0;
    if (!q38_test_qsa_attention(
            q.data(), index.data(), run->state_main_k.data(),
            run->state_main_v.data(), run->state_index_k.data(),
            fixture.state_count, &q_norm, &index_q_norm, &index_k_norm,
            run->attention.data(), run->selected.data(), kSelectedStride,
            &selected_count, error->data(), error->size()))
        return false;
    run->selected.resize(selected_count);
    run->sample.attention_us = now_us() - attention_started;

    if (mode == Mode::Baseline) {
        const double output_started = now_us();
        if (!q38_test_qsa_output_project(
                &o_proj, run->attention.data(), run->output.data(),
                error->data(), error->size()))
            return false;
        run->sample.output_us = now_us() - output_started;
    } else {
        const size_t input_bytes = kAttention * sizeof(float);
        const size_t output_bytes = kOutput * sizeof(float);
        const double dispatch_started = now_us();
        if (!cuda_ok(cudaMemcpyAsync(
                         device->output_input, run->attention.data(),
                         input_bytes, cudaMemcpyHostToDevice, device->stream),
                     "QSA-C1 attention upload", error) ||
            !q38_cuda_bf16_matvec_configured(
                device->o_proj, kOutput, kAttention, device->output_input,
                device->output, 256, device->stream, cuda_error,
                sizeof(cuda_error))) {
            *error = cuda_error[0] ? cuda_error : "QSA-C1 launch failed";
            return false;
        }
        run->sample.dispatch_us += now_us() - dispatch_started;
        run->sample.launches++;
        const double copy_started = now_us();
        if (!cuda_ok(cudaMemcpyAsync(
                         run->output.data(), device->output, output_bytes,
                         cudaMemcpyDeviceToHost, device->stream),
                     "QSA-C1 output download", error))
            return false;
        run->sample.memcpy_us += now_us() - copy_started;
        const double sync_started = now_us();
        if (!cuda_ok(cudaStreamSynchronize(device->stream),
                     "QSA-C1 synchronize", error))
            return false;
        run->sample.sync_wait_us += now_us() - sync_started;
        run->sample.syncs++;
        run->sample.h2d_bytes += input_bytes;
        run->sample.d2h_bytes += output_bytes;
            run->sample.output_us = now_us() - dispatch_started;
    }
    run->sample.total_us = now_us() - total_started;
    return true;
}

static bool validate_run(const Fixture &fixture, const Run &run,
                         std::string *error, Compare *q, Compare *k,
                         Compare *v, Compare *index, Compare *state_k,
                         Compare *state_v, Compare *state_index,
                         Compare *attention, Compare *output,
                         bool *selected_match) {
    *q = compare(run.q.data(), fixture.q_projection.data(), run.q.size());
    *v = compare(run.v.data(), fixture.values.data(), run.v.size());
    *index = compare(run.index.data(), fixture.index_projection.data(),
                     run.index.size());
    *state_k = compare(run.state_main_k.data(), fixture.state_main_k.data(),
                       run.state_main_k.size());
    // The fixture stores keys after RMS/RoPE; state_main_k is therefore the
    // expected representation of the projected K path.
    *k = *state_k;
    *state_v = compare(run.state_main_v.data(), fixture.state_main_v.data(),
                       run.state_main_v.size());
    *state_index = compare(run.state_index_k.data(),
                           fixture.state_index_k.data(),
                           run.state_index_k.size());
    *attention = compare(run.attention.data(), fixture.attention.data(),
                         run.attention.size());
    *output = compare(run.output.data(), fixture.expected_output.data(),
                      run.output.size());
    *selected_match = run.selected == fixture.selected;
    if (!compare_passes(*q) || !compare_passes(*k) || !compare_passes(*v) ||
        !compare_passes(*index) || !compare_passes(*state_k) ||
        !compare_passes(*state_v) || !compare_passes(*state_index) ||
        !compare_passes(*attention) || !compare_passes(*output) ||
        !*selected_match) {
        *error = "QSA correctness validation failed: q_abs=" +
                 std::to_string(q->max_abs) +
                 " k_abs=" + std::to_string(k->max_abs) +
                 " v_abs=" + std::to_string(v->max_abs) +
                 " index_abs=" + std::to_string(index->max_abs) +
                 " state_k_abs=" + std::to_string(state_k->max_abs) +
                 " state_v_abs=" + std::to_string(state_v->max_abs) +
                 " state_index_abs=" + std::to_string(state_index->max_abs) +
                 " attention_abs=" + std::to_string(attention->max_abs) +
                 " output_abs=" + std::to_string(output->max_abs);
        return false;
    }
    return true;
}

static Stats sample_stats(const std::vector<Sample> &samples,
                          double Sample::*field) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const Sample &sample : samples)
        values.push_back(sample.*field);
    return summarize(std::move(values));
}

static void emit_metric(FILE *artifact, const char *name, const Stats &stats) {
    std::fprintf(artifact,
                 "\"%s\":{\"median_us\":%.6f,\"p95_us\":%.6f,"
                 "\"min_us\":%.6f,\"max_us\":%.6f}",
                 name, stats.median, stats.p95, stats.min, stats.max);
}

static bool run_case(const std::string &root, const char *name, FILE *artifact,
                     bool *all_correct, Mode candidate_mode) {
    Fixture fixture;
    std::string error;
    const std::string dir = root + "/" + name;
    if (!load_fixture(dir, &fixture, &error)) {
        std::fprintf(stderr, "%s: %s\n", name, error.c_str());
        return false;
    }
    Device device;
    if (!alloc_device(fixture, &device, &error) ||
        !upload_weights(fixture, &device, &error)) {
        std::fprintf(stderr, "%s: %s\n", name, error.c_str());
        free_device(&device);
        return false;
    }
    device.chain_workspace.qfull = device.q;
    device.chain_workspace.q = device.output_input;
    device.chain_workspace.k = device.k;
    device.chain_workspace.v = device.v;
    device.chain_workspace.selected_capacity = kSelectedStride;
    std::vector<Sample> baseline_samples, c1_samples;
    baseline_samples.reserve(kSamples);
    c1_samples.reserve(kSamples);
    Compare c1_output;
    bool baseline_correct = true;
    bool c1_correct = true;
    bool baseline_selected = false;
    bool c1_selected = false;
    for (size_t i = 0; i < kWarmups; ++i) {
        Run warmup;
        if (!run_once(fixture, &device, Mode::Baseline, &warmup, &error)) {
            std::fprintf(stderr, "%s baseline warmup: %s\n", name,
                         error.c_str());
            free_device(&device);
            return false;
        }
        if (!run_once(fixture, &device, candidate_mode, &warmup, &error)) {
            std::fprintf(stderr, "%s candidate warmup: %s\n", name,
                         error.c_str());
            free_device(&device);
            return false;
        }
    }
    for (size_t i = 0; i < kSamples; ++i) {
        Run run;
        if (!run_once(fixture, &device, Mode::Baseline, &run, &error)) {
            std::fprintf(stderr, "%s baseline: %s\n", name, error.c_str());
            free_device(&device);
            return false;
        }
        baseline_samples.push_back(run.sample);
        Compare q, k, v, index, state_k, state_v, state_index, attention,
            output;
        bool selected_match = false;
        const bool baseline_valid = validate_run(
            fixture, run, &error, &q, &k, &v, &index, &state_k, &state_v,
            &state_index,
            &attention, &output, &selected_match);
        if (!baseline_valid)
            std::fprintf(stderr, "%s baseline correctness: %s\n", name,
                         error.c_str());
        baseline_correct &= baseline_valid;
        baseline_selected |= selected_match;
        if (!run_once(fixture, &device, candidate_mode, &run, &error)) {
            std::fprintf(stderr, "%s candidate: %s\n", name, error.c_str());
            free_device(&device);
            return false;
        }
        c1_samples.push_back(run.sample);
        bool c1_valid;
        if (candidate_mode == Mode::ChainC1) {
            output = compare(run.output.data(), fixture.expected_output.data(),
                             run.output.size());
            c1_valid = compare_passes(output);
            selected_match = run.selected == fixture.selected;
            c1_valid = c1_valid && selected_match;
            if (!c1_valid)
                error = "QSA chain output validation failed: max_abs=" +
                        std::to_string(output.max_abs);
        } else {
            c1_valid = validate_run(
                fixture, run, &error, &q, &k, &v, &index, &state_k, &state_v,
                &state_index, &attention, &output, &selected_match);
        }
        if (!c1_valid)
            std::fprintf(stderr, "%s C1 correctness: %s\n", name,
                         error.c_str());
        c1_correct &= c1_valid;
        c1_selected |= selected_match;
        c1_output = output;
    }
    free_device(&device);
    const Stats baseline_total = sample_stats(baseline_samples, &Sample::total_us);
    const Stats c1_total = sample_stats(c1_samples, &Sample::total_us);
    const double saved = baseline_total.median - c1_total.median;
    const double reduction = baseline_total.median > 0.0
        ? saved / baseline_total.median * 100.0 : 0.0;
    *all_correct &= baseline_correct && c1_correct;
    std::fprintf(
        artifact,
        "{\"fixture\":\"%s\",\"layer\":%u,\"baseline\":{",
        name, fixture.layer);
    emit_metric(artifact, "total", baseline_total);
    std::fprintf(artifact, ",");
    emit_metric(artifact, "qkv", sample_stats(baseline_samples, &Sample::qkv_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "index_compress",
                sample_stats(baseline_samples, &Sample::index_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "state_update",
                sample_stats(baseline_samples, &Sample::state_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "attention",
                sample_stats(baseline_samples, &Sample::attention_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "output_projection",
                sample_stats(baseline_samples, &Sample::output_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "dispatch",
                sample_stats(baseline_samples, &Sample::dispatch_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "sync_wait",
                sample_stats(baseline_samples, &Sample::sync_wait_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "memcpy",
                sample_stats(baseline_samples, &Sample::memcpy_us));
    std::fprintf(artifact, ",\"launches\":%" PRIu64
                     ",\"syncs\":%" PRIu64
                     ",\"h2d_bytes\":%" PRIu64
                     ",\"d2h_bytes\":%" PRIu64
                     "},\"%s\":{",
                     baseline_samples.front().launches,
                     baseline_samples.front().syncs,
                     baseline_samples.front().h2d_bytes,
                     baseline_samples.front().d2h_bytes,
                     candidate_mode == Mode::ChainC1 ? "chain_c1" : "c1");
    emit_metric(artifact, "total", c1_total);
    std::fprintf(artifact, ",");
    emit_metric(artifact, "qkv", sample_stats(c1_samples, &Sample::qkv_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "index_compress",
                sample_stats(c1_samples, &Sample::index_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "state_update",
                sample_stats(c1_samples, &Sample::state_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "attention",
                sample_stats(c1_samples, &Sample::attention_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "output_projection",
                sample_stats(c1_samples, &Sample::output_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "dispatch", sample_stats(c1_samples,
                                                    &Sample::dispatch_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "sync_wait", sample_stats(c1_samples,
                                                    &Sample::sync_wait_us));
    std::fprintf(artifact, ",");
    emit_metric(artifact, "memcpy", sample_stats(c1_samples,
                                                 &Sample::memcpy_us));
    std::fprintf(artifact, ",\"launches\":%" PRIu64
                     ",\"syncs\":%" PRIu64
                     ",\"h2d_bytes\":%" PRIu64
                     ",\"d2h_bytes\":%" PRIu64
                     "},\"promotion\":{\"median_saved_us\":%.6f,"
                     "\"relative_reduction_percent\":%.6f,"
                     "\"promoted\":%s},\"correctness\":{\"baseline\":%s,"
                     "\"%s\":%s,\"selected_ids_exact\":%s,"
                     "\"max_abs\":%.9g,\"max_rel\":%.9g,\"rmse\":%.9g,"
                     "\"nonfinite\":0}}\n",
                 c1_samples.front().launches, c1_samples.front().syncs,
                 c1_samples.front().h2d_bytes, c1_samples.front().d2h_bytes,
                 saved, reduction, c1_correct && reduction >= 10.0 ? "true" :
                 "false", baseline_correct ? "true" : "false",
                 candidate_mode == Mode::ChainC1 ? "chain_c1" : "c1",
                 c1_correct ? "true" : "false",
                 baseline_selected && c1_selected ? "true" : "false",
                 c1_output.max_abs, c1_output.max_rel, c1_output.rmse);
    std::printf(
        "{\"fixture\":\"%s\",\"layer\":%u,\"mode\":\"%s\","
        "\"baseline_median_us\":%.3f,"
        "\"c1_median_us\":%.3f,\"saved_us\":%.3f,"
        "\"reduction_percent\":%.3f,\"baseline_correct\":%s,"
        "\"c1_correct\":%s,\"selected_ids_exact\":%s}\n",
        name, fixture.layer,
        candidate_mode == Mode::ChainC1 ? "qsa_chain_c1" : "qsa_c1",
        baseline_total.median, c1_total.median, saved,
        reduction, baseline_correct ? "true" : "false",
        c1_correct ? "true" : "false",
        baseline_selected && c1_selected ? "true" : "false");
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string root = argc > 1 ? argv[1] : "tests/fixtures/qsa";
    const std::string artifact =
        argc > 2 ? argv[2] : "artifacts/perf/subsystems/qsa_reference.json";
    Mode candidate_mode = Mode::C1;
    if (argc > 3) {
        if (std::strcmp(argv[3], "qsa_chain_c1") == 0)
            candidate_mode = Mode::ChainC1;
        else if (std::strcmp(argv[3], "qsa_c1") != 0) {
            std::fprintf(stderr,
                         "usage: qsa_bench FIXTURE_ROOT ARTIFACT "
                         "[qsa_c1|qsa_chain_c1]\n");
            return 2;
        }
    }
    FILE *out = std::fopen(artifact.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "failed to open %s\n", artifact.c_str());
        return 1;
    }
    std::fprintf(out,
                 "{\"schema\":\"QSA_SUBSYSTEM_V1\","
                 "\"mode\":\"%s\","
                 "\"fixture_root\":\"%s\",\"warmups\":%zu,"
                 "\"samples\":%zu,\"fixtures\":[\n",
                 candidate_mode == Mode::ChainC1 ? "qsa_chain_c1" : "qsa_c1",
                 root.c_str(), kWarmups, kSamples);
    bool all_correct = true;
    const char *names[] = {"early", "middle", "late"};
    bool ok = true;
    for (size_t i = 0; i < 3; ++i) {
        if (i) std::fprintf(out, ",");
        if (!run_case(root, names[i], out, &all_correct, candidate_mode))
            ok = false;
    }
    std::fprintf(out, "]}\n");
    std::fclose(out);
    return ok && all_correct ? 0 : 1;
}
