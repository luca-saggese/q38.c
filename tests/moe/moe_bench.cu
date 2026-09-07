#include "moe_reference.h"

#include "../../q38_cuda_primitives.h"
#include "../../q38_moe_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <inttypes.h>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr size_t kWarmup = 100;
constexpr size_t kSamples = 1000;
constexpr size_t kRouterBytes =
    Q38_MOE_EXPERTS * Q38_MOE_HIDDEN * sizeof(uint16_t);
constexpr size_t kHiddenBytes = Q38_MOE_HIDDEN * sizeof(float);
constexpr size_t kIntermediateBytes = Q38_MOE_INTERMEDIATE * sizeof(float);
constexpr size_t kOutputBytes = Q38_MOE_HIDDEN * sizeof(float);
constexpr size_t kGateUpBytes =
    Q38_TEST_MOE_GATE_UP_BLOCKS * sizeof(q38_q2_k_block);
constexpr size_t kDownBytes =
    Q38_TEST_MOE_DOWN_BLOCKS * sizeof(q38_q2_k_block);

static float bf16_to_float(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

struct Fixture {
    std::vector<float> hidden;
    std::vector<uint16_t> router;
    std::vector<uint16_t> selected_experts;
    std::vector<float> selected_weights;
    std::vector<q38_q2_k_block> gate_up;
    std::vector<q38_q2_k_block> down;
    std::vector<uint16_t> shared_gate_bf16;
    std::vector<uint16_t> shared_up_bf16;
    std::vector<uint16_t> shared_down_bf16;
    std::vector<uint16_t> shared_weight_bf16;
    std::vector<float> expected_routed;
    std::vector<float> expected_shared;
    std::vector<float> expected;
};

struct Device {
    cudaStream_t stream = nullptr;
    uint16_t *router = nullptr;
    q38_q2_k_block *gate_up = nullptr;
    q38_q2_k_block *down = nullptr;
    uint16_t *shared_gate = nullptr;
    uint16_t *shared_up = nullptr;
    uint16_t *shared_down = nullptr;
    float *hidden = nullptr;
    float *logits = nullptr;
    float *mid = nullptr;
    float *expert = nullptr;
    float *accum = nullptr;
    float *grouped_mid = nullptr;
    float *expert_outputs = nullptr;
    uint16_t *route_ids = nullptr;
    float *route_weights = nullptr;
    float *shared_mid = nullptr;
    float *shared_output = nullptr;
};

struct Scope {
    double median_us = 0.0;
    double p95_us = 0.0;
    size_t launches = 0;
    size_t syncs = 0;
    size_t h2d_bytes = 0;
    size_t d2h_bytes = 0;
    size_t d2d_bytes = 0;
    size_t bytes_read = 0;
};

struct Determinism {
    size_t samples = 0;
    size_t unique_output_hashes = 0;
    size_t unique_mid_hashes = 0;
    uint64_t first_output_hash = 0;
    uint64_t first_mid_hash = 0;
    double max_abs_vs_first = 0.0;
    double max_rel_vs_first = 0.0;
    double rmse_vs_first = 0.0;
    bool finite = true;
};

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const unsigned char *cursor =
        static_cast<const unsigned char *>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= cursor[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static bool compare_repeat(const std::vector<float> &reference,
                           const std::vector<float> &current,
                           double *max_abs, double *max_rel, double *rmse,
                           bool *finite) {
    if (reference.size() != current.size() || !max_abs || !max_rel ||
        !rmse || !finite)
        return false;
    double sum = 0.0;
    *max_abs = 0.0;
    *max_rel = 0.0;
    *finite = true;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(current[i])) *finite = false;
        const double difference =
            std::abs((double)current[i] - reference[i]);
        const double scale = std::max(std::abs((double)reference[i]), 1e-12);
        *max_abs = std::max(*max_abs, difference);
        *max_rel = std::max(*max_rel, difference / scale);
        sum += difference * difference;
    }
    *rmse = std::sqrt(sum / reference.size());
    return true;
}

static std::string path_join(const std::string &dir, const char *name) {
    return dir + "/" + name;
}

static bool read_file(const std::string &path, void *data, size_t bytes,
                      std::string *error) {
    FILE *file = fopen(path.c_str(), "rb");
    if (!file) {
        *error = "missing fixture file: " + path;
        return false;
    }
    const size_t read = fread(data, 1, bytes, file);
    const bool closed = fclose(file) == 0;
    const bool ok = read == bytes && closed;
    if (!ok) *error = "short or unreadable fixture file: " + path;
    return ok;
}

static bool load_fixture(const std::string &dir, Fixture *fixture,
                         std::string *error) {
    fixture->hidden.resize(Q38_MOE_HIDDEN);
    fixture->router.resize(Q38_TEST_MOE_ROUTER_VALUES);
    fixture->selected_experts.resize(Q38_MOE_TOP_K);
    fixture->selected_weights.resize(Q38_MOE_TOP_K);
    fixture->gate_up.resize(Q38_MOE_TOP_K * Q38_TEST_MOE_GATE_UP_BLOCKS);
    fixture->down.resize(Q38_MOE_TOP_K * Q38_TEST_MOE_DOWN_BLOCKS);
    fixture->shared_gate_bf16.resize(
        Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN);
    fixture->shared_up_bf16.resize(
        Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN);
    fixture->shared_down_bf16.resize(
        Q38_MOE_HIDDEN * Q38_MOE_INTERMEDIATE);
    fixture->shared_weight_bf16.resize(Q38_MOE_HIDDEN);
    fixture->expected_routed.resize(Q38_MOE_HIDDEN);
    fixture->expected_shared.resize(Q38_MOE_HIDDEN);
    fixture->expected.resize(Q38_MOE_HIDDEN);
    return
        read_file(path_join(dir, "hidden.f32"), fixture->hidden.data(),
                  kHiddenBytes, error) &&
        read_file(path_join(dir, "router.bf16"), fixture->router.data(),
                  kRouterBytes, error) &&
        read_file(path_join(dir, "selected_experts.u16"),
                  fixture->selected_experts.data(),
                  Q38_MOE_TOP_K * sizeof(uint16_t), error) &&
        read_file(path_join(dir, "selected_weights.f32"),
                  fixture->selected_weights.data(),
                  Q38_MOE_TOP_K * sizeof(float), error) &&
        read_file(path_join(dir, "selected_gate_up.q2_k"),
                  fixture->gate_up.data(),
                  kGateUpBytes * Q38_MOE_TOP_K, error) &&
        read_file(path_join(dir, "selected_down.q2_k"), fixture->down.data(),
                  kDownBytes * Q38_MOE_TOP_K, error) &&
        read_file(path_join(dir, "shared_gate.bf16"),
                  fixture->shared_gate_bf16.data(),
                  fixture->shared_gate_bf16.size() * sizeof(uint16_t), error) &&
        read_file(path_join(dir, "shared_up.bf16"),
                  fixture->shared_up_bf16.data(),
                  fixture->shared_up_bf16.size() * sizeof(uint16_t), error) &&
        read_file(path_join(dir, "shared_down.bf16"),
                  fixture->shared_down_bf16.data(),
                  fixture->shared_down_bf16.size() * sizeof(uint16_t), error) &&
        read_file(path_join(dir, "shared_gate_weight.bf16"),
                  fixture->shared_weight_bf16.data(),
                  fixture->shared_weight_bf16.size() * sizeof(uint16_t),
                  error) &&
        read_file(path_join(dir, "expected_routed.f32"),
                  fixture->expected_routed.data(), kOutputBytes, error) &&
        read_file(path_join(dir, "expected_shared.f32"),
                  fixture->expected_shared.data(), kOutputBytes, error) &&
        read_file(path_join(dir, "expected.f32"), fixture->expected.data(),
                  kOutputBytes, error);
}

static bool cuda_ok(cudaError_t status, const char *what, std::string *error) {
    if (status == cudaSuccess) return true;
    *error = std::string(what) + ": " + cudaGetErrorString(status);
    return false;
}

static bool alloc_device(Device *device, const Fixture &fixture,
                         std::string *error) {
    if (!cuda_ok(cudaStreamCreate(&device->stream), "cudaStreamCreate",
                 error))
        return false;
    const auto alloc = [&](void **ptr, size_t bytes, const char *what) {
        return cuda_ok(cudaMalloc(ptr, bytes), what, error);
    };
    return alloc((void **)&device->router, kRouterBytes, "router allocation") &&
           alloc((void **)&device->gate_up, kGateUpBytes * Q38_MOE_TOP_K,
                 "gate/up allocation") &&
           alloc((void **)&device->down, kDownBytes * Q38_MOE_TOP_K,
                 "down allocation") &&
           alloc((void **)&device->shared_gate,
                 fixture.shared_gate_bf16.size() * sizeof(uint16_t),
                 "shared gate allocation") &&
           alloc((void **)&device->shared_up,
                 fixture.shared_up_bf16.size() * sizeof(uint16_t),
                 "shared up allocation") &&
           alloc((void **)&device->shared_down,
                 fixture.shared_down_bf16.size() * sizeof(uint16_t),
                 "shared down allocation") &&
           alloc((void **)&device->hidden, kHiddenBytes, "hidden allocation") &&
           alloc((void **)&device->logits,
                 Q38_MOE_EXPERTS * sizeof(float), "logit allocation") &&
           alloc((void **)&device->mid, kIntermediateBytes,
                 "intermediate allocation") &&
           alloc((void **)&device->expert, kOutputBytes,
                 "expert allocation") &&
           alloc((void **)&device->accum, kOutputBytes,
                 "accumulator allocation") &&
           alloc((void **)&device->grouped_mid,
                 kIntermediateBytes * Q38_MOE_TOP_K,
                 "grouped intermediate allocation") &&
           alloc((void **)&device->expert_outputs,
                 kOutputBytes * Q38_MOE_TOP_K,
                 "grouped expert output allocation") &&
           alloc((void **)&device->route_ids,
                 Q38_MOE_TOP_K * sizeof(uint16_t), "route ID allocation") &&
           alloc((void **)&device->route_weights,
                 Q38_MOE_TOP_K * sizeof(float), "route weight allocation") &&
           alloc((void **)&device->shared_mid, kIntermediateBytes,
                 "shared intermediate allocation") &&
           alloc((void **)&device->shared_output, kOutputBytes,
                 "shared output allocation");
}

static void free_device(Device *device) {
    cudaFree(device->router);
    cudaFree(device->gate_up);
    cudaFree(device->down);
    cudaFree(device->shared_gate);
    cudaFree(device->shared_up);
    cudaFree(device->shared_down);
    cudaFree(device->hidden);
    cudaFree(device->logits);
    cudaFree(device->mid);
    cudaFree(device->expert);
    cudaFree(device->accum);
    cudaFree(device->grouped_mid);
    cudaFree(device->expert_outputs);
    cudaFree(device->route_ids);
    cudaFree(device->route_weights);
    cudaFree(device->shared_mid);
    cudaFree(device->shared_output);
    if (device->stream) cudaStreamDestroy(device->stream);
}

static bool upload_fixture(Device *device, const Fixture &fixture,
                           std::string *error) {
    const auto copy = [&](void *dst, const void *src, size_t bytes,
                          const char *what) {
        return cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                       device->stream),
                       what, error);
    };
    return copy(device->router, fixture.router.data(), kRouterBytes,
                "router upload") &&
           copy(device->gate_up, fixture.gate_up.data(),
                kGateUpBytes * Q38_MOE_TOP_K, "gate/up upload") &&
           copy(device->down, fixture.down.data(),
                kDownBytes * Q38_MOE_TOP_K, "down upload") &&
           copy(device->shared_gate, fixture.shared_gate_bf16.data(),
                fixture.shared_gate_bf16.size() * sizeof(uint16_t),
                "shared gate upload") &&
           copy(device->shared_up, fixture.shared_up_bf16.data(),
                fixture.shared_up_bf16.size() * sizeof(uint16_t),
                "shared up upload") &&
           copy(device->shared_down, fixture.shared_down_bf16.data(),
                fixture.shared_down_bf16.size() * sizeof(uint16_t),
                "shared down upload") &&
           copy(device->hidden, fixture.hidden.data(), kHiddenBytes,
                "hidden upload") &&
           cuda_ok(cudaStreamSynchronize(device->stream), "fixture sync",
                   error);
}

static bool run_router(Device *device, const Fixture &fixture,
                       std::string *error) {
    std::vector<float> logits(Q38_MOE_EXPERTS);
    q38_test_moe_route route = {};
    float effective[Q38_MOE_EXPERTS];
    float weights_pre[Q38_MOE_EXPERTS];
    float weights_effective[Q38_MOE_EXPERTS];
    return cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                   kHiddenBytes, cudaMemcpyHostToDevice,
                                   device->stream),
                   "router input upload", error) &&
           q38_cuda_bf16_matvec(
               device->router, Q38_MOE_EXPERTS, Q38_MOE_HIDDEN, device->hidden,
               device->logits, device->stream, nullptr, 0) &&
           cuda_ok(cudaMemcpyAsync(logits.data(), device->logits,
                                   Q38_MOE_EXPERTS * sizeof(float),
                                   cudaMemcpyDeviceToHost, device->stream),
                   "router logits download", error) &&
           cuda_ok(cudaStreamSynchronize(device->stream), "router sync",
                   error) &&
           q38_test_moe_select_logits(
               logits.data(), &route, effective, weights_pre, weights_effective,
               nullptr, 0);
}

static bool run_routed(Device *device, const Fixture &fixture,
                       std::vector<float> *output, std::string *error) {
    if (!cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                 kHiddenBytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "routed input upload", error) ||
        !cuda_ok(cudaMemsetAsync(device->accum, 0, kOutputBytes,
                                 device->stream),
                 "accumulator clear", error))
        return false;
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        const q38_q2_k_block *gate_up =
            device->gate_up + k * Q38_TEST_MOE_GATE_UP_BLOCKS;
        const q38_q2_k_block *down =
            device->down + k * Q38_TEST_MOE_DOWN_BLOCKS;
        if (!q38_moe_cuda_q2_gate_up(
                gate_up, device->hidden, device->mid, device->stream, nullptr,
                0) ||
            !q38_moe_cuda_q2_down(
                down, device->mid, device->expert, device->stream, nullptr,
                0) ||
            !q38_moe_cuda_accumulate_weighted(
                device->accum, device->expert, fixture.selected_weights[k],
                device->stream, nullptr, 0))
            return false;
    }
    output->resize(Q38_MOE_HIDDEN);
    return cuda_ok(cudaMemcpyAsync(output->data(), device->accum, kOutputBytes,
                                   cudaMemcpyDeviceToHost, device->stream),
                   "routed download", error) &&
           cuda_ok(cudaStreamSynchronize(device->stream), "routed sync",
                   error);
}

static bool run_routed_grouped(Device *device, const Fixture &fixture,
                               std::vector<float> *output,
                               std::string *error,
                               bool deterministic = false) {
    uint16_t route_ids[Q38_MOE_TOP_K];
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
        route_ids[k] = (uint16_t)k;
    if (!cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                 kHiddenBytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "grouped routed input upload", error) ||
        !cuda_ok(cudaMemcpyAsync(device->route_ids, route_ids,
                                 Q38_MOE_TOP_K * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice, device->stream),
                 "grouped route-ID upload", error) ||
        !cuda_ok(cudaMemcpyAsync(
                     device->route_weights, fixture.selected_weights.data(),
                     Q38_MOE_TOP_K * sizeof(float), cudaMemcpyHostToDevice,
                     device->stream),
                 "grouped route-weight upload", error) ||
        !(deterministic
              ? q38_moe_cuda_q2_grouped_indexed_deterministic(
                    device->gate_up, device->down, device->hidden,
                    device->route_ids, device->route_weights, Q38_MOE_TOP_K,
                    Q38_TEST_MOE_GATE_UP_BLOCKS, Q38_TEST_MOE_DOWN_BLOCKS,
                    device->accum, device->grouped_mid,
                    device->expert_outputs, device->stream, nullptr, 0)
              : q38_moe_cuda_q2_grouped_indexed(
                    device->gate_up, device->down, device->hidden,
                    device->route_ids, device->route_weights, Q38_MOE_TOP_K,
                    Q38_TEST_MOE_GATE_UP_BLOCKS, Q38_TEST_MOE_DOWN_BLOCKS,
                    device->accum, device->grouped_mid, device->stream,
                    nullptr, 0)))
        return false;
    output->resize(Q38_MOE_HIDDEN);
    return cuda_ok(cudaMemcpyAsync(output->data(), device->accum, kOutputBytes,
                                   cudaMemcpyDeviceToHost, device->stream),
                   "grouped routed download", error) &&
           cuda_ok(cudaStreamSynchronize(device->stream), "grouped routed sync",
                   error);
}

static bool run_shared(Device *device, const Fixture &fixture,
                       std::vector<float> *output, std::string *error) {
    std::vector<float> gate(Q38_MOE_INTERMEDIATE);
    std::vector<float> up(Q38_MOE_INTERMEDIATE);
    std::vector<float> mid(Q38_MOE_INTERMEDIATE);
    if (!cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                 kHiddenBytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "shared input upload", error) ||
        !q38_cuda_bf16_matvec(
            device->shared_gate, Q38_MOE_INTERMEDIATE, Q38_MOE_HIDDEN,
            device->hidden, device->shared_mid, device->stream, nullptr, 0) ||
        !cuda_ok(cudaMemcpyAsync(gate.data(), device->shared_mid,
                                 kIntermediateBytes, cudaMemcpyDeviceToHost,
                                 device->stream),
                 "shared gate download", error) ||
        !cuda_ok(cudaStreamSynchronize(device->stream), "shared gate sync",
                 error))
        return false;
    if (!cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                 kHiddenBytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "shared input upload", error) ||
        !q38_cuda_bf16_matvec(
            device->shared_up, Q38_MOE_INTERMEDIATE, Q38_MOE_HIDDEN,
            device->hidden, device->shared_mid, device->stream, nullptr, 0) ||
        !cuda_ok(cudaMemcpyAsync(up.data(), device->shared_mid,
                                 kIntermediateBytes, cudaMemcpyDeviceToHost,
                                 device->stream),
                 "shared up download", error) ||
        !cuda_ok(cudaStreamSynchronize(device->stream), "shared up sync",
                 error))
        return false;
    for (size_t i = 0; i < mid.size(); ++i)
        mid[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
    if (!cuda_ok(cudaMemcpyAsync(device->shared_mid, mid.data(),
                                 kIntermediateBytes,
                                 cudaMemcpyHostToDevice, device->stream),
                 "shared mid upload", error) ||
        !q38_cuda_bf16_matvec(
            device->shared_down, Q38_MOE_HIDDEN, Q38_MOE_INTERMEDIATE,
            device->shared_mid, device->shared_output, device->stream, nullptr,
            0))
        return false;
    output->resize(Q38_MOE_HIDDEN);
    return cuda_ok(cudaMemcpyAsync(output->data(), device->shared_output,
                                   kOutputBytes, cudaMemcpyDeviceToHost,
                                   device->stream),
                   "shared output download", error) &&
           cuda_ok(cudaStreamSynchronize(device->stream), "shared down sync",
                   error);
}

static bool run_complete(Device *device, const Fixture &fixture,
                         std::vector<float> *output, std::string *error,
                         bool grouped = false, bool deterministic = false) {
    std::vector<float> logits(Q38_MOE_EXPERTS);
    std::vector<float> shared_gate(Q38_MOE_INTERMEDIATE);
    std::vector<float> shared_up(Q38_MOE_INTERMEDIATE);
    std::vector<float> shared_mid(Q38_MOE_INTERMEDIATE);
    std::vector<float> shared_output(Q38_MOE_HIDDEN);
    q38_test_moe_route route = {};
    float effective[Q38_MOE_EXPERTS];
    float pre_weights[Q38_MOE_EXPERTS];
    float effective_weights[Q38_MOE_EXPERTS];
    if (!cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                 kHiddenBytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "layer router upload", error) ||
        !q38_cuda_bf16_matvec(
            device->router, Q38_MOE_EXPERTS, Q38_MOE_HIDDEN, device->hidden,
            device->logits, device->stream, nullptr, 0) ||
        !cuda_ok(cudaMemcpyAsync(logits.data(), device->logits,
                                 Q38_MOE_EXPERTS * sizeof(float),
                                 cudaMemcpyDeviceToHost, device->stream),
                 "router logits download", error) ||
        !cuda_ok(cudaStreamSynchronize(device->stream), "router layer sync",
                 error) ||
        !q38_test_moe_select_logits(
            logits.data(), &route, effective, pre_weights, effective_weights,
            nullptr, 0))
        return false;
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
        if (route.expert[k] != fixture.selected_experts[k] ||
            std::abs(route.weight[k] - fixture.selected_weights[k]) > 1e-5f) {
            *error = "GPU router selection disagrees with fixture metadata";
            return false;
        }
    if (!cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                 kHiddenBytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "shared gate input upload", error) ||
        !q38_cuda_bf16_matvec(
            device->shared_gate, Q38_MOE_INTERMEDIATE, Q38_MOE_HIDDEN,
            device->hidden, device->shared_mid, device->stream, nullptr, 0) ||
        !cuda_ok(cudaMemcpyAsync(shared_gate.data(), device->shared_mid,
                                 kIntermediateBytes, cudaMemcpyDeviceToHost,
                                 device->stream),
                 "shared gate layer download", error) ||
        !cuda_ok(cudaStreamSynchronize(device->stream), "shared gate layer sync",
                 error))
        return false;
    if (!cuda_ok(cudaMemcpyAsync(device->hidden, fixture.hidden.data(),
                                 kHiddenBytes, cudaMemcpyHostToDevice,
                                 device->stream),
                 "shared up input upload", error) ||
        !q38_cuda_bf16_matvec(
            device->shared_up, Q38_MOE_INTERMEDIATE, Q38_MOE_HIDDEN,
            device->hidden, device->shared_mid, device->stream, nullptr, 0) ||
        !cuda_ok(cudaMemcpyAsync(shared_up.data(), device->shared_mid,
                                 kIntermediateBytes, cudaMemcpyDeviceToHost,
                                 device->stream),
                 "shared up layer download", error) ||
        !cuda_ok(cudaStreamSynchronize(device->stream), "shared up layer sync",
                 error))
        return false;
    for (size_t i = 0; i < shared_mid.size(); ++i)
        shared_mid[i] =
            shared_gate[i] / (1.0f + expf(-shared_gate[i])) * shared_up[i];
    std::vector<float> routed;
    if (grouped) {
        if (!run_routed_grouped(device, fixture, &routed, error,
                                deterministic))
            return false;
    } else if (!run_routed(device, fixture, &routed, error)) {
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(device->shared_mid, shared_mid.data(),
                                 kIntermediateBytes,
                                 cudaMemcpyHostToDevice, device->stream),
                 "shared down input upload", error) ||
        !q38_cuda_bf16_matvec(
            device->shared_down, Q38_MOE_HIDDEN, Q38_MOE_INTERMEDIATE,
            device->shared_mid, device->shared_output, device->stream, nullptr,
            0) ||
        !cuda_ok(cudaMemcpyAsync(shared_output.data(), device->shared_output,
                                 kOutputBytes, cudaMemcpyDeviceToHost,
                                 device->stream),
                 "shared down layer download", error) ||
        !cuda_ok(cudaStreamSynchronize(device->stream), "shared down layer sync",
                 error))
        return false;
    float shared_gate_value = 0.0f;
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
        shared_gate_value +=
            bf16_to_float(fixture.shared_weight_bf16[d]) *
            fixture.hidden[d];
    shared_gate_value = 1.0f / (1.0f + expf(-shared_gate_value));
    output->resize(Q38_MOE_HIDDEN);
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
        (*output)[d] = shared_gate_value * shared_output[d] + routed[d];
    }
    return true;
}

static bool measure(const std::function<bool()> &operation, Scope *scope,
                    std::string *error) {
    std::vector<double> samples;
    samples.reserve(kSamples);
    for (size_t i = 0; i < kWarmup; ++i)
        if (!operation()) {
            *error = "warmup operation failed";
            return false;
        }
    for (size_t i = 0; i < kSamples; ++i) {
        const auto started = std::chrono::steady_clock::now();
        if (!operation()) {
            *error = "measured operation failed";
            return false;
        }
        const auto ended = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(
                              ended - started)
                              .count());
    }
    std::sort(samples.begin(), samples.end());
    scope->median_us = samples[samples.size() / 2];
    scope->p95_us = samples[(samples.size() * 95) / 100];
    return true;
}

static bool compare_output(const std::vector<float> &actual,
                           const std::vector<float> &expected, double *max_abs,
                           double *rmse) {
    *max_abs = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) return false;
        const double diff = std::abs((double)actual[i] - expected[i]);
        *max_abs = std::max(*max_abs, diff);
        sum += diff * diff;
    }
    *rmse = std::sqrt(sum / actual.size());
    return true;
}

static void print_scope(const char *name, const Scope &scope,
                        bool comma = true) {
    printf("    \"%s\":{\"median_us\":%.3f,\"p95_us\":%.3f,"
           "\"kernel_launches\":%zu,\"host_syncs\":%zu,"
           "\"h2d_bytes\":%zu,\"d2h_bytes\":%zu,\"d2d_bytes\":%zu,"
           "\"bytes_read\":%zu}%s\n",
           name, scope.median_us, scope.p95_us, scope.launches, scope.syncs,
           scope.h2d_bytes, scope.d2h_bytes, scope.d2d_bytes,
           scope.bytes_read, comma ? "," : "");
}

static bool probe_grouped_determinism(Device *device, const Fixture &fixture,
                                      Determinism *result,
                                      std::string *error,
                                      bool deterministic = false) {
    constexpr size_t kDeterminismSamples = 1000;
    std::vector<float> first_output;
    std::vector<float> current_output;
    std::vector<float> first_mid(Q38_MOE_TOP_K * Q38_MOE_INTERMEDIATE);
    std::vector<float> current_mid(first_mid.size());
    std::set<uint64_t> output_hashes;
    std::set<uint64_t> mid_hashes;
    if (!device || !result || !error) return false;
    *result = Determinism{};
    for (size_t sample = 0; sample < kDeterminismSamples; ++sample) {
        if (!run_routed_grouped(device, fixture, &current_output, error,
                                deterministic) ||
            !cuda_ok(cudaMemcpyAsync(
                         current_mid.data(), device->grouped_mid,
                         current_mid.size() * sizeof(float),
                         cudaMemcpyDeviceToHost, device->stream),
                     "determinism mid download", error) ||
            !cuda_ok(cudaStreamSynchronize(device->stream),
                     "determinism mid sync", error))
            return false;
        const uint64_t output_hash = hash_bytes(
            current_output.data(), current_output.size() * sizeof(float));
        const uint64_t mid_hash =
            hash_bytes(current_mid.data(), current_mid.size() * sizeof(float));
        output_hashes.insert(output_hash);
        mid_hashes.insert(mid_hash);
        if (sample == 0) {
            first_output = current_output;
            first_mid = current_mid;
            result->first_output_hash = output_hash;
            result->first_mid_hash = mid_hash;
            continue;
        }
        double max_abs = 0.0, max_rel = 0.0, rmse = 0.0;
        bool finite = true;
        if (!compare_repeat(first_output, current_output, &max_abs, &max_rel,
                            &rmse, &finite) ||
            !finite)
            result->finite = false;
        result->max_abs_vs_first =
            std::max(result->max_abs_vs_first, max_abs);
        result->max_rel_vs_first =
            std::max(result->max_rel_vs_first, max_rel);
        result->rmse_vs_first = std::max(result->rmse_vs_first, rmse);
    }
    result->samples = kDeterminismSamples;
    result->unique_output_hashes = output_hashes.size();
    result->unique_mid_hashes = mid_hashes.size();
    return true;
}

static bool bench_fixture(const char *name, const std::string &dir) {
    Fixture fixture;
    std::string error;
    if (!load_fixture(dir, &fixture, &error)) {
        fprintf(stderr, "%s: %s\n", name, error.c_str());
        return false;
    }
    Device device;
    if (!alloc_device(&device, fixture, &error) ||
        !upload_fixture(&device, fixture, &error)) {
        fprintf(stderr, "%s: %s\n", name, error.c_str());
        free_device(&device);
        return false;
    }

    std::vector<float> host_gate(
        Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN);
    std::vector<float> host_up(host_gate.size());
    std::vector<float> host_down(
        Q38_MOE_HIDDEN * Q38_MOE_INTERMEDIATE);
    std::vector<float> host_weight(Q38_MOE_HIDDEN);
    for (size_t i = 0; i < host_gate.size(); ++i) {
        host_gate[i] = bf16_to_float(fixture.shared_gate_bf16[i]);
        host_up[i] = bf16_to_float(fixture.shared_up_bf16[i]);
    }
    for (size_t i = 0; i < host_down.size(); ++i)
        host_down[i] = bf16_to_float(fixture.shared_down_bf16[i]);
    for (size_t i = 0; i < host_weight.size(); ++i)
        host_weight[i] = bf16_to_float(fixture.shared_weight_bf16[i]);
    std::vector<const q38_q2_k_block *> selected_gate(Q38_MOE_TOP_K);
    std::vector<const q38_q2_k_block *> selected_down(Q38_MOE_TOP_K);
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k) {
        selected_gate[k] =
            fixture.gate_up.data() + k * Q38_TEST_MOE_GATE_UP_BLOCKS;
        selected_down[k] =
            fixture.down.data() + k * Q38_TEST_MOE_DOWN_BLOCKS;
    }
    std::vector<float> oracle(Q38_MOE_HIDDEN);
    q38_test_moe_route oracle_route = {};
    if (!q38_test_moe_layer_q2(
            fixture.hidden.data(), fixture.router.data(), selected_gate.data(),
            selected_down.data(), host_gate.data(), host_up.data(),
            host_down.data(), host_weight.data(), oracle.data(), &oracle_route,
            error.empty() ? nullptr : nullptr, 0)) {
        fprintf(stderr, "%s: independent oracle failed\n", name);
        free_device(&device);
        return false;
    }
    for (size_t k = 0; k < Q38_MOE_TOP_K; ++k)
        if (oracle_route.expert[k] != fixture.selected_experts[k] ||
            std::abs(oracle_route.weight[k] - fixture.selected_weights[k]) >
                1e-5f) {
            fprintf(stderr, "%s: fixture routing metadata disagrees with oracle\n",
                    name);
            free_device(&device);
            return false;
        }

    Scope router = {0};
    router.launches = 1;
    router.syncs = 1;
    router.h2d_bytes = kHiddenBytes;
    router.d2h_bytes = Q38_MOE_EXPERTS * sizeof(float);
    router.bytes_read = kRouterBytes;
    Scope gate_up = {0};
    gate_up.launches = 1;
    gate_up.syncs = 1;
    gate_up.bytes_read = kGateUpBytes;
    Scope down = {0};
    down.launches = 1;
    down.syncs = 1;
    down.bytes_read = kDownBytes;
    Scope expert = {0};
    expert.launches = 2;
    expert.syncs = 1;
    expert.bytes_read = kGateUpBytes + kDownBytes;
    Scope routed = {0};
    routed.launches = Q38_MOE_TOP_K * 3;
    routed.syncs = 1;
    routed.h2d_bytes = kHiddenBytes;
    routed.d2h_bytes = kOutputBytes;
    routed.bytes_read = Q38_MOE_TOP_K * (kGateUpBytes + kDownBytes);
    Scope grouped_routed = {0};
    grouped_routed.launches = 2;
    grouped_routed.syncs = 1;
    grouped_routed.h2d_bytes =
        kHiddenBytes + Q38_MOE_TOP_K *
            (sizeof(uint16_t) + sizeof(float));
    grouped_routed.d2h_bytes = kOutputBytes;
    grouped_routed.bytes_read =
        Q38_MOE_TOP_K * (kGateUpBytes + kDownBytes);
    Scope deterministic_grouped_routed = grouped_routed;
    deterministic_grouped_routed.launches = 3;
    Scope shared = {0};
    shared.launches = 3;
    shared.syncs = 3;
    shared.h2d_bytes = 3 * kHiddenBytes + kIntermediateBytes;
    shared.d2h_bytes = 2 * kIntermediateBytes + kOutputBytes;
    shared.bytes_read =
        2 * fixture.shared_gate_bf16.size() * sizeof(uint16_t) +
        fixture.shared_down_bf16.size() * sizeof(uint16_t);
    Scope complete = {0};
    complete.launches = 1 + 2 + 1 + routed.launches;
    complete.syncs = 1 + 2 + 1 + 1;
    complete.h2d_bytes = 4 * kHiddenBytes + kIntermediateBytes;
    complete.d2h_bytes =
        Q38_MOE_EXPERTS * sizeof(float) + 2 * kIntermediateBytes +
        kOutputBytes + routed.d2h_bytes;
    complete.bytes_read =
        kRouterBytes + shared.bytes_read + routed.bytes_read;
    Scope grouped_complete = complete;
    grouped_complete.launches = 1 + 2 + 1 + grouped_routed.launches;
    grouped_complete.h2d_bytes +=
        Q38_MOE_TOP_K * (sizeof(uint16_t) + sizeof(float));
    Scope deterministic_grouped_complete = grouped_complete;
    deterministic_grouped_complete.launches += 1;

    const bool measured =
        measure([&] { return run_router(&device, fixture, &error); }, &router,
                &error) &&
        measure(
            [&] {
                return q38_moe_cuda_q2_gate_up(
                           device.gate_up, device.hidden, device.mid,
                           device.stream, nullptr, 0) &&
                       cudaStreamSynchronize(device.stream) == cudaSuccess;
            },
            &gate_up, &error) &&
        measure(
            [&] {
                return q38_moe_cuda_q2_down(
                           device.down, device.mid, device.expert, device.stream,
                           nullptr, 0) &&
                       cudaStreamSynchronize(device.stream) == cudaSuccess;
            },
            &down, &error) &&
        measure(
            [&] {
                return q38_moe_cuda_q2_gate_up(
                           device.gate_up, device.hidden, device.mid,
                           device.stream, nullptr, 0) &&
                       q38_moe_cuda_q2_down(
                           device.down, device.mid, device.expert,
                           device.stream, nullptr, 0) &&
                       cudaStreamSynchronize(device.stream) == cudaSuccess;
            },
            &expert, &error) &&
        measure(
            [&] {
                std::vector<float> output;
                return run_routed(&device, fixture, &output, &error);
            },
            &routed, &error) &&
        measure(
            [&] {
                std::vector<float> output;
                return run_shared(&device, fixture, &output, &error);
            },
            &shared, &error) &&
        measure(
            [&] {
                std::vector<float> output;
                return run_complete(&device, fixture, &output, &error);
            },
            &complete, &error) &&
        measure(
            [&] {
                std::vector<float> output;
                return run_routed_grouped(&device, fixture, &output, &error);
            },
            &grouped_routed, &error) &&
        measure(
            [&] {
                std::vector<float> output;
                return run_complete(&device, fixture, &output, &error, true);
            },
            &grouped_complete, &error) &&
        measure(
            [&] {
                std::vector<float> output;
                return run_routed_grouped(&device, fixture, &output, &error,
                                          true);
            },
            &deterministic_grouped_routed, &error) &&
        measure(
            [&] {
                std::vector<float> output;
                return run_complete(&device, fixture, &output, &error, true,
                                    true);
            },
            &deterministic_grouped_complete, &error);

    if (!measured) {
        fprintf(stderr, "%s: %s\n", name, error.c_str());
        free_device(&device);
        return false;
    }
    std::vector<float> actual;
    if (!run_complete(&device, fixture, &actual, &error)) {
        fprintf(stderr, "%s: final correctness run failed: %s\n", name,
                error.c_str());
        free_device(&device);
        return false;
    }
    std::vector<float> grouped_actual;
    if (!run_complete(&device, fixture, &grouped_actual, &error, true)) {
        fprintf(stderr, "%s: grouped candidate correctness run failed: %s\n",
                name, error.c_str());
        free_device(&device);
        return false;
    }
    std::vector<float> deterministic_grouped_actual;
    if (!run_complete(&device, fixture, &deterministic_grouped_actual, &error,
                      true, true)) {
        fprintf(stderr,
                "%s: deterministic grouped candidate correctness run failed: %s\n",
                name, error.c_str());
        free_device(&device);
        return false;
    }
    double oracle_abs = 0.0, oracle_rmse = 0.0;
    double captured_abs = 0.0, captured_rmse = 0.0;
    const bool oracle_ok = compare_output(actual, oracle, &oracle_abs,
                                          &oracle_rmse);
    const bool captured_ok = compare_output(actual, fixture.expected,
                                            &captured_abs, &captured_rmse);
    double grouped_abs = 0.0, grouped_rmse = 0.0;
    const bool grouped_oracle_ok =
        compare_output(grouped_actual, oracle, &grouped_abs, &grouped_rmse);
    double grouped_capture_abs = 0.0, grouped_capture_rmse = 0.0;
    const bool grouped_capture_ok = compare_output(
        grouped_actual, fixture.expected, &grouped_capture_abs,
        &grouped_capture_rmse);
    double deterministic_grouped_abs = 0.0;
    double deterministic_grouped_rmse = 0.0;
    const bool deterministic_grouped_oracle_ok = compare_output(
        deterministic_grouped_actual, oracle, &deterministic_grouped_abs,
        &deterministic_grouped_rmse);
    double deterministic_grouped_capture_abs = 0.0;
    double deterministic_grouped_capture_rmse = 0.0;
    const bool deterministic_grouped_capture_ok = compare_output(
        deterministic_grouped_actual, fixture.expected,
        &deterministic_grouped_capture_abs,
        &deterministic_grouped_capture_rmse);
    if (!oracle_ok || oracle_abs > 2e-2 || !captured_ok ||
        captured_abs > 2e-2 || !grouped_oracle_ok || grouped_abs > 2e-2 ||
        !grouped_capture_ok || grouped_capture_abs > 2e-2 ||
        !deterministic_grouped_oracle_ok ||
        deterministic_grouped_abs > 2e-2 ||
        !deterministic_grouped_capture_ok ||
        deterministic_grouped_capture_abs > 2e-2) {
        std::vector<float> routed_debug;
        std::vector<float> shared_debug;
        double routed_abs = 0.0, routed_rmse = 0.0;
        double shared_abs = 0.0, shared_rmse = 0.0;
        float debug_gate = 0.0f;
        if (run_routed(&device, fixture, &routed_debug, &error)) {
            compare_output(routed_debug, fixture.expected_routed, &routed_abs,
                           &routed_rmse);
        }
        if (run_shared(&device, fixture, &shared_debug, &error)) {
            for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
                debug_gate +=
                    bf16_to_float(fixture.shared_weight_bf16[d]) *
                    fixture.hidden[d];
            debug_gate = 1.0f / (1.0f + expf(-debug_gate));
            for (float &value : shared_debug)
                value *= debug_gate;
            compare_output(shared_debug, fixture.expected_shared, &shared_abs,
                           &shared_rmse);
        }
        fprintf(stderr,
                "%s: correctness failed actual_vs_oracle_abs=%.9g "
                "actual_vs_oracle_rmse=%.9g actual_vs_capture_abs=%.9g "
                "actual_vs_capture_rmse=%.9g routed_abs=%.9g "
                "shared_abs=%.9g debug_gate=%.9g shared0=%.9g "
                "expected_shared0=%.9g\n",
                name, oracle_abs, oracle_rmse, captured_abs, captured_rmse,
                routed_abs, shared_abs, debug_gate,
                shared_debug.empty() ? 0.0 : shared_debug[0],
                fixture.expected_shared[0]);
        free_device(&device);
        return false;
    }

    Determinism determinism;
    if (!probe_grouped_determinism(&device, fixture, &determinism, &error)) {
        fprintf(stderr, "%s: grouped determinism probe failed: %s\n", name,
                error.c_str());
        free_device(&device);
        return false;
    }
    Determinism deterministic_determinism;
    if (!probe_grouped_determinism(&device, fixture, &deterministic_determinism,
                                   &error, true)) {
        fprintf(stderr, "%s: deterministic grouped probe failed: %s\n", name,
                error.c_str());
        free_device(&device);
        return false;
    }
    const double dispatch_residual = std::max(
        0.0, complete.median_us - router.median_us - routed.median_us -
                  shared.median_us);
    const double grouped_speedup_pct =
        100.0 * (complete.median_us - grouped_complete.median_us) /
        complete.median_us;
    const double deterministic_speedup_pct =
        100.0 * (grouped_complete.median_us -
                  deterministic_grouped_complete.median_us) /
        grouped_complete.median_us;
    printf("  \"%s\":{\"correctness\":{\"max_abs\":%.9g,\"rmse\":%.9g,"
           "\"oracle_max_abs\":%.9g,\"oracle_rmse\":%.9g,\"nan_inf\":0},\n"
           "   \"candidates\":{\"moe_c2_grouped\":{\"correctness\":"
           "{\"max_abs\":%.9g,\"rmse\":%.9g,\"nan_inf\":0},"
           "\"speedup_pct\":%.3f,\"scopes\":{\n",
           name, captured_abs, captured_rmse, oracle_abs, oracle_rmse,
           grouped_capture_abs, grouped_capture_rmse, grouped_speedup_pct);
    print_scope("routed_experts", grouped_routed);
    print_scope("complete_moe_layer", grouped_complete, false);
    printf("   }\n"
           "   },\n"
           "   \"moe_c3_deterministic_reduction\":{\"correctness\":"
           "{\"max_abs\":%.9g,\"rmse\":%.9g,\"nan_inf\":0},"
           "\"speedup_pct_vs_grouped\":%.3f,\"scopes\":{\n",
           deterministic_grouped_capture_abs,
           deterministic_grouped_capture_rmse, deterministic_speedup_pct);
    print_scope("routed_experts", deterministic_grouped_routed);
    print_scope("complete_moe_layer", deterministic_grouped_complete, false);
    printf("   }\n"
           "   }\n"
           "   },\n"
    "   \"determinism\":{\"atomic\":{\"samples\":%zu,"
    "\"unique_output_hashes\":%zu,\"unique_mid_hashes\":%zu,"
    "\"first_output_hash\":\"%016" PRIx64
    "\",\"first_mid_hash\":\"%016" PRIx64
    "\",\"max_abs_vs_first\":%.9g,"
    "\"max_rel_vs_first\":%.9g,\"rmse_vs_first\":%.9g,"
    "\"nan_inf\":%s},\"deterministic\":{\"samples\":%zu,"
    "\"unique_output_hashes\":%zu,\"unique_mid_hashes\":%zu,"
    "\"first_output_hash\":\"%016" PRIx64
    "\",\"first_mid_hash\":\"%016" PRIx64
    "\",\"max_abs_vs_first\":%.9g,"
    "\"max_rel_vs_first\":%.9g,\"rmse_vs_first\":%.9g,"
    "\"nan_inf\":%s},\"atomic_accumulation_is_nondeterministic\":%s},\n"
    "   \"accounting\":{\"router_topk_us\":%.3f,"
           "\"routed_experts_us\":%.3f,\"shared_expert_us\":%.3f,"
           "\"dispatch_sync_memcpy_residual_us\":%.3f,"
           "\"accounted_wall_us\":%.3f,\"accounted_fraction\":1.0},\n"
           "   \"scopes\":{\n",
           determinism.samples, determinism.unique_output_hashes,
           determinism.unique_mid_hashes, determinism.first_output_hash,
           determinism.first_mid_hash, determinism.max_abs_vs_first,
           determinism.max_rel_vs_first, determinism.rmse_vs_first,
           determinism.finite ? "0" : "1",
           deterministic_determinism.samples,
           deterministic_determinism.unique_output_hashes,
           deterministic_determinism.unique_mid_hashes,
           deterministic_determinism.first_output_hash,
           deterministic_determinism.first_mid_hash,
           deterministic_determinism.max_abs_vs_first,
           deterministic_determinism.max_rel_vs_first,
           deterministic_determinism.rmse_vs_first,
           deterministic_determinism.finite ? "0" : "1",
           determinism.unique_output_hashes > 1 ? "true" : "false",
           router.median_us,
           routed.median_us, shared.median_us, dispatch_residual,
           complete.median_us);
    print_scope("router_topk_projection", router);
    print_scope("one_expert_gate_up", gate_up);
    print_scope("one_expert_down", down);
    print_scope("one_complete_expert", expert);
    print_scope("ten_selected_experts", routed);
    print_scope("shared_expert", shared);
    print_scope("complete_moe_layer", complete, false);
    printf("   }}%s\n", strcmp(name, "late") == 0 ? "" : ",");
    free_device(&device);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string root = argc > 1 ? argv[1] : "tests/fixtures/moe";
    printf("{\"fixture_policy\":\"compact_real_early_middle_late\","
           "\"warmup\":%zu,\"samples\":%zu,\"fixtures\":{\n",
           kWarmup, kSamples);
    if (!bench_fixture("early", root + "/early") ||
        !bench_fixture("middle", root + "/middle") ||
        !bench_fixture("late", root + "/late"))
        return 2;
    printf("}}\n");
    return 0;
}
