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
                         std::vector<float> *output, std::string *error) {
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
    if (!run_routed(device, fixture, &routed, error))
        return false;
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
            q38_half_to_float(fixture.shared_weight_bf16[d]) *
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
        host_gate[i] = q38_half_to_float(fixture.shared_gate_bf16[i]);
        host_up[i] = q38_half_to_float(fixture.shared_up_bf16[i]);
    }
    for (size_t i = 0; i < host_down.size(); ++i)
        host_down[i] = q38_half_to_float(fixture.shared_down_bf16[i]);
    for (size_t i = 0; i < host_weight.size(); ++i)
        host_weight[i] = q38_half_to_float(fixture.shared_weight_bf16[i]);
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
            &complete, &error);

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
    double max_abs = 0.0;
    double rmse = 0.0;
    if (!compare_output(actual, oracle, &max_abs, &rmse) ||
        max_abs > 2e-2 || !compare_output(actual, fixture.expected, &max_abs,
                                           &rmse) ||
        max_abs > 2e-2) {
        fprintf(stderr, "%s: correctness failed max_abs=%.9g rmse=%.9g\n", name,
                max_abs, rmse);
        free_device(&device);
        return false;
    }

    printf("  \"%s\":{\"correctness\":{\"max_abs\":%.9g,\"rmse\":%.9g,"
           "\"nan_inf\":0},\n"
           "   \"scopes\":{\n",
           name, max_abs, rmse);
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
