#include "../q38_forward.h"
#include "../q38_gguf.h"
#include "../q38_moe.h"
#include "../q38_moe_cuda.h"
#include "../q38_quant.h"
#include "../q38_weights.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    return false;
}

static double event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float value = 0.0f;
    if (cudaEventElapsedTime(&value, start, stop) != cudaSuccess) return -1.0;
    return value;
}

struct Capture {
    std::vector<float> hidden;
    uint32_t layer;
};

static bool capture_moe(uint32_t layer, const q38_moe_trace *trace,
                        void *user, char *error, size_t error_len) {
    Capture *capture = static_cast<Capture *>(user);
    if (!trace || !trace->router_input ||
        trace->router_input_count != Q38_MOE_HIDDEN) {
        if (error && error_len) std::snprintf(error, error_len,
                                              "missing real MoE hidden");
        return false;
    }
    if (capture->hidden.empty()) {
        capture->hidden.assign(trace->router_input,
                               trace->router_input + trace->router_input_count);
        capture->layer = layer;
    }
    return true;
}

static bool decode_gate_up(const void *weights, uint32_t qtype,
                           std::vector<float> *dense) {
    dense->assign(2 * Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN, 0.0f);
    char error[128];
    for (size_t row = 0; row < 2 * Q38_MOE_INTERMEDIATE; ++row) {
        const size_t stride = qtype == Q38_QUANT_Q2_K
            ? sizeof(q38_q2_k_block) : sizeof(q38_q4_k_block);
        if (!q38_quant_dequantize_row(
                qtype, static_cast<const uint8_t *>(weights) + row * 10 * stride, 10,
                dense->data() + row * Q38_MOE_HIDDEN, Q38_MOE_HIDDEN,
                error, sizeof(error)))
            return fail(error);
    }
    return true;
}

static bool decode_down(const void *weights, uint32_t qtype,
                        std::vector<float> *dense) {
    dense->assign(Q38_MOE_INTERMEDIATE * Q38_MOE_HIDDEN, 0.0f);
    std::vector<float> row(Q38_MOE_HIDDEN);
    char error[128];
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        const size_t stride = qtype == Q38_QUANT_Q2_K
            ? sizeof(q38_q2_k_block) : sizeof(q38_q4_k_block);
        if (!q38_quant_dequantize_row(
                qtype, static_cast<const uint8_t *>(weights) + i * 10 * stride,
                10, row.data(),
                row.size(), error, sizeof(error)))
            return fail(error);
        std::copy(row.begin(), row.end(),
                  dense->begin() + i * Q38_MOE_HIDDEN);
    }
    return true;
}

static bool run_gate_up(const void *weights, uint32_t qtype,
                        const float *hidden,
                        float *mid, cudaStream_t stream, cudaEvent_t start,
                        cudaEvent_t stop, std::vector<double> *samples) {
    char error[256];
    for (int i = 0; i < 1; ++i) {
        const bool ok = qtype == Q38_QUANT_Q2_K
            ? q38_moe_cuda_q2_gate_up(weights, hidden, mid, stream, error,
                                      sizeof(error))
            : q38_moe_cuda_q4_gate_up(weights, hidden, mid, stream, error,
                                      sizeof(error));
        if (!ok)
            return fail(error);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return fail("gate/up warmup failed");
    for (int i = 0; i < 10; ++i) {
        cudaEventRecord(start, stream);
        const bool ok = qtype == Q38_QUANT_Q2_K
            ? q38_moe_cuda_q2_gate_up(weights, hidden, mid, stream, error,
                                      sizeof(error))
            : q38_moe_cuda_q4_gate_up(weights, hidden, mid, stream, error,
                                      sizeof(error));
        if (!ok)
            return fail(error);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        samples->push_back(event_ms(start, stop));
    }
    return true;
}

static bool run_down(const void *weights, uint32_t qtype,
                     const float *mid, float *output,
                     cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
                     std::vector<double> *samples) {
    char error[256];
    const bool warm_ok = qtype == Q38_QUANT_Q2_K
        ? q38_moe_cuda_q2_down(weights, mid, output, stream, error,
                               sizeof(error))
        : q38_moe_cuda_q4_down(weights, mid, output, stream, error,
                               sizeof(error));
    if (!warm_ok)
        return fail(error);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return fail("down warmup failed");
    for (int i = 0; i < 10; ++i) {
        cudaEventRecord(start, stream);
        const bool ok = qtype == Q38_QUANT_Q2_K
            ? q38_moe_cuda_q2_down(weights, mid, output, stream, error,
                                   sizeof(error))
            : q38_moe_cuda_q4_down(weights, mid, output, stream, error,
                                   sizeof(error));
        if (!ok)
            return fail(error);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        samples->push_back(event_ms(start, stop));
    }
    return true;
}

static void summarize(std::vector<double> *samples) {
    std::sort(samples->begin(), samples->end());
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
        "artifacts/m8/qwen38-runtime-R1-Q4Experts-BF16Core-BF16PLE.gguf";
    char error[256] = {};
    q38_gguf *model = q38_gguf_open(path, error, sizeof(error));
    if (!model) return fail(error) ? 1 : 1;
    q38_weights weights = {};
    if (!q38_weights_bind_subset(model, Q38_MODEL_LAYERS - 1, &weights,
                                 error, sizeof(error)))
        return fail(error) ? 1 : 1;
    q38_forward_state state = {};
    if (!q38_forward_state_init(&state, &weights, 248044, error,
                                sizeof(error)))
        return fail(error) ? 1 : 1;
    Capture capture = {};
    const char *hidden_path = argc > 2 ? argv[2] : nullptr;
    if (hidden_path) {
        FILE *hidden_file = std::fopen(hidden_path, "rb");
        if (!hidden_file ||
            std::fread(capture.hidden.data(), sizeof(float), 0, hidden_file) != 0) {
            if (hidden_file) std::fclose(hidden_file);
            hidden_file = nullptr;
        }
        if (hidden_file) {
            capture.hidden.resize(Q38_MOE_HIDDEN);
            const size_t count = std::fread(capture.hidden.data(), sizeof(float),
                                            capture.hidden.size(), hidden_file);
            std::fclose(hidden_file);
            if (count != capture.hidden.size())
                capture.hidden.clear();
        }
    }
    if (capture.hidden.empty()) {
        q38_forward_diagnostics diagnostics = {};
        diagnostics.moe_trace = capture_moe;
        diagnostics.trace_user = &capture;
        const uint32_t token = 9419;
        std::vector<float> logits(248320);
        if (!q38_forward_full(model, &weights, &state, &token, 1, logits.data(),
                              logits.size(), &diagnostics, error, sizeof(error)))
            return fail(error) ? 1 : 1;
        if (capture.hidden.empty())
            return fail("forward did not capture hidden") ? 1 : 1;
        if (hidden_path) {
            FILE *hidden_file = std::fopen(hidden_path, "wb");
            if (!hidden_file ||
                std::fwrite(capture.hidden.data(), sizeof(float),
                            capture.hidden.size(), hidden_file) !=
                    capture.hidden.size()) {
                if (hidden_file) std::fclose(hidden_file);
                return fail("failed to save captured hidden") ? 1 : 1;
            }
            std::fclose(hidden_file);
        }
    }
    capture.layer = 0;

    q38_tensor *gate_up = weights.layer[capture.layer].experts.bank[0].gate_up;
    q38_tensor *down = weights.layer[capture.layer].experts.bank[0].down;
    uint64_t gate_offset, gate_bytes, down_offset, down_bytes;
    if (!q38_moe_expert_slice(model, gate_up, 0, &gate_offset, &gate_bytes,
                              error, sizeof(error)) ||
        !q38_moe_expert_slice(model, down, 0, &down_offset, &down_bytes,
                               error, sizeof(error)))
        return fail(error) ? 1 : 1;
    if (gate_up->type != down->type ||
        (gate_up->type != Q38_QUANT_Q2_K &&
         gate_up->type != Q38_QUANT_Q4_K))
        return fail("unsupported routed expert qtype") ? 1 : 1;

    const void *host_gate = model->map + gate_offset;
    const void *host_down = model->map + down_offset;
    const uint32_t qtype = gate_up->type;
    std::vector<float> gate_dense, down_dense;
    if (!decode_gate_up(host_gate, qtype, &gate_dense) ||
        !decode_down(host_down, qtype, &down_dense))
        return 1;
    std::vector<float> expected_mid(Q38_MOE_INTERMEDIATE);
    std::vector<float> expected_output(Q38_MOE_HIDDEN);
    for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i) {
        float gate = 0.0f, up = 0.0f;
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
            gate += gate_dense[i * Q38_MOE_HIDDEN + d] * capture.hidden[d];
            up += gate_dense[(Q38_MOE_INTERMEDIATE + i) *
                             Q38_MOE_HIDDEN + d] * capture.hidden[d];
        }
        expected_mid[i] = gate / (1.0f + std::exp(-gate)) * up;
    }
    for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
        for (size_t i = 0; i < Q38_MOE_INTERMEDIATE; ++i)
            expected_output[d] += down_dense[i * Q38_MOE_HIDDEN + d] *
                                  expected_mid[i];

    cudaStream_t stream = nullptr;
    cudaEvent_t start, stop;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    void *device_gate = nullptr, *device_down = nullptr;
    float *device_hidden = nullptr, *device_mid = nullptr, *device_output = nullptr;
    cudaMalloc(&device_gate, gate_bytes);
    cudaMalloc(&device_down, down_bytes);
    cudaMalloc(&device_hidden, capture.hidden.size() * sizeof(float));
    cudaMalloc(&device_mid, expected_mid.size() * sizeof(float));
    cudaMalloc(&device_output, expected_output.size() * sizeof(float));
    cudaMemcpy(device_gate, host_gate, gate_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_down, host_down, down_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_hidden, capture.hidden.data(),
               capture.hidden.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<double> gate_samples, down_samples;
    if (!run_gate_up(device_gate, qtype, device_hidden, device_mid, stream, start,
                     stop, &gate_samples) ||
        !run_down(device_down, qtype, device_mid, device_output, stream, start,
                  stop, &down_samples))
        return 1;
    std::vector<float> actual_mid(expected_mid.size());
    std::vector<float> actual_output(expected_output.size());
    cudaMemcpy(actual_mid.data(), device_mid, gate_samples.size() * 0 +
               actual_mid.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_output.data(), device_output,
               actual_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float mid_error = 0.0f, output_error = 0.0f;
    for (size_t i = 0; i < actual_mid.size(); ++i)
        mid_error = std::max(mid_error, std::fabs(actual_mid[i] - expected_mid[i]));
    for (size_t i = 0; i < actual_output.size(); ++i)
        output_error = std::max(output_error,
                                std::fabs(actual_output[i] - expected_output[i]));
    summarize(&gate_samples);
    summarize(&down_samples);
    const double gate_bytes_per_call = (double)gate_bytes;
    const double down_bytes_per_call = (double)down_bytes;
    std::printf(
        "{\"format\":\"q38-routed-expert-q-gemv-v1\",\"model\":\"%s\","
        "\"layer\":%u,\"expert\":0,\"hidden_source\":\"real_m8_forward\","
        "\"qtype\":\"%s\",\"gate_up\":{\"rows\":%zu,\"cols\":%zu,"
        "\"packed_bytes\":%llu,\"kernel_median_ms\":%.6f,\"p95_ms\":%.6f,"
        "\"effective_GBps\":%.6f,\"geometry\":\"grid=3 block=256\","
        "\"correctness_max_abs\":%.9g},\"down\":{\"rows\":%zu,\"cols\":%zu,"
        "\"packed_bytes\":%llu,\"kernel_median_ms\":%.6f,\"p95_ms\":%.6f,"
        "\"effective_GBps\":%.6f,\"geometry\":\"grid=10 block=256\","
        "\"correctness_max_abs\":%.9g},\"forward_layer\":%u}\n",
        path, capture.layer, qtype == Q38_QUANT_Q2_K ? "Q2_K" : "Q4_K",
        (size_t)(2 * Q38_MOE_INTERMEDIATE),
        (size_t)Q38_MOE_HIDDEN,
        (unsigned long long)gate_bytes, gate_samples[5], gate_samples[9],
        gate_bytes_per_call / (gate_samples[5] * 1e6),
        mid_error, (size_t)Q38_MOE_HIDDEN, (size_t)Q38_MOE_INTERMEDIATE,
        (unsigned long long)down_bytes, down_samples[5], down_samples[9],
        down_bytes_per_call / (down_samples[5] * 1e6),
        output_error, capture.layer);

    cudaFree(device_gate); cudaFree(device_down); cudaFree(device_hidden);
    cudaFree(device_mid); cudaFree(device_output);
    cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
    q38_forward_state_destroy(&state);
    q38_weights_release(&weights);
    q38_gguf_close(model);
    return 0;
}
