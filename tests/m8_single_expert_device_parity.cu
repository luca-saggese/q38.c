#include "../q38_forward_cuda.h"
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
#include <chrono>

static bool fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    return false;
}

static bool decode_row(const void *data, uint32_t qtype, size_t row,
                       std::vector<float> *out) {
    const size_t block_size = qtype == Q38_QUANT_Q2_K
        ? sizeof(q38_q2_k_block) : sizeof(q38_q4_k_block);
    char error[128];
    out->resize(Q38_MOE_HIDDEN);
    return q38_quant_dequantize_row(
        qtype, static_cast<const unsigned char *>(data) +
                   row * 10u * block_size, 10, out->data(), out->size(),
        error, sizeof(error));
}

static void compare(const char *name, const float *actual,
                    const std::vector<float> &expected, double *max_abs,
                    double *max_rel) {
    double abs_error = 0.0, rel_error = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double diff = std::fabs((double)actual[i] - expected[i]);
        abs_error = std::max(abs_error, diff);
        rel_error = std::max(rel_error,
                             diff / std::max(1.0, std::fabs((double)expected[i])));
    }
    *max_abs = abs_error;
    *max_rel = rel_error;
    std::printf("{\"boundary\":\"%s\",\"max_abs\":%.9g,\"max_rel\":%.9g,"
                "\"pass\":%s}\n", name, abs_error, rel_error,
                abs_error <= 2e-4 && rel_error <= 2e-4 ? "true" : "false");
}

static bool capture_hidden(uint32_t, const q38_moe_trace *trace, void *user,
                           char *error, size_t error_len) {
    std::vector<float> *hidden = static_cast<std::vector<float> *>(user);
    if (!trace || !trace->router_input ||
        trace->router_input_count != Q38_MOE_HIDDEN)
        return fail("missing forward hidden") &&
               !(error && error_len);
    hidden->assign(trace->router_input,
                   trace->router_input + trace->router_input_count);
    return true;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] :
        "artifacts/m1/qwen38-runtime-only-Q2Experts-BF16Core-BF16PLE.gguf";
    const char *hidden_path = argc > 2 ? argv[2] :
        "artifacts/post_m8_opt/shared_hidden_layer0.bin";
    char error[256] = {};
    q38_gguf *model = q38_gguf_open(model_path, error, sizeof(error));
    if (!model) return fail(error) ? 0 : 1;
    std::vector<float> hidden(Q38_MOE_HIDDEN);
    FILE *hidden_file = std::fopen(hidden_path, "rb");
    if (!hidden_file ||
        std::fread(hidden.data(), sizeof(float), hidden.size(), hidden_file) !=
            hidden.size()) {
        if (hidden_file) std::fclose(hidden_file);
        hidden.clear();
    } else {
        std::fclose(hidden_file);
    }

    q38_weights weights = {};
    if (!q38_weights_bind_subset(model, Q38_MODEL_LAYERS - 1, &weights,
                                 error, sizeof(error))) {
        q38_gguf_close(model);
        return fail(error) ? 0 : 1;
    }
    if (hidden.empty()) {
        q38_forward_diagnostics diagnostics = {};
        diagnostics.moe_trace = capture_hidden;
        diagnostics.trace_user = &hidden;
        std::vector<float> logits(248320);
        const uint32_t token = 9419;
        q38_forward_state state = {};
        if (!q38_forward_state_init(&state, &weights, 248044, error,
                                    sizeof(error)) ||
            !q38_forward_full(model, &weights, &state, &token, 1,
                              logits.data(), logits.size(), &diagnostics,
                              error, sizeof(error)) || hidden.empty()) {
            q38_forward_state_destroy(&state);
            hidden_file = std::fopen(hidden_path, "wb");
            if (hidden_file) {
                std::fwrite(hidden.data(), sizeof(float), hidden.size(), hidden_file);
                std::fclose(hidden_file);
            }
            q38_gguf_close(model);
            return fail(error[0] ? error : "forward did not capture hidden")
                ? 0 : 1;
        }
        q38_forward_state_destroy(&state);
    }
    q38_tensor *gate_up = weights.layer[0].experts.bank[0].gate_up;
    q38_tensor *down = weights.layer[0].experts.bank[0].down;
    uint64_t gate_offset, gate_bytes, down_offset, down_bytes;
    if (!q38_moe_expert_slice(model, gate_up, 0, &gate_offset, &gate_bytes,
                               error, sizeof(error)) ||
        !q38_moe_expert_slice(model, down, 0, &down_offset, &down_bytes,
                              error, sizeof(error))) {
        q38_gguf_close(model);
        return fail(error) ? 0 : 1;
    }
    const uint32_t qtype = gate_up->type;
    const void *host_gate = model->map + gate_offset;
    const void *host_down = model->map + down_offset;
    std::vector<float> row, gate(640 * Q38_MOE_HIDDEN),
        up(640 * Q38_MOE_HIDDEN), down_dense(640 * Q38_MOE_HIDDEN);
    for (size_t i = 0; i < 640; ++i) {
        if (!decode_row(host_gate, qtype, i, &row)) return 1;
        std::copy(row.begin(), row.end(), gate.begin() + i * Q38_MOE_HIDDEN);
        if (!decode_row(host_gate, qtype, 640 + i, &row)) return 1;
        std::copy(row.begin(), row.end(), up.begin() + i * Q38_MOE_HIDDEN);
        if (!decode_row(host_down, qtype, i, &row)) return 1;
        std::copy(row.begin(), row.end(),
                  down_dense.begin() + i * Q38_MOE_HIDDEN);
    }
    std::vector<float> expected_gate(640), expected_up(640),
        expected_mid(640), expected_output(Q38_MOE_HIDDEN, 0.0f);
    for (size_t i = 0; i < 640; ++i) {
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d) {
            expected_gate[i] += gate[i * Q38_MOE_HIDDEN + d] * hidden[d];
            expected_up[i] += up[i * Q38_MOE_HIDDEN + d] * hidden[d];
        }
        expected_mid[i] = expected_gate[i] /
            (1.0f + std::exp(-expected_gate[i])) * expected_up[i];
        for (size_t d = 0; d < Q38_MOE_HIDDEN; ++d)
            expected_output[d] +=
                down_dense[i * Q38_MOE_HIDDEN + d] * expected_mid[i];
    }

    q38_forward_cuda_context *context =
        q38_forward_cuda_context_create(error, sizeof(error));
    bool resident = context &&
        q38_forward_cuda_enable_all_non_ple_residency(
            context, model, error, sizeof(error));
    if (!context) {
        q38_gguf_close(model);
        return fail(error) ? 0 : 1;
    }
    if (!resident)
        std::fprintf(stderr, "resident setup unavailable: %s\n", error);
    std::vector<float> canonical(Q38_MOE_HIDDEN);
    const auto canonical_started = std::chrono::steady_clock::now();
    if (!q38_forward_cuda_expert_backend(
            model, gate_up, down, 0, hidden.data(), canonical.data(), context,
            error, sizeof(error))) {
        q38_forward_cuda_context_destroy(context);
        q38_gguf_close(model);
        return fail(error) ? 0 : 1;
    }
    const double canonical_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - canonical_started).count();

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaEvent_t gate_start = nullptr, gate_stop = nullptr;
    cudaEvent_t down_start = nullptr, down_stop = nullptr;
    void *device_gate = nullptr, *device_down = nullptr;
    float *device_hidden = nullptr, *device_mid = nullptr, *device_output = nullptr;
    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&gate_start);
    cudaEventCreate(&gate_stop);
    cudaEventCreate(&down_start);
    cudaEventCreate(&down_stop);
    cudaMalloc(&device_gate, gate_bytes);
    cudaMalloc(&device_down, down_bytes);
    cudaMalloc(&device_hidden, hidden.size() * sizeof(float));
    cudaMalloc(&device_mid, expected_mid.size() * sizeof(float));
    cudaMalloc(&device_output, expected_output.size() * sizeof(float));
    cudaMemcpy(device_gate, host_gate, gate_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_down, host_down, down_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_hidden, hidden.data(), hidden.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    if (qtype == Q38_QUANT_Q2_K
            ? !q38_moe_cuda_q2_gate_up(
                  device_gate, device_hidden, device_mid, stream, error,
                  sizeof(error))
            : !q38_moe_cuda_q4_gate_up(
                  device_gate, device_hidden, device_mid, stream, error,
                  sizeof(error))) {
        return fail(error) ? 0 : 1;
    }
    if (qtype == Q38_QUANT_Q2_K
            ? !q38_moe_cuda_q2_down(
                  device_down, device_mid, device_output, stream, error,
                  sizeof(error))
            : !q38_moe_cuda_q4_down(
                  device_down, device_mid, device_output, stream, error,
                  sizeof(error))) {
        return fail(error) ? 0 : 1;
    }
    const auto device_started = std::chrono::steady_clock::now();
    cudaEventRecord(start, stream);
    cudaEventRecord(gate_start, stream);
    if (qtype == Q38_QUANT_Q2_K
            ? !q38_moe_cuda_q2_gate_up(
                  device_gate, device_hidden, device_mid, stream, error,
                  sizeof(error))
            : !q38_moe_cuda_q4_gate_up(
                  device_gate, device_hidden, device_mid, stream, error,
                  sizeof(error)))
        return fail(error) ? 0 : 1;
    cudaEventRecord(gate_stop, stream);
    cudaEventRecord(down_start, stream);
    if (qtype == Q38_QUANT_Q2_K
            ? !q38_moe_cuda_q2_down(
                  device_down, device_mid, device_output, stream, error,
                  sizeof(error))
            : !q38_moe_cuda_q4_down(
                  device_down, device_mid, device_output, stream, error,
                  sizeof(error)))
        return fail(error) ? 0 : 1;
    cudaEventRecord(down_stop, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    float gate_ms = 0.0f, down_ms = 0.0f;
    cudaEventElapsedTime(&gate_ms, gate_start, gate_stop);
    cudaEventElapsedTime(&down_ms, down_start, down_stop);
    const double device_total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - device_started).count();
    std::vector<float> device_mid_host(expected_mid.size());
    std::vector<float> device_output_host(expected_output.size());
    cudaMemcpy(device_mid_host.data(), device_mid,
               device_mid_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(device_output_host.data(), device_output,
               device_output_host.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    double abs_error = 0.0, rel_error = 0.0;
    compare("post_activation", device_mid_host.data(), expected_mid,
            &abs_error, &rel_error);
    compare("final_expert", device_output_host.data(), expected_output,
            &abs_error, &rel_error);
    compare("canonical_vs_device", device_output_host.data(), canonical,
            &abs_error, &rel_error);
    std::printf("{\"qtype\":%u,\"resident\":%s,\"canonical_expert_ms\":%.6f,"
                "\"device_only_total_ms\":%.6f,\"device_only_kernel_ms\":%.6f,"
                "\"gate_up_kernel_ms\":%.6f,\"activation_ms\":0.0,"
                "\"down_kernel_ms\":%.6f,\"H2D_bytes\":0,"
                "\"D2H_bytes\":0,\"host_syncs\":1,\"kernel_launches\":2,"
                "\"correctness\":%s}\n", qtype, resident ? "true" : "false",
                canonical_ms, device_total_ms,
                kernel_ms, gate_ms, down_ms,
                abs_error <= 2e-4 && rel_error <= 2e-4 ? "true" : "false");

    cudaFree(device_gate);
    cudaFree(device_down);
    cudaFree(device_hidden);
    cudaFree(device_mid);
    cudaFree(device_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(gate_start);
    cudaEventDestroy(gate_stop);
    cudaEventDestroy(down_start);
    cudaEventDestroy(down_stop);
    cudaStreamDestroy(stream);
    q38_forward_cuda_context_destroy(context);
    q38_gguf_close(model);
    return abs_error <= 2e-4 && rel_error <= 2e-4 ? 0 : 1;
}
