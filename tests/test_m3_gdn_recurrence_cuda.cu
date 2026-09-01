#include "q38_gdn.h"
#include "q38_gdn_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool cuda_check(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

static bool compare(const std::vector<float> &expected,
                    const std::vector<float> &actual, const char *name,
                    float tolerance) {
    float max_abs = 0.0f;
    size_t max_index = 0;
    for (size_t i = 0; i < expected.size(); i++) {
        const float difference = std::fabs(expected[i] - actual[i]);
        if (difference > max_abs) {
            max_abs = difference;
            max_index = i;
        }
    }
    if (max_abs > tolerance) {
        std::fprintf(stderr, "%s mismatch: max_abs=%g at %zu expected=%g actual=%g\n",
                     name, max_abs, max_index, expected[max_index],
                     actual[max_index]);
        return false;
    }
    return true;
}

static void make_inputs(size_t tokens, std::vector<float> &q,
                        std::vector<float> &k, std::vector<float> &v,
                        std::vector<float> &decay,
                        std::vector<float> &beta) {
    const size_t channels = Q38_GDN_VALUE_CHANNELS;
    for (size_t token = 0; token < tokens; token++) {
        for (size_t head = 0; head < Q38_GDN_VALUE_HEADS; head++) {
            decay[token * Q38_GDN_VALUE_HEADS + head] =
                0.82f + 0.01f * (float)((head + token) % 7u);
            beta[token * Q38_GDN_VALUE_HEADS + head] =
                0.35f + 0.025f * (float)((head * 3u + token) % 6u);
        }
        for (size_t channel = 0; channel < channels; channel++) {
            const size_t q_code = (channel * 17u + token * 11u) % 101u;
            const size_t k_code = (channel * 29u + token * 7u + 13u) % 97u;
            const size_t v_code = (channel * 31u + token * 5u + 3u) % 89u;
            q[token * channels + channel] = ((float)q_code - 50.0f) * 0.001f;
            k[token * channels + channel] = ((float)k_code - 48.0f) * 0.001f;
            v[token * channels + channel] = ((float)v_code - 44.0f) * 0.0015f;
        }
    }
}

static bool run_case(size_t tokens, cudaStream_t stream) {
    const size_t channels = Q38_GDN_VALUE_CHANNELS;
    const size_t state_elements =
        (size_t)Q38_GDN_VALUE_HEADS * Q38_GDN_HEAD_DIM * Q38_GDN_HEAD_DIM;
    const size_t state_bytes = state_elements * sizeof(float);
    const size_t vector_bytes = tokens * channels * sizeof(float);
    const size_t gate_bytes = tokens * Q38_GDN_VALUE_HEADS * sizeof(float);
    std::vector<float> q(tokens * channels), k(tokens * channels),
        v(tokens * channels), decay(tokens * Q38_GDN_VALUE_HEADS),
        beta(tokens * Q38_GDN_VALUE_HEADS);
    make_inputs(tokens, q, k, v, decay, beta);

    std::vector<float> expected_state(state_elements, 0.0f);
    std::vector<float> expected_output(tokens * channels, 0.0f);
    if (!q38_gdn_ref_run(expected_state.data(), 1, tokens, q.data(), k.data(),
                         v.data(), decay.data(), beta.data(), 0.75f,
                         expected_output.data()))
        return false;

    float *d_state = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_decay = nullptr, *d_beta = nullptr, *d_output = nullptr;
    bool ok =
        cuda_check(cudaMalloc(&d_state, state_bytes), "cudaMalloc state") &&
        cuda_check(cudaMalloc(&d_q, vector_bytes), "cudaMalloc q") &&
        cuda_check(cudaMalloc(&d_k, vector_bytes), "cudaMalloc k") &&
        cuda_check(cudaMalloc(&d_v, vector_bytes), "cudaMalloc v") &&
        cuda_check(cudaMalloc(&d_decay, gate_bytes), "cudaMalloc decay") &&
        cuda_check(cudaMalloc(&d_beta, gate_bytes), "cudaMalloc beta") &&
        cuda_check(cudaMalloc(&d_output, vector_bytes), "cudaMalloc output");
    if (!ok) goto cleanup;

    ok = cuda_check(cudaMemcpy(d_q, q.data(), vector_bytes,
                               cudaMemcpyHostToDevice),
                    "copy q") &&
         cuda_check(cudaMemcpy(d_k, k.data(), vector_bytes,
                               cudaMemcpyHostToDevice),
                    "copy k") &&
         cuda_check(cudaMemcpy(d_v, v.data(), vector_bytes,
                               cudaMemcpyHostToDevice),
                    "copy v") &&
         cuda_check(cudaMemcpy(d_decay, decay.data(), gate_bytes,
                               cudaMemcpyHostToDevice),
                    "copy decay") &&
         cuda_check(cudaMemcpy(d_beta, beta.data(), gate_bytes,
                               cudaMemcpyHostToDevice),
                    "copy beta");
    if (!ok) goto cleanup;

    {
        char error[256];
        if (!q38_cuda_gdn_recurrence_reset(d_state, stream, error,
                                           sizeof(error))) {
            std::fprintf(stderr, "tokens=%zu reset failed: %s\n", tokens, error);
            ok = false;
            goto cleanup;
        }
        if (!cuda_check(cudaStreamSynchronize(stream), "reset synchronize")) {
            ok = false;
            goto cleanup;
        }
        std::vector<float> reset_state(state_elements);
        ok = cuda_check(cudaMemcpy(reset_state.data(), d_state, state_bytes,
                                   cudaMemcpyDeviceToHost),
                        "copy reset state");
        if (!ok) goto cleanup;
        if (!compare(std::vector<float>(state_elements, 0.0f), reset_state,
                     "zero reset", 0.0f)) {
            ok = false;
            goto cleanup;
        }

        if (!q38_cuda_gdn_recurrence(
                d_state, tokens, d_q, d_k, d_v, d_decay, d_beta, 0.75f,
                d_output, stream, error, sizeof(error))) {
            std::fprintf(stderr, "tokens=%zu recurrence failed: %s\n", tokens,
                         error);
            ok = false;
            goto cleanup;
        }
        if (!cuda_check(cudaStreamSynchronize(stream),
                        "recurrence synchronize")) {
            ok = false;
            goto cleanup;
        }
    }

    {
        std::vector<float> actual_state(state_elements);
        std::vector<float> actual_output(tokens * channels);
        ok = cuda_check(cudaMemcpy(actual_state.data(), d_state, state_bytes,
                                   cudaMemcpyDeviceToHost),
                        "copy state") &&
             cuda_check(cudaMemcpy(actual_output.data(), d_output, vector_bytes,
                                   cudaMemcpyDeviceToHost),
                        "copy output");
        if (ok)
            ok = compare(expected_output, actual_output, "recurrence output",
                         2e-5f);
        if (ok)
            ok = compare(expected_state, actual_state, "recurrence state",
                         2e-5f);
    }

cleanup:
    cudaFree(d_state);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_decay);
    cudaFree(d_beta);
    cudaFree(d_output);
    return ok;
}

int main() {
    int devices = 0;
    cudaError_t status = cudaGetDeviceCount(&devices);
    if (!cuda_check(status, "cudaGetDeviceCount") || devices == 0) {
        std::fprintf(stderr, "M3-C08: CUDA device unavailable\n");
        return 2;
    }
    cudaStream_t stream = nullptr;
    if (!cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;

    bool ok = true;
    const size_t token_cases[] = {1, 2, 4, 5};
    for (size_t tokens : token_cases) {
        if (!run_case(tokens, stream)) {
            std::fprintf(stderr, "test_m3_gdn_recurrence_cuda: %zu tokens failed\n",
                         tokens);
            ok = false;
            break;
        }
    }
    ok = cuda_check(cudaStreamSynchronize(stream), "final synchronize") && ok;
    ok = cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy") && ok;
    if (!ok) return 1;
    std::puts("test_m3_gdn_recurrence_cuda: 1, 2, 4, and 5 token FP32 recurrence/state cases passed");
    return 0;
}
