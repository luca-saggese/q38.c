#include "q38_gdn.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <time.h>
#include <vector>

static bool read_file(const std::string &path, void *data, size_t bytes) {
    std::ifstream input(path, std::ios::binary);
    return input.good() &&
           static_cast<bool>(input.read(static_cast<char *>(data), bytes));
}

static double now_ms() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<double>(ts.tv_sec) * 1000.0 +
           static_cast<double>(ts.tv_nsec) / 1000000.0;
}

static bool check_cuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
}

int main(int argc, char **argv) {
    const std::string fixture =
        argc > 1 ? argv[1] : "artifacts/m4/ple_projection_q8_fixture";
    const size_t iterations = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 100;
    constexpr size_t cases = 3;
    constexpr size_t cols = 2560;
    constexpr size_t key_rows = 10240;
    constexpr size_t value_rows = 2560;
    constexpr size_t key_bytes = key_rows * cols / 32 * 34;
    constexpr size_t value_bytes = value_rows * cols / 32 * 34;
    constexpr size_t input_bytes = cols * sizeof(float);
    constexpr size_t output_bytes = (key_rows + value_rows) * sizeof(float);

    std::vector<uint8_t> key(key_bytes), value(value_bytes);
    std::vector<float> inputs(cases * cols), expected_key(cases * key_rows),
        expected_value(cases * value_rows), actual_key(key_rows),
        actual_value(value_rows);
    if (!read_file(fixture + "/key_proj_q8.bin", key.data(), key.size()) ||
        !read_file(fixture + "/value_proj_q8.bin", value.data(), value.size()) ||
        !read_file(fixture + "/rows_q8_f32.bin", inputs.data(),
                   inputs.size() * sizeof(float)) ||
        !read_file(fixture + "/key_expected_f32.bin", expected_key.data(),
                   expected_key.size() * sizeof(float)) ||
        !read_file(fixture + "/value_expected_f32.bin", expected_value.data(),
                   expected_value.size() * sizeof(float))) {
        std::fprintf(stderr, "failed to read Q8 projection fixture\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    void *device_key = nullptr, *device_value = nullptr;
    float *device_input = nullptr, *device_key_output = nullptr,
          *device_value_output = nullptr;
    if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate") ||
        !check_cuda(cudaMalloc(&device_key, key.size()), "cudaMalloc key") ||
        !check_cuda(cudaMalloc(&device_value, value.size()), "cudaMalloc value") ||
        !check_cuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc input") ||
        !check_cuda(cudaMalloc(&device_key_output, key_rows * sizeof(float)),
                    "cudaMalloc key output") ||
        !check_cuda(cudaMalloc(&device_value_output, value_rows * sizeof(float)),
                    "cudaMalloc value output") ||
        !check_cuda(cudaMemcpyAsync(device_key, key.data(), key.size(),
                                    cudaMemcpyHostToDevice, stream),
                    "upload key") ||
        !check_cuda(cudaMemcpyAsync(device_value, value.data(), value.size(),
                                    cudaMemcpyHostToDevice, stream),
                    "upload value") ||
        !check_cuda(cudaStreamSynchronize(stream), "weight synchronization")) {
        return 1;
    }

    std::printf("{\"format\":\"q38-ple-projection-c1-v1\","
                "\"baseline\":\"PLE_PROJ_C1\","
                "\"fixture\":\"%s\",\"iterations\":%zu,"
                "\"weights_h2d_bytes_before_timing\":%zu,"
                "\"cases\":[",
                fixture.c_str(), iterations, key.size() + value.size());
    const char *names[] = {"early", "middle", "late"};
    char error[256];
    for (size_t case_index = 0; case_index < cases; ++case_index) {
        for (size_t warmup = 0; warmup < 5; ++warmup) {
            if (!check_cuda(cudaMemcpyAsync(
                                device_input, inputs.data() + case_index * cols,
                                input_bytes, cudaMemcpyHostToDevice, stream),
                            "warmup input") ||
                !q38_cuda_gdn_project(Q38_GDN_WEIGHT_Q8_0, device_key, key_rows,
                                      cols, device_input, 1, device_key_output,
                                      stream, error, sizeof(error)) ||
                !q38_cuda_gdn_project(Q38_GDN_WEIGHT_Q8_0, device_value,
                                      value_rows, cols, device_input, 1,
                                      device_value_output, stream, error,
                                      sizeof(error)) ||
                !check_cuda(cudaStreamSynchronize(stream), "warmup sync")) {
                std::fprintf(stderr, "warmup failed: %s\n", error);
                return 1;
            }
        }
        cudaEvent_t execution_start = nullptr, execution_stop = nullptr;
        if (!check_cuda(cudaEventCreate(&execution_start), "cudaEventCreate") ||
            !check_cuda(cudaEventCreate(&execution_stop), "cudaEventCreate"))
            return 1;
        double enqueue_ms = 0.0;
        double complete_start = now_ms();
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
            if (!check_cuda(cudaMemcpyAsync(
                                device_input, inputs.data() + case_index * cols,
                                input_bytes, cudaMemcpyHostToDevice, stream),
                            "input upload"))
                return 1;
            double api_start = now_ms();
            if (!check_cuda(cudaEventRecord(execution_start, stream),
                            "execution start") ||
                !q38_cuda_gdn_project(Q38_GDN_WEIGHT_Q8_0, device_key, key_rows,
                                      cols, device_input, 1, device_key_output,
                                      stream, error, sizeof(error)) ||
                !q38_cuda_gdn_project(Q38_GDN_WEIGHT_Q8_0, device_value,
                                      value_rows, cols, device_input, 1,
                                      device_value_output, stream, error,
                                      sizeof(error)) ||
                !check_cuda(cudaEventRecord(execution_stop, stream),
                            "execution stop"))
                return 1;
            enqueue_ms += now_ms() - api_start;
            if (!check_cuda(cudaMemcpyAsync(actual_key.data(),
                                            device_key_output,
                                            key_rows * sizeof(float),
                                            cudaMemcpyDeviceToHost, stream),
                            "key download") ||
                !check_cuda(cudaMemcpyAsync(actual_value.data(),
                                            device_value_output,
                                            value_rows * sizeof(float),
                                            cudaMemcpyDeviceToHost, stream),
                            "value download") ||
                !check_cuda(cudaStreamSynchronize(stream), "iteration sync"))
                return 1;
        }
        double complete_ms = now_ms() - complete_start;
        float execution_ms = 0.0f;
        if (!check_cuda(cudaEventElapsedTime(&execution_ms, execution_start,
                                             execution_stop),
                        "execution elapsed"))
            return 1;
        double max_abs = 0.0, squared = 0.0;
        size_t count = key_rows + value_rows;
        for (size_t i = 0; i < key_rows; ++i) {
            double diff = static_cast<double>(actual_key[i]) -
                          expected_key[case_index * key_rows + i];
            max_abs = fmax(max_abs, fabs(diff));
            squared += diff * diff;
        }
        for (size_t i = 0; i < value_rows; ++i) {
            double diff = static_cast<double>(actual_value[i]) -
                          expected_value[case_index * value_rows + i];
            max_abs = fmax(max_abs, fabs(diff));
            squared += diff * diff;
        }
        if (case_index) std::printf(",");
        std::printf("{\"case\":\"%s\",\"correctness\":\"%s\","
                    "\"max_abs\":%.9g,\"rmse\":%.9g,"
                    "\"complete_projection_wall_ms\":%.6f,"
                    "\"complete_projection_wall_ms_per_iteration\":%.6f,"
                    "\"cuda_execution_ms_total\":%.6f,"
                    "\"cuda_execution_ms_per_iteration\":%.6f,"
                    "\"enqueue_api_wall_ms_total\":%.6f,"
                    "\"enqueue_api_wall_ms_per_iteration\":%.6f,"
                    "\"h2d_bytes_per_iteration\":%zu,"
                    "\"d2h_bytes_per_iteration\":%zu,"
                    "\"sync_count_per_iteration\":1,"
                    "\"matrix_calls_per_iteration\":2,"
                    "\"row_callbacks_per_iteration\":0,"
                    "\"weights_h2d_bytes_during_timing\":0}",
                    names[case_index], max_abs < 2e-3 ? "GREEN" : "RED",
                    max_abs, std::sqrt(squared / static_cast<double>(count)),
                    complete_ms, complete_ms / iterations, execution_ms,
                    execution_ms / iterations, enqueue_ms, enqueue_ms / iterations,
                    input_bytes, output_bytes);
        cudaEventDestroy(execution_start);
        cudaEventDestroy(execution_stop);
    }
    std::puts("]}");
    cudaFree(device_key);
    cudaFree(device_value);
    cudaFree(device_input);
    cudaFree(device_key_output);
    cudaFree(device_value_output);
    cudaStreamDestroy(stream);
    return 0;
}
