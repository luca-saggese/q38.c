#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr float kAbsTolerance = 1.0e-5f;
constexpr int kThreads = 128;
constexpr uint8_t kMagic[8] = {'N', 'V', 'F', '4', 'T', 'S', 'T', '1'};

struct Record {
    uint32_t stage;
    uint32_t projection;
    uint32_t m;
    uint32_t k;
    std::vector<uint8_t> weight;
    std::vector<uint8_t> weight_scale;
    float weight_scale_2;
    float input_scale;
    std::vector<uint8_t> activation;
    std::vector<uint8_t> activation_scale;
    std::vector<float> activation_input;
    std::vector<float> expected;
};

struct DeviceBuffers {
    uint8_t *weight = nullptr;
    uint8_t *weight_scale = nullptr;
    uint8_t *activation = nullptr;
    uint8_t *activation_scale = nullptr;
    uint8_t *quantized_activation = nullptr;
    uint8_t *quantized_scale = nullptr;
    float *activation_input = nullptr;
    float *output = nullptr;
    float *gate_output = nullptr;
    float *up_output = nullptr;
};

const char *stage_name(uint32_t stage) {
    return stage == 0 ? "early" : stage == 1 ? "middle" : "late";
}

const char *projection_name(uint32_t projection) {
    return projection == 0 ? "gate" : projection == 1 ? "up" : "down";
}

bool cuda_ok(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

template <typename T>
bool read_value(std::ifstream &input, T *value) {
    input.read(reinterpret_cast<char *>(value), sizeof(*value));
    return input.good();
}

bool read_bytes(std::ifstream &input, std::vector<uint8_t> *bytes,
                size_t count) {
    bytes->resize(count);
    input.read(reinterpret_cast<char *>(bytes->data()), count);
    return input.good();
}

bool read_fixture(const char *path, std::vector<Record> *records) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    uint8_t magic[8];
    uint32_t version = 0;
    uint32_t count = 0;
    input.read(reinterpret_cast<char *>(magic), sizeof(magic));
    if (!input.good() || std::memcmp(magic, kMagic, sizeof(magic)) != 0 ||
        !read_value(input, &version) || version != 1 ||
        !read_value(input, &count)) {
        return false;
    }
    records->clear();
    records->reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Record record;
        uint32_t weight_bytes = 0;
        uint32_t scale_bytes = 0;
        uint32_t activation_bytes = 0;
        uint32_t activation_scale_bytes = 0;
        if (!read_value(input, &record.stage) ||
            !read_value(input, &record.projection) ||
            !read_value(input, &record.m) ||
            !read_value(input, &record.k) ||
            !read_value(input, &weight_bytes) ||
            !read_value(input, &scale_bytes) ||
            !read_value(input, &activation_bytes) ||
            !read_value(input, &activation_scale_bytes) ||
            !read_value(input, &record.weight_scale_2) ||
            !read_value(input, &record.input_scale) ||
            !read_bytes(input, &record.weight, weight_bytes) ||
            !read_bytes(input, &record.weight_scale, scale_bytes) ||
            !read_bytes(input, &record.activation, activation_bytes) ||
            !read_bytes(input, &record.activation_scale,
                        activation_scale_bytes)) {
            return false;
        }
        record.activation_input.resize(record.k);
        record.expected.resize(record.m);
        input.read(reinterpret_cast<char *>(record.activation_input.data()),
                   record.activation_input.size() * sizeof(float));
        input.read(reinterpret_cast<char *>(record.expected.data()),
                   record.expected.size() * sizeof(float));
        if (!input.good()) return false;
        records->push_back(std::move(record));
    }
    return true;
}

struct FullFixture {
    uint32_t stage = 0;
    uint32_t layer = 0;
    uint32_t expert = 0;
    std::vector<Record> projections;
    std::vector<float> expected_intermediate;
};

constexpr uint8_t kFullMagic[8] = {'N', 'V', 'F', '4', 'F', 'U', 'L', 'L'};

bool read_full_fixture(const char *path, FullFixture *fixture) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    uint8_t magic[8];
    uint32_t version = 0;
    uint32_t count = 0;
    input.read(reinterpret_cast<char *>(magic), sizeof(magic));
    if (!input.good() || std::memcmp(magic, kFullMagic, sizeof(magic)) != 0 ||
        !read_value(input, &version) || version != 1 ||
        !read_value(input, &fixture->stage) ||
        !read_value(input, &fixture->layer) ||
        !read_value(input, &fixture->expert) ||
        !read_value(input, &count) || count != 3) {
        return false;
    }
    fixture->projections.clear();
    fixture->projections.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Record record;
        uint32_t weight_bytes = 0;
        uint32_t scale_bytes = 0;
        uint32_t input_count = 0;
        uint32_t activation_bytes = 0;
        uint32_t activation_scale_bytes = 0;
        if (!read_value(input, &record.projection) ||
            !read_value(input, &record.m) ||
            !read_value(input, &record.k) ||
            !read_value(input, &weight_bytes) ||
            !read_value(input, &scale_bytes) ||
            !read_value(input, &input_count) ||
            !read_value(input, &activation_bytes) ||
            !read_value(input, &activation_scale_bytes) ||
            !read_value(input, &record.weight_scale_2) ||
            !read_value(input, &record.input_scale) ||
            !read_bytes(input, &record.weight, weight_bytes) ||
            !read_bytes(input, &record.weight_scale, scale_bytes)) {
            return false;
        }
        record.stage = fixture->stage;
        record.activation_input.resize(input_count);
        input.read(reinterpret_cast<char *>(record.activation_input.data()),
                   input_count * sizeof(float));
        if (!input.good() ||
            !read_bytes(input, &record.activation, activation_bytes) ||
            !read_bytes(input, &record.activation_scale,
                        activation_scale_bytes)) {
            return false;
        }
        record.expected.resize(record.m);
        input.read(reinterpret_cast<char *>(record.expected.data()),
                   record.expected.size() * sizeof(float));
        if (!input.good()) return false;
        fixture->projections.push_back(std::move(record));
    }
    fixture->expected_intermediate.resize(640);
    input.read(reinterpret_cast<char *>(fixture->expected_intermediate.data()),
               fixture->expected_intermediate.size() * sizeof(float));
    return input.good();
}

__device__ __forceinline__ float decode_fp4(uint32_t nibble) {
    constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[nibble & 0xF];
}

__device__ __forceinline__ float decode_e4m3fn(uint32_t value) {
    const int sign = (value & 0x80) ? -1 : 1;
    const int exponent = (value >> 3) & 0xF;
    const int mantissa = value & 0x7;
    float decoded;
    if (exponent == 0) {
        decoded = (static_cast<float>(mantissa) / 8.0f) * exp2f(-6.0f);
    } else if (exponent == 0xF) {
        decoded = (1.0f + static_cast<float>(mantissa) / 8.0f) * 256.0f;
    } else {
        decoded = (1.0f + static_cast<float>(mantissa) / 8.0f) *
                  exp2f(static_cast<float>(exponent - 7));
    }
    return static_cast<float>(sign) * decoded;
}

__device__ __forceinline__ int round_to_even(float value) {
    const float lower = floorf(value);
    const float fraction = value - lower;
    if (fraction < 0.5f) return static_cast<int>(lower);
    if (fraction > 0.5f) return static_cast<int>(lower + 1.0f);
    const int lower_int = static_cast<int>(lower);
    return (lower_int & 1) == 0 ? lower_int : lower_int + 1;
}

__device__ __forceinline__ uint32_t encode_e4m3fn(float value) {
    const uint32_t sign = value < 0.0f ? 0x80u : 0u;
    const float magnitude = fabsf(value);
    if (magnitude == 0.0f) return sign;
    if (!isfinite(magnitude) || magnitude >= 448.0f) return sign | 0x7Eu;
    if (magnitude < exp2f(-6.0f)) {
        const int mantissa = round_to_even(magnitude * exp2f(9.0f));
        return sign | static_cast<uint32_t>(
            mantissa >= 8 ? 8 : mantissa);
    }
    int exponent = static_cast<int>(floorf(log2f(magnitude)));
    int exponent_field = exponent + 7;
    int mantissa = round_to_even(
        (magnitude / exp2f(static_cast<float>(exponent)) - 1.0f) * 8.0f);
    if (mantissa >= 8) {
        ++exponent_field;
        mantissa = 0;
    }
    if (exponent_field >= 0xF) return sign | 0x7Eu;
    return sign | (static_cast<uint32_t>(exponent_field) << 3) |
           static_cast<uint32_t>(mantissa);
}

__device__ __forceinline__ uint32_t encode_fp4(float value) {
    const float magnitude = fabsf(value);
    uint32_t code;
    if (magnitude <= 0.25f) code = 0;
    else if (magnitude < 0.75f) code = 1;
    else if (magnitude <= 1.25f) code = 2;
    else if (magnitude < 1.75f) code = 3;
    else if (magnitude <= 2.5f) code = 4;
    else if (magnitude < 3.5f) code = 5;
    else if (magnitude <= 5.0f) code = 6;
    else code = 7;
    return code | (value < 0.0f ? 0x8u : 0u);
}

__global__ void quantize_activation_kernel(
    const float *input, uint8_t *packed, uint8_t *scales,
    float input_scale, uint32_t k) {
    const uint32_t block = blockIdx.x;
    const uint32_t start = block * 16;
    if (start >= k) return;
    __shared__ float values[16];
    __shared__ float block_scale;
    __shared__ uint32_t codes[16];
    const uint32_t lane = threadIdx.x;
    if (lane < 16) values[lane] = input[start + lane];
    __syncthreads();
    if (lane == 0) {
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(values[i]));
        const uint32_t scale_byte =
            encode_e4m3fn(amax / (6.0f * input_scale));
        scales[block] = static_cast<uint8_t>(scale_byte);
        block_scale = decode_e4m3fn(scale_byte) * input_scale;
        if (block_scale < 1.0e-5f) block_scale = 1.0f;
    }
    __syncthreads();
    if (lane < 16) codes[lane] = encode_fp4(values[lane] / block_scale);
    __syncthreads();
    if (lane < 8) {
        packed[block * 8 + lane] = static_cast<uint8_t>(
            codes[lane * 2] | (codes[lane * 2 + 1] << 4));
    }
}

__global__ void nvfp4_matvec_kernel(
    const uint8_t *weight, const uint8_t *weight_scale,
    float weight_scale_2, const uint8_t *activation,
    const uint8_t *activation_scale, float input_scale,
    float *output, uint32_t m, uint32_t k) {
    const uint32_t row = blockIdx.x;
    if (row >= m) return;
    __shared__ float partial[kThreads];
    float sum = 0.0f;
    for (uint32_t logical_k = threadIdx.x; logical_k < k;
         logical_k += blockDim.x) {
        const uint8_t weight_byte = weight[row * (k / 2) + logical_k / 2];
        const uint8_t activation_byte = activation[logical_k / 2];
        const float weight_value = decode_fp4(
            logical_k & 1 ? weight_byte >> 4 : weight_byte & 0xF);
        const float activation_value = decode_fp4(
            logical_k & 1 ? activation_byte >> 4 : activation_byte & 0xF);
        const float weight_scale_value =
            decode_e4m3fn(weight_scale[row * (k / 16) + logical_k / 16]) *
            weight_scale_2;
        const float activation_scale_value =
            decode_e4m3fn(activation_scale[logical_k / 16]) * input_scale;
        sum += weight_value * weight_scale_value *
               activation_value * activation_scale_value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partial[0];
}

__global__ void silu_mul_kernel(
    const float *gate, const float *up, float *output, uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float gate_value = gate[index];
    output[index] = gate_value / (1.0f + expf(-gate_value)) * up[index];
}

bool q38_cuda_nvfp4_quantize_activation(
    const Record &record, DeviceBuffers *device) {
    if (!cuda_ok(cudaMemcpy(device->activation_input,
                            record.activation_input.data(),
                            record.activation_input.size() * sizeof(float),
                            cudaMemcpyHostToDevice), "copy activation input")) {
        return false;
    }
    quantize_activation_kernel<<<record.k / 16, 32>>>(
        device->activation_input, device->quantized_activation,
        device->quantized_scale, record.input_scale, record.k);
    return cuda_ok(cudaGetLastError(), "quantize activation launch");
}

bool q38_cuda_nvfp4_quantize_device_input(
    const Record &record, const DeviceBuffers *device) {
    quantize_activation_kernel<<<record.k / 16, 32>>>(
        device->activation_input, device->quantized_activation,
        device->quantized_scale, record.input_scale, record.k);
    return cuda_ok(cudaGetLastError(), "quantize device activation launch");
}

bool q38_cuda_nvfp4_launch_matvec(
    const Record &record, DeviceBuffers *device) {
    if (!cuda_ok(cudaMemcpy(device->weight, record.weight.data(),
                            record.weight.size(), cudaMemcpyHostToDevice),
                 "copy packed weight") ||
        !cuda_ok(cudaMemcpy(device->weight_scale, record.weight_scale.data(),
                            record.weight_scale.size(), cudaMemcpyHostToDevice),
                 "copy weight scale")) {
        return false;
    }
    nvfp4_matvec_kernel<<<record.m, kThreads>>>(
        device->weight, device->weight_scale, record.weight_scale_2,
        device->quantized_activation, device->quantized_scale,
        record.input_scale, device->output, record.m, record.k);
    return cuda_ok(cudaGetLastError(), "matvec launch") &&
           cuda_ok(cudaDeviceSynchronize(), "matvec synchronize");
}

bool q38_cuda_nvfp4_matvec_reference(
    const Record &record, const DeviceBuffers *device,
    const uint8_t *activation, const uint8_t *activation_scale,
    float *output) {
    if (!cuda_ok(cudaMemcpy(device->weight, record.weight.data(),
                            record.weight.size(), cudaMemcpyHostToDevice),
                 "copy packed weight") ||
        !cuda_ok(cudaMemcpy(device->weight_scale, record.weight_scale.data(),
                            record.weight_scale.size(), cudaMemcpyHostToDevice),
                 "copy weight scale") ||
        !cuda_ok(cudaMemcpy(device->activation, activation, record.k / 2,
                            cudaMemcpyHostToDevice),
                 "copy packed activation") ||
        !cuda_ok(cudaMemcpy(device->activation_scale, activation_scale,
                            record.k / 16,
                            cudaMemcpyHostToDevice),
                 "copy activation scale")) {
        return false;
    }
    nvfp4_matvec_kernel<<<record.m, kThreads>>>(
        device->weight, device->weight_scale, record.weight_scale_2,
        device->activation, device->activation_scale, record.input_scale,
        device->output, record.m, record.k);
    if (!cuda_ok(cudaGetLastError(), "matvec launch") ||
        !cuda_ok(cudaDeviceSynchronize(), "matvec synchronize")) {
        return false;
    }
    if (!cuda_ok(cudaMemcpy(output, device->output,
                            record.m * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy matvec output")) {
        return false;
    }
    return true;
}

bool allocate_buffers(DeviceBuffers *device, uint32_t max_m, uint32_t max_k);
void free_buffers(DeviceBuffers *device);
bool compare_values(const float *actual, const std::vector<float> &expected,
                    uint32_t count, double *max_abs, double *rmse,
                    uint32_t *nan_count, uint32_t *inf_count);

bool run_full_fixture(const char *path) {
    FullFixture fixture;
    if (!read_full_fixture(path, &fixture)) {
        std::fprintf(stderr, "failed to read full fixture: %s\n", path);
        return false;
    }
    const Record *gate = nullptr;
    const Record *up = nullptr;
    const Record *down = nullptr;
    for (const Record &record : fixture.projections) {
        if (record.projection == 0) gate = &record;
        else if (record.projection == 1) up = &record;
        else if (record.projection == 2) down = &record;
    }
    if (gate == nullptr || up == nullptr || down == nullptr) return false;

    DeviceBuffers device;
    if (!allocate_buffers(&device, 2560, 2560)) {
        free_buffers(&device);
        return false;
    }
    bool passed = true;
    std::vector<float> gate_output(gate->m);
    std::vector<float> up_output(up->m);
    std::vector<float> intermediate(640);
    std::vector<float> down_output(down->m);

    auto quantize_and_compare = [&](const Record &record,
                                    const char *name) {
        if (!q38_cuda_nvfp4_quantize_activation(record, &device) ||
            !cuda_ok(cudaDeviceSynchronize(), "quantizer synchronize")) {
            return false;
        }
        std::vector<uint8_t> packed(record.k / 2);
        std::vector<uint8_t> scales(record.k / 16);
        if (!cuda_ok(cudaMemcpy(packed.data(), device.quantized_activation,
                                packed.size(), cudaMemcpyDeviceToHost),
                     "copy full quantized activation") ||
            !cuda_ok(cudaMemcpy(scales.data(), device.quantized_scale,
                                scales.size(), cudaMemcpyDeviceToHost),
                     "copy full quantized scales")) {
            return false;
        }
        const bool exact = packed == record.activation &&
                           scales == record.activation_scale;
        std::printf("{\"full_stage\":\"%s\",\"projection\":\"%s\","
                    "\"activation_packed_exact\":%s,"
                    "\"activation_scale_exact\":%s}\n",
                    stage_name(record.stage), name,
                    packed == record.activation ? "true" : "false",
                    scales == record.activation_scale ? "true" : "false");
        return exact;
    };

    if (!quantize_and_compare(*gate, "gate") ||
        !q38_cuda_nvfp4_launch_matvec(*gate, &device) ||
        !cuda_ok(cudaMemcpy(device.gate_output, device.output,
                            gate->m * sizeof(float),
                            cudaMemcpyDeviceToDevice),
                 "save gate output") ||
        !cuda_ok(cudaMemcpy(gate_output.data(), device.output,
                            gate->m * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy gate output")) {
        passed = false;
    }
    if (passed && !quantize_and_compare(*up, "up")) passed = false;
    if (passed &&
        (!q38_cuda_nvfp4_launch_matvec(*up, &device) ||
         !cuda_ok(cudaMemcpy(device.up_output, device.output,
                             up->m * sizeof(float),
                             cudaMemcpyDeviceToDevice),
                  "save up output") ||
         !cuda_ok(cudaMemcpy(up_output.data(), device.output,
                             up->m * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy up output"))) {
        passed = false;
    }
    if (passed) {
        silu_mul_kernel<<<5, 128>>>(
            device.gate_output, device.up_output, device.activation_input,
            640);
        passed = cuda_ok(cudaGetLastError(), "silu multiply launch") &&
                 cuda_ok(cudaDeviceSynchronize(), "silu multiply synchronize") &&
                 cuda_ok(cudaMemcpy(intermediate.data(), device.activation_input,
                                    intermediate.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy intermediate output");
    }

    double intermediate_max_abs = 0.0;
    double intermediate_rmse = 0.0;
    uint32_t intermediate_nan = 0;
    uint32_t intermediate_inf = 0;
    const bool intermediate_ok =
        passed && compare_values(
            intermediate.data(), fixture.expected_intermediate, 640,
            &intermediate_max_abs, &intermediate_rmse, &intermediate_nan,
            &intermediate_inf);
    std::printf(
        "{\"full_stage\":\"%s\",\"projection\":\"silu_mul\","
        "\"max_abs\":%.9g,\"rmse\":%.9g,\"nan\":%u,\"inf\":%u,"
        "\"pass\":%s}\n",
        stage_name(fixture.stage), intermediate_max_abs, intermediate_rmse,
        intermediate_nan, intermediate_inf, intermediate_ok ? "true" : "false");
    passed = passed && intermediate_ok;

    if (passed) {
        if (!q38_cuda_nvfp4_quantize_device_input(*down, &device) ||
            !cuda_ok(cudaDeviceSynchronize(), "down quantizer synchronize")) {
            passed = false;
        } else {
            std::vector<uint8_t> packed(down->k / 2);
            std::vector<uint8_t> scales(down->k / 16);
            passed = cuda_ok(cudaMemcpy(packed.data(), device.quantized_activation,
                                        packed.size(), cudaMemcpyDeviceToHost),
                             "copy down quantized activation") &&
                      cuda_ok(cudaMemcpy(scales.data(), device.quantized_scale,
                                         scales.size(), cudaMemcpyDeviceToHost),
                              "copy down quantized scales") &&
                      packed == down->activation &&
                      scales == down->activation_scale;
            std::printf(
                "{\"full_stage\":\"%s\",\"projection\":\"down\","
                "\"activation_packed_exact\":%s,"
                "\"activation_scale_exact\":%s}\n",
                stage_name(fixture.stage),
                packed == down->activation ? "true" : "false",
                scales == down->activation_scale ? "true" : "false");
        }
    }
    if (passed &&
        (!q38_cuda_nvfp4_launch_matvec(*down, &device) ||
         !cuda_ok(cudaMemcpy(down_output.data(), device.output,
                             down->m * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "copy down output"))) {
        passed = false;
    }

    auto report_projection = [&](const char *name, const std::vector<float> &actual,
                                 const Record &record) {
        double max_abs = 0.0;
        double rmse = 0.0;
        uint32_t nan = 0;
        uint32_t inf = 0;
        const bool ok = passed && compare_values(
            actual.data(), record.expected, record.m, &max_abs, &rmse, &nan, &inf);
        std::printf(
            "{\"full_stage\":\"%s\",\"projection\":\"%s\","
            "\"max_abs\":%.9g,\"rmse\":%.9g,\"nan\":%u,\"inf\":%u,"
            "\"pass\":%s}\n",
            stage_name(fixture.stage), name, max_abs, rmse, nan, inf,
            ok ? "true" : "false");
        return ok;
    };
    const bool gate_ok = report_projection("gate", gate_output, *gate);
    const bool up_ok = report_projection("up", up_output, *up);
    const bool down_ok = report_projection("down", down_output, *down);
    passed = passed && gate_ok && up_ok && down_ok;
    free_buffers(&device);
    std::printf(
        "{\"full_stage\":\"%s\",\"full_expert\":\"layer_%u_expert_%u\","
        "\"cuda_allocations_during_call\":0,\"cuda_allocations\":10,"
        "\"pass\":%s,\"abs_tolerance\":%.9g}\n",
        stage_name(fixture.stage), fixture.layer, fixture.expert,
        passed ? "true" : "false", kAbsTolerance);
    return passed;
}

bool allocate_buffers(DeviceBuffers *device, uint32_t max_m, uint32_t max_k) {
    return cuda_ok(cudaMalloc(&device->weight, max_m * (max_k / 2)),
                   "allocate weight") &&
           cuda_ok(cudaMalloc(&device->weight_scale, max_m * (max_k / 16)),
                   "allocate weight scale") &&
           cuda_ok(cudaMalloc(&device->activation, max_k / 2),
                   "allocate activation") &&
           cuda_ok(cudaMalloc(&device->activation_scale, max_k / 16),
                   "allocate activation scale") &&
           cuda_ok(cudaMalloc(&device->quantized_activation, max_k / 2),
                   "allocate quantized activation") &&
           cuda_ok(cudaMalloc(&device->quantized_scale, max_k / 16),
                   "allocate quantized scale") &&
           cuda_ok(cudaMalloc(&device->activation_input,
                              max_k * sizeof(float)),
                   "allocate activation input") &&
           cuda_ok(cudaMalloc(&device->output, max_m * sizeof(float)),
                   "allocate output") &&
           cuda_ok(cudaMalloc(&device->gate_output, max_m * sizeof(float)),
                   "allocate gate output") &&
           cuda_ok(cudaMalloc(&device->up_output, max_m * sizeof(float)),
                   "allocate up output");
}

void free_buffers(DeviceBuffers *device) {
    cudaFree(device->weight);
    cudaFree(device->weight_scale);
    cudaFree(device->activation);
    cudaFree(device->activation_scale);
    cudaFree(device->quantized_activation);
    cudaFree(device->quantized_scale);
    cudaFree(device->activation_input);
    cudaFree(device->output);
    cudaFree(device->gate_output);
    cudaFree(device->up_output);
    *device = {};
}

bool compare_values(const float *actual, const std::vector<float> &expected,
                    uint32_t count, double *max_abs, double *rmse,
                    uint32_t *nan_count, uint32_t *inf_count) {
    double sum_sq = 0.0;
    *max_abs = 0.0;
    *nan_count = 0;
    *inf_count = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (std::isnan(actual[i])) ++*nan_count;
        if (std::isinf(actual[i])) ++*inf_count;
        const double difference =
            std::fabs(static_cast<double>(actual[i]) - expected[i]);
        *max_abs = std::max(*max_abs, difference);
        sum_sq += difference * difference;
    }
    *rmse = std::sqrt(sum_sq / count);
    return *max_abs <= kAbsTolerance && *nan_count == 0 && *inf_count == 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s fixture.bin [full_expert_payload.bin]\n",
                     argv[0]);
        return 2;
    }
    std::vector<Record> records;
    if (!read_fixture(argv[1], &records)) {
        std::fprintf(stderr, "failed to read fixture: %s\n", argv[1]);
        return 1;
    }
    DeviceBuffers device;
    if (!allocate_buffers(&device, 4, 64)) {
        free_buffers(&device);
        return 1;
    }
    bool passed = true;
    for (const Record &record : records) {
        std::vector<float> c0_output(record.m);
        if (!q38_cuda_nvfp4_matvec_reference(
                record, &device, record.activation.data(),
                record.activation_scale.data(), c0_output.data())) {
            passed = false;
            break;
        }
        double c0_max_abs = 0.0;
        double c0_rmse = 0.0;
        uint32_t c0_nan = 0;
        uint32_t c0_inf = 0;
        const bool c0_ok = compare_values(
            c0_output.data(), record.expected, record.m, &c0_max_abs,
            &c0_rmse, &c0_nan, &c0_inf);

        if (!q38_cuda_nvfp4_quantize_activation(record, &device) ||
            !cuda_ok(cudaDeviceSynchronize(), "quantizer synchronize")) {
            passed = false;
            break;
        }
        std::vector<uint8_t> quantized_activation(32);
        std::vector<uint8_t> quantized_scale(4);
        if (!cuda_ok(cudaMemcpy(quantized_activation.data(),
                                device.quantized_activation, 32,
                                cudaMemcpyDeviceToHost),
                     "copy quantized activation") ||
            !cuda_ok(cudaMemcpy(quantized_scale.data(),
                                device.quantized_scale, 4,
                                cudaMemcpyDeviceToHost),
                     "copy quantized scale")) {
            passed = false;
            break;
        }
        const bool quantized_ok =
            quantized_activation == record.activation &&
            quantized_scale == record.activation_scale;

        std::vector<float> c2_output(record.m);
        if (!q38_cuda_nvfp4_matvec_reference(
                record, &device, quantized_activation.data(),
                quantized_scale.data(), c2_output.data())) {
            passed = false;
            break;
        }
        double c2_max_abs = 0.0;
        double c2_rmse = 0.0;
        uint32_t c2_nan = 0;
        uint32_t c2_inf = 0;
        const bool c2_ok = compare_values(
            c2_output.data(), record.expected, record.m, &c2_max_abs,
            &c2_rmse, &c2_nan, &c2_inf);
        std::printf(
            "{\"stage\":\"%s\",\"projection\":\"%s\","
            "\"c0\":{\"max_abs\":%.9g,\"rmse\":%.9g,\"nan\":%u,\"inf\":%u,"
            "\"pass\":%s},"
            "\"activation\":{\"packed_exact\":%s,\"scale_exact\":%s},"
            "\"c2\":{\"max_abs\":%.9g,\"rmse\":%.9g,\"nan\":%u,\"inf\":%u,"
            "\"pass\":%s}}\n",
            stage_name(record.stage), projection_name(record.projection),
            c0_max_abs, c0_rmse, c0_nan, c0_inf, c0_ok ? "true" : "false",
            quantized_activation == record.activation ? "true" : "false",
            quantized_scale == record.activation_scale ? "true" : "false",
            c2_max_abs, c2_rmse, c2_nan, c2_inf, c2_ok ? "true" : "false");
        passed = passed && c0_ok && quantized_ok && c2_ok;
    }
    free_buffers(&device);
    std::printf("{\"records\":%zu,\"pass\":%s,\"abs_tolerance\":%.9g,"
                "\"full_expert\":\"not_requested\"}\n",
                records.size(), passed ? "true" : "false", kAbsTolerance);
    if (argc == 3) passed = passed && run_full_fixture(argv[2]);
    return passed ? 0 : 1;
}
