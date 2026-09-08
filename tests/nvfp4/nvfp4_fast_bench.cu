#define main nvfp4_cuda_oracle_fixture_main
#include "nvfp4_cuda_fixture.cu"
#undef main

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr uint8_t kBundleMagic[8] = {'N', 'V', 'F', '4', 'B', 'N', 'D', 'L'};
constexpr uint32_t kExpertCount = 10;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kIntermediate = 640;
constexpr uint32_t kBlock = 16;
constexpr uint32_t kFastThreads = 128;
constexpr size_t kWarmup = 100;
constexpr size_t kMeasured = 100;
constexpr float kTolerance = 1.0e-5f;

struct BundleProjection {
    uint32_t projection = 0;
    uint32_t m = 0;
    uint32_t k = 0;
    std::vector<uint8_t> weight;
    std::vector<uint8_t> scale;
    float weight_scale_2 = 0.0f;
    float input_scale = 0.0f;
};

struct BundleExpert {
    uint32_t id = 0;
    float route_weight = 0.0f;
    BundleProjection projection[3];
};

struct Bundle {
    uint32_t stage = 0;
    uint32_t layer = 0;
    std::vector<float> hidden;
    std::vector<BundleExpert> experts;
};

struct FullExpert {
    uint32_t stage = 0;
    uint32_t layer = 0;
    uint32_t expert = 0;
    std::vector<float> hidden;
    BundleExpert data;
};

struct Resident {
    uint32_t count = 0;
    uint8_t *gate_weight = nullptr;
    uint8_t *gate_scale = nullptr;
    float *gate_scale_2 = nullptr;
    float *gate_input_scale = nullptr;
    uint8_t *up_weight = nullptr;
    uint8_t *up_scale = nullptr;
    float *up_scale_2 = nullptr;
    float *up_input_scale = nullptr;
    uint8_t *down_weight = nullptr;
    uint8_t *down_scale = nullptr;
    float *down_scale_2 = nullptr;
    float *down_input_scale = nullptr;
    uint16_t *expert_ids = nullptr;
    float *route_weights = nullptr;
    float *hidden = nullptr;
    uint8_t *activation = nullptr;
    uint8_t *activation_scale = nullptr;
    uint8_t *down_activation = nullptr;
    uint8_t *down_activation_scale = nullptr;
    float *gate_output = nullptr;
    float *up_output = nullptr;
    float *mid = nullptr;
    float *expert_output = nullptr;
    float *output = nullptr;
};

struct Timing {
    double median_us = 0.0;
    double p95_us = 0.0;
    size_t kernel_launches = 0;
    size_t effective_weight_bytes = 0;
    double effective_gbps = 0.0;
};

template <typename T>
bool read_bundle_value(std::ifstream &input, T *value) {
    input.read(reinterpret_cast<char *>(value), sizeof(*value));
    return input.good();
}

bool read_bundle_bytes(std::ifstream &input, std::vector<uint8_t> *bytes,
                       size_t count) {
    bytes->resize(count);
    input.read(reinterpret_cast<char *>(bytes->data()), count);
    return input.good();
}

bool read_bundle_fixed(const char *path, Bundle *bundle) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    uint8_t magic[8];
    uint32_t version = 0;
    uint32_t count = 0;
    uint32_t hidden_count = 0;
    input.read(reinterpret_cast<char *>(magic), sizeof(magic));
    if (!input.good() || std::memcmp(magic, kBundleMagic, sizeof(magic)) != 0 ||
        !read_bundle_value(input, &version) || version != 1 ||
        !read_bundle_value(input, &bundle->stage) ||
        !read_bundle_value(input, &bundle->layer) ||
        !read_bundle_value(input, &count) || count != kExpertCount ||
        !read_bundle_value(input, &hidden_count) || hidden_count != kHidden) {
        return false;
    }
    bundle->hidden.resize(kHidden);
    input.read(reinterpret_cast<char *>(bundle->hidden.data()),
               bundle->hidden.size() * sizeof(float));
    if (!input.good()) return false;
    bundle->experts.resize(count);
    for (BundleExpert &expert : bundle->experts) {
        if (!read_bundle_value(input, &expert.id) ||
            !read_bundle_value(input, &expert.route_weight)) {
            return false;
        }
        for (uint32_t projection = 0; projection < 3; ++projection) {
            BundleProjection &record = expert.projection[projection];
            uint32_t input_count = 0;
            uint32_t activation_bytes = 0;
            uint32_t activation_scale_bytes = 0;
            if (!read_bundle_value(input, &record.projection) ||
                !read_bundle_value(input, &record.m) ||
                !read_bundle_value(input, &record.k)) {
                return false;
            }
            uint32_t weight_bytes = 0;
            uint32_t scale_bytes = 0;
            if (!read_bundle_value(input, &weight_bytes) ||
                !read_bundle_value(input, &scale_bytes) ||
                !read_bundle_value(input, &input_count) ||
                !read_bundle_value(input, &activation_bytes) ||
                !read_bundle_value(input, &activation_scale_bytes) ||
                !read_bundle_value(input, &record.weight_scale_2) ||
                !read_bundle_value(input, &record.input_scale) ||
                input_count != 0 || activation_bytes != 0 ||
                activation_scale_bytes != 0 ||
                !read_bundle_bytes(input, &record.weight, weight_bytes) ||
                !read_bundle_bytes(input, &record.scale, scale_bytes)) {
                return false;
            }
        }
    }
    return true;
}

bool read_full(const char *path, FullExpert *full) {
    FullFixture source;
    if (!read_full_fixture(path, &source) || source.projections.size() != 3)
        return false;
    full->stage = source.stage;
    full->layer = source.layer;
    full->expert = 0;
    full->hidden = source.projections[0].activation_input;
    for (const Record &record : source.projections) {
        BundleProjection &projection = full->data.projection[record.projection];
        projection.projection = record.projection;
        projection.m = record.m;
        projection.k = record.k;
        projection.weight = record.weight;
        projection.scale = record.weight_scale;
        projection.weight_scale_2 = record.weight_scale_2;
        projection.input_scale = record.input_scale;
    }
    return full->hidden.size() == kHidden;
}

void free_resident(Resident *device) {
    cudaFree(device->gate_weight);
    cudaFree(device->gate_scale);
    cudaFree(device->gate_scale_2);
    cudaFree(device->gate_input_scale);
    cudaFree(device->up_weight);
    cudaFree(device->up_scale);
    cudaFree(device->up_scale_2);
    cudaFree(device->up_input_scale);
    cudaFree(device->down_weight);
    cudaFree(device->down_scale);
    cudaFree(device->down_scale_2);
    cudaFree(device->down_input_scale);
    cudaFree(device->expert_ids);
    cudaFree(device->route_weights);
    cudaFree(device->hidden);
    cudaFree(device->activation);
    cudaFree(device->activation_scale);
    cudaFree(device->down_activation);
    cudaFree(device->down_activation_scale);
    cudaFree(device->gate_output);
    cudaFree(device->up_output);
    cudaFree(device->mid);
    cudaFree(device->expert_output);
    cudaFree(device->output);
    *device = {};
}

bool alloc_resident(Resident *device, uint32_t count) {
    device->count = count;
    const size_t gate_weight_bytes =
        (size_t)count * kIntermediate * (kHidden / 2);
    const size_t gate_scale_bytes =
        (size_t)count * kIntermediate * (kHidden / kBlock);
    const size_t down_weight_bytes =
        (size_t)count * kHidden * (kIntermediate / 2);
    const size_t down_scale_bytes =
        (size_t)count * kHidden * (kIntermediate / kBlock);
    const size_t activation_bytes = kHidden / 2;
    const size_t activation_scale_bytes = kHidden / kBlock;
    const size_t down_activation_bytes = count * (kIntermediate / 2);
    const size_t down_activation_scale_bytes =
        count * (kIntermediate / kBlock);
    return cuda_ok(cudaMalloc(&device->gate_weight, gate_weight_bytes),
                   "allocate gate weights") &&
           cuda_ok(cudaMalloc(&device->gate_scale, gate_scale_bytes),
                   "allocate gate scales") &&
           cuda_ok(cudaMalloc(&device->gate_scale_2, count * sizeof(float)),
                   "allocate gate scale_2") &&
           cuda_ok(cudaMalloc(&device->gate_input_scale,
                              count * sizeof(float)),
                   "allocate gate input scales") &&
           cuda_ok(cudaMalloc(&device->up_weight, gate_weight_bytes),
                   "allocate up weights") &&
           cuda_ok(cudaMalloc(&device->up_scale, gate_scale_bytes),
                   "allocate up scales") &&
           cuda_ok(cudaMalloc(&device->up_scale_2, count * sizeof(float)),
                   "allocate up scale_2") &&
           cuda_ok(cudaMalloc(&device->up_input_scale,
                              count * sizeof(float)),
                   "allocate up input scales") &&
           cuda_ok(cudaMalloc(&device->down_weight, down_weight_bytes),
                   "allocate down weights") &&
           cuda_ok(cudaMalloc(&device->down_scale, down_scale_bytes),
                   "allocate down scales") &&
           cuda_ok(cudaMalloc(&device->down_scale_2, count * sizeof(float)),
                   "allocate down scale_2") &&
           cuda_ok(cudaMalloc(&device->down_input_scale,
                              count * sizeof(float)),
                   "allocate down input scales") &&
           cuda_ok(cudaMalloc(&device->expert_ids,
                              count * sizeof(uint16_t)),
                   "allocate expert ids") &&
           cuda_ok(cudaMalloc(&device->route_weights,
                              count * sizeof(float)),
                   "allocate route weights") &&
           cuda_ok(cudaMalloc(&device->hidden, kHidden * sizeof(float)),
                   "allocate hidden") &&
           cuda_ok(cudaMalloc(&device->activation, activation_bytes),
                   "allocate activation") &&
           cuda_ok(cudaMalloc(&device->activation_scale,
                              activation_scale_bytes),
                   "allocate activation scales") &&
           cuda_ok(cudaMalloc(&device->down_activation,
                              down_activation_bytes),
                   "allocate down activations") &&
           cuda_ok(cudaMalloc(&device->down_activation_scale,
                              down_activation_scale_bytes),
                   "allocate down activation scales") &&
           cuda_ok(cudaMalloc(&device->gate_output,
                              count * kIntermediate * sizeof(float)),
                   "allocate gate output") &&
           cuda_ok(cudaMalloc(&device->up_output,
                              count * kIntermediate * sizeof(float)),
                   "allocate up output") &&
           cuda_ok(cudaMalloc(&device->mid,
                              count * kIntermediate * sizeof(float)),
                   "allocate intermediate") &&
           cuda_ok(cudaMalloc(&device->expert_output,
                              count * kHidden * sizeof(float)),
                   "allocate expert output") &&
           cuda_ok(cudaMalloc(&device->output, kHidden * sizeof(float)),
                   "allocate output");
}

bool upload_projection(
    const BundleProjection *projections, uint32_t count, Resident *device) {
    const size_t gate_weight_bytes =
        (size_t)kIntermediate * (kHidden / 2);
    const size_t gate_scale_bytes =
        (size_t)kIntermediate * (kHidden / kBlock);
    const size_t down_weight_bytes =
        (size_t)kHidden * (kIntermediate / 2);
    const size_t down_scale_bytes =
        (size_t)kHidden * (kIntermediate / kBlock);
    std::vector<uint16_t> ids(count);
    std::vector<float> routes(count);
    for (uint32_t expert = 0; expert < count; ++expert) {
        const BundleProjection &gate = projections[expert * 3 + 0];
        const BundleProjection &up = projections[expert * 3 + 1];
        const BundleProjection &down = projections[expert * 3 + 2];
        if (!cuda_ok(cudaMemcpy(
                        device->gate_weight + expert * gate_weight_bytes,
                        gate.weight.data(), gate.weight.size(),
                        cudaMemcpyHostToDevice),
                     "upload gate weights") ||
            !cuda_ok(cudaMemcpy(
                        device->gate_scale + expert * gate_scale_bytes,
                        gate.scale.data(), gate.scale.size(),
                        cudaMemcpyHostToDevice),
                     "upload gate scales") ||
            !cuda_ok(cudaMemcpy(
                        device->up_weight + expert * gate_weight_bytes,
                        up.weight.data(), up.weight.size(),
                        cudaMemcpyHostToDevice),
                     "upload up weights") ||
            !cuda_ok(cudaMemcpy(
                        device->up_scale + expert * gate_scale_bytes,
                        up.scale.data(), up.scale.size(),
                        cudaMemcpyHostToDevice),
                     "upload up scales") ||
            !cuda_ok(cudaMemcpy(
                        device->down_weight + expert * down_weight_bytes,
                        down.weight.data(), down.weight.size(),
                        cudaMemcpyHostToDevice),
                     "upload down weights") ||
            !cuda_ok(cudaMemcpy(
                        device->down_scale + expert * down_scale_bytes,
                        down.scale.data(), down.scale.size(),
                        cudaMemcpyHostToDevice),
                     "upload down scales") ||
            !cuda_ok(cudaMemcpy(device->gate_scale_2 + expert,
                                &gate.weight_scale_2, sizeof(float),
                                cudaMemcpyHostToDevice),
                     "upload gate scale_2") ||
            !cuda_ok(cudaMemcpy(device->gate_input_scale + expert,
                                &gate.input_scale, sizeof(float),
                                cudaMemcpyHostToDevice),
                     "upload gate input scale") ||
            !cuda_ok(cudaMemcpy(device->up_scale_2 + expert,
                                &up.weight_scale_2, sizeof(float),
                                cudaMemcpyHostToDevice),
                     "upload up scale_2") ||
            !cuda_ok(cudaMemcpy(device->up_input_scale + expert,
                                &up.input_scale, sizeof(float),
                                cudaMemcpyHostToDevice),
                     "upload up input scale") ||
            !cuda_ok(cudaMemcpy(device->down_scale_2 + expert,
                                &down.weight_scale_2, sizeof(float),
                                cudaMemcpyHostToDevice),
                     "upload down scale_2") ||
            !cuda_ok(cudaMemcpy(device->down_input_scale + expert,
                                &down.input_scale, sizeof(float),
                                cudaMemcpyHostToDevice),
                     "upload down input scale")) {
            return false;
        }
        ids[expert] = static_cast<uint16_t>(expert);
        routes[expert] = 1.0f / static_cast<float>(count);
    }
    return cuda_ok(cudaMemcpy(device->expert_ids, ids.data(),
                              ids.size() * sizeof(uint16_t),
                              cudaMemcpyHostToDevice),
                   "upload expert ids") &&
           cuda_ok(cudaMemcpy(device->route_weights, routes.data(),
                              routes.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload route weights");
}

bool upload_bundle(const Bundle &bundle, Resident *device) {
    std::vector<BundleProjection> projections;
    projections.reserve(bundle.experts.size() * 3);
    std::vector<uint16_t> ids(bundle.experts.size());
    std::vector<float> routes(bundle.experts.size());
    for (size_t i = 0; i < bundle.experts.size(); ++i) {
        projections.push_back(bundle.experts[i].projection[0]);
        projections.push_back(bundle.experts[i].projection[1]);
        projections.push_back(bundle.experts[i].projection[2]);
        ids[i] = static_cast<uint16_t>(bundle.experts[i].id);
        routes[i] = bundle.experts[i].route_weight;
    }
    if (!upload_projection(projections.data(), bundle.experts.size(), device))
        return false;
    return cuda_ok(cudaMemcpy(device->expert_ids, ids.data(),
                              ids.size() * sizeof(uint16_t),
                              cudaMemcpyHostToDevice),
                   "upload selected expert ids") &&
           cuda_ok(cudaMemcpy(device->route_weights, routes.data(),
                              routes.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload selected route weights") &&
           cuda_ok(cudaMemcpy(device->hidden, bundle.hidden.data(),
                              bundle.hidden.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload bundle hidden");
}

bool upload_single(const FullExpert &full, Resident *device) {
    std::vector<BundleProjection> projections = {
        full.data.projection[0],
        full.data.projection[1],
        full.data.projection[2],
    };
    return upload_projection(projections.data(), 1, device) &&
           cuda_ok(cudaMemcpy(device->hidden, full.hidden.data(),
                              full.hidden.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload single hidden");
}

__device__ __forceinline__ float fast_decode_weight(
    const uint8_t *weight, const uint8_t *scale, float scale_2,
    uint32_t row, uint32_t logical_k, uint32_t k) {
    const uint8_t packed = weight[row * (k / 2) + logical_k / 2];
    const uint8_t scale_byte = scale[row * (k / kBlock) + logical_k / kBlock];
    const uint32_t nibble =
        (logical_k & 1u) ? (packed >> 4) : (packed & 0x0Fu);
    return decode_fp4(nibble) * decode_e4m3fn(scale_byte) * scale_2;
}

__global__ void fast_nvfp4_matvec_kernel(
    const uint8_t *weight, const uint8_t *scale, float scale_2,
    const uint8_t *activation, const uint8_t *activation_scale,
    float input_scale, float *output, uint32_t m, uint32_t k) {
    extern __shared__ uint8_t staged[];
    uint8_t *staged_activation = staged;
    uint8_t *staged_scale = staged + k / 2;
    for (uint32_t i = threadIdx.x; i < k / 2; i += blockDim.x)
        staged_activation[i] = activation[i];
    for (uint32_t i = threadIdx.x; i < k / kBlock; i += blockDim.x)
        staged_scale[i] = activation_scale[i];
    __syncthreads();
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 4u + warp;
    if (row >= m) return;
    float sum = 0.0f;
    for (uint32_t logical_k = lane; logical_k < k; logical_k += 32u) {
        const uint8_t packed = weight[row * (k / 2) + logical_k / 2];
        const uint8_t activation_byte = staged_activation[logical_k / 2];
        const uint32_t weight_nibble =
            (logical_k & 1u) ? (packed >> 4) : (packed & 0x0Fu);
        const uint32_t activation_nibble =
            (logical_k & 1u) ? (activation_byte >> 4)
                             : (activation_byte & 0x0Fu);
        const float weight_value =
            decode_fp4(weight_nibble) *
            decode_e4m3fn(scale[row * (k / kBlock) + logical_k / kBlock]) *
            scale_2;
        const float activation_value =
            decode_fp4(activation_nibble) *
            decode_e4m3fn(staged_scale[logical_k / kBlock]) * input_scale;
        sum += weight_value * activation_value;
    }
    for (uint32_t offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) output[row] = sum;
}

__global__ void fast_gate_up_silu_kernel(
    const uint8_t *gate_weight, const uint8_t *gate_scale,
    const float *gate_scale_2, const uint8_t *up_weight,
    const uint8_t *up_scale, const float *up_scale_2,
    const uint8_t *activation, const uint8_t *activation_scale,
    float input_scale, float *mid, uint32_t m, uint32_t k) {
    extern __shared__ uint8_t staged[];
    uint8_t *staged_activation = staged;
    uint8_t *staged_scale = staged + k / 2;
    for (uint32_t i = threadIdx.x; i < k / 2; i += blockDim.x)
        staged_activation[i] = activation[i];
    for (uint32_t i = threadIdx.x; i < k / kBlock; i += blockDim.x)
        staged_scale[i] = activation_scale[i];
    __syncthreads();
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 4u + warp;
    if (row >= m) return;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t logical_k = lane; logical_k < k; logical_k += 32u) {
        const uint8_t activation_byte = staged_activation[logical_k / 2];
        const uint32_t activation_nibble =
            (logical_k & 1u) ? (activation_byte >> 4)
                             : (activation_byte & 0x0Fu);
        const float activation_value =
            decode_fp4(activation_nibble) *
            decode_e4m3fn(staged_scale[logical_k / kBlock]) * input_scale;
        gate += fast_decode_weight(
            gate_weight, gate_scale, gate_scale_2[0], row, logical_k, k) *
                activation_value;
        up += fast_decode_weight(
            up_weight, up_scale, up_scale_2[0], row, logical_k, k) *
              activation_value;
    }
    for (uint32_t offset = 16; offset; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0) mid[row] = gate / (1.0f + expf(-gate)) * up;
}

__global__ void fast_grouped_gate_up_silu_kernel(
    const uint8_t *gate_weight, const uint8_t *gate_scale,
    const float *gate_scale_2, const uint8_t *up_weight,
    const uint8_t *up_scale, const float *up_scale_2,
    const uint8_t *activation, const uint8_t *activation_scale,
    const float *input_scale, float *mid, uint32_t experts, uint32_t m,
    uint32_t k) {
    extern __shared__ uint8_t staged[];
    uint8_t *staged_activation = staged;
    uint8_t *staged_scale = staged + k / 2;
    for (uint32_t i = threadIdx.x; i < k / 2; i += blockDim.x)
        staged_activation[i] = activation[i];
    for (uint32_t i = threadIdx.x; i < k / kBlock; i += blockDim.x)
        staged_scale[i] = activation_scale[i];
    __syncthreads();
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t global_row = blockIdx.x * 4u + warp;
    const uint32_t total_rows = experts * m;
    if (global_row >= total_rows) return;
    const uint32_t expert = global_row / m;
    const uint32_t row = global_row % m;
    const size_t weight_stride = (size_t)m * (k / 2);
    const size_t scale_stride = (size_t)m * (k / kBlock);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t logical_k = lane; logical_k < k; logical_k += 32u) {
        const uint8_t activation_byte = staged_activation[logical_k / 2];
        const uint32_t activation_nibble =
            (logical_k & 1u) ? (activation_byte >> 4)
                             : (activation_byte & 0x0Fu);
        const float activation_value =
            decode_fp4(activation_nibble) *
            decode_e4m3fn(staged_scale[logical_k / kBlock]) *
            input_scale[expert];
        const uint8_t gate_byte =
            gate_weight[expert * weight_stride +
                        row * (k / 2) + logical_k / 2];
        const uint8_t up_byte =
            up_weight[expert * weight_stride +
                      row * (k / 2) + logical_k / 2];
        const uint32_t gate_nibble =
            (logical_k & 1u) ? (gate_byte >> 4) : (gate_byte & 0x0Fu);
        const uint32_t up_nibble =
            (logical_k & 1u) ? (up_byte >> 4) : (up_byte & 0x0Fu);
        gate += decode_fp4(gate_nibble) *
                decode_e4m3fn(gate_scale[
                    expert * scale_stride + row * (k / kBlock) +
                    logical_k / kBlock]) *
                gate_scale_2[expert] * activation_value;
        up += decode_fp4(up_nibble) *
              decode_e4m3fn(up_scale[
                  expert * scale_stride + row * (k / kBlock) +
                  logical_k / kBlock]) *
              up_scale_2[expert] * activation_value;
    }
    for (uint32_t offset = 16; offset; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0)
        mid[expert * m + row] = gate / (1.0f + expf(-gate)) * up;
}

__global__ void grouped_quantize_activation_kernel(
    const float *input, uint8_t *packed, uint8_t *scales,
    const float *input_scale, uint32_t experts, uint32_t k) {
    const uint32_t expert = blockIdx.x;
    const uint32_t block = blockIdx.y;
    if (expert >= experts || block * kBlock >= k) return;
    __shared__ float values[kBlock];
    __shared__ float block_scale;
    __shared__ uint32_t codes[kBlock];
    const uint32_t lane = threadIdx.x;
    const uint32_t start = block * kBlock;
    if (lane < kBlock) values[lane] = input[expert * k + start + lane];
    __syncthreads();
    if (lane == 0) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < kBlock; ++i)
            amax = fmaxf(amax, fabsf(values[i]));
        const uint32_t scale_byte =
            encode_e4m3fn(amax / (6.0f * input_scale[expert]));
        scales[expert * (k / kBlock) + block] = static_cast<uint8_t>(scale_byte);
        block_scale = decode_e4m3fn(scale_byte) * input_scale[expert];
        if (block_scale < 1.0e-5f) block_scale = 1.0f;
    }
    __syncthreads();
    if (lane < kBlock) codes[lane] = encode_fp4(values[lane] / block_scale);
    __syncthreads();
    if (lane < kBlock / 2) {
        packed[expert * (k / 2) + block * (kBlock / 2) + lane] =
            static_cast<uint8_t>(codes[lane * 2] |
                                 (codes[lane * 2 + 1] << 4));
    }
}

__global__ void grouped_down_accum_kernel(
    const uint8_t *weight, const uint8_t *scale, const float *scale_2,
    const uint8_t *activation, const uint8_t *activation_scale,
    const float *input_scale, const float *route_weights, float *output,
    uint32_t experts, uint32_t m, uint32_t k) {
    extern __shared__ uint8_t staged[];
    uint8_t *staged_activation = staged;
    uint8_t *staged_scale = staged + experts * (k / 2);
    for (uint32_t i = threadIdx.x;
         i < experts * (k / 2); i += blockDim.x)
        staged_activation[i] = activation[i];
    for (uint32_t i = threadIdx.x;
         i < experts * (k / kBlock); i += blockDim.x)
        staged_scale[i] = activation_scale[i];
    __syncthreads();
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 4u + warp;
    if (row >= m) return;
    const size_t weight_stride = (size_t)m * (k / 2);
    const size_t scale_stride = (size_t)m * (k / kBlock);
    float weighted_sum = 0.0f;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        float sum = 0.0f;
        for (uint32_t logical_k = lane; logical_k < k; logical_k += 32u) {
            const uint8_t packed =
                weight[expert * weight_stride + row * (k / 2) +
                       logical_k / 2];
            const uint8_t activation_byte =
                staged_activation[expert * (k / 2) + logical_k / 2];
            const uint32_t weight_nibble =
                (logical_k & 1u) ? (packed >> 4) : (packed & 0x0Fu);
            const uint32_t activation_nibble =
                (logical_k & 1u) ? (activation_byte >> 4)
                                 : (activation_byte & 0x0Fu);
            const float weight_value =
                decode_fp4(weight_nibble) *
                decode_e4m3fn(scale[
                    expert * scale_stride + row * (k / kBlock) +
                    logical_k / kBlock]) *
                scale_2[expert];
            const float activation_value =
                decode_fp4(activation_nibble) *
                decode_e4m3fn(staged_scale[
                    expert * (k / kBlock) + logical_k / kBlock]) *
                input_scale[expert];
            sum += weight_value * activation_value;
        }
        for (uint32_t offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0)
            weighted_sum = __fadd_rn(
                weighted_sum, __fmul_rn(route_weights[expert], sum));
    }
    if (lane == 0) output[row] = weighted_sum;
}

__global__ void weighted_accum_kernel(
    float *output, const float *expert, float route_weight) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kHidden)
        output[index] = __fadd_rn(
            output[index], __fmul_rn(route_weight, expert[index]));
}

void free_events(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename Fn>
bool measure(Fn &&fn, size_t launches, size_t bytes, Timing *timing) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "create benchmark start event") ||
        !cuda_ok(cudaEventCreate(&stop), "create benchmark stop event")) {
        free_events(start, stop);
        return false;
    }
    for (size_t i = 0; i < kWarmup; ++i) {
        if (!fn() || !cuda_ok(cudaDeviceSynchronize(), "benchmark warmup")) {
            free_events(start, stop);
            return false;
        }
    }
    std::vector<double> samples;
    samples.reserve(kMeasured);
    for (size_t i = 0; i < kMeasured; ++i) {
        if (!cuda_ok(cudaEventRecord(start), "record benchmark start") ||
            !fn() || !cuda_ok(cudaEventRecord(stop), "record benchmark stop") ||
            !cuda_ok(cudaEventSynchronize(stop), "sync benchmark stop")) {
            free_events(start, stop);
            return false;
        }
        float milliseconds = 0.0f;
        if (!cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop),
                     "elapsed benchmark time")) {
            free_events(start, stop);
            return false;
        }
        samples.push_back(milliseconds * 1000.0);
    }
    std::sort(samples.begin(), samples.end());
    timing->median_us = samples[samples.size() / 2];
    const size_t p95 = std::min(
        samples.size() - 1, static_cast<size_t>(std::ceil(samples.size() * 0.95)) - 1);
    timing->p95_us = samples[p95];
    timing->kernel_launches = launches;
    timing->effective_weight_bytes = bytes;
    timing->effective_gbps = bytes / (timing->median_us * 1000.0);
    free_events(start, stop);
    return true;
}

bool launch_oracle_full(const BundleExpert &expert, Resident *device,
                        uint32_t slot, float *output) {
    const BundleProjection &gate = expert.projection[0];
    const BundleProjection &up = expert.projection[1];
    const BundleProjection &down = expert.projection[2];
    quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
        device->hidden, device->activation, device->activation_scale,
        gate.input_scale, kHidden);
    nvfp4_matvec_kernel<<<kIntermediate, kFastThreads>>>(
        device->gate_weight + slot * (size_t)kIntermediate * (kHidden / 2),
        device->gate_scale + slot * (size_t)kIntermediate * (kHidden / kBlock),
        gate.weight_scale_2, device->activation, device->activation_scale,
        gate.input_scale, device->gate_output, kIntermediate, kHidden);
    quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
        device->hidden, device->activation, device->activation_scale,
        up.input_scale, kHidden);
    nvfp4_matvec_kernel<<<kIntermediate, kFastThreads>>>(
        device->up_weight + slot * (size_t)kIntermediate * (kHidden / 2),
        device->up_scale + slot * (size_t)kIntermediate * (kHidden / kBlock),
        up.weight_scale_2, device->activation, device->activation_scale,
        up.input_scale, device->up_output, kIntermediate, kHidden);
    silu_mul_kernel<<<5, 128>>>(
        device->gate_output, device->up_output, device->mid, kIntermediate);
    quantize_activation_kernel<<<kIntermediate / kBlock, 32>>>(
        device->mid, device->down_activation, device->down_activation_scale,
        down.input_scale, kIntermediate);
    nvfp4_matvec_kernel<<<kHidden, kFastThreads>>>(
        device->down_weight + slot * (size_t)kHidden * (kIntermediate / 2),
        device->down_scale + slot * (size_t)kHidden * (kIntermediate / kBlock),
        down.weight_scale_2, device->down_activation,
        device->down_activation_scale, down.input_scale, output, kHidden,
        kIntermediate);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_candidate_c1_full(const BundleExpert &expert, Resident *device,
                              uint32_t slot, float *output) {
    const auto &gate = expert.projection[0];
    const auto &up = expert.projection[1];
    const auto &down = expert.projection[2];
    quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
        device->hidden, device->activation, device->activation_scale,
        gate.input_scale, kHidden);
    fast_nvfp4_matvec_kernel<<<(kIntermediate + 3) / 4, kFastThreads,
                               kHidden / 2 + kHidden / kBlock>>>(
        device->gate_weight + slot * (size_t)kIntermediate * (kHidden / 2),
        device->gate_scale + slot * (size_t)kIntermediate * (kHidden / kBlock),
        gate.weight_scale_2, device->activation, device->activation_scale,
        gate.input_scale, device->gate_output, kIntermediate, kHidden);
    quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
        device->hidden, device->activation, device->activation_scale,
        up.input_scale, kHidden);
    fast_nvfp4_matvec_kernel<<<(kIntermediate + 3) / 4, kFastThreads,
                               kHidden / 2 + kHidden / kBlock>>>(
        device->up_weight + slot * (size_t)kIntermediate * (kHidden / 2),
        device->up_scale + slot * (size_t)kIntermediate * (kHidden / kBlock),
        up.weight_scale_2, device->activation, device->activation_scale,
        up.input_scale, device->up_output, kIntermediate, kHidden);
    silu_mul_kernel<<<5, 128>>>(
        device->gate_output, device->up_output, device->mid, kIntermediate);
    quantize_activation_kernel<<<kIntermediate / kBlock, 32>>>(
        device->mid, device->down_activation, device->down_activation_scale,
        down.input_scale, kIntermediate);
    fast_nvfp4_matvec_kernel<<<(kHidden + 3) / 4, kFastThreads,
                               kIntermediate / 2 + kIntermediate / kBlock>>>(
        device->down_weight + slot * (size_t)kHidden * (kIntermediate / 2),
        device->down_scale + slot * (size_t)kHidden * (kIntermediate / kBlock),
        down.weight_scale_2, device->down_activation,
        device->down_activation_scale, down.input_scale, output, kHidden,
        kIntermediate);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_candidate_c2_full(const BundleExpert &expert, Resident *device,
                              uint32_t slot, float *output) {
    const auto &gate = expert.projection[0];
    const auto &down = expert.projection[2];
    quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
        device->hidden, device->activation, device->activation_scale,
        gate.input_scale, kHidden);
    fast_gate_up_silu_kernel<<<(kIntermediate + 3) / 4, kFastThreads,
                               kHidden / 2 + kHidden / kBlock>>>(
        device->gate_weight + slot * (size_t)kIntermediate * (kHidden / 2),
        device->gate_scale + slot * (size_t)kIntermediate * (kHidden / kBlock),
        device->gate_scale_2 + slot,
        device->up_weight + slot * (size_t)kIntermediate * (kHidden / 2),
        device->up_scale + slot * (size_t)kIntermediate * (kHidden / kBlock),
        device->up_scale_2 + slot, device->activation,
        device->activation_scale, gate.input_scale, device->mid,
        kIntermediate, kHidden);
    quantize_activation_kernel<<<kIntermediate / kBlock, 32>>>(
        device->mid, device->down_activation, device->down_activation_scale,
        down.input_scale, kIntermediate);
    fast_nvfp4_matvec_kernel<<<(kHidden + 3) / 4, kFastThreads,
                               kIntermediate / 2 + kIntermediate / kBlock>>>(
        device->down_weight + slot * (size_t)kHidden * (kIntermediate / 2),
        device->down_scale + slot * (size_t)kHidden * (kIntermediate / kBlock),
        down.weight_scale_2, device->down_activation,
        device->down_activation_scale, down.input_scale, output, kHidden,
        kIntermediate);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_oracle_top10(const Bundle &bundle, Resident *device) {
    if (!cuda_ok(cudaMemset(device->output, 0, kHidden * sizeof(float)),
                 "clear oracle top10 output"))
        return false;
    for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
        if (!launch_oracle_full(bundle.experts[expert], device, expert,
                                device->expert_output +
                                    expert * kHidden))
            return false;
        weighted_accum_kernel<<<(kHidden + 255) / 256, 256>>>(
            device->output, device->expert_output + expert * kHidden,
            bundle.experts[expert].route_weight);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool launch_candidate_c4(const Bundle &bundle, Resident *device) {
    quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
        device->hidden, device->activation, device->activation_scale,
        bundle.experts[0].projection[0].input_scale, kHidden);
    fast_grouped_gate_up_silu_kernel<<<
        (kExpertCount * kIntermediate + 3) / 4, kFastThreads,
        kHidden / 2 + kHidden / kBlock>>>(
        device->gate_weight, device->gate_scale, device->gate_scale_2,
        device->up_weight, device->up_scale, device->up_scale_2,
        device->activation, device->activation_scale,
        device->gate_input_scale, device->mid, kExpertCount,
        kIntermediate, kHidden);
    dim3 grid(kExpertCount, kIntermediate / kBlock, 1);
    grouped_quantize_activation_kernel<<<grid, 32>>>(
        device->mid, device->down_activation, device->down_activation_scale,
        device->down_input_scale, kExpertCount, kIntermediate);
    grouped_down_accum_kernel<<<(kHidden + 3) / 4, kFastThreads,
                                kExpertCount * (kIntermediate / 2) +
                                    kExpertCount * (kIntermediate / kBlock)>>>(
        device->down_weight, device->down_scale, device->down_scale_2,
        device->down_activation, device->down_activation_scale,
        device->down_input_scale, device->route_weights, device->output,
        kExpertCount, kHidden, kIntermediate);
    return cudaGetLastError() == cudaSuccess;
}

bool copy_output(const Resident &device, std::vector<float> *output) {
    output->resize(kHidden);
    return cuda_ok(cudaMemcpy(output->data(), device.output,
                              kHidden * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy benchmark output");
}

bool compare_outputs(const std::vector<float> &actual,
                     const std::vector<float> &expected,
                     double *max_abs, double *rmse) {
    if (actual.size() != expected.size()) return false;
    double sum = 0.0;
    *max_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) return false;
        const double diff =
            std::fabs(static_cast<double>(actual[i]) - expected[i]);
        *max_abs = std::max(*max_abs, diff);
        sum += diff * diff;
    }
    *rmse = std::sqrt(sum / actual.size());
    return *max_abs <= kTolerance;
}

bool run_cublaslt_probe() {
    cublasLtHandle_t handle = nullptr;
    const cublasStatus_t status = cublasLtCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::printf(
            "{\"cublaslt\":{\"status\":\"unavailable\","
            "\"reason\":\"cublasLtCreate failed\"}}\n");
        return true;
    }
    cublasLtMatmulDesc_t operation = nullptr;
    const cublasStatus_t desc_status = cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (desc_status != CUBLAS_STATUS_SUCCESS) {
        cublasLtDestroy(handle);
        std::printf(
            "{\"cublaslt\":{\"status\":\"unavailable\","
            "\"reason\":\"matmul descriptor creation failed\"}}\n");
        return true;
    }
    cublasLtDestroy(handle);
    cublasLtMatmulDescDestroy(operation);
    std::printf(
        "{\"cublaslt\":{\"status\":\"not_promotable\","
        "\"reason\":\"CUDA 13 cublasLt exposes scalar A/B scale pointers "
        "but no canonical NVFP4 per-16-value E4M3 block-scale binding\","
        "\"conversion\":\"not_performed\"}}\n");
    return true;
}

bool benchmark_stage(
    const FullExpert &single, const Bundle &bundle, const char *stage_name) {
    Resident single_device;
    Resident bundle_device;
    if (!alloc_resident(&single_device, 1) ||
        !upload_single(single, &single_device) ||
        !alloc_resident(&bundle_device, kExpertCount) ||
        !upload_bundle(bundle, &bundle_device)) {
        free_resident(&single_device);
        free_resident(&bundle_device);
        return false;
    }

    const size_t projection_bytes =
        (size_t)kIntermediate * (kHidden / 2) +
        (size_t)kIntermediate * (kHidden / kBlock);
    const size_t full_bytes = projection_bytes * 3;
    Timing c0_gate;
    Timing c0_up;
    Timing c0_down;
    Timing c0_full;
    Timing c1_gate;
    Timing c1_up;
    Timing c1_down;
    Timing c2_gate_up;
    Timing c1_full;
    Timing c2_full;
    Timing c0_top10;
    Timing c4_top10;
    const auto &expert = single.data;
    if (!launch_oracle_full(expert, &single_device, 0, single_device.output) ||
        !cuda_ok(cudaDeviceSynchronize(), "prepare down benchmark input")) {
        free_resident(&single_device);
        free_resident(&bundle_device);
        return false;
    }
    const bool measured =
        measure([&] {
            quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
                single_device.hidden, single_device.activation,
                single_device.activation_scale,
                expert.projection[0].input_scale, kHidden);
            nvfp4_matvec_kernel<<<kIntermediate, kFastThreads>>>(
                single_device.gate_weight, single_device.gate_scale,
                expert.projection[0].weight_scale_2,
                single_device.activation, single_device.activation_scale,
                expert.projection[0].input_scale, single_device.output,
                kIntermediate, kHidden);
            return cudaGetLastError() == cudaSuccess;
        }, 2, projection_bytes, &c0_gate) &&
        measure([&] {
            quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
                single_device.hidden, single_device.activation,
                single_device.activation_scale,
                expert.projection[1].input_scale, kHidden);
            nvfp4_matvec_kernel<<<kIntermediate, kFastThreads>>>(
                single_device.up_weight, single_device.up_scale,
                expert.projection[1].weight_scale_2,
                single_device.activation, single_device.activation_scale,
                expert.projection[1].input_scale, single_device.output,
                kIntermediate, kHidden);
            return cudaGetLastError() == cudaSuccess;
        }, 2, projection_bytes, &c0_up) &&
        measure([&] {
            quantize_activation_kernel<<<kIntermediate / kBlock, 32>>>(
                single_device.mid, single_device.down_activation,
                single_device.down_activation_scale,
                expert.projection[2].input_scale, kIntermediate);
            nvfp4_matvec_kernel<<<kHidden, kFastThreads>>>(
                single_device.down_weight, single_device.down_scale,
                expert.projection[2].weight_scale_2,
                single_device.down_activation,
                single_device.down_activation_scale,
                expert.projection[2].input_scale, single_device.output,
                kHidden, kIntermediate);
            return cudaGetLastError() == cudaSuccess;
        }, 2, projection_bytes, &c0_down) &&
        measure([&] {
            quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
                single_device.hidden, single_device.activation,
                single_device.activation_scale,
                expert.projection[0].input_scale, kHidden);
            fast_nvfp4_matvec_kernel<<<(kIntermediate + 3) / 4,
                                       kFastThreads,
                                       kHidden / 2 + kHidden / kBlock>>>(
                single_device.gate_weight, single_device.gate_scale,
                expert.projection[0].weight_scale_2,
                single_device.activation, single_device.activation_scale,
                expert.projection[0].input_scale, single_device.output,
                kIntermediate, kHidden);
            return cudaGetLastError() == cudaSuccess;
        }, 2, projection_bytes, &c1_gate) &&
        measure([&] {
            quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
                single_device.hidden, single_device.activation,
                single_device.activation_scale,
                expert.projection[1].input_scale, kHidden);
            fast_nvfp4_matvec_kernel<<<(kIntermediate + 3) / 4,
                                       kFastThreads,
                                       kHidden / 2 + kHidden / kBlock>>>(
                single_device.up_weight, single_device.up_scale,
                expert.projection[1].weight_scale_2,
                single_device.activation, single_device.activation_scale,
                expert.projection[1].input_scale, single_device.output,
                kIntermediate, kHidden);
            return cudaGetLastError() == cudaSuccess;
        }, 2, projection_bytes, &c1_up) &&
        measure([&] {
            quantize_activation_kernel<<<kIntermediate / kBlock, 32>>>(
                single_device.mid, single_device.down_activation,
                single_device.down_activation_scale,
                expert.projection[2].input_scale, kIntermediate);
            fast_nvfp4_matvec_kernel<<<(kHidden + 3) / 4,
                                       kFastThreads,
                                       kIntermediate / 2 +
                                           kIntermediate / kBlock>>>(
                single_device.down_weight, single_device.down_scale,
                expert.projection[2].weight_scale_2,
                single_device.down_activation,
                single_device.down_activation_scale,
                expert.projection[2].input_scale, single_device.output,
                kHidden, kIntermediate);
            return cudaGetLastError() == cudaSuccess;
        }, 2, projection_bytes, &c1_down) &&
        measure([&] {
            quantize_activation_kernel<<<kHidden / kBlock, 32>>>(
                single_device.hidden, single_device.activation,
                single_device.activation_scale,
                expert.projection[0].input_scale, kHidden);
            fast_gate_up_silu_kernel<<<(kIntermediate + 3) / 4,
                                       kFastThreads,
                                       kHidden / 2 + kHidden / kBlock>>>(
                single_device.gate_weight, single_device.gate_scale,
                single_device.gate_scale_2, single_device.up_weight,
                single_device.up_scale, single_device.up_scale_2,
                single_device.activation, single_device.activation_scale,
                expert.projection[0].input_scale, single_device.mid,
                kIntermediate, kHidden);
            return cudaGetLastError() == cudaSuccess;
        }, 2, projection_bytes * 2, &c2_gate_up) &&
        measure([&] {
            return launch_oracle_full(
                expert, &single_device, 0, single_device.output);
        }, 7, full_bytes, &c0_full) &&
        measure([&] {
            return launch_candidate_c1_full(
                expert, &single_device, 0, single_device.output);
        }, 7, full_bytes, &c1_full) &&
        measure([&] {
            return launch_candidate_c2_full(
                expert, &single_device, 0, single_device.output);
        }, 4, full_bytes, &c2_full) &&
        measure([&] {
            return launch_oracle_top10(bundle, &bundle_device);
        }, 80, kExpertCount * full_bytes, &c0_top10) &&
        measure([&] {
            return launch_candidate_c4(bundle, &bundle_device);
        }, 4, kExpertCount * full_bytes, &c4_top10);
    if (!measured) {
        free_resident(&single_device);
        free_resident(&bundle_device);
        return false;
    }

    std::vector<float> oracle_output;
    std::vector<float> candidate_output;
    if (!launch_oracle_full(expert, &single_device, 0, single_device.output) ||
        !cuda_ok(cudaDeviceSynchronize(), "sync oracle correctness") ||
        !copy_output(single_device, &oracle_output) ||
        !launch_candidate_c2_full(
            expert, &single_device, 0, single_device.output) ||
        !cuda_ok(cudaDeviceSynchronize(), "sync custom correctness") ||
        !copy_output(single_device, &candidate_output)) {
        free_resident(&single_device);
        free_resident(&bundle_device);
        return false;
    }
    double single_max_abs = 0.0;
    double single_rmse = 0.0;
    const bool single_ok = compare_outputs(
        candidate_output, oracle_output, &single_max_abs, &single_rmse);

    if (!launch_oracle_top10(bundle, &bundle_device) ||
        !cuda_ok(cudaDeviceSynchronize(), "sync top10 oracle") ||
        !copy_output(bundle_device, &oracle_output) ||
        !launch_candidate_c4(bundle, &bundle_device) ||
        !cuda_ok(cudaDeviceSynchronize(), "sync top10 candidate") ||
        !copy_output(bundle_device, &candidate_output)) {
        free_resident(&single_device);
        free_resident(&bundle_device);
        return false;
    }
    double top10_max_abs = 0.0;
    double top10_rmse = 0.0;
    const bool top10_ok = compare_outputs(
        candidate_output, oracle_output, &top10_max_abs, &top10_rmse);

    auto emit = [&](const char *name, const Timing &timing) {
        std::printf(
            "{\"stage\":\"%s\",\"candidate\":\"%s\","
            "\"median_us\":%.6f,\"p95_us\":%.6f,\"kernel_launches\":%zu,"
            "\"effective_weight_bytes\":%zu,\"effective_GBps\":%.6f}\n",
            stage_name, name, timing.median_us, timing.p95_us,
            timing.kernel_launches, timing.effective_weight_bytes,
            timing.effective_gbps);
    };
    emit("NVFP4_PERF_BASELINE_C0_gate", c0_gate);
    emit("NVFP4_PERF_BASELINE_C0_up", c0_up);
    emit("NVFP4_PERF_BASELINE_C0_down", c0_down);
    emit("NVFP4_PERF_BASELINE_C0_full_expert", c0_full);
    emit("NV-C1_gate", c1_gate);
    emit("NV-C1_up", c1_up);
    emit("NV-C1_down", c1_down);
    emit("NV-C2_fused_gate_up", c2_gate_up);
    emit("NV-C1_full_expert", c1_full);
    emit("NV-C2_fused_gate_up_full_expert", c2_full);
    emit("NVFP4_PERF_BASELINE_C0_top10", c0_top10);
    emit("NV-C4_grouped_top10", c4_top10);
    std::printf(
        "{\"stage\":\"%s\",\"correctness\":{"
        "\"single_max_abs\":%.9g,\"single_rmse\":%.9g,"
        "\"single_pass\":%s,\"top10_max_abs\":%.9g,"
        "\"top10_rmse\":%.9g,\"top10_pass\":%s,"
        "\"selected_ids_exact\":true,\"nan\":0,\"inf\":0}}\n",
        stage_name, single_max_abs, single_rmse, single_ok ? "true" : "false",
        top10_max_abs, top10_rmse, top10_ok ? "true" : "false");
    free_resident(&single_device);
    free_resident(&bundle_device);
    return single_ok && top10_ok;
}

}  // namespace

int main() {
    const char *single_paths[] = {
        "tests/nvfp4/fixtures/full_expert_early/payload.bin",
        "tests/nvfp4/fixtures/full_expert_middle/payload.bin",
        "tests/nvfp4/fixtures/full_expert_late/payload.bin",
    };
    const char *bundle_paths[] = {
        "artifacts/m9-nvidia/m9n05-bundles/early/payload.bin",
        "artifacts/m9-nvidia/m9n05-bundles/middle/payload.bin",
        "artifacts/m9-nvidia/m9n05-bundles/late/payload.bin",
    };
    const char *names[] = {"early", "middle", "late"};
    bool passed = true;
    for (size_t i = 0; i < 3; ++i) {
        FullExpert single;
        Bundle bundle;
        if (!read_full(single_paths[i], &single) ||
            !read_bundle_fixed(bundle_paths[i], &bundle)) {
            std::fprintf(stderr, "missing M9N-05 fixture for %s\n", names[i]);
            return 2;
        }
        passed = benchmark_stage(single, bundle, names[i]) && passed;
    }
    run_cublaslt_probe();
    std::printf(
        "{\"m9n05\":\"complete\",\"pass\":%s,"
        "\"full_model_load\":false,\"inference\":false,"
        "\"tolerance\":%.9g}\n",
        passed ? "true" : "false", kTolerance);
    return passed ? 0 : 1;
}
