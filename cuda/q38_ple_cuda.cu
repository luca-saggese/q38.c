#include "q38_ple_cuda.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__device__ static float half_to_float_device(uint16_t bits) {
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t fraction = bits & 0x3ffu;
    uint32_t value;
    if (!exponent) {
        if (!fraction) value = sign;
        else {
            exponent = 1;
            while (!(fraction & 0x400u)) {
                fraction <<= 1;
                exponent--;
            }
            fraction &= 0x3ffu;
            value = sign | ((exponent + 112u) << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (fraction << 13);
    } else {
        value = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    return __uint_as_float(value);
}

__device__ static float q2_value(const q38_q2_k_block *block,
                                 unsigned element) {
    const unsigned half = element / 128;
    const unsigned within = element % 128;
    const unsigned group = within / 16;
    const unsigned l = within % 16;
    const unsigned shift = (group / 2) * 2;
    const unsigned scale_index = half * 8 + group;
    const uint8_t scale = block->scales[scale_index];
    const unsigned qindex = half * 32 + (group % 2) * 16 + l;
    const unsigned quant = (block->qs[qindex] >> shift) & 3u;
    return half_to_float_device(block->d) * (scale & 0xfu) * quant -
           half_to_float_device(block->dmin) * (scale >> 4);
}

__device__ static void q4_scale_min(unsigned index, const uint8_t *scales,
                                    uint8_t *scale, uint8_t *minimum) {
    if (index < 4) {
        *scale = scales[index] & 63u;
        *minimum = scales[index + 4] & 63u;
    } else {
        *scale = (scales[index + 4] & 0xfu) |
                 ((scales[index - 4] >> 6) << 4);
        *minimum = (scales[index + 4] >> 4) |
                   ((scales[index] >> 6) << 4);
    }
}

__device__ static float q4_value(const q38_q4_k_block *block,
                                 unsigned element) {
    const unsigned group = element / 32;
    const unsigned l = element % 32;
    uint8_t scale, minimum;
    q4_scale_min(group, block->scales, &scale, &minimum);
    const uint8_t packed = block->qs[(group / 2) * 32 + l];
    const unsigned quant = group % 2 ? packed >> 4 : packed & 0xfu;
    return half_to_float_device(block->d) * scale * quant -
           half_to_float_device(block->dmin) * minimum;
}

__global__ static void lookup_rows_kernel(uint32_t qtype,
                                          const void *table,
                                          uint64_t table_rows,
                                          uint32_t row_width,
                                          const uint32_t *ids,
                                          size_t id_count, float *out) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = id_count * (size_t)row_width;
    if (index >= elements) return;

    const size_t selected = index / row_width;
    const unsigned element = (unsigned)(index % row_width);
    const uint32_t row = ids[selected];
    if ((uint64_t)row >= table_rows) {
        out[index] = __int_as_float(0x7fc00000);
        return;
    }

    const size_t block_count = row_width / Q38_QUANT_QK_K;
    if (qtype == Q38_QUANT_Q2_K) {
        const size_t block = (size_t)row * block_count +
                             element / Q38_QUANT_QK_K;
        out[index] = q2_value((const q38_q2_k_block *)table + block,
                               element % Q38_QUANT_QK_K);
    } else {
        const size_t block = (size_t)row * block_count +
                             element / Q38_QUANT_QK_K;
        out[index] = q4_value((const q38_q4_k_block *)table + block,
                               element % Q38_QUANT_QK_K);
    }
}

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len > 0) snprintf(error, error_len, "%s", message);
    return false;
}

static void cuda_fatal(cudaError_t status, const char *phase) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "q38: fatal CUDA error during %s: %s\n",
            phase, cudaGetErrorString(status));
    fflush(stderr);
    _exit(134);
}

__global__ static void expand_rows_kernel(const uint32_t *map,
                                          size_t id_count,
                                          uint32_t row_width,
                                          const float *unique_rows,
                                          float *out) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t elements = id_count * (size_t)row_width;
    if (index >= elements) return;
    const size_t selected = index / row_width;
    const size_t element = index % row_width;
    const size_t unique = (size_t)map[selected];
    out[index] = unique_rows[unique * (size_t)row_width + element];
}

extern "C" bool q38_ple_cuda_lookup_rows(uint32_t qtype,
                                         const void *device_table,
                                         uint64_t table_rows,
                                         uint32_t row_width,
                                         const uint32_t *device_ids,
                                         size_t id_count, float *device_rows,
                                         cudaStream_t stream, char *error,
                                         size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (qtype != Q38_QUANT_Q2_K && qtype != Q38_QUANT_Q4_K) {
        return fail(error, error_len, "unsupported CUDA PLE row type");
    }
    if (!device_table || !table_rows || !row_width ||
        row_width % Q38_QUANT_QK_K != 0 || !device_ids || !id_count ||
        !device_rows) {
        return fail(error, error_len, "invalid CUDA PLE lookup arguments");
    }
    if (id_count > SIZE_MAX / row_width) {
        return fail(error, error_len, "CUDA PLE output size overflows");
    }
    const size_t elements = id_count * (size_t)row_width;
    lookup_rows_kernel<<<(unsigned)((elements + 255) / 256), 256, 0, stream>>>(
        qtype, device_table, table_rows, row_width, device_ids, id_count,
        device_rows);
    cuda_fatal(cudaGetLastError(), "PLE row lookup launch");
    cuda_fatal(cudaStreamSynchronize(stream), "PLE row lookup execution");
    return true;
}

extern "C" bool q38_ple_cuda_lookup_rows_dedup(
    uint32_t qtype, const void *device_table, uint64_t table_rows,
    uint32_t row_width, const uint32_t *host_ids, size_t id_count,
    float *device_rows, cudaStream_t stream, q38_ple_cuda_lookup_stats *stats,
    char *error, size_t error_len) {
    if (error && error_len > 0) error[0] = '\0';
    if (stats) {
        stats->input_count = 0;
        stats->unique_count = 0;
    }
    if (qtype != Q38_QUANT_Q2_K && qtype != Q38_QUANT_Q4_K) {
        return fail(error, error_len, "unsupported CUDA PLE row type");
    }
    if (!device_table || table_rows == 0 || row_width == 0 ||
        row_width % Q38_QUANT_QK_K != 0 || !host_ids || id_count == 0 ||
        id_count > UINT32_MAX || !device_rows) {
        return fail(error, error_len, "invalid CUDA PLE dedup arguments");
    }
    if (id_count > SIZE_MAX / row_width) {
        return fail(error, error_len, "PLE output size overflows");
    }
    if (id_count > SIZE_MAX / sizeof(*host_ids)) {
        return fail(error, error_len, "invalid host PLE ID batch");
    }
    const size_t output_elements = id_count * (size_t)row_width;

    uint32_t *unique_ids =
        (uint32_t *)malloc(id_count * sizeof(*unique_ids));
    uint32_t *map = (uint32_t *)malloc(id_count * sizeof(*map));
    if (!unique_ids || !map) {
        free(unique_ids);
        free(map);
        return fail(error, error_len, "PLE ID deduplication allocation failed");
    }
    size_t unique_count = 0;
    for (size_t i = 0; i < id_count; ++i) {
        size_t found = unique_count;
        for (size_t j = 0; j < unique_count; ++j) {
            if (unique_ids[j] == host_ids[i]) {
                found = j;
                break;
            }
        }
        if (found == unique_count) unique_ids[unique_count++] = host_ids[i];
        map[i] = (uint32_t)found;
    }
    if (unique_count > SIZE_MAX / row_width ||
        unique_count * (size_t)row_width > SIZE_MAX / sizeof(float)) {
        free(map);
        free(unique_ids);
        return fail(error, error_len, "PLE unique output size overflows");
    }

    uint32_t *device_ids = nullptr;
    uint32_t *device_map = nullptr;
    float *device_unique_rows = nullptr;
    bool ok = false;
    if (cudaMalloc(&device_ids, unique_count * sizeof(*device_ids)) !=
            cudaSuccess ||
        cudaMalloc(&device_map, id_count * sizeof(*device_map)) !=
            cudaSuccess ||
        cudaMalloc(&device_unique_rows,
                   unique_count * (size_t)row_width * sizeof(float)) !=
            cudaSuccess) {
        fail(error, error_len, "PLE deduplication CUDA allocation failed");
        goto cleanup;
    }
    if (cudaMemcpyAsync(device_ids, unique_ids,
                        unique_count * sizeof(*unique_ids),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(device_map, map, id_count * sizeof(*map),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        fail(error, error_len, "PLE deduplication CUDA copy failed");
        goto cleanup;
    }
    if (!q38_ple_cuda_lookup_rows(
            qtype, device_table, table_rows, row_width, device_ids,
            unique_count, device_unique_rows, stream, error, error_len)) {
        goto cleanup;
    }
    expand_rows_kernel<<<(unsigned)((output_elements + 255) / 256), 256, 0,
                         stream>>>(
        device_map, id_count, row_width, device_unique_rows, device_rows);
    cuda_fatal(cudaGetLastError(), "PLE deduplication expansion launch");
    cuda_fatal(cudaStreamSynchronize(stream), "PLE deduplication expansion");
    ok = true;
    if (stats) {
        stats->input_count = id_count;
        stats->unique_count = unique_count;
    }

cleanup:
    if (device_unique_rows) cudaFree(device_unique_rows);
    if (device_map) cudaFree(device_map);
    if (device_ids) cudaFree(device_ids);
    free(map);
    free(unique_ids);
    return ok;
}
