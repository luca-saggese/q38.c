#include "q38_ple_cuda.h"

#include <cuda_runtime.h>

#include <stdio.h>

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
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        if (error && error_len > 0)
            snprintf(error, error_len, "CUDA PLE lookup launch failed: %s",
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}
