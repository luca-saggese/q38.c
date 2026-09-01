#include "../q38_ple_stage.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    q38_ple_stage_pool pool;
    char error[128];
    if (!q38_ple_stage_pool_init(&pool, 2, 1 << 20, error, sizeof(error)))
        return 1;
    uint32_t a = 0, b = 0;
    if (!q38_ple_stage_pool_acquire(&pool, &a, error, sizeof(error)) ||
        !q38_ple_stage_pool_acquire(&pool, &b, error, sizeof(error)) ||
        a == b || !q38_ple_stage_pool_data(&pool, a) ||
        !q38_ple_stage_pool_data(&pool, b)) return 1;
    const size_t bytes = 4096;
    std::memset(q38_ple_stage_pool_data(&pool, a), 0x5a, bytes);
    uint8_t *device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess ||
        !q38_ple_stage_pool_submit(&pool, a, device, bytes, 0, error,
                                    sizeof(error)) ||
        cudaDeviceSynchronize() != cudaSuccess) return 1;
    std::vector<uint8_t> host(bytes);
    cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < bytes; ++i) if (host[i] != 0x5a) return 1;
    uint32_t reused = 99;
    if (!q38_ple_stage_pool_acquire(&pool, &reused, error, sizeof(error)) ||
        reused != a || pool.wait_count != 0 ||
        pool.high_watermark != bytes || pool.h2d_bytes != bytes) return 1;
    cudaFree(device);
    q38_ple_stage_pool_destroy(&pool);
    std::puts("test_m5_ter_stage: persistent bounded pinned staging reuse passed");
    return 0;
}
