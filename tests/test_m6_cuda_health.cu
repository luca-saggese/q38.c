#include <cuda_runtime.h>

#include <cstdio>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) return 1;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 1;
    if (prop.major < 12) return 1;
    std::printf("test_m6_cuda_health: device=%s cc=%d.%d\n",
                prop.name, prop.major, prop.minor);
    return 0;
}
