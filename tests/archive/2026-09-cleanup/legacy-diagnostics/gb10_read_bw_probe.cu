#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

__global__ static void fill_probe(float4 *data, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < count; i += stride)
        data[i] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
}

__global__ static void read_probe(const float4 *data, float *sink,
                                  size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < count; i += stride) {
        float4 value = data[i];
        acc += value.x + value.y + value.z + value.w;
    }
    if (acc == -1234567.0f) *sink = acc;
}

int main(int argc, char **argv) {
    const size_t gib = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 2;
    const size_t bytes = gib * (size_t)1024 * 1024 * 1024;
    const size_t count = bytes / sizeof(float4);
    float4 *data = NULL;
    float *sink = NULL;
    cudaStream_t stream = NULL;
    cudaEvent_t start = NULL, stop = NULL;
    if (cudaMalloc(&data, bytes) != cudaSuccess ||
        cudaMalloc(&sink, sizeof(float)) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess ||
        cudaEventCreate(&start) != cudaSuccess ||
        cudaEventCreate(&stop) != cudaSuccess)
        return 2;
    const unsigned threads = 256;
    const unsigned blocks = 4096;
    fill_probe<<<blocks, threads, 0, stream>>>(data, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return 3;
    read_probe<<<blocks, threads, 0, stream>>>(data, sink, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(start, stream) != cudaSuccess)
        return 3;
    read_probe<<<blocks, threads, 0, stream>>>(data, sink, count);
    if (cudaGetLastError() != cudaSuccess ||
        cudaEventRecord(stop, stream) != cudaSuccess ||
        cudaEventSynchronize(stop) != cudaSuccess)
        return 3;
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess || ms <= 0.0f)
        return 3;
    printf("{\"format\":\"q38-p8o4-gb10-read-ceiling-v1\","
           "\"allocation_bytes\":%zu,\"kernel_ms\":%.9g,"
           "\"effective_GBps\":%.9g,\"blocks\":%u,\"threads\":%u}\n",
           bytes, ms, (double)bytes / ((double)ms * 1e6), blocks, threads);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(sink);
    cudaFree(data);
    return 0;
}
