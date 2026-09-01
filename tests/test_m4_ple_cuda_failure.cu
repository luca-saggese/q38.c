#include "../q38_ple_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

static int child_fault(void) {
    uint32_t *ids = nullptr;
    if (cudaMalloc(&ids, sizeof(*ids)) != cudaSuccess) return 2;
    const uint32_t id = 0;
    if (cudaMemcpy(ids, &id, sizeof(id), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        cudaFree(ids);
        return 3;
    }
    float *rows = nullptr;
    if (cudaMalloc(&rows, 256 * sizeof(float)) != cudaSuccess) {
        cudaFree(ids);
        return 4;
    }
    char error[128];
    /* The lookup must not return after an illegal device access. */
    (void)q38_ple_cuda_lookup_rows(
        Q38_QUANT_Q2_K, reinterpret_cast<const void *>(uintptr_t(1)), 1,
        256, ids, 1, rows, 0, error, sizeof(error));
    cudaFree(rows);
    cudaFree(ids);
    return 5;
}

int main(void) {
    const pid_t pid = fork();
    if (pid < 0) {
        std::perror("fork");
        return 1;
    }
    if (pid == 0) _exit(child_fault());
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        std::perror("waitpid");
        return 1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) == 0) {
        std::fprintf(stderr, "faulting lookup did not fail fatally\n");
        return 1;
    }
    std::puts("test_m4_ple_cuda_failure: CUDA lookup faults terminate child");
    return 0;
}
