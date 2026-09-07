#include "q38_residency_plan.h"

#include <cuda_runtime.h>

#include <inttypes.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <string>
#include <vector>

struct RunStats {
    double wall_ms = 0.0;
    uint64_t read_calls = 0;
    uint64_t memcpy_async_calls = 0;
    uint64_t sync_count = 0;
    uint64_t allocations = 0;
    uint64_t read_bytes = 0;
    uint64_t h2d_bytes = 0;
    uint64_t total_transfer_bytes = 0;
    uint64_t transfer_count = 0;
    uint64_t p95_transfer_bytes = 0;
};

static bool is_ple(const q38_tensor *tensor, void *) {
    return tensor && tensor->name.ptr &&
           std::strstr(tensor->name.ptr,
                       ".ple.ple_embedding.ngram_embedding.shard_") != nullptr;
}

static double now_ms() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint64_t hash_bytes(const unsigned char *data, size_t bytes) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i)
        hash = (hash ^ data[i]) * UINT64_C(1099511628211);
    return hash;
}

static bool check_cuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    return false;
}

static void record_transfer(RunStats *stats, uint64_t bytes) {
    stats->total_transfer_bytes += bytes;
    stats->transfer_count++;
}

static bool verify_destinations(const q38_gguf *model,
                                const q38_residency_plan *plan,
                                const std::vector<void *> &destinations) {
    size_t largest = 0;
    for (size_t i = 0; i < plan->entry_count; ++i)
        largest = std::max(largest, (size_t)plan->entries[i].bytes);
    std::vector<unsigned char> copy(largest);
    for (size_t i = 0; i < plan->entry_count; ++i) {
        const auto &entry = plan->entries[i];
        if (!check_cuda(cudaMemcpy(copy.data(), destinations[i],
                                   (size_t)entry.bytes,
                                   cudaMemcpyDeviceToHost),
                        "verification D2H"))
            return false;
        const uint64_t source_hash = hash_bytes(
            model->map + entry.file_offset, (size_t)entry.bytes);
        const uint64_t destination_hash =
            hash_bytes(copy.data(), (size_t)entry.bytes);
        if (source_hash != destination_hash) {
            std::fprintf(stderr, "verification hash mismatch at entry %zu\n", i);
            return false;
        }
    }
    return true;
}

static bool run_per_tensor(const q38_gguf *model,
                           const q38_residency_plan *plan,
                           RunStats *stats) {
    std::vector<void *> destinations(plan->entry_count, nullptr);
    const double started = now_ms();
    for (size_t i = 0; i < plan->entry_count; ++i) {
        const auto &entry = plan->entries[i];
        if (!check_cuda(cudaMalloc(&destinations[i], (size_t)entry.bytes),
                        "per-tensor cudaMalloc"))
            return false;
        stats->allocations++;
        if (!check_cuda(cudaMemcpyAsync(
                destinations[i], model->map + entry.file_offset,
                (size_t)entry.bytes, cudaMemcpyHostToDevice, 0),
                        "per-tensor cudaMemcpyAsync"))
            return false;
        stats->memcpy_async_calls++;
        stats->read_bytes += entry.bytes;
        stats->h2d_bytes += entry.bytes;
        record_transfer(stats, entry.bytes);
        if (!check_cuda(cudaStreamSynchronize(0),
                        "per-tensor cudaStreamSynchronize"))
            return false;
        stats->sync_count++;
    }
    stats->wall_ms = now_ms() - started;
    stats->p95_transfer_bytes = plan->entry_count ? plan->entries[
        (plan->entry_count * 95u) / 100u].bytes : 0;
    if (!verify_destinations(model, plan, destinations)) return false;
    for (void *destination : destinations) cudaFree(destination);
    return true;
}

static bool run_coalesced(const q38_gguf *model,
                          const q38_residency_plan *plan,
                          RunStats *stats) {
    std::vector<void *> destinations(plan->entry_count, nullptr);
    size_t largest_span = 0;
    std::vector<uint64_t> transfer_sizes;
    for (size_t i = 0; i < plan->span_count; ++i) {
        largest_span = std::max(largest_span, (size_t)plan->spans[i].bytes);
        transfer_sizes.push_back(plan->spans[i].bytes);
    }
    void *stage = nullptr;
    void *transfer = nullptr;
    if (!check_cuda(cudaMallocHost(&stage, largest_span),
                    "coalesced cudaMallocHost") ||
        !check_cuda(cudaMalloc(&transfer, largest_span),
                    "coalesced transfer cudaMalloc"))
        return false;
    stats->allocations += 2;
    for (size_t i = 0; i < plan->entry_count; ++i) {
        const auto &entry = plan->entries[i];
        if (!check_cuda(cudaMalloc(&destinations[i], (size_t)entry.bytes),
                        "coalesced tensor cudaMalloc"))
            return false;
        stats->allocations++;
    }
    const double started = now_ms();
    for (size_t s = 0; s < plan->span_count; ++s) {
        const auto &span = plan->spans[s];
        std::memcpy(stage, model->map + span.file_offset, (size_t)span.bytes);
        stats->read_calls++;
        stats->read_bytes += span.bytes;
        if (!check_cuda(cudaMemcpyAsync(
                transfer, stage, (size_t)span.bytes,
                cudaMemcpyHostToDevice, 0),
                        "coalesced H2D"))
            return false;
        stats->memcpy_async_calls++;
        stats->h2d_bytes += span.bytes;
        record_transfer(stats, span.bytes);
        for (size_t j = 0; j < span.entry_count; ++j) {
            const size_t index = span.first_entry + j;
            const auto &entry = plan->entries[index];
            const uint64_t relative = entry.file_offset - span.file_offset;
            if (!check_cuda(cudaMemcpyAsync(
                    destinations[index],
                    (const char *)transfer + relative, (size_t)entry.bytes,
                    cudaMemcpyDeviceToDevice, 0),
                            "coalesced D2D"))
                return false;
            stats->memcpy_async_calls++;
            record_transfer(stats, entry.bytes);
        }
    }
    if (!check_cuda(cudaStreamSynchronize(0),
                    "coalesced final cudaStreamSynchronize"))
        return false;
    stats->sync_count = 1;
    stats->wall_ms = now_ms() - started;
    if (!verify_destinations(model, plan, destinations)) return false;
    std::sort(transfer_sizes.begin(), transfer_sizes.end());
    stats->p95_transfer_bytes =
        transfer_sizes.empty()
            ? 0
            : transfer_sizes[(transfer_sizes.size() * 95u) / 100u];
    for (void *destination : destinations) cudaFree(destination);
    cudaFree(transfer);
    cudaFreeHost(stage);
    return true;
}

int main() {
    constexpr size_t kFileBytes = 1024ull * 1024ull * 1024ull + 4096;
    constexpr size_t kTensorBytes = 2ull * 1024ull * 1024ull;
    constexpr size_t kTensorCount = 512;
    char path[] = "/tmp/q38-startup1-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, (off_t)kFileBytes) != 0) return 1;
    void *mapping = mmap(nullptr, kFileBytes, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) return 1;
    auto *bytes = (unsigned char *)mapping;
    for (size_t offset = 0; offset < kFileBytes; offset += 4096)
        bytes[offset] = (unsigned char)((offset / 4096u) * 131u);

    std::vector<std::string> names;
    names.reserve(kTensorCount);
    std::vector<q38_tensor> tensors(kTensorCount);
    for (size_t i = 0; i < kTensorCount; ++i) {
        names.emplace_back("blk.tensor_" + std::to_string(i));
        tensors[i].name = {names.back().data(), names.back().size()};
        tensors[i].abs_offset = 4096 + i * kTensorBytes;
        tensors[i].bytes = kTensorBytes;
    }
    names[kTensorCount / 2] =
        "blk.ple.ple_embedding.ngram_embedding.shard_0";
    tensors[kTensorCount / 2].name =
        {names[kTensorCount / 2].data(), names[kTensorCount / 2].size()};
    q38_gguf model{};
    model.map = (const uint8_t *)mapping;
    model.size = kFileBytes;
    model.n_tensors = tensors.size();
    model.tensors = tensors.data();

    q38_residency_plan plan;
    q38_residency_plan_init(&plan);
    char error[256] = {};
    if (!q38_residency_plan_build(
            &model, is_ple, nullptr, 64u * 1024u,
            256u * 1024u * 1024u, &plan, error, sizeof(error))) {
        std::fprintf(stderr, "planner: %s\n", error);
        return 1;
    }
    RunStats current, c1;
    const bool ok = run_per_tensor(&model, &plan, &current) &&
                    run_coalesced(&model, &plan, &c1);
    std::printf(
        "{\"fixture_file_bytes\":%zu,\"resident_bytes\":%" PRIu64
        ",\"ple_bytes\":%" PRIu64
        ",\"ple_upload_bytes\":0"
        ",\"entries\":%zu,\"spans\":%zu,\"correctness\":\"%s\""
        ",\"speedup\":%.3f,"
        "\"current\":{\"wall_ms\":%.3f,\"read_calls\":%" PRIu64
        ",\"read_bytes\":%" PRIu64 ",\"effective_read_gbps\":%.3f"
        ",\"h2d_bytes\":%" PRIu64 ",\"effective_h2d_gbps\":%.3f"
        ",\"memcpy_async_calls\":%" PRIu64 ",\"stream_syncs\":%" PRIu64
        ",\"allocations\":%" PRIu64 ",\"avg_transfer_bytes\":%.1f"
        ",\"p95_transfer_bytes\":%" PRIu64 "},"
        "\"c1\":{\"wall_ms\":%.3f,\"read_calls\":%" PRIu64
        ",\"read_bytes\":%" PRIu64 ",\"effective_read_gbps\":%.3f"
        ",\"h2d_bytes\":%" PRIu64 ",\"effective_h2d_gbps\":%.3f"
        ",\"memcpy_async_calls\":%" PRIu64 ",\"stream_syncs\":%" PRIu64
        ",\"allocations\":%" PRIu64 ",\"avg_transfer_bytes\":%.1f,"
        "\"p95_transfer_bytes\":%" PRIu64 "}}\n",
        kFileBytes, plan.resident_bytes, plan.excluded_ple_bytes,
        plan.entry_count, plan.span_count, ok ? "GREEN" : "RED",
        current.wall_ms > 0.0 && c1.wall_ms > 0.0
            ? current.wall_ms / c1.wall_ms : 0.0,
        current.wall_ms, current.read_calls,
        current.read_bytes, current.wall_ms > 0.0
            ? (double)current.read_bytes / (current.wall_ms * 1.0e6)
            : 0.0,
        current.h2d_bytes, current.wall_ms > 0.0
            ? (double)current.h2d_bytes / (current.wall_ms * 1.0e6)
            : 0.0,
        current.memcpy_async_calls, current.sync_count, current.allocations,
        current.transfer_count
            ? (double)current.total_transfer_bytes / current.transfer_count
            : 0.0,
        current.p95_transfer_bytes,
        c1.wall_ms, c1.read_calls, c1.read_bytes, c1.wall_ms > 0.0
            ? (double)c1.read_bytes / (c1.wall_ms * 1.0e6)
            : 0.0,
        c1.h2d_bytes, c1.wall_ms > 0.0
            ? (double)c1.h2d_bytes / (c1.wall_ms * 1.0e6)
            : 0.0,
        c1.memcpy_async_calls, c1.sync_count,
        c1.allocations,
        c1.transfer_count
            ? (double)c1.total_transfer_bytes / c1.transfer_count
            : 0.0,
        c1.p95_transfer_bytes);
    q38_residency_plan_destroy(&plan);
    munmap(mapping, kFileBytes);
    close(fd);
    unlink(path);
    return ok ? 0 : 1;
}
