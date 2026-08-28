/* test_memory.c — M0 memory telemetry tests.
 *
 * Verifies the snapshot schema keys and the monotonic peak counter without
 * requiring a CUDA device, so it can run on any host.
 */

#include "q38_memory.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

int main(void) {
    q38_memory_tracker t;
    q38_memory_tracker_init(&t);

    /* Peak is monotonic: alloc 100 -> peak 100; alloc 50 -> peak 150;
     * free 50 -> live 100, peak stays 150. */
    q38_memory_track_alloc(&t, 100);
    CHECK(t.peak_internal_bytes == 100, "peak after first alloc");

    q38_memory_track_alloc(&t, 50);
    CHECK(t.peak_internal_bytes == 150, "peak monotonic up");

    q38_memory_track_free(&t, 50);
    CHECK(t.internal_allocated_bytes == 100, "live bytes after free");
    CHECK(t.peak_internal_bytes == 150, "peak unchanged after free");

    /* Over-free is clamped to zero, not underflow. */
    q38_memory_track_free(&t, 1000);
    CHECK(t.internal_allocated_bytes == 0, "over-free clamped to zero");

    /* Snapshot schema. */
    q38_memory_snapshot snap;
    q38_memory_capture(&t, "gguf_mapped", 1234, 1234, 0, &snap);
    CHECK(strcmp(snap.phase, "gguf_mapped") == 0, "snapshot phase set");
    CHECK(snap.model_file_bytes == 1234, "snapshot model_file_bytes");
    CHECK(snap.model_mapped_bytes == 1234, "snapshot model_mapped_bytes");
    CHECK(snap.peak_internal_bytes == 150, "snapshot carries peak");

    /* JSON serialization has deterministic keys. */
    char buf[512];
    q38_memory_snapshot_json(&snap, buf, sizeof(buf));
    CHECK(strstr(buf, "\"phase\"") != NULL, "json has phase key");
    CHECK(strstr(buf, "\"rss_bytes\"") != NULL, "json has rss_bytes key");
    CHECK(strstr(buf, "\"mem_available_bytes\"") != NULL, "json has mem_available_bytes key");
    CHECK(strstr(buf, "\"cuda_free_bytes\"") != NULL, "json has cuda_free_bytes key");
    CHECK(strstr(buf, "\"cuda_total_bytes\"") != NULL, "json has cuda_total_bytes key");
    CHECK(strstr(buf, "\"model_file_bytes\"") != NULL, "json has model_file_bytes key");
    CHECK(strstr(buf, "\"model_mapped_bytes\"") != NULL, "json has model_mapped_bytes key");
    CHECK(strstr(buf, "\"cuda_allocated_bytes\"") != NULL, "json has cuda_allocated_bytes key");
    CHECK(strstr(buf, "\"peak_internal_bytes\"") != NULL, "json has peak_internal_bytes key");

    if (failures == 0) {
        printf("test_memory: all tests passed\n");
        return 0;
    }
    printf("test_memory: %d failure(s)\n", failures);
    return 1;
}
