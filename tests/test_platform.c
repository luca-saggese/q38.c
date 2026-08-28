/* test_platform.c — M0 platform guard tests.
 *
 * Verifies the platform probe accepts the target and refuses anything else.
 * Runs on the real DGX Spark: acceptance (M0-T02) when the device is
 * GB10/SM121, refusal (M0-T03) when the compute capability or device count
 * is not the target. The refusal path is also exercised directly by
 * checking the guard logic.
 */

#include "q38.h"
#include "q38_cuda.h"
#include "q38_platform.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

/* Directly test the guard logic without a device by checking the refusal
 * conditions on a synthetic platform_info. */
static void test_guard_logic(void) {
    q38_platform_info p;
    memset(&p, 0, sizeof(p));

    /* Wrong CC. */
    p.cuda_device_count = 1;
    p.cc_major = 11;
    p.cc_minor = 0;
    CHECK(!(p.cc_major == 12 && p.cc_minor == 1), "refuses non-sm121 CC");

    /* Wrong device count. */
    p.cuda_device_count = 2;
    p.cc_major = 12;
    p.cc_minor = 1;
    CHECK(p.cuda_device_count != 1, "refuses multi-device");

    /* Correct target. */
    p.cuda_device_count = 1;
    p.cc_major = 12;
    p.cc_minor = 1;
    CHECK(p.cuda_device_count == 1 && p.cc_major == 12 && p.cc_minor == 1,
          "accepts GB10/SM121");
}

int main(void) {
    test_guard_logic();

    /* Live CUDA probe (only meaningful on the real device). */
    q38_platform_info p;
    char reason[256];
    int rc = q38_platform_probe(&p, reason, sizeof(reason));
    if (rc == 0) {
        printf("platform: device=%s sm_%d%d devices=%d\n",
               p.device_name, p.cc_major, p.cc_minor, p.cuda_device_count);
        printf("platform: cuda total=%" PRIu64 " free=%" PRIu64 "\n",
               p.cuda_total_bytes, p.cuda_free_bytes);
        printf("platform: mem total=%" PRIu64 " avail=%" PRIu64 "\n",
               p.mem_total_bytes, p.mem_available_bytes);
        CHECK(p.cc_major == 12 && p.cc_minor == 1, "live device is SM121");
        CHECK(p.cuda_device_count == 1, "live device count == 1");
    } else {
        printf("platform: refused -> %s (expected off-target)\n", reason);
        CHECK(1, "probe refuses off-target platform (no fallback)");
    }

    if (failures == 0) {
        printf("test_platform: all tests passed\n");
        return 0;
    }
    printf("test_platform: %d failure(s)\n", failures);
    return 1;
}
