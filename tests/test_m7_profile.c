#include "q38_profile.h"
#include "q38_forward_cuda.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    q38_profile profile;
    char json[8192];
    char error[64] = {0};
    q38_profile_init(&profile);
    q38_profile_record_launch(&profile, Q38_PROFILE_GDN);
    q38_profile_record_sync(&profile, Q38_PROFILE_QSA);
    q38_profile_record_allocation(&profile, Q38_PROFILE_MOE, 4096);
    q38_profile_set_token_count(&profile, 2);
    q38_forward_cuda_telemetry telemetry = {
        .subsystem = "moe", .layer = 2, .logical_stage = "moe_router",
        .tensor_name = "router.weight", .rows = 4, .cols = 8,
        .bytes = 128, .weight_bytes = 128, .activation_read_bytes = 32,
        .activation_write_bytes = 16, .d2h_bytes = 16, .kernel_ms = 2.0f,
        .host_syncs = 1
    };
    q38_profile_record_cuda_telemetry(&profile, &telemetry);
    q38_forward_stage_usage stage = {"moe_router", 3, 0, 0, 0, 1.5};
    if (!q38_profile_stage_trace(&stage, &profile, error, sizeof(error)) ||
        !q38_profile_json(&profile, json, sizeof(json)) ||
        !strstr(json, "\"name\":\"gdn\"") ||
        !strstr(json, "\"name\":\"lm_head\"") ||
        profile.subsystem[Q38_PROFILE_MOE].allocation_bytes != 4096 ||
        profile.subsystem[Q38_PROFILE_MOE].kernel_launches != 3 ||
        profile.weight_bytes_per_token != 64.0 ||
        profile.host_syncs_per_token != 0.5 ||
        profile.d2h_bytes_per_token != 8.0) {
        fprintf(stderr, "test_m7_profile: host profile failed\n");
        q38_profile_destroy(&profile);
        return 1;
    }
    if (q38_profile_cuda_init(&profile) == 0) {
        if (q38_profile_cuda_begin(&profile, Q38_PROFILE_GDN, NULL) != 0 ||
            q38_profile_cuda_end(&profile, Q38_PROFILE_GDN, NULL) != 0) {
            fprintf(stderr, "test_m7_profile: CUDA timing failed\n");
            q38_profile_destroy(&profile);
            return 1;
        }
    }
    q38_profile_destroy(&profile);
    puts("M7 profile telemetry passed");
    return 0;
}
