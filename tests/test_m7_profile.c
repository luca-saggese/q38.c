#include "q38_profile.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    q38_profile profile;
    char json[4096];
    char error[64] = {0};
    q38_profile_init(&profile);
    q38_profile_record_launch(&profile, Q38_PROFILE_GDN);
    q38_profile_record_sync(&profile, Q38_PROFILE_QSA);
    q38_profile_record_allocation(&profile, Q38_PROFILE_MOE, 4096);
    q38_forward_stage_usage stage = {"moe_router", 3, 0, 0, 0, 1.5};
    if (!q38_profile_stage_trace(&stage, &profile, error, sizeof(error)) ||
        !q38_profile_json(&profile, json, sizeof(json)) ||
        !strstr(json, "\"name\":\"gdn\"") ||
        !strstr(json, "\"name\":\"lm_head\"") ||
        profile.subsystem[Q38_PROFILE_MOE].allocation_bytes != 4096 ||
        profile.subsystem[Q38_PROFILE_MOE].kernel_launches != 3) {
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
