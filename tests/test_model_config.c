#include "q38_model_config.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char error[256];
    q38_model_config config = *q38_model_config_default();
    if (!q38_model_config_validate(&config, error, sizeof(error))) {
        fprintf(stderr, "default config rejected: %s\n", error);
        return 1;
    }
    config.layer_types[3] = Q38_LAYER_LINEAR_ATTENTION;
    if (q38_model_config_validate(&config, error, sizeof(error)) ||
        strstr(error, "layer_types[3]") == NULL) {
        fprintf(stderr, "layer pattern mutation was not rejected\n");
        return 1;
    }
    config = *q38_model_config_default();
    config.num_experts = 511;
    if (q38_model_config_validate(&config, error, sizeof(error)) ||
        strstr(error, "num_experts") == NULL) {
        fprintf(stderr, "scalar mutation was not rejected\n");
        return 1;
    }
    puts("test_model_config: all tests passed");
    return 0;
}
