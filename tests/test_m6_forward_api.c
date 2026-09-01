#include "q38_decode.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char error[128] = "stale";
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    if (q38_forward_full(NULL, NULL, NULL, NULL, 0, NULL, 0,
                         &diagnostics, error, sizeof(error)) ||
        strstr(error, "invalid full forward arguments") == NULL)
        return 1;
    if (q38_decode(NULL, NULL, NULL, 0, NULL, 0, NULL, NULL, error,
                   sizeof(error)) ||
        strstr(error, "decode output token is null") == NULL)
        return 1;
    puts("test_m6_forward_api: complete forward/decode API validation passed");
    return 0;
}
