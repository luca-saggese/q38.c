#include "q38_residency.h"

#include <assert.h>
#include <stdint.h>

int main(void) {
    q38_model_residency residency;
    char error[128] = {0};
    assert(q38_model_residency_init(&residency, 2, error, sizeof(error)));

    q38_tensor core = {.bytes = 257};
    q38_tensor expert = {.bytes = 513};
    q38_tensor ple = {.bytes = 4096};
    assert(q38_model_residency_account_tensor(
        &residency, &core, false, UINT32_MAX, error, sizeof(error)));
    assert(q38_model_residency_account_tensor(
        &residency, &expert, false, 1, error, sizeof(error)));
    assert(q38_model_residency_account_tensor(
        &residency, &ple, true, UINT32_MAX, error, sizeof(error)));

    assert(residency.core.used == 512);
    assert(residency.expert_banks[1].used == 768);
    assert(residency.non_ple_bytes == 770);
    assert(residency.ple_bytes == 4096);
    assert(residency.core.tensor_count == 1);
    assert(residency.expert_banks[1].tensor_count == 1);
    assert(residency.aligned_bytes == 1280);

    q38_model_residency_destroy(&residency);
    return 0;
}
