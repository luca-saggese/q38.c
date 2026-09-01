#ifndef Q38_TOPK_REF_H
#define Q38_TOPK_REF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool q38_topk_select_ref(const float *scores, size_t count, size_t k,
                         uint32_t *indices, char *error, size_t error_len);

#ifdef __cplusplus
}
#endif

#endif
