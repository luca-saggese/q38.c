#ifndef Q38_QSA_CANDIDATE_H
#define Q38_QSA_CANDIDATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*q38_qsa_candidate_fn)(
    const void *wq, const void *wk, const void *wv,
    const float *input, float *q, float *k, float *v,
    size_t token_count, size_t cols, void *stream);

typedef int (*q38_qsa_candidate_abi_fn)(void);

#define Q38_QSA_CANDIDATE_SYMBOL "q38_qsa_candidate_project"
#define Q38_QSA_CANDIDATE_ABI_SYMBOL "q38_qsa_candidate_abi"
#define Q38_QSA_CANDIDATE_ABI_VERSION 1

#ifdef __cplusplus
}
#endif

#endif
