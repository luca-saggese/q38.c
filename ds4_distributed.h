#ifndef DS4_DISTRIBUTED_H
#define DS4_DISTRIBUTED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

typedef ds4_distributed_options ds4_dist_options;
typedef struct ds4_dist_session ds4_dist_session;

typedef struct {
    const char *prompt;
    const char *system;
    const char *dump_logits_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    ds4_think_mode think_mode;
} ds4_dist_generation_options;

typedef enum {
    DS4_DIST_CLI_ERROR = -1,
    DS4_DIST_CLI_NOT_MATCHED = 0,
    DS4_DIST_CLI_MATCHED = 1,
} ds4_dist_cli_parse_result;

bool ds4_dist_enabled(const ds4_dist_options *opt);
ds4_dist_options *ds4_dist_options_create(void);
void ds4_dist_options_free(ds4_dist_options *opt);
void ds4_dist_usage(FILE *fp);
ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen);
int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen);
int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen);
void ds4_dist_session_free(ds4_dist_session *d);
int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen);
int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen);
int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen);
int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen);

#endif
