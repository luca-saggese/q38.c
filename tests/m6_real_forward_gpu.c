#define main m6_real_forward_cpu_main
#include "m6_real_forward.c"
#undef main

#include "../q38_forward_cuda.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf output.json\n", argv[0]);
        return 2;
    }
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open: %s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 47, &weights, error, sizeof(error))) {
        fprintf(stderr, "bind: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    q38_forward_state state;
    if (!q38_forward_state_init(&state, &weights, 248044, error,
                                sizeof(error))) {
        fprintf(stderr, "state: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    q38_forward_cuda_context *cuda =
        q38_forward_cuda_context_create(error, sizeof(error));
    if (!cuda) {
        fprintf(stderr, "cuda: %s\n", error);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    FILE *out = fopen(argv[2], "w");
    if (!out) {
        perror("output");
        q38_forward_cuda_context_destroy(cuda);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    fputs("{\"format\":\"q38-m6-full-trace-v4\",\"tokens\":[9419],"
          "\"stages\":[\n",
          out);
    trace_context context = {.out = out, .first = true};
    q38_forward_diagnostics diagnostics;
    memset(&diagnostics, 0, sizeof(diagnostics));
    diagnostics.trace = trace;
    diagnostics.route_trace = route_trace;
    diagnostics.router_trace = router_trace;
    diagnostics.moe_trace = moe_trace;
    diagnostics.pre_router_trace = pre_router_trace;
    diagnostics.qsa_trace = qsa_trace;
    diagnostics.trace_user = &context;
    float *logits = calloc(248320, sizeof(float));
    const uint32_t token = 9419;
    const bool ok =
        logits &&
        q38_forward_full_with_backend(
            model, &weights, &state, &token, 1, logits, 248320, &diagnostics,
            q38_forward_cuda_matvec_backend, cuda, error, sizeof(error)) &&
        trace_complete(&context);
    if (!ok) {
        if (!error[0])
            snprintf(error, sizeof(error),
                     "forward trace is incomplete; refusing partial output");
        fprintf(stderr, "forward: %s\n", error);
        remove(argv[2]);
        free(logits);
        fclose(out);
        q38_forward_cuda_context_destroy(cuda);
        q38_forward_state_destroy(&state);
        q38_gguf_close(model);
        return 1;
    }
    write_logits(out, logits, 248320);
    fputs("\n],", out);
    write_decisions(out, &context);
    fputs("\n}\n", out);
    fclose(out);
    free(logits);
    q38_forward_cuda_context_destroy(cuda);
    q38_forward_state_destroy(&state);
    q38_gguf_close(model);
    return 0;
}
