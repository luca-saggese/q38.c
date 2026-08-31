#include "q38_gguf.h"
#include "q38_weights.h"

#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char error[256];
    q38_gguf *model = q38_gguf_open(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    q38_weights weights;
    if (!q38_weights_bind_subset(model, 0, &weights, error, sizeof(error))) {
        fprintf(stderr, "bind failed: %s\n", error);
        q38_gguf_close(model);
        return 1;
    }
    const q38_gdn_weights *gdn = &weights.layer[0].gdn;
    if (!gdn->in_proj_qkv || !gdn->in_proj_z || !gdn->in_proj_a ||
        !gdn->in_proj_b || !gdn->conv1d || !gdn->A_log ||
        !gdn->dt_bias || !gdn->norm || !gdn->out_proj) {
        fprintf(stderr, "layer 0 GDN tensor family is incomplete\n");
        q38_gguf_close(model);
        return 1;
    }
    if (gdn->in_proj_qkv->dim[0] != 10240 ||
        gdn->in_proj_qkv->dim[1] != 2560 ||
        gdn->conv1d->dim[2] != 4 ||
        gdn->out_proj->dim[0] != 2560 ||
        gdn->out_proj->dim[1] != 6144) {
        fprintf(stderr, "layer 0 GDN shape mismatch\n");
        q38_gguf_close(model);
        return 1;
    }
    q38_gguf_close(model);
    puts("test_m3_gdn_binding: strict projection/conv/gate binding passed");
    return 0;
}
