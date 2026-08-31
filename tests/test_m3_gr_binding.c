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
    const q38_layer_weights *layer = &weights.layer[0];
    if (!layer->attn_gr.block_inject_weight || !layer->attn_gr.hc_norm ||
        !layer->attn_gr.input_mix_weight_down ||
        !layer->attn_gr.input_mix_weight_up ||
        !layer->mlp_gr.block_inject_weight || !layer->mlp_gr.hc_norm ||
        !layer->mlp_gr.input_mix_weight_down ||
        !layer->mlp_gr.input_mix_weight_up) {
        fprintf(stderr, "layer 0 GR tensor family is incomplete\n");
        q38_gguf_close(model);
        return 1;
    }
    if (layer->attn_gr.block_inject_weight->dim[0] != 4 ||
        layer->attn_gr.block_inject_weight->dim[1] != 10240 ||
        layer->attn_gr.input_mix_weight_down->dim[0] != 320 ||
        layer->attn_gr.input_mix_weight_down->dim[1] != 10240) {
        fprintf(stderr, "layer 0 GR shape mismatch\n");
        q38_gguf_close(model);
        return 1;
    }
    q38_gguf_close(model);
    puts("test_m3_gr_binding: strict attention/MLP GR binding passed");
    return 0;
}
