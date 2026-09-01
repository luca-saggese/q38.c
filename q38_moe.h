#ifndef Q38_MOE_H
#define Q38_MOE_H

#include "q38_gguf.h"
#include "q38_weights.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    q38_tensor *router;
    q38_tensor *router_bias;
    q38_tensor *routed_gate_up;
    q38_tensor *routed_down;
    q38_tensor *shared_gate_proj;
    q38_tensor *shared_up_proj;
    q38_tensor *shared_down_proj;
    q38_tensor *shared_gate;
    bool routed_quantized;
} q38_moe_weights;

bool q38_moe_weights_validate(const q38_moe_weights *weights,
                              char *error, size_t error_len);
bool q38_moe_bind_layer(const q38_layer_weights *layer, bool quantized,
                        q38_moe_weights *out, char *error, size_t error_len);
bool q38_moe_expert_slice(const q38_gguf *model, const q38_tensor *tensor,
                          uint32_t expert, uint64_t *offset, uint64_t *bytes,
                          char *error, size_t error_len);

#endif
