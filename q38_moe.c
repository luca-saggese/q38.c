#include "q38_moe.h"

#include <stdio.h>
#include <string.h>

static bool fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static bool shape(const q38_tensor *t, uint32_t ndim, const uint64_t *dims) {
    if (!t || t->ndim != ndim) return false;
    for (uint32_t i = 0; i < ndim; ++i) if (t->dim[i] != dims[i]) return false;
    return true;
}

bool q38_moe_weights_validate(const q38_moe_weights *w,
                              char *error, size_t error_len) {
    static const uint64_t router[] = {512, 2560};
    static const uint64_t gate_up[] = {512, 1280, 2560};
    static const uint64_t down[] = {512, 2560, 640};
    static const uint64_t down_t[] = {512, 640, 2560};
    static const uint64_t shared[] = {640, 2560};
    static const uint64_t shared_down[] = {2560, 640};
    static const uint64_t gate[] = {1, 2560};
    if (error && error_len) error[0] = '\0';
    if (!w || !w->router || !w->routed_gate_up || !w->routed_down ||
        !w->shared_gate_proj || !w->shared_up_proj || !w->shared_down_proj ||
        !w->shared_gate || w->router_bias)
        return fail(error, error_len, "MoE tensor family is incomplete");
    if (!shape(w->router, 2, router) || w->router->type != 30 ||
        !shape(w->routed_gate_up, 3, gate_up) ||
        !shape(w->routed_down, 3, w->routed_quantized ? down_t : down) ||
        !shape(w->shared_gate_proj, 2, shared) ||
        !shape(w->shared_up_proj, 2, shared) ||
        !shape(w->shared_down_proj, 2, shared_down) ||
        !shape(w->shared_gate, 2, gate))
        return fail(error, error_len, "MoE tensor shape mismatch");
    if (w->routed_quantized &&
        (w->routed_gate_up->type != 10u ||
         w->routed_down->type != 10u) &&
        (w->routed_gate_up->type != 12u ||
         w->routed_down->type != 12u))
        return fail(error, error_len, "MoE routed quant type mismatch");
    if ((!w->routed_quantized &&
         (w->routed_gate_up->type != 30u ||
          w->routed_down->type != 30u)) ||
        w->shared_gate_proj->type != 30 || w->shared_up_proj->type != 30 ||
        w->shared_down_proj->type != 30 || w->shared_gate->type != 30)
        return fail(error, error_len, "MoE tensor type mismatch");
    return true;
}

bool q38_moe_bind_layer(const q38_layer_weights *layer, bool quantized,
                        q38_moe_weights *out, char *error, size_t error_len) {
    if (!layer || !out) return fail(error, error_len, "invalid MoE bind arguments");
    memset(out, 0, sizeof(*out));
    out->router = layer->router;
    out->routed_gate_up = layer->experts.bank_count
        ? layer->experts.bank[0].gate_up : NULL;
    out->routed_down = layer->experts.bank_count
        ? layer->experts.bank[0].down : NULL;
    out->shared_gate_proj = layer->shared_gate_proj;
    out->shared_up_proj = layer->shared_up_proj;
    out->shared_down_proj = layer->shared_down_proj;
    out->shared_gate = layer->shared_expert_gate;
    out->routed_quantized = quantized;
    return q38_moe_weights_validate(out, error, error_len);
}

bool q38_moe_expert_slice(const q38_gguf *model, const q38_tensor *tensor,
                          uint32_t expert, uint64_t *offset, uint64_t *bytes,
                          char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    if (!model || !tensor || !offset || !bytes || tensor->ndim != 3 ||
        tensor->dim[0] != Q38_MODEL_EXPERTS || expert >= Q38_MODEL_EXPERTS ||
        tensor->bytes % tensor->dim[0] != 0)
        return fail(error, error_len, "invalid routed expert slice");
    const uint64_t slice = tensor->bytes / tensor->dim[0];
    if (expert > UINT64_MAX / slice ||
        tensor->abs_offset > UINT64_MAX - expert * slice)
        return fail(error, error_len, "routed expert slice overflows");
    *offset = tensor->abs_offset + expert * slice;
    *bytes = slice;
    if (*offset > model->size || *bytes > model->size - *offset)
        return fail(error, error_len, "routed expert slice exceeds model mapping");
    return true;
}
