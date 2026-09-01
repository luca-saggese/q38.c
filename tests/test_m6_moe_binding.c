#include "q38_moe.h"

#include <stdio.h>
#include <string.h>

static q38_tensor tensor(uint32_t type, uint32_t ndim,
                         uint64_t a, uint64_t b, uint64_t c) {
    q38_tensor t;
    memset(&t, 0, sizeof(t));
    t.type = type; t.ndim = ndim; t.dim[0] = a; t.dim[1] = b; t.dim[2] = c;
    return t;
}

int main(void) {
    q38_tensor r = tensor(30,2,512,2560,0), gu = tensor(10,3,512,1280,2560);
    q38_tensor d = tensor(10,3,512,640,2560), sg = tensor(30,2,640,2560,0);
    q38_tensor su = tensor(30,2,640,2560,0), sd = tensor(30,2,2560,640,0);
    q38_tensor sgate = tensor(30,2,1,2560,0);
    q38_layer_weights layer;
    memset(&layer, 0, sizeof(layer));
    layer.router = &r; layer.shared_gate_proj = &sg; layer.shared_up_proj = &su;
    layer.shared_down_proj = &sd; layer.shared_expert_gate = &sgate;
    q38_expert_store_init_uniform(&layer.experts, 10);
    layer.experts.bank[0].gate_up = &gu; layer.experts.bank[0].down = &d;
    q38_moe_weights out;
    char error[128];
    if (!q38_moe_bind_layer(&layer, true, &out, error, sizeof(error)))
        return 1;
    puts("test_m6_moe_binding: routed/router/shared tensor domains bound strictly");
    return 0;
}
