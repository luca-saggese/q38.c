#include "q38_nvfp4_pack.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int check_view(const q38_nvfp4_pack *pack, uint32_t layer,
                      uint32_t projection, uint32_t component,
                      uint64_t bytes, uint64_t rows, uint64_t cols,
                      uint32_t dtype) {
    char error[256] = {0};
    q38_nvfp4_view view;
    if (!q38_nvfp4_pack_get_expert_view(
            pack, layer, projection, 0, component, &view,
            error, sizeof(error))) {
        fprintf(stderr, "expert view failed: %s\n", error);
        return 1;
    }
    if (!view.data || view.bytes != bytes || view.rows != rows ||
        view.cols != cols || view.dtype != dtype ||
        view.quant_type != Q38_QUANT_NVIDIA_NVFP4) {
        fprintf(stderr,
                "unexpected view l=%u p=%u c=%u bytes=%" PRIu64
                " rows=%" PRIu64 " cols=%" PRIu64 " dtype=%u\n",
                layer, projection, component, view.bytes, view.rows,
                view.cols, view.dtype);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *pack_path = argc > 1
        ? argv[1] : "artifacts/m9-nvidia/q38_nvfp4_source_backed.pack";
    const char *source_root = argc > 2
        ? argv[2] : "models/Qwen3.8-Flash-Next-NVFP4";
    char error[256] = {0};
    q38_nvfp4_pack *pack = NULL;
    if (!q38_nvfp4_pack_open(
            pack_path, source_root, &pack, error, sizeof(error))) {
        fprintf(stderr, "pack open failed: %s\n", error);
        return 1;
    }
    int failed = 0;
    if (!q38_nvfp4_pack_is_source_backed(pack) ||
        q38_nvfp4_pack_main_resident_bytes(pack) != UINT64_C(77843711744) ||
        q38_nvfp4_pack_ple_bytes(pack) != UINT64_C(51200245762)) {
        fprintf(stderr, "pack accounting mismatch\n");
        failed = 1;
    }
    for (uint32_t layer = 0; layer < 48; ++layer) {
        if (layer != 0 && layer != 24 && layer != 47) continue;
        for (uint32_t projection = 0; projection < 3; ++projection) {
            failed |= check_view(
                pack, layer, projection, Q38_NVFP4_COMPONENT_WEIGHT,
                UINT64_C(819200),
                projection == Q38_NVFP4_PROJECTION_DOWN ? 2560 : 640,
                projection == Q38_NVFP4_PROJECTION_DOWN ? 320 : 1280,
                2);
            failed |= check_view(
                pack, layer, projection, Q38_NVFP4_COMPONENT_WEIGHT_SCALE,
                UINT64_C(102400),
                projection == Q38_NVFP4_PROJECTION_DOWN ? 2560 : 640,
                projection == Q38_NVFP4_PROJECTION_DOWN ? 40 : 160,
                3);
            failed |= check_view(
                pack, layer, projection, Q38_NVFP4_COMPONENT_WEIGHT_SCALE_2,
                UINT64_C(4), 1, 1, 4);
            failed |= check_view(
                pack, layer, projection, Q38_NVFP4_COMPONENT_INPUT_SCALE,
                UINT64_C(4), 1, 1, 4);
        }
    }
    for (uint32_t shard = 0; shard < 128; ++shard) {
        q38_nvfp4_view view;
        if (!q38_nvfp4_pack_get_ple_view(
                pack, shard, &view, error, sizeof(error)) ||
            view.bytes != UINT64_C(400001920) ||
            view.rows != UINT64_C(2500012) || view.cols != 160 ||
            view.dtype != 3) {
            fprintf(stderr, "PLE view failed for shard %u: %s\n",
                    shard, error);
            failed = 1;
            break;
        }
    }
    uint32_t bf16_count = 0;
    const char *name = NULL;
    uint32_t ndim = 0;
    uint64_t shape[4] = {0};
    q38_nvfp4_view bf16;
    if (!q38_nvfp4_pack_get_bf16_count(pack, &bf16_count) ||
        bf16_count != 1067 ||
        !q38_nvfp4_pack_get_bf16_view(
            pack, 0, &bf16, &name, &ndim, shape, error, sizeof(error)) ||
        !name || !name[0] || !bf16.data || ndim == 0) {
        fprintf(stderr, "BF16 binding failed: %s\n", error);
        failed = 1;
    }
    q38_nvfp4_pack_close(pack);
    if (failed) return 1;
    puts("test_m9_nvfp4_pack: source-backed binding passed");
    return 0;
}
