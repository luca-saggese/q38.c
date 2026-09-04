#include "../q38_replay.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int expect(bool condition, const char *message) {
    if (!condition) fprintf(stderr, "test_m9_checkpoint_mismatch: %s\n", message);
    return condition ? 0 : 1;
}

static int patch_u32(const char *path, long offset, uint32_t value) {
    FILE *file = fopen(path, "r+b");
    if (!file || fseek(file, offset, SEEK_SET) != 0 ||
        fwrite(&value, sizeof(value), 1, file) != 1 ||
        fclose(file) != 0)
        return 1;
    return 0;
}

int main(void) {
    const char *path = "artifacts/m9/checkpoint_mismatch.snapshot";
    q38_forward_state source, restored;
    q38_session_state layout;
    char error[256];
    memset(&source, 0, sizeof(source));
    memset(&restored, 0, sizeof(restored));
    if (expect(q38_session_state_init(&layout, 0, error, sizeof(error)),
               "layout init failed"))
        return 1;
    source.storage.layout = layout;
    restored.storage.layout = layout;
    if (expect(q38_state_alloc(&layout, &source.storage, error, sizeof(error)) &&
               q38_state_alloc(&layout, &restored.storage, error, sizeof(error)),
               "state allocation failed"))
        return 1;
    source.initialized = true;
    restored.initialized = true;
    if (expect(q38_replay_snapshot_save(path, &source, error, sizeof(error)),
               "snapshot save failed"))
        return 1;
    if (expect(q38_replay_snapshot_load(path, &restored, error, sizeof(error)),
               "snapshot roundtrip failed"))
        return 1;
    if (expect(patch_u32(path, 8, 99u) == 0, "version patch failed") ||
        expect(!q38_replay_snapshot_load(path, &restored, error, sizeof(error)),
               "unsupported version was accepted"))
        return 1;
    if (expect(q38_replay_snapshot_save(path, &source, error, sizeof(error)),
               "snapshot rewrite failed"))
        return 1;
    if (expect(patch_u32(path, 12, 99u) == 0, "layout patch failed") ||
        expect(!q38_replay_snapshot_load(path, &restored, error, sizeof(error)),
               "layout mismatch was accepted"))
        return 1;
    q38_state_free(&source.storage);
    q38_state_free(&restored.storage);
    remove(path);
    puts("test_m9_checkpoint_mismatch: version/layout rejection passed");
    return 0;
}
