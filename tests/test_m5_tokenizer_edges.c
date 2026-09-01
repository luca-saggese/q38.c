#include "q38_tokenizer.h"

#include <stdio.h>
#include <string.h>

static int check(const q38_tokenizer *t, const char *name, const char *text,
                 const uint32_t *expected, size_t count) {
    q38_token_batch got = {0};
    char error[256];
    int bad = !q38_tokenizer_encode(t, text, false, &got, error,
                                    sizeof(error)) ||
              got.token_count != count ||
              memcmp(got.tokens, expected, count * sizeof(*expected)) != 0;
    if (bad) {
        fprintf(stderr, "tokenizer edge mismatch (%s): %s got=%u expected=%zu ids=",
                name, error, got.token_count, count);
        for (uint32_t i = 0; i < got.token_count; ++i) fprintf(stderr, "%u,", got.tokens[i]);
        fputc('\n', stderr);
    }
    q38_token_batch_free(&got);
    return bad;
}

int main(void) {
    const uint32_t escaped[] = {2855, 7018, 35414, 24193, 37734, 1639};
    const uint32_t unicode[] = {933, 87209, 211096, 150581, 151188,
                                151552, 220, 172, 238, 238, 115};
    const uint32_t audio[] = {248070, 248076, 248071};
    q38_tokenizer t;
    char error[256];
    if (!q38_tokenizer_init(&t, "/home/lvx/q38model", NULL, error,
                            sizeof(error)) ||
        !q38_tokenizer_verify_specials(&t, error, sizeof(error)) ||
        t.bos_id != 248044 || t.eos_id != 248044 ||
        check(&t, "escaped", "quote \\\" slash \\\\ newline\\n", escaped,
              sizeof(escaped) / sizeof(*escaped)) ||
        check(&t, "unicode", "é 😀 عربي עברית 𐐷", unicode,
              sizeof(unicode) / sizeof(*unicode)) ||
        check(&t, "audio", "<|audio_start|><|audio_pad|><|audio_end|>", audio,
              sizeof(audio) / sizeof(*audio))) {
        fprintf(stderr, "tokenizer edge test failed: %s\n", error);
        q38_tokenizer_destroy(&t);
        return 1;
    }
    q38_tokenizer_destroy(&t);
    puts("test_m5_tokenizer_edges: Unicode, escaping, audio markers, and special IDs passed");
    return 0;
}
