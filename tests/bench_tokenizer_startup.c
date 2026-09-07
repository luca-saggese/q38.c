#include "q38_tokenizer.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

typedef struct {
    uint64_t malloc_count;
    uint64_t calloc_count;
    uint64_t realloc_count;
    uint64_t strdup_count;
    uint64_t allocated_bytes;
    uint64_t copied_string_bytes;
} allocation_stats;

static allocation_stats g_alloc;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
char *__real_strdup(const char *);

void *__wrap_malloc(size_t bytes) {
    g_alloc.malloc_count++;
    g_alloc.allocated_bytes += bytes;
    return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
    g_alloc.calloc_count++;
    g_alloc.allocated_bytes += count * bytes;
    return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
    g_alloc.realloc_count++;
    g_alloc.allocated_bytes += bytes;
    return __real_realloc(ptr, bytes);
}

char *__wrap_strdup(const char *value) {
    g_alloc.strdup_count++;
    if (value) g_alloc.copied_string_bytes += strlen(value) + 1;
    return __real_strdup(value);
}

static bool run_fixture(const q38_tokenizer *tokenizer) {
    static const char *const cases[] = {
        "ASCII text.",
        "  whitespace\twith  runs\nand punctuation!!!",
        "UTF-8: café déjà vu — naïve.",
        "CJK 日本語 中文 한국어",
        "emoji: 😀 🚀 👩‍💻",
        "<|endoftext|> <|im_start|> user",
        "A long sentence with repeated words and punctuation, "
        "used to exercise the complete tokenizer path."
    };
    char error[256] = {};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        q38_token_batch first = {};
        q38_token_batch second = {};
        char *decoded_first = NULL;
        char *decoded_second = NULL;
        size_t first_bytes = 0;
        size_t second_bytes = 0;
        bool ok =
            q38_tokenizer_encode(tokenizer, cases[i], false, &first,
                                 error, sizeof(error)) &&
            q38_tokenizer_encode(tokenizer, cases[i], false, &second,
                                 error, sizeof(error)) &&
            first.token_count == second.token_count &&
            memcmp(first.tokens, second.tokens,
                   first.token_count * sizeof(*first.tokens)) == 0 &&
            q38_tokenizer_decode(tokenizer, first.tokens, first.token_count,
                                  &decoded_first, &first_bytes,
                                  error, sizeof(error)) &&
            q38_tokenizer_decode(tokenizer, second.tokens, second.token_count,
                                  &decoded_second, &second_bytes,
                                  error, sizeof(error)) &&
            first_bytes == second_bytes &&
            memcmp(decoded_first, decoded_second, first_bytes) == 0;
        free(decoded_first);
        free(decoded_second);
        q38_token_batch_free(&first);
        q38_token_batch_free(&second);
        if (!ok) {
            fprintf(stderr, "fixture failed at case %zu: %s\n", i, error);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    const char *tokenizer_dir =
        argc > 1 ? argv[1] : "/home/lvx/q38model";
    q38_tokenizer_profile profile;
    q38_tokenizer_profile_set(&profile);
    q38_tokenizer tokenizer;
    char error[256] = {};
    if (!q38_tokenizer_init(&tokenizer, tokenizer_dir, NULL,
                            error, sizeof(error))) {
        fprintf(stderr, "tokenizer init failed: %s\n", error);
        return 1;
    }
    const bool fixture_green = run_fixture(&tokenizer);
    char tokenizer_path[512];
    snprintf(tokenizer_path, sizeof(tokenizer_path),
             "%s/tokenizer.json", tokenizer_dir);
    struct stat tokenizer_stat = {};
    const size_t tokenizer_json_bytes =
        stat(tokenizer_path, &tokenizer_stat) == 0
            ? (size_t)tokenizer_stat.st_size : 0;
    printf("{\"tokenizer_json_bytes\":%zu,\"vocab_entries\":%" PRIu64
           ",\"merges_entries\":%" PRIu64 ",\"wall_ms\":%.3f,"
           "\"file_read_ms\":%.3f,\"json_scan_ms\":%.3f,"
           "\"vocab_parse_ms\":%.3f,\"vocab_string_alloc_ms\":%.3f,"
           "\"vocab_hash_build_ms\":%.3f,\"merges_parse_ms\":%.3f,"
           "\"merges_string_alloc_ms\":%.3f,\"merges_index_build_ms\":%.3f,"
           "\"special_token_parse_ms\":%.3f,\"regex_pretokenizer_init_ms\":%.3f,"
           "\"fingerprint_ms\":%.3f,\"other_ms\":%.3f,"
           "\"malloc_count\":%" PRIu64 ",\"calloc_count\":%" PRIu64
           ",\"realloc_count\":%" PRIu64 ",\"strdup_count\":%" PRIu64
           ",\"allocated_bytes\":%" PRIu64
           ",\"copied_string_bytes\":%" PRIu64
           ",\"hash_insertions\":%" PRIu64 ",\"hash_lookups\":%" PRIu64
           ",\"string_comparisons\":%" PRIu64 ",\"strlen_calls\":%" PRIu64
           ",\"fixture\":\"%s\"}\n",
           tokenizer_json_bytes, profile.vocab_entries, profile.merge_entries,
           profile.wall_ms, profile.file_read_ms, profile.json_scan_ms,
           profile.vocab_parse_ms, profile.vocab_string_alloc_ms,
           profile.vocab_hash_build_ms, profile.merges_parse_ms,
           profile.merges_string_alloc_ms, profile.merges_index_build_ms,
           profile.special_token_parse_ms,
           profile.regex_pretokenizer_init_ms, profile.fingerprint_ms,
           profile.other_ms, g_alloc.malloc_count, g_alloc.calloc_count,
           g_alloc.realloc_count, g_alloc.strdup_count,
           g_alloc.allocated_bytes, profile.string_copy_bytes +
               g_alloc.copied_string_bytes, profile.hash_insertions,
           profile.hash_lookups, profile.string_comparisons,
           profile.strlen_calls, fixture_green ? "GREEN" : "RED");
    q38_tokenizer_destroy(&tokenizer);
    return fixture_green ? 0 : 1;
}
