#include "q38_ple_ref.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *text;
    size_t length;
} file_text;

static int fail(const char *message) {
    fprintf(stderr, "q38_forward_probe: %s\n", message);
    return 1;
}

static file_text read_file(const char *path) {
    file_text out = {0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return out;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return out; }
    long size = ftell(fp);
    if (size < 0 || fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return out; }
    out.text = (char *)malloc((size_t)size + 1);
    if (!out.text || fread(out.text, 1, (size_t)size, fp) != (size_t)size) {
        free(out.text);
        out.text = NULL;
        fclose(fp);
        return out;
    }
    out.text[size] = '\0';
    out.length = (size_t)size;
    fclose(fp);
    return out;
}

static const char *key_value(const char *object, const char *key) {
    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *at = strstr(object, needle);
    if (!at) return NULL;
    at = strchr(at + strlen(needle), ':');
    return at ? at + 1 : NULL;
}

static const char *array_end(const char *at) {
    int depth = 0;
    int in_string = 0;
    for (; *at; ++at) {
        if (*at == '"' && (at == 0 || at[-1] != '\\')) in_string = !in_string;
        if (in_string) continue;
        if (*at == '[') ++depth;
        else if (*at == ']' && --depth == 0) return at;
    }
    return NULL;
}

static int parse_u64_array(const char *object, const char *key,
                           uint64_t *values, size_t count) {
    const char *at = key_value(object, key);
    if (!at) return 0;
    while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
    if (*at != '[') return 0;
    for (size_t i = 0; i < count; ++i) {
        char *end = NULL;
        while (*at && (*at == '[' || *at == ']' || *at == ',' || *at == ' ' ||
                       *at == '\n' || *at == '\r' || *at == '\t')) ++at;
        if (!*at || *at == ']') return 0;
        values[i] = strtoull(at, &end, 10);
        if (end == at) return 0;
        at = end;
    }
    return 1;
}

static int parse_u32_array(const char *object, const char *key,
                           uint32_t *values, size_t count) {
    uint64_t parsed[Q38_PLE_MAX_HEADS];
    if (count > Q38_PLE_MAX_HEADS ||
        !parse_u64_array(object, key, parsed, count)) return 0;
    for (size_t i = 0; i < count; ++i) {
        if (parsed[i] > UINT32_MAX) return 0;
        values[i] = (uint32_t)parsed[i];
    }
    return 1;
}

static int parse_u32_scalar(const char *object, const char *key,
                            uint32_t *value) {
    const char *at = key_value(object, key);
    if (!at) return 0;
    while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
    char *end = NULL;
    unsigned long parsed = strtoul(at, &end, 10);
    if (end == at || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int parse_tokens(const char *object, const char *key,
                        uint32_t **tokens, size_t *count) {
    const char *at = key_value(object, key);
    if (!at) return 0;
    while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
    if (*at != '[') return 0;
    size_t capacity = 16;
    uint32_t *out = (uint32_t *)malloc(capacity * sizeof(*out));
    if (!out) return 0;
    size_t used = 0;
    ++at;
    for (;;) {
        while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
        if (*at == ']') break;
        if (used == capacity) {
            capacity *= 2;
            uint32_t *grown = (uint32_t *)realloc(out, capacity * sizeof(*out));
            if (!grown) { free(out); return 0; }
            out = grown;
        }
        char *end = NULL;
        unsigned long value = strtoul(at, &end, 10);
        if (end == at || value > UINT32_MAX) { free(out); return 0; }
        out[used++] = (uint32_t)value;
        at = end;
        while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
        if (*at == ',') ++at;
        else if (*at != ']') { free(out); return 0; }
    }
    *tokens = out;
    *count = used;
    return 1;
}

static int parse_rows(const char *object, uint32_t *rows, size_t count) {
    const char *at = key_value(object, "ngram_row_ids");
    if (!at) return 0;
    while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
    if (*at != '[') return 0;
    ++at;
    for (size_t i = 0; i < count; ++i) {
        while (*at && (*at == '[' || *at == ']' || *at == ',' || *at == ' ' ||
                       *at == '\n' || *at == '\r' || *at == '\t')) ++at;
        char *end = NULL;
        unsigned long value = strtoul(at, &end, 10);
        if (end == at || value > UINT32_MAX) return 0;
        rows[i] = (uint32_t)value;
        at = end;
    }
    return 1;
}

static int check_optional_vectors_are_explicitly_unavailable(const char *object) {
    const char *keys[] = {"hidden_before_ple", "hidden_after_ple",
                          "ple_contribution_vector"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        const char *at = key_value(object, keys[i]);
        if (!at) return 0;
        while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
        if (strncmp(at, "null", 4) != 0) return 0;
    }
    return 1;
}

static int check_stream(const q38_ple_hash_config *config, uint32_t eos_token,
                        const uint32_t *tokens, size_t token_count,
                        const uint32_t *expected, char *error,
                        size_t error_len) {
    q38_ngram_history history;
    q38_ngram_history_reset(&history);
    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        uint32_t actual[Q38_PLE_MAX_HEADS];
        if (!q38_ple_ngram_ids_ref(config, &history, tokens[token_index],
                                   eos_token, actual, 16, error, error_len)) {
            return 0;
        }
        for (size_t head = 0; head < 16; ++head) {
            if (actual[head] != expected[token_index * 16 + head]) return 0;
        }
        q38_ngram_history_append(&history, tokens[token_index], eos_token);
    }
    return 1;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "artifacts/m4/ple_injection_golden.json";
    file_text file = read_file(path);
    if (!file.text) return fail("cannot read golden dump");
    if (!strstr(file.text, "\"format\":\"q38-m4-c06-ple-golden-v1\"") &&
        !strstr(file.text, "\"format\": \"q38-m4-c06-ple-golden-v1\"")) {
        free(file.text);
        return fail("unsupported golden format");
    }

    q38_ple_hash_config config;
    memset(&config, 0, sizeof(config));
    const char *config_object = strstr(file.text, "\"config\"");
    uint32_t eos_token = 0;
    if (!config_object) {
        free(file.text);
        return fail("invalid PLE hash configuration in golden dump");
    }
    if (!parse_u32_scalar(config_object, "ngram_size", &config.ngram_size)) {
        free(file.text); return fail("golden ngram_size is invalid");
    }
    if (!parse_u32_scalar(config_object, "heads_per_ngram",
                          &config.heads_per_ngram)) {
        free(file.text); return fail("golden heads_per_ngram is invalid");
    }
    if (!parse_u32_scalar(config_object, "eos_token_id", &eos_token)) {
        free(file.text); return fail("golden eos_token_id is invalid");
    }
    if (!parse_u32_array(config_object, "head_offsets", config.head_offsets, 16)) {
        free(file.text); return fail("golden head_offsets is invalid");
    }
    if (!parse_u32_array(config_object, "head_vocab_sizes", config.head_vocab_sizes, 16)) {
        free(file.text); return fail("golden head_vocab_sizes is invalid");
    }
    if (!parse_u64_array(config_object, "multipliers", config.multipliers, 3)) {
        free(file.text);
        return fail("golden multipliers are invalid");
    }
    char error[128];
    if (!q38_ple_hash_config_validate(&config, error, sizeof(error))) {
        free(file.text);
        return fail(error);
    }

    const char *cases = strstr(file.text, "\"cases\"");
    if (!cases) { free(file.text); return fail("golden dump has no cases"); }
    const char *cases_end = array_end(cases);
    size_t checked = 0;
    const char *cursor = strchr(cases, '{');
    while (cursor && (!cases_end || cursor < cases_end)) {
        const char *end = strchr(cursor, '}');
        if (!end) { free(file.text); return fail("unterminated golden case"); }
        size_t object_len = (size_t)(end - cursor);
        char *object = (char *)malloc(object_len + 1);
        if (!object) { free(file.text); return fail("out of memory"); }
        memcpy(object, cursor, object_len);
        object[object_len] = '\0';

        uint32_t *tokens = NULL;
        size_t token_count = 0;
        if (!parse_tokens(object, "tokens", &tokens, &token_count) ||
            token_count == 0 ||
            !check_optional_vectors_are_explicitly_unavailable(object)) {
            free(tokens);
            free(object);
            free(file.text);
            return fail("invalid or fabricated hidden vectors in golden case");
        }
        uint32_t *partitions = NULL;
        size_t partition_count = 0;
        if (!parse_tokens(object, "partitions", &partitions, &partition_count) ||
            partition_count == 0) {
            free(tokens);
            free(object);
            free(file.text);
            return fail("golden case has no partitioning");
        }
        size_t partition_total = 0;
        for (size_t i = 0; i < partition_count; ++i) {
            if (partitions[i] == 0 || partition_total > SIZE_MAX - partitions[i]) {
                free(partitions); free(tokens); free(object); free(file.text);
                return fail("invalid golden partitioning");
            }
            partition_total += partitions[i];
        }
        if (partition_total != token_count) {
            free(partitions); free(tokens); free(object); free(file.text);
            return fail("golden partitioning does not cover token stream");
        }
        if (token_count > SIZE_MAX / 16) {
            free(partitions); free(tokens);
            free(object);
            free(file.text);
            return fail("golden case is too large");
        }
        uint32_t *expected = (uint32_t *)malloc(token_count * 16 * sizeof(*expected));
        if (!expected || !parse_rows(object, expected, token_count * 16)) {
            free(expected);
            free(partitions);
            free(tokens);
            free(object);
            free(file.text);
            return fail("invalid row IDs in golden case");
        }

        if (!check_stream(&config, eos_token, tokens, token_count, expected,
                          error, sizeof(error))) {
            free(expected); free(partitions); free(tokens); free(object);
            free(file.text);
            return fail(error[0] ? error : "q38 PLE IDs differ from independent golden");
        }
        q38_ngram_history history;
        q38_ngram_history_reset(&history);
        size_t start = 0;
        for (size_t part = 0; part < partition_count; ++part) {
            size_t end = start + partitions[part];
            for (size_t token_index = start; token_index < end; ++token_index) {
                uint32_t actual[Q38_PLE_MAX_HEADS];
                if (!q38_ple_ngram_ids_ref(&config, &history, tokens[token_index],
                                           eos_token, actual, 16, error,
                                           sizeof(error))) {
                    free(expected); free(partitions); free(tokens); free(object);
                    free(file.text);
                    return fail(error);
                }
                for (size_t head = 0; head < 16; ++head) {
                    if (actual[head] != expected[token_index * 16 + head]) {
                        free(expected); free(partitions); free(tokens); free(object);
                        free(file.text);
                        return fail("chunked q38 PLE IDs differ from golden");
                    }
                }
                q38_ngram_history_append(&history, tokens[token_index], eos_token);
            }
            start = end;
        }
        free(expected);
        free(partitions);
        free(tokens);
        free(object);
        ++checked;
        cursor = strchr(end + 1, '{');
    }
    free(file.text);
    if (checked == 0) return fail("golden dump has no parseable cases");
    printf("q38_forward_probe: %zu PLE ID cases match independent goldens\n", checked);
    puts("q38_forward_probe: hidden_before_ple/hidden_after_ple/PLE contribution are unavailable; no vectors fabricated");
    return 0;
}
