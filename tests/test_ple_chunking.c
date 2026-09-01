#include "q38_ple_ref.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *text;
    size_t length;
} file_text;

typedef struct {
    uint32_t *values;
    size_t count;
} u32_array;

static int fail(const char *message) {
    fprintf(stderr, "test_ple_chunking: %s\n", message);
    return 1;
}

static file_text read_file(const char *path) {
    file_text out = {0};
    FILE *fp = fopen(path, "rb");
    if (!fp) return out;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return out;
    }
    long size = ftell(fp);
    if (size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return out;
    }
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

static const char *skip_json_space(const char *at) {
    while (*at == ' ' || *at == '\n' || *at == '\r' || *at == '\t') ++at;
    return at;
}

static int parse_u32_scalar(const char *object, const char *key,
                            uint32_t *value) {
    const char *at = key_value(object, key);
    if (!at) return 0;
    at = skip_json_space(at);
    char *end = NULL;
    unsigned long parsed = strtoul(at, &end, 10);
    if (end == at || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int parse_u64_array(const char *object, const char *key,
                           uint64_t *values, size_t count) {
    const char *at = key_value(object, key);
    if (!at) return 0;
    at = skip_json_space(at);
    if (*at != '[') return 0;
    ++at;
    for (size_t i = 0; i < count; ++i) {
        at = skip_json_space(at);
        while (*at == ',' || *at == '[' || *at == ']') {
            ++at;
            at = skip_json_space(at);
        }
        char *end = NULL;
        unsigned long long parsed = strtoull(at, &end, 10);
        if (end == at) return 0;
        values[i] = (uint64_t)parsed;
        at = end;
    }
    return 1;
}

static int parse_u32_array(const char *object, const char *key,
                           uint32_t *values, size_t count) {
    uint64_t parsed[Q38_PLE_MAX_HEADS];
    if (count > Q38_PLE_MAX_HEADS ||
        !parse_u64_array(object, key, parsed, count)) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (parsed[i] > UINT32_MAX) return 0;
        values[i] = (uint32_t)parsed[i];
    }
    return 1;
}

static int parse_u32_list(const char *object, const char *key,
                          u32_array *array) {
    const char *at = key_value(object, key);
    if (!at) return 0;
    at = skip_json_space(at);
    if (*at != '[') return 0;
    ++at;

    size_t capacity = 16;
    uint32_t *values = (uint32_t *)malloc(capacity * sizeof(*values));
    if (!values) return 0;
    size_t count = 0;
    for (;;) {
        at = skip_json_space(at);
        if (*at == ']') break;
        if (count == capacity) {
            capacity *= 2;
            uint32_t *grown =
                (uint32_t *)realloc(values, capacity * sizeof(*values));
            if (!grown) {
                free(values);
                return 0;
            }
            values = grown;
        }
        char *end = NULL;
        unsigned long parsed = strtoul(at, &end, 10);
        if (end == at || parsed > UINT32_MAX) {
            free(values);
            return 0;
        }
        values[count++] = (uint32_t)parsed;
        at = skip_json_space(end);
        if (*at == ',') ++at;
        else if (*at != ']') {
            free(values);
            return 0;
        }
    }
    array->values = values;
    array->count = count;
    return 1;
}

static int parse_row_ids(const char *object, size_t count, uint32_t *rows) {
    const char *at = key_value(object, "ngram_row_ids");
    if (!at) return 0;
    at = skip_json_space(at);
    if (*at != '[') return 0;
    ++at;
    for (size_t i = 0; i < count; ++i) {
        at = skip_json_space(at);
        while (*at == ',' || *at == '[' || *at == ']') {
            ++at;
            at = skip_json_space(at);
        }
        char *end = NULL;
        unsigned long parsed = strtoul(at, &end, 10);
        if (end == at || parsed > UINT32_MAX) return 0;
        rows[i] = (uint32_t)parsed;
        at = end;
    }
    return 1;
}

static int append_partition(uint32_t *parts, size_t *count, size_t capacity,
                            size_t value) {
    if (value == 0 || value > UINT32_MAX || *count == capacity) return 0;
    parts[(*count)++] = (uint32_t)value;
    return 1;
}

static size_t make_random_partition(size_t total, uint32_t seed,
                                    uint32_t *parts, size_t capacity) {
    size_t remaining = total;
    size_t count = 0;
    uint32_t state = seed;
    while (remaining != 0) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        size_t size = (size_t)(state % 5u) + 1u;
        if (size > remaining) size = remaining;
        if (!append_partition(parts, &count, capacity, size)) return 0;
        remaining -= size;
    }
    return count;
}

static int check_partition(const q38_ple_hash_config *config,
                           uint32_t eos_token, const uint32_t *tokens,
                           size_t token_count, const uint32_t *expected,
                           const uint32_t *parts, size_t part_count,
                           const char *case_name, const char *partition_name) {
    q38_ngram_history history;
    q38_ngram_history_reset(&history);
    size_t token_index = 0;
    char error[128];
    for (size_t part = 0; part < part_count; ++part) {
        size_t part_size = parts[part];
        if (part_size == 0 || part_size > token_count - token_index) {
            return fail("invalid partition generated by harness");
        }
        for (size_t i = 0; i < part_size; ++i, ++token_index) {
            uint32_t actual[Q38_PLE_MAX_HEADS];
            if (!q38_ple_ngram_ids_ref(
                    config, &history, tokens[token_index], eos_token, actual,
                    Q38_PLE_MAX_HEADS, error, sizeof(error))) {
                fprintf(stderr, "test_ple_chunking: %s/%s: %s\n", case_name,
                        partition_name, error);
                return 0;
            }
            if (memcmp(actual, expected + token_index * Q38_PLE_MAX_HEADS,
                       Q38_PLE_MAX_HEADS * sizeof(*actual)) != 0) {
                fprintf(stderr,
                        "test_ple_chunking: %s/%s: row IDs differ at token %zu\n",
                        case_name, partition_name, token_index);
                return 0;
            }
            q38_ngram_history_append(&history, tokens[token_index], eos_token);
        }
    }
    return token_index == token_count;
}

static int check_case(const q38_ple_hash_config *config, uint32_t eos_token,
                      const char *object, size_t *partition_checks) {
    u32_array tokens = {0};
    if (!parse_u32_list(object, "tokens", &tokens) || tokens.count == 0) {
        free(tokens.values);
        return fail("golden case has no token stream");
    }
    if (tokens.count > SIZE_MAX / Q38_PLE_MAX_HEADS) {
        free(tokens.values);
        return fail("golden token stream is too large");
    }
    size_t row_count = tokens.count * Q38_PLE_MAX_HEADS;
    uint32_t *expected = (uint32_t *)malloc(row_count * sizeof(*expected));
    if (!expected || !parse_row_ids(object, row_count, expected)) {
        free(expected);
        free(tokens.values);
        return fail("golden case has invalid row IDs");
    }

    const char *name_at = key_value(object, "name");
    char case_name[96] = "unnamed";
    if (name_at) {
        name_at = skip_json_space(name_at);
        if (*name_at == '"') {
            ++name_at;
            size_t i = 0;
            while (name_at[i] && name_at[i] != '"' && i + 1 < sizeof(case_name)) {
                case_name[i] = name_at[i];
                ++i;
            }
            case_name[i] = '\0';
        }
    }

    uint32_t parts[2048];
    size_t count = 0;
    if (!append_partition(parts, &count, 2048, tokens.count) ||
        !check_partition(config, eos_token, tokens.values, tokens.count,
                         expected, parts, count, case_name, "single")) {
        free(expected);
        free(tokens.values);
        return 0;
    }
    ++*partition_checks;

    count = 0;
    for (size_t i = 0; i < tokens.count; ++i) {
        if (!append_partition(parts, &count, 2048, 1)) {
            free(expected);
            free(tokens.values);
            return fail("one-token partition exceeds harness capacity");
        }
    }
    if (!check_partition(config, eos_token, tokens.values, tokens.count,
                         expected, parts, count, case_name, "one-token")) {
        free(expected);
        free(tokens.values);
        return 0;
    }
    ++*partition_checks;

    count = 0;
    if (tokens.count >= 3 &&
        (!append_partition(parts, &count, 2048, 2) ||
         !append_partition(parts, &count, 2048, tokens.count - 2) ||
         !check_partition(config, eos_token, tokens.values, tokens.count,
                          expected, parts, count, case_name, "two-tail"))) {
        free(expected);
        free(tokens.values);
        return 0;
    }
    if (tokens.count >= 3) ++*partition_checks;

    count = 0;
    if (tokens.count >= 8 &&
        (!append_partition(parts, &count, 2048, 3) ||
         !append_partition(parts, &count, 2048, 5) ||
         !append_partition(parts, &count, 2048, tokens.count - 8) ||
         !check_partition(config, eos_token, tokens.values, tokens.count,
                          expected, parts, count, case_name, "three-five-tail"))) {
        free(expected);
        free(tokens.values);
        return 0;
    }
    if (tokens.count >= 8) ++*partition_checks;

    count = 0;
    size_t remaining = tokens.count;
    while (remaining != 0) {
        size_t part = remaining < 4 ? remaining : 4;
        if (!append_partition(parts, &count, 2048, part)) {
            free(expected);
            free(tokens.values);
            return fail("four-token partition exceeds harness capacity");
        }
        remaining -= part;
    }
    if (!check_partition(config, eos_token, tokens.values, tokens.count,
                         expected, parts, count, case_name, "four-token")) {
        free(expected);
        free(tokens.values);
        return 0;
    }
    ++*partition_checks;

    for (uint32_t seed = 1; seed <= 2; ++seed) {
        count = make_random_partition(tokens.count, seed, parts, 2048);
        if (count == 0 ||
            !check_partition(config, eos_token, tokens.values, tokens.count,
                             expected, parts, count, case_name,
                             seed == 1 ? "random-1" : "random-2")) {
            free(expected);
            free(tokens.values);
            return 0;
        }
        ++*partition_checks;
    }

    free(expected);
    free(tokens.values);
    return 1;
}

int main(int argc, char **argv) {
    const char *path =
        argc > 1 ? argv[1] : "artifacts/m4/ple_injection_golden.json";
    file_text file = read_file(path);
    if (!file.text) return fail("cannot read independent golden corpus");
    if (!strstr(file.text, "q38-m4-c06-ple-golden-v1")) {
        free(file.text);
        return fail("unsupported golden format");
    }

    q38_ple_hash_config config;
    memset(&config, 0, sizeof(config));
    const char *config_object = strstr(file.text, "\"config\"");
    uint32_t eos_token = 0;
    if (!config_object ||
        !parse_u32_scalar(config_object, "ngram_size", &config.ngram_size) ||
        !parse_u32_scalar(config_object, "heads_per_ngram",
                          &config.heads_per_ngram) ||
        !parse_u32_scalar(config_object, "eos_token_id", &eos_token) ||
        !parse_u32_array(config_object, "head_offsets", config.head_offsets,
                         Q38_PLE_MAX_HEADS) ||
        !parse_u32_array(config_object, "head_vocab_sizes",
                         config.head_vocab_sizes, Q38_PLE_MAX_HEADS) ||
        !parse_u64_array(config_object, "multipliers", config.multipliers,
                         Q38_PLE_MAX_NGRAM)) {
        free(file.text);
        return fail("invalid PLE hash configuration in golden corpus");
    }
    char error[128];
    if (!q38_ple_hash_config_validate(&config, error, sizeof(error))) {
        free(file.text);
        return fail(error);
    }

    const char *cases = strstr(file.text, "\"cases\"");
    if (!cases) {
        free(file.text);
        return fail("golden corpus has no cases");
    }
    const char *cursor = strchr(cases, '{');
    const char *cases_end = strstr(cases, "],\n  \"checksums\"");
    size_t checked_cases = 0;
    size_t partition_checks = 0;
    while (cursor && (!cases_end || cursor < cases_end)) {
        const char *object_end = strchr(cursor, '}');
        if (!object_end || (cases_end && object_end > cases_end)) {
            free(file.text);
            return fail("unterminated golden case");
        }
        size_t object_length = (size_t)(object_end - cursor);
        char *object = (char *)malloc(object_length + 1);
        if (!object) {
            free(file.text);
            return fail("out of memory");
        }
        memcpy(object, cursor, object_length);
        object[object_length] = '\0';
        if (!check_case(&config, eos_token, object, &partition_checks)) {
            free(object);
            free(file.text);
            return 1;
        }
        free(object);
        ++checked_cases;
        cursor = strchr(object_end + 1, '{');
    }
    free(file.text);
    if (checked_cases == 0) return fail("golden corpus has no parseable cases");
    printf("test_ple_chunking: %zu cases and %zu deterministic partitions passed\n",
           checked_cases, partition_checks);
    puts("test_ple_chunking: row-ID/history invariance passed; activation "
         "invariance is covered by the checkpoint-backed M4-C06 injection probe");
    return 0;
}
