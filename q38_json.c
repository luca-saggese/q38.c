#include "q38_json.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q38_JSON_MAX_DEPTH 128

static void json_ws(const char **cursor) {
    while (cursor && *cursor && isspace((unsigned char)**cursor))
        (*cursor)++;
}

static bool json_fail(char *error, size_t error_len, const char *message) {
    if (error && error_len) snprintf(error, error_len, "%s", message);
    return false;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool json_append_char(char **data, size_t *len, size_t *cap,
                             unsigned char value) {
    if (!data || !len || !cap) return false;
    if (*len + 1 >= *cap) {
        size_t next = *cap ? *cap * 2 : 32;
        if (next <= *len + 1) next = *len + 2;
        char *grown = realloc(*data, next);
        if (!grown) return false;
        *data = grown;
        *cap = next;
    }
    (*data)[(*len)++] = (char)value;
    (*data)[*len] = '\0';
    return true;
}

static bool json_append_utf8(char **data, size_t *len, size_t *cap,
                             uint32_t codepoint) {
    if (codepoint <= 0x7f)
        return json_append_char(data, len, cap, (unsigned char)codepoint);
    if (codepoint <= 0x7ff)
        return json_append_char(data, len, cap, (unsigned char)(0xc0 |
                                  (codepoint >> 6))) &&
               json_append_char(data, len, cap, (unsigned char)(0x80 |
                                  (codepoint & 0x3f)));
    if (codepoint <= 0xffff)
        return json_append_char(data, len, cap, (unsigned char)(0xe0 |
                                  (codepoint >> 12))) &&
               json_append_char(data, len, cap, (unsigned char)(0x80 |
                                  ((codepoint >> 6) & 0x3f))) &&
               json_append_char(data, len, cap, (unsigned char)(0x80 |
                                  (codepoint & 0x3f)));
    if (codepoint <= 0x10ffff)
        return json_append_char(data, len, cap, (unsigned char)(0xf0 |
                                  (codepoint >> 18))) &&
               json_append_char(data, len, cap, (unsigned char)(0x80 |
                                  ((codepoint >> 12) & 0x3f))) &&
               json_append_char(data, len, cap, (unsigned char)(0x80 |
                                  ((codepoint >> 6) & 0x3f))) &&
               json_append_char(data, len, cap, (unsigned char)(0x80 |
                                  (codepoint & 0x3f)));
    return false;
}

static bool json_parse_u16(const char **cursor, uint32_t *value,
                           char *error, size_t error_len) {
    uint32_t result = 0;
    if (!cursor || !*cursor || **cursor != 'u')
        return json_fail(error, error_len, "expected JSON unicode escape");
    (*cursor)++;
    for (int i = 0; i < 4; ++i) {
        const int digit = json_hex((*cursor)[i]);
        if (digit < 0)
            return json_fail(error, error_len, "invalid JSON unicode escape");
        result = (result << 4) | (uint32_t)digit;
    }
    *cursor += 4;
    *value = result;
    return true;
}

bool q38_json_parse_string(const char **cursor, char **out,
                           char *error, size_t error_len) {
    char *result = NULL;
    size_t len = 0, cap = 0;
    if (error && error_len) error[0] = '\0';
    if (!cursor || !*cursor || **cursor != '"')
        return json_fail(error, error_len, "expected JSON string");
    (*cursor)++;
    while (**cursor && **cursor != '"') {
        unsigned char value = (unsigned char)*(*cursor)++;
        if (value != '\\') {
            if (value < 0x20 ||
                !json_append_char(&result, &len, &cap, value))
                goto fail;
            continue;
        }
        if (!**cursor) goto fail;
        value = (unsigned char)*(*cursor)++;
        if (value == 'u') {
            (*cursor)--;
            uint32_t codepoint;
            if (!json_parse_u16(cursor, &codepoint, error, error_len))
                goto fail;
            if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                const char *lookahead = *cursor;
                if (lookahead[0] == '\\' && lookahead[1] == 'u') {
                    uint32_t low;
                    lookahead++;
                    if (!json_parse_u16(&lookahead, &low, error, error_len) ||
                        low < 0xdc00 || low > 0xdfff)
                        goto fail;
                    *cursor = lookahead;
                    codepoint = 0x10000 +
                                ((codepoint - 0xd800) << 10) +
                                (low - 0xdc00);
                } else {
                    goto fail;
                }
            } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                goto fail;
            }
            if (!json_append_utf8(&result, &len, &cap, codepoint))
                goto fail;
            continue;
        }
        switch (value) {
        case '"': case '\\': case '/':
            if (!json_append_char(&result, &len, &cap, value)) goto fail;
            break;
        case 'b':
            if (!json_append_char(&result, &len, &cap, '\b')) goto fail;
            break;
        case 'f':
            if (!json_append_char(&result, &len, &cap, '\f')) goto fail;
            break;
        case 'n':
            if (!json_append_char(&result, &len, &cap, '\n')) goto fail;
            break;
        case 'r':
            if (!json_append_char(&result, &len, &cap, '\r')) goto fail;
            break;
        case 't':
            if (!json_append_char(&result, &len, &cap, '\t')) goto fail;
            break;
        default:
            goto fail;
        }
    }
    if (**cursor != '"') goto fail;
    (*cursor)++;
    if (!result) {
        result = malloc(1);
        if (!result) goto fail;
        result[0] = '\0';
    }
    *out = result;
    return true;
fail:
    free(result);
    if (error && error_len && !error[0])
        snprintf(error, error_len, "invalid JSON string");
    return false;
}

static bool json_skip_string(const char **cursor, char *error,
                             size_t error_len) {
    char *value = NULL;
    const bool ok = q38_json_parse_string(cursor, &value, error, error_len);
    free(value);
    return ok;
}

static bool json_skip_value_depth(const char **cursor, int depth,
                                  char *error, size_t error_len) {
    if (depth > Q38_JSON_MAX_DEPTH)
        return json_fail(error, error_len, "JSON nesting limit exceeded");
    json_ws(cursor);
    if (!cursor || !*cursor || !**cursor)
        return json_fail(error, error_len, "unexpected end of JSON");
    if (**cursor == '"') return json_skip_string(cursor, error, error_len);
    if (**cursor == '{') {
        (*cursor)++;
        json_ws(cursor);
        if (**cursor == '}') {
            (*cursor)++;
            return true;
        }
        while (**cursor) {
            if (!json_skip_string(cursor, error, error_len))
                return false;
            json_ws(cursor);
            if (**cursor != ':')
                return json_fail(error, error_len, "expected JSON object colon");
            (*cursor)++;
            if (!json_skip_value_depth(cursor, depth + 1, error, error_len))
                return false;
            json_ws(cursor);
            if (**cursor == '}') {
                (*cursor)++;
                return true;
            }
            if (**cursor != ',')
                return json_fail(error, error_len, "expected JSON object comma");
            (*cursor)++;
            json_ws(cursor);
        }
        return json_fail(error, error_len, "unterminated JSON object");
    }
    if (**cursor == '[') {
        (*cursor)++;
        json_ws(cursor);
        if (**cursor == ']') {
            (*cursor)++;
            return true;
        }
        while (**cursor) {
            if (!json_skip_value_depth(cursor, depth + 1, error, error_len))
                return false;
            json_ws(cursor);
            if (**cursor == ']') {
                (*cursor)++;
                return true;
            }
            if (**cursor != ',')
                return json_fail(error, error_len, "expected JSON array comma");
            (*cursor)++;
            json_ws(cursor);
        }
        return json_fail(error, error_len, "unterminated JSON array");
    }
    if (!strncmp(*cursor, "true", 4)) {
        *cursor += 4;
        return true;
    }
    if (!strncmp(*cursor, "false", 5)) {
        *cursor += 5;
        return true;
    }
    if (!strncmp(*cursor, "null", 4)) {
        *cursor += 4;
        return true;
    }
    char *end = NULL;
    errno = 0;
    (void)strtod(*cursor, &end);
    if (end != *cursor && errno != ERANGE) {
        *cursor = end;
        return true;
    }
    return json_fail(error, error_len, "invalid JSON value");
}

bool q38_json_skip_value(const char **cursor, char *error, size_t error_len) {
    if (error && error_len) error[0] = '\0';
    return json_skip_value_depth(cursor, 0, error, error_len);
}

bool q38_json_raw_value(const char **cursor, char **out,
                        char *error, size_t error_len) {
    const char *start;
    const char *end;
    if (error && error_len) error[0] = '\0';
    if (!cursor || !*cursor || !out)
        return json_fail(error, error_len, "invalid JSON raw value arguments");
    json_ws(cursor);
    start = *cursor;
    if (!q38_json_skip_value(cursor, error, error_len)) return false;
    end = *cursor;
    while (end > start && isspace((unsigned char)end[-1])) end--;
    *out = malloc((size_t)(end - start) + 1);
    if (!*out) return json_fail(error, error_len, "JSON raw value allocation failed");
    memcpy(*out, start, (size_t)(end - start));
    (*out)[end - start] = '\0';
    return true;
}

static bool json_object_open(const char **cursor, char *error,
                             size_t error_len) {
    json_ws(cursor);
    if (!cursor || !*cursor || **cursor != '{')
        return json_fail(error, error_len, "expected JSON object");
    (*cursor)++;
    return true;
}

bool q38_json_object_field(const char *object, const char *key, char **raw,
                           char *error, size_t error_len) {
    const char *cursor = object;
    if (error && error_len) error[0] = '\0';
    if (!key || !raw || !json_object_open(&cursor, error, error_len))
        return false;
    *raw = NULL;
    json_ws(&cursor);
    if (*cursor == '}') {
        cursor++;
        return true;
    }
    while (*cursor) {
        char *name = NULL;
        if (!q38_json_parse_string(&cursor, &name, error, error_len))
            return false;
        json_ws(&cursor);
        if (*cursor != ':') {
            free(name);
            return json_fail(error, error_len, "expected JSON object colon");
        }
        cursor++;
        json_ws(&cursor);
        if (!strcmp(name, key)) {
            free(name);
            if (!q38_json_raw_value(&cursor, raw, error, error_len))
                return false;
        } else {
            free(name);
            if (!q38_json_skip_value(&cursor, error, error_len))
                return false;
        }
        json_ws(&cursor);
        if (*cursor == '}') {
            cursor++;
            return true;
        }
        if (*cursor != ',')
            return json_fail(error, error_len, "expected JSON object comma");
        cursor++;
        json_ws(&cursor);
    }
    return json_fail(error, error_len, "unterminated JSON object");
}

bool q38_json_get_string(const char *object, const char *key, char **value,
                         char *error, size_t error_len) {
    char *raw = NULL;
    const char *cursor;
    if (!value) return json_fail(error, error_len, "JSON output is null");
    *value = NULL;
    if (!q38_json_object_field(object, key, &raw, error, error_len))
        return false;
    if (!raw) return true;
    cursor = raw;
    if (*cursor != '"' ||
        !q38_json_parse_string(&cursor, value, error, error_len)) {
        free(raw);
        return json_fail(error, error_len, "JSON field is not a string");
    }
    json_ws(&cursor);
    const bool complete = *cursor == '\0';
    free(raw);
    return complete || json_fail(error, error_len, "trailing JSON string data");
}

bool q38_json_get_bool(const char *object, const char *key, bool *value,
                       char *error, size_t error_len) {
    char *raw = NULL;
    if (!value) return json_fail(error, error_len, "JSON output is null");
    if (!q38_json_object_field(object, key, &raw, error, error_len))
        return false;
    if (!raw) return true;
    if (!strcmp(raw, "true")) *value = true;
    else if (!strcmp(raw, "false")) *value = false;
    else {
        free(raw);
        return json_fail(error, error_len, "JSON field is not boolean");
    }
    free(raw);
    return true;
}

bool q38_json_get_number(const char *object, const char *key, double *value,
                         char *error, size_t error_len) {
    char *raw = NULL;
    char *end = NULL;
    if (!value) return json_fail(error, error_len, "JSON output is null");
    if (!q38_json_object_field(object, key, &raw, error, error_len))
        return false;
    if (!raw) return true;
    errno = 0;
    *value = strtod(raw, &end);
    if (errno == ERANGE || end == raw || *end) {
        free(raw);
        return json_fail(error, error_len, "JSON field is not finite number");
    }
    free(raw);
    return true;
}

bool q38_json_array_each(const char *array, q38_json_item_cb callback,
                         void *user, char *error, size_t error_len) {
    const char *cursor = array;
    json_ws(&cursor);
    if (!cursor || *cursor != '[')
        return json_fail(error, error_len, "expected JSON array");
    cursor++;
    json_ws(&cursor);
    if (*cursor == ']') return true;
    while (*cursor) {
        const char *start = cursor;
        if (!q38_json_skip_value(&cursor, error, error_len)) return false;
        const size_t len = (size_t)(cursor - start);
        if (callback && !callback(start, len, user, error, error_len))
            return false;
        json_ws(&cursor);
        if (*cursor == ']') return true;
        if (*cursor != ',')
            return json_fail(error, error_len, "expected JSON array comma");
        cursor++;
        json_ws(&cursor);
    }
    return json_fail(error, error_len, "unterminated JSON array");
}

bool q38_json_object_each(const char *object, q38_json_item_cb callback,
                          void *user, char *error, size_t error_len) {
    const char *cursor = object;
    json_ws(&cursor);
    if (!cursor || *cursor != '{')
        return json_fail(error, error_len, "expected JSON object");
    cursor++;
    json_ws(&cursor);
    if (*cursor == '}') return true;
    while (*cursor) {
        char *name = NULL;
        const char *start;
        if (!q38_json_parse_string(&cursor, &name, error, error_len))
            return false;
        free(name);
        json_ws(&cursor);
        if (*cursor != ':')
            return json_fail(error, error_len, "expected JSON object colon");
        cursor++;
        json_ws(&cursor);
        start = cursor;
        if (!q38_json_skip_value(&cursor, error, error_len)) return false;
        if (callback && !callback(start, (size_t)(cursor - start), user,
                                  error, error_len))
            return false;
        json_ws(&cursor);
        if (*cursor == '}') return true;
        if (*cursor != ',')
            return json_fail(error, error_len, "expected JSON object comma");
        cursor++;
        json_ws(&cursor);
    }
    return json_fail(error, error_len, "unterminated JSON object");
}

bool q38_json_raw_is_null(const char *raw) {
    return raw && !strcmp(raw, "null");
}
