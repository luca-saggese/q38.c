#include "q38_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static bool count_item(const char *raw, size_t len, void *user,
                       char *error, size_t error_len) {
    size_t *count = user;
    (void)error;
    (void)error_len;
    if (!raw || !len) return false;
    (*count)++;
    return true;
}

int main(void) {
    char error[256] = {0};
    const char *object =
        "{\"text\":\"hello\\nworld\",\"number\":1.5,"
        "\"flag\":true,\"nested\":{\"x\":[1,{\"y\":\"z\"}]},"
        "\"unicode\":\"\\ud83d\\ude80\"}";
    char *text = NULL;
    char *raw = NULL;
    double number = 0.0;
    bool flag = false;
    size_t count = 0;
    check(q38_json_get_string(object, "text", &text, error, sizeof(error)),
          "JSON string field");
    check(text && !strcmp(text, "hello\nworld"), "JSON escaped newline");
    free(text);
    check(q38_json_get_number(object, "number", &number, error, sizeof(error)) &&
              number == 1.5,
          "JSON number field");
    check(q38_json_get_bool(object, "flag", &flag, error, sizeof(error)) &&
              flag,
          "JSON boolean field");
    check(q38_json_object_field(object, "nested", &raw, error, sizeof(error)) &&
              raw && strstr(raw, "\"y\":\"z\""),
          "JSON nested raw field");
    free(raw);
    check(q38_json_array_each("[1,{\"x\":2},[3]]", count_item, &count,
                              error, sizeof(error)) && count == 3,
          "JSON array iteration");
    check(q38_json_get_string(object, "unicode", &text, error, sizeof(error)) &&
              text && !strcmp(text, "\xF0\x9F\x9A\x80"),
          "JSON surrogate pair");
    free(text);
    const char *malformed = "{";
    check(!q38_json_skip_value(&malformed, error, sizeof(error)),
          "JSON malformed object rejection");
    if (failures) return 1;
    puts("test_q38_json: all tests passed");
    return 0;
}
