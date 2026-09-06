#ifndef Q38_JSON_H
#define Q38_JSON_H

#include <stdbool.h>
#include <stddef.h>

typedef bool (*q38_json_item_cb)(const char *raw, size_t raw_len,
                                 void *user, char *error, size_t error_len);

bool q38_json_parse_string(const char **cursor, char **out,
                           char *error, size_t error_len);
bool q38_json_skip_value(const char **cursor, char *error, size_t error_len);
bool q38_json_raw_value(const char **cursor, char **out,
                        char *error, size_t error_len);

bool q38_json_object_field(const char *object, const char *key, char **raw,
                           char *error, size_t error_len);
bool q38_json_get_string(const char *object, const char *key, char **value,
                         char *error, size_t error_len);
bool q38_json_get_bool(const char *object, const char *key, bool *value,
                       char *error, size_t error_len);
bool q38_json_get_number(const char *object, const char *key, double *value,
                         char *error, size_t error_len);

bool q38_json_array_each(const char *array, q38_json_item_cb callback,
                         void *user, char *error, size_t error_len);
bool q38_json_object_each(const char *object, q38_json_item_cb callback,
                          void *user, char *error, size_t error_len);

bool q38_json_raw_is_null(const char *raw);

#endif
