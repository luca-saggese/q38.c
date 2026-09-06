#include "q38_prompt.h"
#include "q38_server_engine.h"

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

static bool collect_event(const q38_server_event *event, void *user,
                          char *error, size_t error_len) {
    size_t *count = user;
    (void)error;
    (void)error_len;
    if (event && count) (*count)++;
    return true;
}

int main(void) {
    char error[256] = {0};
    q38_server_request request;
    q38_server_request_init(&request);
    request.thinking = true;
    request.prompt = strdup("hello");
    check(request.prompt != NULL, "request prompt allocation");

    char *rendered = NULL;
    size_t rendered_len = 0;
    check(q38_prompt_render_chat(&request, &rendered, &rendered_len,
                                 error, sizeof(error)),
          "Qwen prompt rendering");
    check(rendered && rendered_len && strstr(rendered, "<|im_start|>"),
          "Qwen prompt has native chat boundaries");
    free(rendered);
    q38_server_tool_call extracted = {0};
    check(q38_prompt_extract_tool_call(
              "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Rome\"}}"
              "</tool_call>",
              strlen("<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Rome\"}}"
                     "</tool_call>"),
              &extracted, error, sizeof(error)),
          "Qwen tool-call extraction");
    check(extracted.name && !strcmp(extracted.name, "weather") &&
              extracted.arguments_json &&
              strstr(extracted.arguments_json, "\"city\":\"Rome\""),
          "Qwen tool-call fields");
    free(extracted.name);
    free(extracted.arguments_json);

    q38_server_engine *engine = q38_server_mock_engine_create(NULL, error,
                                                                sizeof(error));
    check(engine != NULL, "mock engine creation");
    size_t event_count = 0;
    q38_server_usage usage = {0};
    check(q38_server_engine_generate(engine, &request, collect_event,
                                     &event_count, &usage, error,
                                     sizeof(error)) == 0,
          "mock generation");
    check(event_count == 3, "mock reasoning/text/done event count");
    check(usage.completion_tokens != 0, "mock usage accounting");
    q38_server_engine_destroy(engine);
    q38_server_request_free(&request);
    if (failures) return 1;
    puts("test_q38_server_engine: all tests passed");
    return 0;
}
