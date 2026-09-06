#include "q38_server_protocol.h"
#include "q38_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

int main(void) {
    char error[256] = {0};
    q38_server_request request;
    const char *chat =
        "{"
        "\"model\":\"qwen3.8-flash-next\","
        "\"stream\":true,"
        "\"max_tokens\":32,"
        "\"messages\":["
          "{\"role\":\"system\",\"content\":\"be concise\"},"
          "{\"role\":\"user\",\"content\":["
            "{\"type\":\"text\",\"text\":\"hello\"},"
            "{\"type\":\"image_url\",\"image_url\":"
              "{\"url\":\"data:image/png;base64,AAAA\"}}"
          "]},"
          "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call-1\","
            "\"type\":\"function\",\"function\":{\"name\":\"weather\","
            "\"arguments\":\"{\\\"city\\\":\\\"Rome\\\"}\"}}]}"
        "],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
          "\"name\":\"weather\",\"description\":\"forecast\","
          "\"parameters\":{\"type\":\"object\",\"properties\":{}}"
        "}}]"
        "}";
    q38_server_request_init(&request);
    check(q38_server_parse_request(
              Q38_SERVER_ENDPOINT_CHAT_COMPLETIONS, chat, &request,
              error, sizeof(error)), error);
    check(request.api == Q38_SERVER_API_OPENAI, "OpenAI API selection");
    check(request.messages.count == 3, "chat message count");
    check(request.messages.items[1].image_count == 1, "data URI image");
    check(request.has_image, "image capability marker");
    check(request.tools.count == 1 &&
              !strcmp(request.tools.items[0].name, "weather"),
          "function tool parsing");
    check(request.messages.items[2].tool_calls.count == 1 &&
              !strcmp(request.messages.items[2].tool_calls.items[0].id,
                      "call-1"),
          "OpenAI tool-call history parsing");
    check(request.stream && request.max_tokens == 32, "common options");
    q38_server_request_free(&request);

    const char *responses =
        "{"
        "\"model\":\"qwen3.8-flash-next\","
        "\"instructions\":\"follow policy\","
        "\"input\":["
          "{\"type\":\"input_text\",\"text\":\"hello\"},"
          "{\"type\":\"function_call\",\"call_id\":\"c1\","
            "\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Rome\\\"}\"},"
          "{\"type\":\"function_call_output\",\"call_id\":\"c1\","
            "\"output\":\"sunny\"}"
        "],"
        "\"reasoning\":{\"effort\":\"low\"}"
        "}";
    int parsed = q38_server_parse_request(
        Q38_SERVER_ENDPOINT_RESPONSES, responses, &request,
        error, sizeof(error));
    if (!parsed) fprintf(stderr, "responses parse error: %s\n", error);
    check(parsed, "Responses request parsing");
    check(request.api == Q38_SERVER_API_RESPONSES, "Responses API selection");
    check(request.messages.count == 4, "Responses input conversion");
    check(request.messages.items[2].tool_calls.count == 1,
          "Responses function call conversion");
    check(request.messages.items[3].tool_call_id &&
              !strcmp(request.messages.items[3].tool_call_id, "c1"),
          "Responses function output conversion");
    check(request.thinking && request.reasoning_effort &&
              !strcmp(request.reasoning_effort, "low"),
          "Responses reasoning conversion");
    q38_server_request_free(&request);

    const char *anthropic =
        "{"
        "\"model\":\"qwen3.8-flash-next\","
        "\"max_tokens\":12,"
        "\"system\":\"system text\","
        "\"messages\":[{\"role\":\"user\",\"content\":["
          "{\"type\":\"text\",\"text\":\"hello\"},"
          "{\"type\":\"tool_use\",\"id\":\"u1\",\"name\":\"weather\","
            "\"input\":{\"city\":\"Rome\"}}"
        "]}"
        "]"
        "}";
    parsed = q38_server_parse_request(
        Q38_SERVER_ENDPOINT_MESSAGES, anthropic, &request,
        error, sizeof(error));
    if (!parsed) fprintf(stderr, "anthropic parse error: %s\n", error);
    check(parsed, "Anthropic request parsing");
    check(request.api == Q38_SERVER_API_ANTHROPIC, "Anthropic API selection");
    check(request.messages.count == 2, "Anthropic system insertion");
    check(request.messages.items[1].tool_calls.count == 1,
          "Anthropic tool-use conversion");
    q38_server_request_free(&request);

    if (failures) return 1;
    puts("test_q38_server_protocol: all tests passed");
    return 0;
}
