#include "q38_tokenizer.h"

#include <stdio.h>
#include <string.h>

static int check_text(const q38_tokenizer *tokenizer, const char *name,
                      const char *text, const uint32_t *expected,
                      uint32_t expected_count) {
    q38_token_batch got = {0};
    char error[256];
    if (!q38_tokenizer_encode(tokenizer, text, false, &got, error,
                              sizeof(error))) {
        fprintf(stderr, "%s: encode failed: %s\n", name, error);
        return 1;
    }
    int failed = got.token_count != expected_count ||
        memcmp(got.tokens, expected, expected_count * sizeof(*expected)) != 0;
    if (failed) {
        fprintf(stderr, "%s: expected %u IDs, got %u\n", name, expected_count,
                got.token_count);
    }
    q38_token_batch_free(&got);
    return failed;
}

static int check_chat(const q38_tokenizer *tokenizer, const char *name,
                      const char *messages, const uint32_t *expected,
                      uint32_t expected_count, bool add_generation_prompt,
                      bool thinking) {
    q38_token_batch got = {0};
    char error[256];
    if (!q38_tokenizer_encode_chat_json(tokenizer, messages,
                                        add_generation_prompt, thinking,
                                        &got, error, sizeof(error))) {
        fprintf(stderr, "%s: chat encode failed: %s\n", name, error);
        return 1;
    }
    int failed = got.token_count != expected_count ||
        memcmp(got.tokens, expected, expected_count * sizeof(*expected)) != 0;
    if (failed)
        fprintf(stderr, "%s: expected %u IDs, got %u\n", name, expected_count,
                got.token_count);
    q38_token_batch_free(&got);
    return failed;
}

int main(void) {
    static const uint32_t ascii[] = {9419, 11, 1814, 0, 198, 15207, 1500, 13};
    static const uint32_t whitespace[] = {220, 6187, 1838, 26453, 198, 1021, 256};
    static const uint32_t unicode[] = {34, 2562, 4311, 44311, 31441, 1892, 220,
                                       109266, 96748, 26594, 3825};
    static const uint32_t repetitive[] = {
        13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        37730, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        37730, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        37730, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        37730, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        37730, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        37730, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        37730, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290, 13290,
        220};
    static const uint32_t special[] = {248045, 74455, 198, 248068, 198};
    static const uint32_t tool[] = {4754, 591, 3147, 20377, 2129, 15889, 21624,
                                    80, 3147, 27, 87, 5608, 29958, 198, 248058,
                                    198, 248059};
    static const uint32_t chat[] = {248045, 8678, 198, 2523, 513, 61446, 13,
                                    248046, 198, 248045, 846, 198, 9419, 30,
                                    248046, 198, 248045, 74455, 198, 248068,
                                    271, 248069, 271};
    static const uint32_t chat_empty[] = {248045, 846, 198, 34, 21817, 248046,
                                          198, 248045, 74455, 198, 248068,
                                          271, 248069, 271};
    static const uint32_t chat_assistant[] = {
        248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13,
        5044, 1683, 15060, 1472, 279, 3274, 11, 9307, 1328, 30800, 11, 2814,
        47675, 25605, 11, 321, 60445, 55404, 11, 27224, 11, 321, 30246, 303,
        279, 1534, 4087, 13, 248046, 198, 248045, 846, 198, 44712, 220, 17,
        10, 17, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        19, 248046, 198};
    static const uint32_t chat_tool[] = {
        248045, 846, 198, 6994, 18054, 13, 248046, 198, 248045, 74455, 198, 248068,
        271, 248069, 271, 248058, 198, 27, 1628, 28, 20377, 29, 198, 27,
        15704, 60922, 29, 198, 13290, 198, 510, 15704, 29, 198, 510, 1628,
        29, 198, 248059, 248046, 198, 248045, 846, 198, 248066, 198, 4754, 547,
        763, 1802, 92, 198, 248067, 248046, 198};
    q38_tokenizer tokenizer;
    char error[256];
    if (!q38_tokenizer_init(&tokenizer, "/home/lvx/q38model",
                            "tools/q38_tokenizer_ref.py", error, sizeof(error))) {
        fprintf(stderr, "tokenizer init failed: %s\n", error);
        return 1;
    }
    int failed = 0;
    failed |= check_text(&tokenizer, "ascii", "Hello, world!\nSecond line.",
                         ascii, sizeof(ascii) / sizeof(*ascii));
    failed |= check_text(&tokenizer, "whitespace", "  leading\tspaces\nline  ",
                         whitespace, sizeof(whitespace) / sizeof(*whitespace));
    failed |= check_text(&tokenizer, "unicode",
                         "Caffè déjà vu — 你好世界 🙂 e\u0301", unicode,
                         sizeof(unicode) / sizeof(*unicode));
    failed |= check_text(&tokenizer, "repetitive",
                         "abcabcabcabcabcabcabcabcabcabc "
                         "abcabcabcabcabcabcabcabcabcabc "
                         "abcabcabcabcabcabcabcabcabcabc "
                         "abcabcabcabcabcabcabcabcabcabc "
                         "abcabcabcabcabcabcabcabcabcabc "
                         "abcabcabcabcabcabcabcabcabcabc "
                         "abcabcabcabcabcabcabcabcabcabc "
                         "abcabcabcabcabcabcabcabcabcabc ",
                         repetitive, sizeof(repetitive) / sizeof(*repetitive));
    failed |= check_text(&tokenizer, "special",
                         "<|im_start|>assistant\n<think>\n", special,
                         sizeof(special) / sizeof(*special));
    failed |= check_text(&tokenizer, "tool",
                         "{\"name\":\"lookup\",\"arguments\":{\"q\":\"<x>&\"}}\n"
                         "<tool_call>\n</tool_call>",
                         tool, sizeof(tool) / sizeof(*tool));
    failed |= check_chat(
        &tokenizer, "chat",
        "[{\"role\":\"system\",\"content\":\"You are concise.\"},"
        "{\"role\":\"user\",\"content\":\"Hello?\"}]",
        chat, sizeof(chat) / sizeof(*chat), true, false);
    failed |= check_chat(
        &tokenizer, "chat_empty",
        "[{\"role\":\"system\",\"content\":\"\"},"
        "{\"role\":\"user\",\"content\":\"Ciao\"}]",
        chat_empty, sizeof(chat_empty) / sizeof(*chat_empty), true, false);
    failed |= check_chat(
        &tokenizer, "chat_assistant",
        "[{\"role\":\"user\",\"content\":\"Compute 2+2.\"},"
        "{\"role\":\"assistant\",\"content\":\"4\"}]",
        chat_assistant, sizeof(chat_assistant) / sizeof(*chat_assistant), false,
        true);
    failed |= check_chat(
        &tokenizer, "chat_tool",
        "[{\"role\":\"user\",\"content\":\"Call lookup.\"},"
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":"
        "{\"name\":\"lookup\",\"arguments\":{\"q\":\"abc\"}}}]},"
        "{\"role\":\"tool\",\"content\":\"{\\\"ok\\\":true}\"}]",
        chat_tool, sizeof(chat_tool) / sizeof(*chat_tool), false, false);
    q38_tokenizer_destroy(&tokenizer);
    if (failed) return 1;
    puts("test_m2_tokenizer: frozen raw and chat token IDs passed");
    return 0;
}
