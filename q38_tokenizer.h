#ifndef Q38_TOKENIZER_H
#define Q38_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t token_count;
    uint32_t *tokens;
    uint64_t prompt_hash;
    uint64_t model_hash;
} q38_token_batch;

typedef struct {
    char *model_dir;
    char *reference_script;
} q38_tokenizer;

/*
 * The frozen Python tokenizer is an intentional compatibility bridge: the
 * tokenizer JSON uses a large Unicode regex/BPE model, while q38 has no JSON
 * or regex runtime dependency. All IDs still come from the local reference.
 */
bool q38_tokenizer_init(q38_tokenizer *tokenizer, const char *model_dir,
                        const char *reference_script, char *error,
                        size_t error_len);
void q38_tokenizer_destroy(q38_tokenizer *tokenizer);

bool q38_tokenizer_encode(const q38_tokenizer *tokenizer, const char *text,
                          bool add_special_tokens, q38_token_batch *out,
                          char *error, size_t error_len);

bool q38_tokenizer_encode_chat_json(const q38_tokenizer *tokenizer,
                                    const char *messages_json,
                                    bool add_generation_prompt,
                                    bool enable_thinking,
                                    q38_token_batch *out, char *error,
                                    size_t error_len);

void q38_token_batch_free(q38_token_batch *batch);

#ifdef __cplusplus
}
#endif

#endif
