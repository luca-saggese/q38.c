#ifndef Q38_TOKENIZER_H
#define Q38_TOKENIZER_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { uint32_t token_count; uint32_t *tokens; uint64_t prompt_hash, model_hash; } q38_token_batch;
typedef struct q38_vocab_entry q38_vocab_entry;
typedef struct q38_merge_entry q38_merge_entry;
typedef struct {
    char *model_dir;
    q38_vocab_entry *vocab; size_t vocab_cap, vocab_count;
    q38_merge_entry *merges; size_t merge_cap, merge_count;
    char **special_text; uint32_t *special_id; size_t special_count;
    uint32_t bos_id, eos_id;
    char *chat_template;
} q38_tokenizer;
bool q38_tokenizer_init(q38_tokenizer*, const char*, const char*, char*, size_t);
bool q38_tokenizer_verify_specials(const q38_tokenizer*, char*, size_t);
void q38_tokenizer_destroy(q38_tokenizer*);
bool q38_tokenizer_encode(const q38_tokenizer*, const char*, bool, q38_token_batch*, char*, size_t);
bool q38_tokenizer_encode_chat_json(const q38_tokenizer*, const char*, bool, bool, q38_token_batch*, char*, size_t);
bool q38_tokenizer_decode(const q38_tokenizer*, const uint32_t*, size_t,
                          char**, size_t*, char*, size_t);
void q38_token_batch_free(q38_token_batch*);
#ifdef __cplusplus
}
#endif
#endif
