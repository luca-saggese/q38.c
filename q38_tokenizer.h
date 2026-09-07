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
typedef struct {
    double wall_ms;
    double file_read_ms;
    double json_scan_ms;
    double vocab_parse_ms;
    double vocab_string_alloc_ms;
    double vocab_hash_build_ms;
    double merges_parse_ms;
    double merges_string_alloc_ms;
    double merges_index_build_ms;
    double special_token_parse_ms;
    double regex_pretokenizer_init_ms;
    double fingerprint_ms;
    double other_ms;
    uint64_t vocab_entries;
    uint64_t merge_entries;
    uint64_t hash_insertions;
    uint64_t hash_lookups;
    uint64_t string_comparisons;
    uint64_t strlen_calls;
    uint64_t string_copy_bytes;
} q38_tokenizer_profile;
bool q38_tokenizer_init(q38_tokenizer*, const char*, const char*, char*, size_t);
void q38_tokenizer_profile_reset(q38_tokenizer_profile *);
void q38_tokenizer_profile_set(q38_tokenizer_profile *);
void q38_tokenizer_profile_note_hash_insertion(void);
void q38_tokenizer_profile_note_hash_lookup(void);
void q38_tokenizer_profile_note_string_comparison(void);
void q38_tokenizer_profile_note_strlen(void);
void q38_tokenizer_profile_note_string_copy(size_t);
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
