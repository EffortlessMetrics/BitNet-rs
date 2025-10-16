#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bitnet_model bitnet_model_t;
typedef struct bitnet_ctx   bitnet_ctx_t;

typedef struct {
  int32_t n_ctx;
  int32_t n_threads;
  int32_t seed;
  float   rope_freq;
} bitnet_params_t;

bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path);
void            bitnet_model_free(bitnet_model_t*);

bitnet_ctx_t*   bitnet_context_new(bitnet_model_t*, const bitnet_params_t*);
void            bitnet_context_free(bitnet_ctx_t*);

int bitnet_tokenize(bitnet_model_t*, const char* text, int add_bos, int parse_special,
                    int32_t* out_ids, int out_cap);

int bitnet_eval(bitnet_ctx_t*, const int32_t* ids, int n_ids,
                float* logits_out, int logits_cap);

int bitnet_decode_greedy(bitnet_ctx_t*, int32_t* io_ids, int max_new_tokens,
                         int eos_id, float temperature);

#ifdef __cplusplus
}
#endif
