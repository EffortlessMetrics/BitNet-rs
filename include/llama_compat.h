/*
 * llama.cpp Compatible C API for bitnet-rs
 *
 * This header provides a drop-in replacement for llama.cpp's C API.
 * Simply replace #include "llama.h" with #include "llama_compat.h"
 * and link against libbitnet_ffi instead of libllama.
 */

#ifndef LLAMA_COMPAT_H
#define LLAMA_COMPAT_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types matching llama.cpp
typedef struct llama_model llama_model;
typedef struct llama_context llama_context;

// Model parameters - matches llama_model_params
struct llama_model_params {
    int32_t n_gpu_layers;          // Number of layers to offload to GPU
    int32_t main_gpu;              // Main GPU device id
    const float* tensor_split;     // Tensor split across multiple GPUs

    // Progress callback
    bool (*progress_callback)(float progress, void* ctx);
    void* progress_callback_user_data;

    // Override key-value pairs
    const void* kv_overrides;

    bool vocab_only;               // Load only vocabulary
    bool use_mmap;                 // Use memory mapping
    bool use_mlock;                // Lock model in RAM
};

// Context parameters - matches llama_context_params
struct llama_context_params {
    uint32_t seed;                 // RNG seed
    uint32_t n_ctx;                // Text context size
    uint32_t n_batch;              // Batch size for prompt processing
    uint32_t n_threads;            // Number of threads
    uint32_t n_threads_batch;      // Number of threads for batch processing

    // RoPE parameters
    int32_t rope_scaling_type;
    float rope_freq_base;
    float rope_freq_scale;

    // YaRN parameters
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;

    // Callback for eval
    bool (*cb_eval)(void* ctx, bool ask);
    void* cb_eval_user_data;

    // KV cache type
    int32_t type_k;
    int32_t type_v;

    // Options
    bool logits_all;               // Return logits for all tokens
    bool embedding;                // Embedding mode
    bool offload_kqv;              // Offload KQV ops to GPU
};

// Default parameters
struct llama_model_params llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);

// Model loading
llama_model* llama_load_model_from_file(
    const char* path_model,
    struct llama_model_params params
);

void llama_free_model(llama_model* model);

// Context management
llama_context* llama_new_context_with_model(
    llama_model* model,
    struct llama_context_params params
);

void llama_free(llama_context* ctx);

// Tokenization
// Returns number of tokens on success, negative on error
// If |return| > n_max_tokens, call again with tokens = NULL to get size
int32_t llama_tokenize(
    const llama_model* model,
    const char* text,
    int32_t text_len,
    int32_t* tokens,
    int32_t n_max_tokens,
    bool add_bos,
    bool special
);

// Token to string conversion
int32_t llama_token_to_piece(
    const llama_model* model,
    int32_t token,
    char* buf,
    int32_t length
);

// Special tokens
int32_t llama_token_bos(const llama_model* model);  // Beginning of sentence
int32_t llama_token_eos(const llama_model* model);  // End of sentence
int32_t llama_token_nl(const llama_model* model);   // Newline
int32_t llama_token_prefix(const llama_model* model); // Prefix
int32_t llama_token_middle(const llama_model* model); // Middle
int32_t llama_token_suffix(const llama_model* model); // Suffix
int32_t llama_token_eot(const llama_model* model);   // End of turn

// Evaluation
int llama_eval(
    llama_context* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t n_past,
    int32_t n_threads
);

// Get logits
float* llama_get_logits(llama_context* ctx);
float* llama_get_logits_ith(llama_context* ctx, int32_t i);

// Get embeddings
float* llama_get_embeddings(llama_context* ctx);

// Vocabulary
int32_t llama_n_vocab(const llama_model* model);
int32_t llama_n_ctx(const llama_context* ctx);
int32_t llama_n_ctx_train(const llama_model* model);
int32_t llama_n_embd(const llama_model* model);

// Model metadata
int32_t llama_model_n_params(const llama_model* model);
int64_t llama_model_size(const llama_model* model);
int32_t llama_model_n_layers(const llama_model* model);

// Model description
int32_t llama_model_desc(const llama_model* model, char* buf, size_t buf_size);

// KV cache management
size_t llama_get_state_size(const llama_context* ctx);
size_t llama_copy_state_data(llama_context* ctx, uint8_t* dst);
size_t llama_set_state_data(llama_context* ctx, uint8_t* src);

// Sampling (simplified - extend as needed)
int32_t llama_sample_token_greedy(
    llama_context* ctx,
    const float* logits
);

// Performance information
void llama_print_timings(llama_context* ctx);
void llama_reset_timings(llama_context* ctx);

// System information
const char* llama_print_system_info(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_COMPAT_H
