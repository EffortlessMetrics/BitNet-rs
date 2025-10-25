// crossval/src/bitnet_cpp_wrapper.cc
//
// Purpose: C FFI shim for BitNet.cpp cross-validation
//
// Assumptions:
// - STUB mode (BITNET_STUB defined): Returns actionable errors, no external dependencies
// - AVAILABLE mode (BITNET_AVAILABLE defined): Provides full BitNet.cpp integration
// - Two-pass buffer negotiation: NULL output pointers trigger size-only queries
// - Per-call context: Model loaded/unloaded in each call (no session state for MVP)
// - C ABI: extern "C" for Rust FFI compatibility
//
// Error handling:
// - Return 0 on success, -1 on error
// - Always NUL-terminate error strings
// - Always set output size parameters (even on error, set to 0)

#include <cstdint>
#include <cstring>
#include <cstdio>

// Conditional includes for AVAILABLE mode
#ifdef BITNET_AVAILABLE
// TODO: Add actual BitNet.cpp headers when wiring implementation
// #include "bitnet.h"
// #include "llama.h"
#endif

extern "C" {

/// Tokenize text using BitNet.cpp tokenizer
///
/// Two-pass pattern:
/// 1. Call with out_tokens=NULL to query size -> fills out_len, returns 0
/// 2. Call with out_tokens buffer -> fills tokens up to out_capacity, returns 0
///
/// Args:
///   model_path: Path to GGUF model file
///   prompt: Input text to tokenize
///   add_bos: Whether to add BOS token (1=yes, 0=no)
///   parse_special: Whether to parse special tokens (1=yes, 0=no)
///   out_tokens: Output buffer for token IDs (NULL for size query)
///   out_capacity: Capacity of out_tokens buffer (ignored if out_tokens=NULL)
///   out_len: [out] Number of tokens produced
///   err: Error message buffer
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_tokenize(
    const char* model_path,
    const char* prompt,
    int add_bos,
    int parse_special,
    int32_t* out_tokens,
    int32_t out_capacity,
    int32_t* out_len,
    char* err,
    int32_t err_len
) {
    // Input validation
    if (!model_path || !prompt || !out_len || !err || err_len <= 0) {
        if (err && err_len > 0) {
            snprintf(err, err_len, "bitnet_tokenize: NULL required parameter");
            err[err_len - 1] = '\0';
        }
        if (out_len) *out_len = 0;
        return -1;
    }

    // Initialize outputs
    *out_len = 0;
    err[0] = '\0';

#ifdef BITNET_STUB
    // STUB mode: return friendly error
    // Silence unused parameter warnings in STUB mode
    (void)add_bos;
    (void)parse_special;
    (void)out_tokens;
    (void)out_capacity;

    snprintf(err, err_len,
             "bitnet_tokenize: STUB mode - BitNet.cpp not available. "
             "Set BITNET_CPP_DIR to enable cross-validation.");
    err[err_len - 1] = '\0';
    return -1;

#elif defined(BITNET_AVAILABLE)
    // AVAILABLE mode: implement two-pass buffer negotiation

    // TODO: Load model context
    // llama_model* model = llama_load_model_from_file(model_path, ...);
    // if (!model) {
    //     snprintf(err, err_len, "bitnet_tokenize: failed to load model: %s", model_path);
    //     err[err_len - 1] = '\0';
    //     return -1;
    // }

    // TODO: Tokenize with BitNet.cpp API
    // std::vector<int32_t> tokens;
    // int result = bitnet_tokenize_internal(model, prompt, add_bos, parse_special, tokens);
    // if (result != 0) {
    //     snprintf(err, err_len, "bitnet_tokenize: tokenization failed");
    //     err[err_len - 1] = '\0';
    //     llama_free_model(model);
    //     return -1;
    // }

    // TODO: Implement two-pass pattern
    // int32_t n_tokens = static_cast<int32_t>(tokens.size());
    // *out_len = n_tokens;
    //
    // // Pass 1: size query
    // if (!out_tokens) {
    //     llama_free_model(model);
    //     return 0;
    // }
    //
    // // Pass 2: fill buffer
    // int32_t n_copy = (n_tokens < out_capacity) ? n_tokens : out_capacity;
    // std::memcpy(out_tokens, tokens.data(), n_copy * sizeof(int32_t));
    // *out_len = n_copy;
    //
    // llama_free_model(model);
    // return 0;

    // Placeholder: return error until wired
    snprintf(err, err_len,
             "bitnet_tokenize: AVAILABLE mode but not yet wired. "
             "TODO: Wire BitNet.cpp tokenizer API.");
    err[err_len - 1] = '\0';
    return -1;

#else
    // Neither STUB nor AVAILABLE defined - compilation error expected
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

/// Evaluate tokens and return logits using BitNet.cpp inference
///
/// Two-pass pattern:
/// 1. Call with out_logits=NULL to query shape -> fills out_rows/out_cols, returns 0
/// 2. Call with out_logits buffer -> fills logits up to logits_capacity, returns 0
///
/// Args:
///   model_path: Path to GGUF model file
///   tokens: Input token IDs
///   n_tokens: Number of tokens
///   n_ctx: Context size for inference
///   out_logits: Output buffer for logits (NULL for size query)
///   logits_capacity: Capacity of out_logits buffer in floats (ignored if out_logits=NULL)
///   out_rows: [out] Number of rows (positions) in logits
///   out_cols: [out] Number of columns (vocab size) in logits
///   err: Error message buffer
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_eval_with_tokens(
    const char* model_path,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t n_ctx,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
) {
    // Input validation
    if (!model_path || !tokens || n_tokens <= 0 || n_ctx <= 0 ||
        !out_rows || !out_cols || !err || err_len <= 0) {
        if (err && err_len > 0) {
            snprintf(err, err_len, "bitnet_eval_with_tokens: NULL or invalid required parameter");
            err[err_len - 1] = '\0';
        }
        if (out_rows) *out_rows = 0;
        if (out_cols) *out_cols = 0;
        return -1;
    }

    // Initialize outputs
    *out_rows = 0;
    *out_cols = 0;
    err[0] = '\0';

#ifdef BITNET_STUB
    // STUB mode: return friendly error
    // Silence unused parameter warnings in STUB mode
    (void)out_logits;
    (void)logits_capacity;

    snprintf(err, err_len,
             "bitnet_eval_with_tokens: STUB mode - BitNet.cpp not available. "
             "Set BITNET_CPP_DIR to enable cross-validation.");
    err[err_len - 1] = '\0';
    return -1;

#elif defined(BITNET_AVAILABLE)
    // AVAILABLE mode: implement two-pass buffer negotiation

    // TODO: Load model and create context
    // llama_model* model = llama_load_model_from_file(model_path, ...);
    // if (!model) {
    //     snprintf(err, err_len, "bitnet_eval_with_tokens: failed to load model: %s", model_path);
    //     err[err_len - 1] = '\0';
    //     return -1;
    // }
    //
    // llama_context_params ctx_params = llama_context_default_params();
    // ctx_params.n_ctx = n_ctx;
    // llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    // if (!ctx) {
    //     snprintf(err, err_len, "bitnet_eval_with_tokens: failed to create context");
    //     err[err_len - 1] = '\0';
    //     llama_free_model(model);
    //     return -1;
    // }

    // TODO: Evaluate tokens
    // int result = bitnet_eval_internal(ctx, tokens, n_tokens);
    // if (result != 0) {
    //     snprintf(err, err_len, "bitnet_eval_with_tokens: evaluation failed");
    //     err[err_len - 1] = '\0';
    //     llama_free(ctx);
    //     llama_free_model(model);
    //     return -1;
    // }

    // TODO: Get logits and shape
    // float* logits = llama_get_logits(ctx);
    // int32_t n_vocab = llama_n_vocab(model);
    // int32_t rows = n_tokens;
    // int32_t cols = n_vocab;
    //
    // *out_rows = rows;
    // *out_cols = cols;
    //
    // // Pass 1: size query
    // if (!out_logits) {
    //     llama_free(ctx);
    //     llama_free_model(model);
    //     return 0;
    // }
    //
    // // Pass 2: fill buffer
    // int32_t total_elements = rows * cols;
    // int32_t n_copy = (total_elements < logits_capacity) ? total_elements : logits_capacity;
    // std::memcpy(out_logits, logits, n_copy * sizeof(float));
    //
    // llama_free(ctx);
    // llama_free_model(model);
    // return 0;

    // Placeholder: return error until wired
    snprintf(err, err_len,
             "bitnet_eval_with_tokens: AVAILABLE mode but not yet wired. "
             "TODO: Wire BitNet.cpp inference API.");
    err[err_len - 1] = '\0';
    return -1;

#else
    // Neither STUB nor AVAILABLE defined - compilation error expected
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

} // extern "C"
