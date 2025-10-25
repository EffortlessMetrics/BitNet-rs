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
#include "llama.h"  // From bitnet.cpp's llama.cpp
#include <vector>   // For token and logits buffers
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
int crossval_bitnet_tokenize(
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
    // AVAILABLE mode: Two-pass buffer negotiation with BitNet.cpp

    // Step 1: Load model context
    // Note: This per-call load is inefficient; consider session API in v0.2
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        snprintf(err, err_len, "bitnet_tokenize: Failed to load model from %s", model_path);
        err[err_len - 1] = '\0';
        return -1;
    }

    // Step 2: Two-pass tokenization pattern
    // Pass 1: Get token count (tokens=NULL, n_tokens_max=0)
    int32_t text_len = static_cast<int32_t>(std::strlen(prompt));
    int32_t n_tokens = llama_tokenize(
        model,
        prompt,
        text_len,
        nullptr,           // tokens=NULL for size query
        0,                 // n_tokens_max=0 for size query
        add_bos != 0,      // Convert C int to bool
        parse_special != 0 // Convert C int to bool
    );

    // llama_tokenize returns negative on error, or the required count
    if (n_tokens < 0) {
        n_tokens = -n_tokens;  // Get actual required count
    }

    if (n_tokens == 0) {
        snprintf(err, err_len, "bitnet_tokenize: Tokenization returned 0 tokens");
        err[err_len - 1] = '\0';
        llama_free_model(model);
        return -1;
    }

    *out_len = n_tokens;

    // If size query (out_tokens == NULL), return now
    if (!out_tokens || out_capacity <= 0) {
        llama_free_model(model);
        return 0;
    }

    // Pass 2: Fill buffer with actual tokens
    if (out_capacity < n_tokens) {
        snprintf(err, err_len,
                 "bitnet_tokenize: Buffer too small (need %d, got %d)",
                 n_tokens, out_capacity);
        err[err_len - 1] = '\0';
        llama_free_model(model);
        return -1;
    }

    int32_t result = llama_tokenize(
        model,
        prompt,
        text_len,
        out_tokens,        // Fill the output buffer
        out_capacity,      // Max tokens we can write
        add_bos != 0,
        parse_special != 0
    );

    if (result < 0) {
        snprintf(err, err_len, "bitnet_tokenize: Tokenization failed on buffer fill");
        err[err_len - 1] = '\0';
        llama_free_model(model);
        return -1;
    }

    // Update out_len with actual number returned (should match n_tokens)
    *out_len = result;

    llama_free_model(model);
    return 0;

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
int crossval_bitnet_eval_with_tokens(
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
    // AVAILABLE mode: Two-pass buffer negotiation with BitNet.cpp

    // Step 1: Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        snprintf(err, err_len, "bitnet_eval: Failed to load model from %s", model_path);
        err[err_len - 1] = '\0';
        return -1;
    }

    // Step 2: Create context with n_ctx and logits_all=true
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    ctx_params.logits_all = true;  // CRITICAL: Enable all-position logits
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        snprintf(err, err_len, "bitnet_eval: Failed to create context (n_ctx=%d)", n_ctx);
        err[err_len - 1] = '\0';
        llama_free_model(model);
        return -1;
    }

    // Step 3: Get vocab size for logits shape
    int32_t n_vocab = llama_n_vocab(model);

    // Step 4: Set output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;

    // Step 5: Implement two-pass pattern
    if (!out_logits || logits_capacity <= 0) {
        // Pass 1: Shape query only - return shape without computing logits
        llama_free(ctx);
        llama_free_model(model);
        return 0;
    }

    // Pass 2: Actually compute and fill logits
    int32_t total_elements = n_tokens * n_vocab;
    if (logits_capacity < total_elements) {
        snprintf(err, err_len,
                 "bitnet_eval: Buffer too small (need %d, got %d)",
                 total_elements, logits_capacity);
        err[err_len - 1] = '\0';
        llama_free(ctx);
        llama_free_model(model);
        return -1;
    }

    // Step 6: Create batch with all tokens and decode in one pass
    // Cast away const for llama_batch_get_one (it doesn't modify the tokens)
    llama_batch batch = llama_batch_get_one(const_cast<int32_t*>(tokens), n_tokens, 0, 0);

    int result = llama_decode(ctx, batch);
    if (result != 0) {
        snprintf(err, err_len, "bitnet_eval: Batch decode failed (result=%d)", result);
        err[err_len - 1] = '\0';
        llama_free(ctx);
        llama_free_model(model);
        return -1;
    }

    // Step 7: Extract logits for all positions
    // With logits_all=true, llama_get_logits_ith(i) returns logits for position i
    for (int32_t i = 0; i < n_tokens; ++i) {
        float* logits_for_pos = llama_get_logits_ith(ctx, i);
        if (!logits_for_pos) {
            snprintf(err, err_len, "bitnet_eval: Failed to get logits for position %d", i);
            err[err_len - 1] = '\0';
            llama_free(ctx);
            llama_free_model(model);
            return -1;
        }
        // Copy to row-major output buffer: out_logits[i * n_vocab : (i+1) * n_vocab]
        std::memcpy(&out_logits[i * n_vocab], logits_for_pos, n_vocab * sizeof(float));
    }

    // Step 8: Cleanup
    llama_free(ctx);
    llama_free_model(model);
    return 0;

#else
    // Neither STUB nor AVAILABLE defined - compilation error expected
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

} // extern "C"
