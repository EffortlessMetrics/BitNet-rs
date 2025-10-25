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
    // New API: Get vocab from model first
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t text_len = static_cast<int32_t>(std::strlen(prompt));
    int32_t n_tokens = llama_tokenize(
        vocab,             // Use vocab instead of model
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
        llama_model_free(model);  // Use new API
        return -1;
    }

    *out_len = n_tokens;

    // If size query (out_tokens == NULL), return now
    if (!out_tokens || out_capacity <= 0) {
        llama_model_free(model);  // Use new API
        return 0;
    }

    // Pass 2: Fill buffer with actual tokens
    if (out_capacity < n_tokens) {
        snprintf(err, err_len,
                 "bitnet_tokenize: Buffer too small (need %d, got %d)",
                 n_tokens, out_capacity);
        err[err_len - 1] = '\0';
        llama_model_free(model);  // Use new API
        return -1;
    }

    int32_t result = llama_tokenize(
        vocab,             // Use vocab instead of model
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
        llama_model_free(model);  // Use new API
        return -1;
    }

    // Update out_len with actual number returned (should match n_tokens)
    *out_len = result;

    llama_model_free(model);  // Use new API
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

    // Step 2: Create context with n_ctx
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        snprintf(err, err_len, "bitnet_eval: Failed to create context (n_ctx=%d)", n_ctx);
        err[err_len - 1] = '\0';
        llama_model_free(model);  // Use new API
        return -1;
    }

    // New API: Set causal attention to false to get all-position logits
    llama_set_causal_attn(ctx, false);

    // Step 3: Get vocab size for logits shape
    // New API: llama_model_get_vocab + llama_vocab_n_tokens
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Step 4: Set output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;

    // Step 5: Implement two-pass pattern
    if (!out_logits || logits_capacity <= 0) {
        // Pass 1: Shape query only - return shape without computing logits
        llama_free(ctx);
        llama_model_free(model);  // Use new API
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
        llama_model_free(model);  // Use new API
        return -1;
    }

    // Step 6: Create batch with all tokens and decode in one pass
    // Cast away const for llama_batch_get_one (it doesn't modify the tokens)
    // New API: llama_batch_get_one now only takes 2 arguments (tokens, n_tokens)
    llama_batch batch = llama_batch_get_one(const_cast<int32_t*>(tokens), n_tokens);

    int result = llama_decode(ctx, batch);
    if (result != 0) {
        snprintf(err, err_len, "bitnet_eval: Batch decode failed (result=%d)", result);
        err[err_len - 1] = '\0';
        llama_free(ctx);
        llama_model_free(model);  // Use new API
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
            llama_model_free(model);  // Use new API
            return -1;
        }
        // Copy to row-major output buffer: out_logits[i * n_vocab : (i+1) * n_vocab]
        std::memcpy(&out_logits[i * n_vocab], logits_for_pos, n_vocab * sizeof(float));
    }

    // Step 8: Cleanup
    llama_free(ctx);
    llama_model_free(model);  // Use new API
    return 0;

#else
    // Neither STUB nor AVAILABLE defined - compilation error expected
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

// ============================================================================
// Socket 1: Context Initialization (Persistent Model Loading)
// ============================================================================

/// Opaque context handle for BitNet.cpp
/// This matches the typedef in the spec: `typedef struct bitnet_context_t bitnet_context_t;`
struct bitnet_context_t {
#ifdef BITNET_AVAILABLE
    llama_model* model;
    llama_context* ctx;
    int32_t n_ctx;
    int32_t n_gpu_layers;
#endif
};

/// Initialize persistent BitNet context
///
/// Socket 1: Eliminates per-call model reload overhead (100-500ms).
/// Expected performance impact: 10-100× speedup for multi-call workflows.
///
/// Args:
///   out_ctx: [out] Opaque context handle (caller frees with bitnet_cpp_free_context)
///   model_path: Path to GGUF model file
///   n_ctx: Context size for inference (e.g., 512, 2048)
///   n_gpu_layers: Number of layers to offload to GPU (0=CPU-only)
///   err: Error message buffer (512 bytes recommended)
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_init_context(
    bitnet_context_t** out_ctx,
    const char* model_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,
    char* err,
    int32_t err_len
) {
    // Input validation
    if (!out_ctx || !model_path || !err || err_len <= 0) {
        if (err && err_len > 0) {
            snprintf(err, err_len, "bitnet_cpp_init_context: NULL required parameter");
            err[err_len - 1] = '\0';
        }
        if (out_ctx) *out_ctx = nullptr;
        return -1;
    }

    // Initialize outputs
    *out_ctx = nullptr;
    err[0] = '\0';

#ifdef BITNET_STUB
    // STUB mode: return friendly error
    (void)n_ctx;
    (void)n_gpu_layers;

    snprintf(err, err_len,
             "bitnet_cpp_init_context: STUB mode - BitNet.cpp not available. "
             "Set BITNET_CPP_DIR to enable cross-validation.");
    err[err_len - 1] = '\0';
    return -1;

#elif defined(BITNET_AVAILABLE)
    // AVAILABLE mode: Create persistent context

    // Step 1: Allocate context structure
    bitnet_context_t* ctx = new (std::nothrow) bitnet_context_t;
    if (!ctx) {
        snprintf(err, err_len, "bitnet_cpp_init_context: Failed to allocate context structure");
        err[err_len - 1] = '\0';
        return -1;
    }

    // Initialize fields
    ctx->model = nullptr;
    ctx->ctx = nullptr;
    ctx->n_ctx = n_ctx;
    ctx->n_gpu_layers = n_gpu_layers;

    // Step 2: Load model
    llama_model_params model_params = llama_model_default_params();
    // Note: GPU layer offloading would be configured here
    // model_params.n_gpu_layers = n_gpu_layers;

    ctx->model = llama_load_model_from_file(model_path, model_params);
    if (!ctx->model) {
        snprintf(err, err_len, "bitnet_cpp_init_context: Failed to load model from %s", model_path);
        err[err_len - 1] = '\0';
        delete ctx;
        return -1;
    }

    // Step 3: Create inference context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
    // Note: GPU configuration would be set here

    ctx->ctx = llama_new_context_with_model(ctx->model, ctx_params);
    if (!ctx->ctx) {
        snprintf(err, err_len, "bitnet_cpp_init_context: Failed to create context (n_ctx=%d)", n_ctx);
        err[err_len - 1] = '\0';
        llama_model_free(ctx->model);
        delete ctx;
        return -1;
    }

    // Success: Return context handle
    *out_ctx = ctx;
    return 0;

#else
    // Neither STUB nor AVAILABLE defined - compilation error expected
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

/// Free BitNet context (releases model and context)
///
/// Socket 1: Cleanup function for RAII pattern via Rust Drop trait.
///
/// Args:
///   ctx: Context handle to free (NULL-safe)
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_free_context(
    bitnet_context_t* ctx
) {
    // NULL-safe: Allow freeing NULL context
    if (!ctx) {
        return 0;
    }

#ifdef BITNET_STUB
    // STUB mode: Nothing to free (context would be NULL in stub mode)
    return 0;

#elif defined(BITNET_AVAILABLE)
    // AVAILABLE mode: Free resources

    if (ctx->ctx) {
        llama_free(ctx->ctx);
        ctx->ctx = nullptr;
    }

    if (ctx->model) {
        llama_model_free(ctx->model);
        ctx->model = nullptr;
    }

    delete ctx;
    return 0;

#else
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

// ============================================================================
// Socket 2: BitNet-Specific Tokenization (Optional)
// ============================================================================

/// Tokenize text using BitNet-native tokenizer with persistent context
///
/// Socket 2: Optional BitNet-specific tokenization. Falls back to llama.cpp if unavailable.
/// Priority: v0.2 optional
///
/// Two-pass pattern:
/// 1. Call with out_tokens=NULL to query size -> fills out_len, returns 0
/// 2. Call with out_tokens buffer -> fills tokens up to out_capacity, returns 0
///
/// Args:
///   ctx: BitNet context handle from bitnet_cpp_init_context
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
int bitnet_cpp_tokenize_with_context(
    const bitnet_context_t* ctx,
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
    if (!ctx || !prompt || !out_len || !err || err_len <= 0) {
        if (err && err_len > 0) {
            snprintf(err, err_len, "bitnet_cpp_tokenize_with_context: NULL required parameter");
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
    (void)add_bos;
    (void)parse_special;
    (void)out_tokens;
    (void)out_capacity;

    snprintf(err, err_len,
             "bitnet_cpp_tokenize_with_context: STUB mode - BitNet.cpp not available.");
    err[err_len - 1] = '\0';
    return -1;

#elif defined(BITNET_AVAILABLE)
    // AVAILABLE mode: Use persistent context for tokenization

    // Validate context
    if (!ctx->model) {
        snprintf(err, err_len, "bitnet_cpp_tokenize_with_context: Invalid context (NULL model)");
        err[err_len - 1] = '\0';
        return -1;
    }

    // Get vocab from model
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    int32_t text_len = static_cast<int32_t>(std::strlen(prompt));

    // Pass 1: Get token count
    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt,
        text_len,
        nullptr,           // tokens=NULL for size query
        0,                 // n_tokens_max=0 for size query
        add_bos != 0,
        parse_special != 0
    );

    if (n_tokens < 0) {
        n_tokens = -n_tokens;  // Get actual required count
    }

    if (n_tokens == 0) {
        snprintf(err, err_len, "bitnet_cpp_tokenize_with_context: Tokenization returned 0 tokens");
        err[err_len - 1] = '\0';
        return -1;
    }

    *out_len = n_tokens;

    // If size query (out_tokens == NULL), return now
    if (!out_tokens || out_capacity <= 0) {
        return 0;
    }

    // Pass 2: Fill buffer with actual tokens
    if (out_capacity < n_tokens) {
        snprintf(err, err_len,
                 "bitnet_cpp_tokenize_with_context: Buffer too small (need %d, got %d)",
                 n_tokens, out_capacity);
        err[err_len - 1] = '\0';
        return -1;
    }

    int32_t result = llama_tokenize(
        vocab,
        prompt,
        text_len,
        out_tokens,
        out_capacity,
        add_bos != 0,
        parse_special != 0
    );

    if (result < 0) {
        snprintf(err, err_len, "bitnet_cpp_tokenize_with_context: Tokenization failed on buffer fill");
        err[err_len - 1] = '\0';
        return -1;
    }

    *out_len = result;
    return 0;

#else
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

// ============================================================================
// Socket 3: BitNet-Specific Inference (1-bit Optimized)
// ============================================================================

/// Evaluate tokens using BitNet-optimized 1-bit kernels with persistent context
///
/// Socket 3: BitNet-native inference with all-position logits support.
/// Priority: v0.2 high (enables BitNet-specific optimizations)
///
/// Two-pass pattern:
/// 1. Call with out_logits=NULL to query shape -> fills out_rows/out_cols, returns 0
/// 2. Call with out_logits buffer -> fills logits up to logits_capacity, returns 0
///
/// Args:
///   ctx: BitNet context handle
///   tokens: Input token IDs
///   n_tokens: Number of tokens
///   seq_id: Sequence ID for batch processing (0 for single sequence)
///   out_logits: Output buffer for logits (NULL for size query)
///   logits_capacity: Capacity of out_logits buffer in floats
///   out_rows: [out] Number of rows (positions) in logits
///   out_cols: [out] Number of columns (vocab size) in logits
///   err: Error message buffer
///   err_len: Capacity of error buffer
///
/// Returns: 0 on success, -1 on error
int bitnet_cpp_eval_with_context(
    const bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t seq_id,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
) {
    // Input validation
    if (!ctx || !tokens || n_tokens <= 0 || !out_rows || !out_cols || !err || err_len <= 0) {
        if (err && err_len > 0) {
            snprintf(err, err_len, "bitnet_cpp_eval_with_context: NULL or invalid required parameter");
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
    (void)seq_id;
    (void)out_logits;
    (void)logits_capacity;

    snprintf(err, err_len,
             "bitnet_cpp_eval_with_context: STUB mode - BitNet.cpp not available.");
    err[err_len - 1] = '\0';
    return -1;

#elif defined(BITNET_AVAILABLE)
    // AVAILABLE mode: Use persistent context for evaluation

    // Validate context
    if (!ctx->model || !ctx->ctx) {
        snprintf(err, err_len, "bitnet_cpp_eval_with_context: Invalid context (NULL model or ctx)");
        err[err_len - 1] = '\0';
        return -1;
    }

    // Validate token count vs context size
    if (n_tokens > ctx->n_ctx) {
        snprintf(err, err_len,
                 "bitnet_cpp_eval_with_context: Token count %d exceeds context size %d",
                 n_tokens, ctx->n_ctx);
        err[err_len - 1] = '\0';
        return -1;
    }

    // Set causal attention to false for all-position logits
    llama_set_causal_attn(ctx->ctx, false);

    // Get vocab size
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Set output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;

    // Pass 1: Shape query only
    if (!out_logits || logits_capacity <= 0) {
        return 0;
    }

    // Pass 2: Compute and fill logits
    int32_t total_elements = n_tokens * n_vocab;
    if (logits_capacity < total_elements) {
        snprintf(err, err_len,
                 "bitnet_cpp_eval_with_context: Buffer too small (need %d, got %d)",
                 total_elements, logits_capacity);
        err[err_len - 1] = '\0';
        return -1;
    }

    // Create batch with all tokens
    // Note: seq_id parameter allows future batch processing support
    llama_batch batch = llama_batch_get_one(const_cast<int32_t*>(tokens), n_tokens);

    int result = llama_decode(ctx->ctx, batch);
    if (result != 0) {
        snprintf(err, err_len, "bitnet_cpp_eval_with_context: Batch decode failed (result=%d)", result);
        err[err_len - 1] = '\0';
        return -1;
    }

    // Extract logits for all positions
    for (int32_t i = 0; i < n_tokens; ++i) {
        float* logits_for_pos = llama_get_logits_ith(ctx->ctx, i);
        if (!logits_for_pos) {
            snprintf(err, err_len, "bitnet_cpp_eval_with_context: Failed to get logits for position %d", i);
            err[err_len - 1] = '\0';
            return -1;
        }
        std::memcpy(&out_logits[i * n_vocab], logits_for_pos, n_vocab * sizeof(float));
    }

    return 0;

#else
    #error "Must define either BITNET_STUB or BITNET_AVAILABLE"
#endif
}

// ============================================================================
// Socket 4: Session API (High-Level Lifecycle Management) - v0.3 STUB
// ============================================================================

/// Opaque session handle (v0.3 - not yet implemented)
struct bitnet_session_t {
    // TODO(v0.3): Implement high-level session API
    // This is an alternative to Socket 1+2+3 composition if BitNet.cpp provides session API
    int _placeholder;
};

/// Create BitNet session with integrated model/context/tokenizer (v0.3)
///
/// Socket 4: High-level session management (alternative to Socket 1+2+3).
/// Priority: v0.3 (decision point: use this OR Socket 1+2+3)
///
/// TODO(v0.3): Implement if BitNet.cpp provides session API
int bitnet_cpp_session_create(
    bitnet_session_t** out_session,
    const char* model_path,
    const char* tokenizer_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,
    char* err,
    int32_t err_len
) {
    (void)out_session;
    (void)model_path;
    (void)tokenizer_path;
    (void)n_ctx;
    (void)n_gpu_layers;

    if (err && err_len > 0) {
        snprintf(err, err_len, "bitnet_cpp_session_create: Not yet implemented (v0.3)");
        err[err_len - 1] = '\0';
    }
    if (out_session) *out_session = nullptr;
    return -1;
}

/// Free BitNet session (v0.3)
int bitnet_cpp_session_free(bitnet_session_t* session) {
    (void)session;
    // TODO(v0.3): Implement session cleanup
    return 0;
}

/// Tokenize using session (v0.3)
int bitnet_cpp_session_tokenize(
    const bitnet_session_t* session,
    const char* prompt,
    int add_bos,
    int parse_special,
    int32_t* out_tokens,
    int32_t out_capacity,
    int32_t* out_len,
    char* err,
    int32_t err_len
) {
    (void)session;
    (void)prompt;
    (void)add_bos;
    (void)parse_special;
    (void)out_tokens;
    (void)out_capacity;

    if (err && err_len > 0) {
        snprintf(err, err_len, "bitnet_cpp_session_tokenize: Not yet implemented (v0.3)");
        err[err_len - 1] = '\0';
    }
    if (out_len) *out_len = 0;
    return -1;
}

/// Evaluate tokens using session (v0.3)
int bitnet_cpp_session_eval(
    const bitnet_session_t* session,
    const int32_t* tokens,
    int32_t n_tokens,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
) {
    (void)session;
    (void)tokens;
    (void)n_tokens;
    (void)out_logits;
    (void)logits_capacity;

    if (err && err_len > 0) {
        snprintf(err, err_len, "bitnet_cpp_session_eval: Not yet implemented (v0.3)");
        err[err_len - 1] = '\0';
    }
    if (out_rows) *out_rows = 0;
    if (out_cols) *out_cols = 0;
    return -1;
}

// ============================================================================
// Socket 5: GPU Support - v0.3 STUB
// ============================================================================

/// Evaluate tokens with GPU acceleration (v0.3)
///
/// Socket 5: GPU-accelerated inference with layer offloading.
/// Priority: v0.3 (post-MVP performance optimization)
///
/// TODO(v0.3): Implement GPU acceleration
int bitnet_cpp_eval_gpu(
    const bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
) {
    (void)ctx;
    (void)tokens;
    (void)n_tokens;
    (void)out_logits;
    (void)logits_capacity;

    if (err && err_len > 0) {
        snprintf(err, err_len, "bitnet_cpp_eval_gpu: Not yet implemented (v0.3)");
        err[err_len - 1] = '\0';
    }
    if (out_rows) *out_rows = 0;
    if (out_cols) *out_cols = 0;
    return -1;
}

// ============================================================================
// Socket 6: Capability Detection - v0.3 STUB
// ============================================================================

/// BitNet.cpp runtime capabilities (v0.3)
struct bitnet_capabilities_t {
    int has_avx2;       // x86 AVX2 SIMD
    int has_avx512;     // x86 AVX-512 SIMD
    int has_neon;       // ARM NEON SIMD
    int has_cuda;       // NVIDIA CUDA GPU
    int has_metal;      // Apple Metal GPU
    int has_hip;        // AMD ROCm GPU
};

/// Get BitNet.cpp runtime capabilities (v0.3)
///
/// Socket 6: Runtime feature detection for optimal kernel selection.
/// Priority: v0.3 (enables runtime optimization)
///
/// TODO(v0.3): Implement capability detection
int bitnet_cpp_get_capabilities(bitnet_capabilities_t* out_caps) {
    if (!out_caps) {
        return -1;
    }

    // TODO(v0.3): Detect actual capabilities
    // For now, return all zeros (no capabilities detected)
    out_caps->has_avx2 = 0;
    out_caps->has_avx512 = 0;
    out_caps->has_neon = 0;
    out_caps->has_cuda = 0;
    out_caps->has_metal = 0;
    out_caps->has_hip = 0;

    return 0;
}

} // extern "C"
