#include "bitnet_c.h"

// NOTE: This C++ implementation forwards to Microsoft BitNet C++ runtime
// It requires BITNET_CPP_DIR to be set and the library to be linked.
// For builds without FFI support, the .c stub is used instead.

#ifdef __cplusplus
#include <memory>
#include <string>
#include <vector>
#include <cstring>

// Forward declarations for BitNet C++ API
// These will be defined by the external bitnet.cpp library
namespace llama {
    struct model;
    struct context;
    struct model_params;
    struct context_params;

    // Model loading
    model* load_model_from_file(const char* path, const model_params& params);
    void free_model(model* m);

    // Context management
    context* new_context(model* m, const context_params& params);
    void free_context(context* ctx);

    // Tokenization
    int tokenize(model* m, const char* text, int* tokens, int n_max, bool add_bos, bool add_special);

    // Inference
    bool eval(context* ctx, const int* tokens, int n_tokens);
    const float* get_logits(context* ctx);
    int get_vocab_size(model* m);

    // Sampling
    int sample_greedy(const float* logits, int n_vocab);
}

// Internal structs to hold C++ objects
struct bitnet_model {
    llama::model* model = nullptr;
    int vocab_size = 0;
};

struct bitnet_ctx {
    llama::context* context = nullptr;
    llama::model* model = nullptr;
    int n_threads = 1;
};

extern "C" {

bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path) {
    if (!gguf_path) return nullptr;

    try {
        auto m = std::make_unique<bitnet_model>();

        // Default model params
        llama::model_params params{};

        // Load model
        m->model = llama::load_model_from_file(gguf_path, params);
        if (!m->model) return nullptr;

        // Get vocab size
        m->vocab_size = llama::get_vocab_size(m->model);

        return m.release();
    } catch (...) {
        return nullptr;
    }
}

void bitnet_model_free(bitnet_model_t* m) {
    if (!m) return;
    try {
        if (m->model) {
            llama::free_model(m->model);
        }
        delete m;
    } catch (...) {
        // Ignore exceptions in cleanup
    }
}

bitnet_ctx_t* bitnet_context_new(bitnet_model_t* m, const bitnet_params_t* p) {
    if (!m || !p || !m->model) return nullptr;

    try {
        auto c = std::make_unique<bitnet_ctx>();

        // Set context params from bitnet_params_t
        llama::context_params params{};
        // Note: Adjust these mappings based on actual llama.cpp API
        // params.n_ctx = p->n_ctx;
        // params.n_threads = p->n_threads;
        // params.seed = p->seed;

        c->context = llama::new_context(m->model, params);
        if (!c->context) return nullptr;

        c->model = m->model;
        c->n_threads = p->n_threads;

        return c.release();
    } catch (...) {
        return nullptr;
    }
}

void bitnet_context_free(bitnet_ctx_t* c) {
    if (!c) return;
    try {
        if (c->context) {
            llama::free_context(c->context);
        }
        delete c;
    } catch (...) {
        // Ignore exceptions in cleanup
    }
}

int bitnet_tokenize(bitnet_model_t* m, const char* text, int add_bos, int add_special,
                    int32_t* out_ids, int out_cap) {
    if (!m || !m->model || !text || !out_ids || out_cap <= 0) return -1;

    try {
        // Tokenize using llama.cpp API
        int n_tokens = llama::tokenize(
            m->model,
            text,
            out_ids,
            out_cap,
            (bool)add_bos,
            (bool)add_special
        );

        if (n_tokens < 0) return -2; // Tokenization failed
        if (n_tokens > out_cap) return -3; // Buffer too small

        return n_tokens;
    } catch (...) {
        return -4;
    }
}

int bitnet_eval(bitnet_ctx_t* c, const int32_t* ids, int n_ids,
                float* logits_out, int logits_cap) {
    if (!c || !c->context || !ids || n_ids <= 0 || !logits_out || logits_cap <= 0) return -1;

    try {
        // Evaluate tokens
        if (!llama::eval(c->context, ids, n_ids)) {
            return -2; // Eval failed
        }

        // Get logits
        const float* logits = llama::get_logits(c->context);
        if (!logits) return -3;

        // Copy logits to output buffer
        // Note: llama.cpp returns logits for the last token position
        int vocab_size = llama::get_vocab_size(c->model);
        if (vocab_size > logits_cap) return -4; // Buffer too small

        std::memcpy(logits_out, logits, vocab_size * sizeof(float));

        return 0; // Success
    } catch (...) {
        return -5;
    }
}

int bitnet_decode_greedy(bitnet_ctx_t* c, int32_t* io_ids, int max_new_tokens,
                         int eos_id, float temperature) {
    if (!c || !c->context || !io_ids || max_new_tokens <= 0) return -1;

    try {
        int generated = 0;
        std::vector<int32_t> tokens;

        // Get initial logits (if context has been evaluated)
        for (int step = 0; step < max_new_tokens; ++step) {
            // Evaluate current token sequence
            if (!llama::eval(c->context, io_ids + generated, 1)) {
                return -2; // Eval failed
            }

            // Get logits for sampling
            const float* logits = llama::get_logits(c->context);
            if (!logits) return -3;

            // Sample next token
            int32_t next_token;
            if (temperature <= 0.0f) {
                // Greedy sampling (argmax)
                int vocab_size = llama::get_vocab_size(c->model);
                next_token = llama::sample_greedy(logits, vocab_size);
            } else {
                // For non-zero temperature, still use greedy for parity
                int vocab_size = llama::get_vocab_size(c->model);
                next_token = llama::sample_greedy(logits, vocab_size);
            }

            // Store generated token
            io_ids[generated] = next_token;
            ++generated;

            // Check for EOS
            if (next_token == eos_id) {
                break;
            }
        }

        return generated;
    } catch (...) {
        return -6;
    }
}

} // extern "C"

#else
// C fallback - should not be reached if this file is compiled as C++
#error "This file must be compiled as C++"
#endif // __cplusplus
