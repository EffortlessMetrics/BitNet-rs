#include "bitnet_c.h"

// NOTE: This C++ implementation forwards to Microsoft BitNet C++ runtime
// It requires BITNET_CPP_DIR to be set and the library to be linked.
// For builds without FFI support, the .c stub is used instead.

#ifdef __cplusplus
#include <memory>
#include <string>
#include <vector>
#include <cstring>

// Include actual llama.cpp headers
// These should be available when BITNET_CPP_DIR is set
extern "C" {
    #include "llama.h"
}

// Internal structs to hold llama.cpp objects
struct bitnet_model {
    llama_model* model = nullptr;
    int vocab_size = 0;
};

struct bitnet_ctx {
    llama_context* context = nullptr;
    llama_model* model = nullptr;
    int n_threads = 1;
};

extern "C" {

bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path) {
    if (!gguf_path) return nullptr;

    try {
        auto m = std::make_unique<bitnet_model>();

        // Default model params
        llama_model_params params = llama_model_default_params();

        // Load model
        m->model = llama_load_model_from_file(gguf_path, params);
        if (!m->model) return nullptr;

        // Get vocab size
        m->vocab_size = llama_n_vocab(m->model);

        return m.release();
    } catch (...) {
        return nullptr;
    }
}

void bitnet_model_free(bitnet_model_t* m) {
    if (!m) return;
    try {
        if (m->model) {
            llama_free_model(m->model);
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
        llama_context_params params = llama_context_default_params();
        params.n_ctx = p->n_ctx;
        params.n_threads = p->n_threads;
        params.n_threads_batch = p->n_threads;
        // Note: seed is set via llama_set_rng_seed() after context creation
        // Enable logits for all tokens (needed for eval)
        params.logits_all = true;

        c->context = llama_new_context_with_model(m->model, params);
        if (!c->context) return nullptr;

        // Note: RNG seed (p->seed) not set here as this version of llama.cpp
        // doesn't expose llama_set_rng_seed. Deterministic behavior is controlled
        // via temperature=0.0 in greedy decode and environment variables on Rust side.

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
            llama_free(c->context);
        }
        delete c;
    } catch (...) {
        // Ignore exceptions in cleanup
    }
}

int bitnet_tokenize(bitnet_model_t* m, const char* text, int add_bos, int parse_special,
                    int32_t* out_ids, int out_cap) {
    if (!m || !m->model || !text) return -1;

    try {
        // Get text length
        int text_len = strlen(text);

        // Handle nullptr/0 pattern: caller wants to know the required buffer size
        if (!out_ids || out_cap == 0) {
            // Preflight call: llama_tokenize returns the number of tokens needed
            int n_tokens = llama_tokenize(
                m->model,
                text,
                text_len,
                nullptr,
                0,
                (bool)add_bos,
                (bool)parse_special
            );

            // Return absolute value (handle both positive and negative conventions)
            return (n_tokens < 0) ? -n_tokens : n_tokens;
        }

        // Normal tokenization with provided buffer
        // Modern llama.cpp signature:
        //   int llama_tokenize(model, text, text_len, tokens, n_max, add_special, parse_special)
        //
        // Parameter mapping:
        //   BitNet add_bos → llama.cpp add_special (controls BOS insertion)
        //   BitNet parse_special → llama.cpp parse_special (parses "<|eot_id|>" etc.)
        int n_tokens = llama_tokenize(
            m->model,
            text,
            text_len,
            out_ids,
            out_cap,
            (bool)add_bos,        // llama.cpp add_special: controls BOS insertion
            (bool)parse_special   // llama.cpp parse_special: parses special token strings
        );

        // If negative, buffer was too small - return error code -2
        // (caller should have allocated enough based on first call)
        if (n_tokens < 0) {
            return -2; // Buffer too small
        }

        return n_tokens;
    } catch (...) {
        return -4;
    }
}

int bitnet_eval(bitnet_ctx_t* c, const int32_t* ids, int n_ids,
                float* logits_out, int logits_cap) {
    if (!c || !c->context || !ids || n_ids <= 0 || !logits_out || logits_cap <= 0) return -1;

    try {
        // Create batch for evaluation
        llama_batch batch = llama_batch_init(n_ids, 0, 1);

        // Prepare seq_ids on stack
        llama_seq_id seq_ids[1] = {0};

        // Populate batch
        for (int i = 0; i < n_ids; i++) {
            batch.token[i] = ids[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = seq_ids;
            batch.logits[i] = (i == n_ids - 1) ? 1 : 0; // Only get logits for last token
        }
        batch.n_tokens = n_ids;

        // Evaluate
        int result = llama_decode(c->context, batch);
        llama_batch_free(batch);

        if (result != 0) {
            return -2; // Eval failed
        }

        // Get logits for last token
        const float* logits = llama_get_logits(c->context);
        if (!logits) return -3;

        // Copy logits to output buffer
        int vocab_size = llama_n_vocab(c->model);
        if (vocab_size > logits_cap) return -4; // Buffer too small

        std::memcpy(logits_out, logits, vocab_size * sizeof(float));

        return 0; // Success
    } catch (...) {
        return -5;
    }
}

int bitnet_prefill(bitnet_ctx_t* c, const int32_t* ids, int n_ids) {
    if (!c || !c->context || !ids || n_ids <= 0) return -1;

    llama_batch batch = llama_batch_init(n_ids, 0, 1);
    llama_seq_id seq_ids[1] = {0};

    for (int i = 0; i < n_ids; ++i) {
        batch.token[i]   = ids[i];
        batch.pos[i]     = i;           // 0..T-1
        batch.n_seq_id[i]= 1;
        batch.seq_id[i]  = seq_ids;
        batch.logits[i]  = (i == n_ids - 1) ? 1 : 0;  // only last token needs logits
    }
    batch.n_tokens = n_ids;

    int rc = llama_decode(c->context, batch);
    llama_batch_free(batch);
    return rc;  // 0 = OK
}

int bitnet_decode_greedy(bitnet_ctx_t* c, int32_t* io_ids, int max_new_tokens,
                         int eos_id, float temperature) {
    if (!c || !c->context || !io_ids || max_new_tokens <= 0) return -1;

    try {
        int generated = 0;
        llama_seq_id seq_ids[1] = {0};

        // Get initial position from KV cache (after prefill)
        int n_past = llama_get_kv_cache_token_count(c->context);

        // Generate tokens one at a time
        for (int step = 0; step < max_new_tokens; ++step) {
            // Create batch for single token
            llama_batch batch = llama_batch_init(1, 0, 1);

            // For first step, we may need to evaluate the prompt
            // For now, assume context was pre-evaluated
            // Just sample from current logits

            // Get logits for sampling
            const float* logits = llama_get_logits(c->context);
            if (!logits) {
                llama_batch_free(batch);
                return -3;
            }

            // Greedy sampling (argmax)
            int vocab_size = llama_n_vocab(c->model);
            int32_t next_token = 0;
            float max_logit = logits[0];
            for (int i = 1; i < vocab_size; i++) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    next_token = i;
                }
            }

            // Store generated token
            io_ids[generated] = next_token;
            ++generated;

            // Check for EOS
            if (next_token == eos_id) {
                llama_batch_free(batch);
                break;
            }

            // Evaluate the new token for next iteration
            batch.token[0] = next_token;
            batch.pos[0] = n_past + (generated - 1); // Correct position for the just-generated token
            batch.n_seq_id[0] = 1;
            batch.seq_id[0] = seq_ids;
            batch.logits[0] = 1;
            batch.n_tokens = 1;

            int result = llama_decode(c->context, batch);
            llama_batch_free(batch);

            if (result != 0) {
                return -2; // Eval failed
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
