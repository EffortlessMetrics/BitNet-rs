/**
 * @file bitnet.h
 * @brief BitNet C API - Drop-in replacement for BitNet C++ bindings
 *
 * This header provides a comprehensive C API that serves as a drop-in replacement
 * for the existing BitNet C++ bindings. It maintains exact signature compatibility
 * while providing enhanced error handling, thread safety, and performance monitoring.
 *
 * @version 0.1.0
 * @author BitNet Contributors
 */

#ifndef BITNET_H
#define BITNET_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* ========================================================================== */
/* Constants and Error Codes                                                 */
/* ========================================================================== */

/** ABI version for compatibility checking */
#define BITNET_ABI_VERSION 1

/** Success return code */
#define BITNET_SUCCESS 0

/** Error codes matching existing C++ API */
#define BITNET_ERROR_INVALID_ARGUMENT -1
#define BITNET_ERROR_MODEL_NOT_FOUND -2
#define BITNET_ERROR_MODEL_LOAD_FAILED -3
#define BITNET_ERROR_INFERENCE_FAILED -4
#define BITNET_ERROR_OUT_OF_MEMORY -5
#define BITNET_ERROR_THREAD_SAFETY -6
#define BITNET_ERROR_INVALID_MODEL_ID -7
#define BITNET_ERROR_CONTEXT_LENGTH_EXCEEDED -8
#define BITNET_ERROR_UNSUPPORTED_OPERATION -9
#define BITNET_ERROR_INTERNAL -10

/* ========================================================================== */
/* Type Definitions                                                          */
/* ========================================================================== */

/** Model format enumeration */
typedef enum {
    BITNET_FORMAT_GGUF = 0,
    BITNET_FORMAT_SAFETENSORS = 1,
    BITNET_FORMAT_HUGGINGFACE = 2
} bitnet_model_format_t;

/** Quantization type enumeration */
typedef enum {
    BITNET_QUANT_I2S = 0,  /**< 2-bit signed quantization */
    BITNET_QUANT_TL1 = 1,  /**< Table lookup 1 (ARM optimized) */
    BITNET_QUANT_TL2 = 2   /**< Table lookup 2 (x86 optimized) */
} bitnet_quantization_type_t;

/** Backend preference enumeration */
typedef enum {
    BITNET_BACKEND_AUTO = 0,  /**< Automatically select best backend */
    BITNET_BACKEND_CPU = 1,   /**< Prefer CPU backend */
    BITNET_BACKEND_GPU = 2    /**< Prefer GPU backend */
} bitnet_backend_preference_t;

/** Model configuration structure */
typedef struct {
    const char* model_path;                    /**< Model file path (null-terminated) */
    uint32_t model_format;                     /**< Model format (bitnet_model_format_t) */
    uint32_t vocab_size;                       /**< Vocabulary size */
    uint32_t hidden_size;                      /**< Hidden size */
    uint32_t num_layers;                       /**< Number of layers */
    uint32_t num_heads;                        /**< Number of attention heads */
    uint32_t intermediate_size;                /**< Intermediate size */
    uint32_t max_position_embeddings;          /**< Maximum position embeddings */
    uint32_t quantization_type;                /**< Quantization type */
    uint32_t block_size;                       /**< Quantization block size */
    float precision;                           /**< Quantization precision */
    uint32_t num_threads;                      /**< Number of threads (0 for auto) */
    uint32_t use_gpu;                          /**< Use GPU acceleration (0=false, 1=true) */
    uint32_t batch_size;                       /**< Batch size */
    uint64_t memory_limit;                     /**< Memory limit in bytes (0 for no limit) */
} bitnet_config_t;

/** Inference configuration structure */
typedef struct {
    uint32_t max_length;                       /**< Maximum sequence length */
    uint32_t max_new_tokens;                   /**< Maximum new tokens to generate */
    float temperature;                         /**< Temperature for sampling */
    uint32_t top_k;                            /**< Top-k sampling (0 to disable) */
    float top_p;                               /**< Top-p sampling (0.0 to disable) */
    float repetition_penalty;                  /**< Repetition penalty */
    float frequency_penalty;                   /**< Frequency penalty */
    float presence_penalty;                    /**< Presence penalty */
    uint64_t seed;                             /**< Random seed (0 for random) */
    uint32_t do_sample;                        /**< Enable sampling (0=greedy, 1=sampling) */
    uint32_t backend_preference;               /**< Backend preference */
    uint32_t enable_streaming;                 /**< Enable streaming output */
    uint32_t stream_buffer_size;               /**< Streaming buffer size */
} bitnet_inference_config_t;

/** Model information structure */
typedef struct {
    const char* name;                          /**< Model name */
    const char* version;                       /**< Model version */
    const char* architecture;                 /**< Model architecture */
    uint32_t vocab_size;                       /**< Vocabulary size */
    uint32_t context_length;                   /**< Context length */
    uint32_t hidden_size;                      /**< Hidden size */
    uint32_t num_layers;                       /**< Number of layers */
    uint32_t num_heads;                        /**< Number of attention heads */
    uint32_t intermediate_size;                /**< Intermediate size */
    uint32_t quantization_type;                /**< Quantization type */
    uint64_t file_size;                        /**< Model file size in bytes */
    uint64_t memory_usage;                     /**< Memory usage in bytes */
    uint32_t is_gpu_loaded;                    /**< Whether model is loaded on GPU */
} bitnet_model_t;

/** Performance metrics structure */
typedef struct {
    float tokens_per_second;                   /**< Tokens per second */
    float latency_ms;                          /**< Latency in milliseconds */
    float memory_usage_mb;                     /**< Memory usage in MB */
    float gpu_utilization;                     /**< GPU utilization (0-100, -1 if N/A) */
    float total_inference_time_ms;             /**< Total inference time in ms */
    float time_to_first_token_ms;              /**< Time to first token in ms */
    uint32_t tokens_generated;                 /**< Number of tokens generated */
    uint32_t prompt_tokens;                    /**< Number of tokens in prompt */
} bitnet_performance_metrics_t;

/** Streaming callback function type */
typedef int (*bitnet_stream_callback_t)(const char* token, void* user_data);

/** Streaming configuration structure */
typedef struct {
    bitnet_stream_callback_t callback;         /**< Callback function */
    void* user_data;                           /**< User data for callback */
    uint32_t buffer_size;                      /**< Buffer size */
    uint32_t yield_interval;                   /**< Yield interval in tokens */
    uint32_t enable_backpressure;              /**< Enable backpressure */
    uint32_t timeout_ms;                       /**< Timeout in milliseconds */
} bitnet_stream_config_t;

/* ========================================================================== */
/* Core API Functions                                                        */
/* ========================================================================== */

/**
 * @brief Get ABI version for compatibility validation
 * @return Current ABI version number
 */
uint32_t bitnet_abi_version(void);

/**
 * @brief Get library version string
 * @return Pointer to null-terminated version string (valid for program lifetime)
 */
const char* bitnet_version(void);

/**
 * @brief Initialize the BitNet library
 *
 * Must be called before any other BitNet functions. Thread-safe and can be
 * called multiple times safely.
 *
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_init(void);

/**
 * @brief Cleanup and shutdown the BitNet library
 *
 * Should be called when the library is no longer needed. Thread-safe.
 *
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_cleanup(void);

/* ========================================================================== */
/* Model Management Functions                                                 */
/* ========================================================================== */

/**
 * @brief Load a model from file (exact signature compatibility)
 *
 * @param path Null-terminated string containing the path to the model file
 * @return Model ID (>= 0) on success, negative error code on failure
 */
int bitnet_model_load(const char* path);

/**
 * @brief Load a model with configuration
 *
 * @param path Null-terminated string containing the path to the model file
 * @param config Pointer to model configuration structure
 * @return Model ID (>= 0) on success, negative error code on failure
 */
int bitnet_model_load_with_config(const char* path, const bitnet_config_t* config);

/**
 * @brief Free a loaded model (exact signature compatibility)
 *
 * @param model_id Model ID returned by bitnet_model_load()
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_model_free(int model_id);

/**
 * @brief Check if a model is loaded
 *
 * @param model_id Model ID to check
 * @return 1 if loaded, 0 if not loaded, negative error code on failure
 */
int bitnet_model_is_loaded(int model_id);

/**
 * @brief Get model information
 *
 * @param model_id Model ID
 * @param info Pointer to structure to fill with model information
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_model_get_info(int model_id, bitnet_model_t* info);

/* ========================================================================== */
/* Inference Functions                                                       */
/* ========================================================================== */

/**
 * @brief Run inference (exact signature compatibility)
 *
 * @param model_id Model ID returned by bitnet_model_load()
 * @param prompt Null-terminated input prompt string
 * @param output Buffer to store generated text (null-terminated)
 * @param max_len Maximum length of output buffer (including null terminator)
 * @return Number of characters written (excluding null terminator) on success,
 *         negative error code on failure
 */
int bitnet_inference(int model_id, const char* prompt, char* output, size_t max_len);

/**
 * @brief Run inference with configuration
 *
 * @param model_id Model ID returned by bitnet_model_load()
 * @param prompt Null-terminated input prompt string
 * @param config Pointer to inference configuration structure
 * @param output Buffer to store generated text (null-terminated)
 * @param max_len Maximum length of output buffer (including null terminator)
 * @return Number of characters written (excluding null terminator) on success,
 *         negative error code on failure
 */
int bitnet_inference_with_config(int model_id, const char* prompt,
                                 const bitnet_inference_config_t* config,
                                 char* output, size_t max_len);

/* ========================================================================== */
/* Error Handling Functions                                                  */
/* ========================================================================== */

/**
 * @brief Get the last error message
 *
 * @return Pointer to null-terminated error message, or NULL if no error occurred
 */
const char* bitnet_get_last_error(void);

/**
 * @brief Clear the last error
 */
void bitnet_clear_last_error(void);

/* ========================================================================== */
/* Configuration Functions                                                   */
/* ========================================================================== */

/**
 * @brief Set the number of threads for CPU inference
 *
 * @param num_threads Number of threads to use (0 for auto-detection)
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_set_num_threads(uint32_t num_threads);

/**
 * @brief Get the current number of threads
 *
 * @return Number of threads currently in use
 */
uint32_t bitnet_get_num_threads(void);

/**
 * @brief Enable or disable GPU acceleration
 *
 * @param enable 1 to enable GPU, 0 to disable
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_set_gpu_enabled(int enable);

/**
 * @brief Check if GPU acceleration is available
 *
 * @return 1 if GPU is available, 0 if not available
 */
int bitnet_is_gpu_available(void);

/* ========================================================================== */
/* Performance Monitoring Functions                                          */
/* ========================================================================== */

/**
 * @brief Get performance metrics for a model
 *
 * @param model_id Model ID
 * @param metrics Pointer to structure to fill with performance metrics
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_get_performance_metrics(int model_id, bitnet_performance_metrics_t* metrics);

/**
 * @brief Reset performance metrics for a model
 *
 * @param model_id Model ID
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_reset_performance_metrics(int model_id);

/* ========================================================================== */
/* Advanced Features (Task 8.2)                                             */
/* ========================================================================== */

/**
 * @brief Start batch inference for multiple prompts
 *
 * @param model_id Model ID
 * @param prompts Array of null-terminated prompt strings
 * @param num_prompts Number of prompts in the array
 * @param config Inference configuration
 * @param outputs Array of output buffers
 * @param max_lens Array of maximum lengths for each output buffer
 * @return Number of successful inferences, negative error code on failure
 */
int bitnet_batch_inference(int model_id, const char** prompts, size_t num_prompts,
                          const bitnet_inference_config_t* config,
                          char** outputs, size_t* max_lens);

/**
 * @brief Start streaming inference
 *
 * @param model_id Model ID
 * @param prompt Input prompt
 * @param config Inference configuration
 * @param stream_config Streaming configuration
 * @return Stream ID (>= 0) on success, negative error code on failure
 */
int bitnet_start_streaming(int model_id, const char* prompt,
                          const bitnet_inference_config_t* config,
                          const bitnet_stream_config_t* stream_config);

/**
 * @brief Stop streaming inference
 *
 * @param stream_id Stream ID returned by bitnet_start_streaming()
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_stop_streaming(int stream_id);

/**
 * @brief Get next token from stream
 *
 * @param stream_id Stream ID
 * @param token Buffer to store the token
 * @param max_len Maximum length of token buffer
 * @return Length of token on success, 0 if stream finished, negative error code on failure
 */
int bitnet_stream_next_token(int stream_id, char* token, size_t max_len);

/* ========================================================================== */
/* Memory Management Functions                                               */
/* ========================================================================== */

/**
 * @brief Set memory limit for the library
 *
 * @param limit_bytes Memory limit in bytes (0 for no limit)
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_set_memory_limit(uint64_t limit_bytes);

/**
 * @brief Get current memory usage
 *
 * @return Current memory usage in bytes
 */
uint64_t bitnet_get_memory_usage(void);

/**
 * @brief Perform garbage collection
 *
 * @return BITNET_SUCCESS on success, error code on failure
 */
int bitnet_garbage_collect(void);

#ifdef __cplusplus
}
#endif

#endif /* BITNET_H */
