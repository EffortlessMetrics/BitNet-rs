/**
 * @file c_integration_test.c
 * @brief C integration test for BitNet C API
 * 
 * This file demonstrates how to use the BitNet C API from C code
 * and validates that the API is compatible with C compilers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Include the BitNet C API header
// In a real build system, this would be: #include <bitnet.h>
#include "../include/bitnet.h"

/**
 * Test basic library initialization and cleanup
 */
void test_initialization(void) {
    printf("Testing library initialization...\n");
    
    int result = bitnet_init();
    assert(result == BITNET_SUCCESS);
    
    result = bitnet_cleanup();
    assert(result == BITNET_SUCCESS);
    
    printf("✓ Initialization test passed\n");
}

/**
 * Test version and ABI information
 */
void test_version_info(void) {
    printf("Testing version information...\n");
    
    // Initialize library
    int result = bitnet_init();
    assert(result == BITNET_SUCCESS);
    
    // Test version string
    const char* version = bitnet_version();
    assert(version != NULL);
    assert(strlen(version) > 0);
    printf("  Library version: %s\n", version);
    
    // Test ABI version
    uint32_t abi_version = bitnet_abi_version();
    assert(abi_version == BITNET_ABI_VERSION);
    printf("  ABI version: %u\n", abi_version);
    
    bitnet_cleanup();
    printf("✓ Version info test passed\n");
}

/**
 * Test error handling
 */
void test_error_handling(void) {
    printf("Testing error handling...\n");
    
    bitnet_init();
    
    // Clear any existing errors
    bitnet_clear_last_error();
    const char* error = bitnet_get_last_error();
    assert(error == NULL);
    
    // Trigger an error with invalid input
    int result = bitnet_model_load(NULL);
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    // Check that error message is available
    error = bitnet_get_last_error();
    assert(error != NULL);
    assert(strlen(error) > 0);
    printf("  Error message: %s\n", error);
    
    // Clear error
    bitnet_clear_last_error();
    error = bitnet_get_last_error();
    assert(error == NULL);
    
    bitnet_cleanup();
    printf("✓ Error handling test passed\n");
}

/**
 * Test configuration management
 */
void test_configuration(void) {
    printf("Testing configuration management...\n");
    
    bitnet_init();
    
    // Test thread count management
    uint32_t original_threads = bitnet_get_num_threads();
    assert(original_threads > 0);
    printf("  Original thread count: %u\n", original_threads);
    
    int result = bitnet_set_num_threads(4);
    assert(result == BITNET_SUCCESS);
    
    uint32_t current_threads = bitnet_get_num_threads();
    assert(current_threads == 4);
    printf("  Set thread count to: %u\n", current_threads);
    
    // Test GPU availability
    int gpu_available = bitnet_is_gpu_available();
    printf("  GPU available: %s\n", gpu_available ? "Yes" : "No");
    
    // Test GPU enable/disable
    result = bitnet_set_gpu_enabled(0);
    assert(result == BITNET_SUCCESS);
    
    bitnet_cleanup();
    printf("✓ Configuration test passed\n");
}

/**
 * Test memory management
 */
void test_memory_management(void) {
    printf("Testing memory management...\n");
    
    bitnet_init();
    
    // Test memory limit setting
    int result = bitnet_set_memory_limit(1024 * 1024 * 1024); // 1GB
    assert(result == BITNET_SUCCESS);
    
    // Test memory usage query
    uint64_t usage = bitnet_get_memory_usage();
    printf("  Current memory usage: %llu bytes\n", (unsigned long long)usage);
    
    // Test garbage collection
    result = bitnet_garbage_collect();
    assert(result == BITNET_SUCCESS);
    
    bitnet_cleanup();
    printf("✓ Memory management test passed\n");
}

/**
 * Test model operations with invalid inputs
 */
void test_model_operations(void) {
    printf("Testing model operations...\n");
    
    bitnet_init();
    
    // Test invalid model loading
    int model_id = bitnet_model_load(NULL);
    assert(model_id == BITNET_ERROR_INVALID_ARGUMENT);
    
    // Test operations on invalid model ID
    int result = bitnet_model_is_loaded(-1);
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    result = bitnet_model_free(-1);
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    bitnet_model_t model_info = {0};
    result = bitnet_model_get_info(-1, &model_info);
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    bitnet_cleanup();
    printf("✓ Model operations test passed\n");
}

/**
 * Test inference operations with invalid inputs
 */
void test_inference_operations(void) {
    printf("Testing inference operations...\n");
    
    bitnet_init();
    
    char output[1024];
    
    // Test invalid model ID
    int result = bitnet_inference(-1, "test", output, sizeof(output));
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    // Test null prompt
    result = bitnet_inference(0, NULL, output, sizeof(output));
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    // Test null output
    result = bitnet_inference(0, "test", NULL, sizeof(output));
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    // Test zero max_len
    result = bitnet_inference(0, "test", output, 0);
    assert(result == BITNET_ERROR_INVALID_ARGUMENT);
    
    bitnet_cleanup();
    printf("✓ Inference operations test passed\n");
}

/**
 * Test configuration structures
 */
void test_configuration_structures(void) {
    printf("Testing configuration structures...\n");
    
    // Test model configuration
    bitnet_config_t config = {0};
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_layers = 32;
    config.num_heads = 32;
    config.temperature = 1.0f;
    config.use_gpu = 0;
    config.batch_size = 1;
    
    assert(config.vocab_size == 32000);
    assert(config.hidden_size == 4096);
    assert(config.temperature == 1.0f);
    
    // Test inference configuration
    bitnet_inference_config_t inf_config = {0};
    inf_config.max_length = 2048;
    inf_config.max_new_tokens = 512;
    inf_config.temperature = 0.8f;
    inf_config.top_k = 50;
    inf_config.top_p = 0.9f;
    inf_config.do_sample = 1;
    
    assert(inf_config.max_length == 2048);
    assert(inf_config.temperature == 0.8f);
    assert(inf_config.do_sample == 1);
    
    // Test performance metrics structure
    bitnet_performance_metrics_t metrics = {0};
    metrics.tokens_per_second = 100.0f;
    metrics.latency_ms = 50.0f;
    metrics.memory_usage_mb = 1024.0f;
    
    assert(metrics.tokens_per_second == 100.0f);
    assert(metrics.latency_ms == 50.0f);
    assert(metrics.memory_usage_mb == 1024.0f);
    
    printf("✓ Configuration structures test passed\n");
}

/**
 * Main test function
 */
int main(void) {
    printf("BitNet C API Integration Test\n");
    printf("=============================\n\n");
    
    // Run all tests
    test_initialization();
    test_version_info();
    test_error_handling();
    test_configuration();
    test_memory_management();
    test_model_operations();
    test_inference_operations();
    test_configuration_structures();
    
    printf("\n✓ All C integration tests passed!\n");
    return 0;
}

/**
 * Example usage function demonstrating typical API usage
 */
void example_usage(void) {
    printf("\nExample Usage:\n");
    printf("--------------\n");
    
    // Initialize the library
    if (bitnet_init() != BITNET_SUCCESS) {
        fprintf(stderr, "Failed to initialize BitNet library\n");
        return;
    }
    
    // Configure the system
    bitnet_set_num_threads(4);
    bitnet_set_memory_limit(2ULL * 1024 * 1024 * 1024); // 2GB
    
    // Load a model (this would fail with a real path in this test)
    const char* model_path = "path/to/model.gguf";
    int model_id = bitnet_model_load(model_path);
    
    if (model_id >= 0) {
        printf("Model loaded successfully with ID: %d\n", model_id);
        
        // Get model information
        bitnet_model_t model_info;
        if (bitnet_model_get_info(model_id, &model_info) == BITNET_SUCCESS) {
            printf("Model vocab size: %u\n", model_info.vocab_size);
            printf("Model hidden size: %u\n", model_info.hidden_size);
        }
        
        // Perform inference
        const char* prompt = "Hello, how are you?";
        char output[1024];
        
        int result = bitnet_inference(model_id, prompt, output, sizeof(output));
        if (result >= 0) {
            printf("Generated text: %s\n", output);
        } else {
            printf("Inference failed with error code: %d\n", result);
            const char* error = bitnet_get_last_error();
            if (error) {
                printf("Error message: %s\n", error);
            }
        }
        
        // Get performance metrics
        bitnet_performance_metrics_t metrics;
        if (bitnet_get_performance_metrics(model_id, &metrics) == BITNET_SUCCESS) {
            printf("Tokens per second: %.2f\n", metrics.tokens_per_second);
            printf("Latency: %.2f ms\n", metrics.latency_ms);
            printf("Memory usage: %.2f MB\n", metrics.memory_usage_mb);
        }
        
        // Clean up the model
        bitnet_model_free(model_id);
    } else {
        printf("Failed to load model, error code: %d\n", model_id);
        const char* error = bitnet_get_last_error();
        if (error) {
            printf("Error message: %s\n", error);
        }
    }
    
    // Clean up the library
    bitnet_cleanup();
}