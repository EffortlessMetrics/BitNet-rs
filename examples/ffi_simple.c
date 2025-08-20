// Simple C example demonstrating BitNet FFI usage
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// FFI declarations (normally from bitnet_ffi.h)
typedef void* bitnet_handle_t;

// API version check
int bitnet_ffi_api_version(void);

// Model operations
bitnet_handle_t bitnet_model_load(const char* path);
void bitnet_model_free(bitnet_handle_t handle);
int bitnet_model_vocab_size(bitnet_handle_t handle);

// Quantization operations
int bitnet_quantize_i2s(const float* input, unsigned char* output, 
                        float* scales, size_t len);

int main(int argc, char* argv[]) {
    printf("BitNet FFI Simple Example\n");
    printf("=========================\n\n");
    
    // Check API version
    int api_version = bitnet_ffi_api_version();
    printf("1. API Version: %d\n", api_version);
    assert(api_version == 1);
    
    // Test quantization
    printf("\n2. Testing quantization...\n");
    const size_t test_size = 128;
    float* input = malloc(test_size * sizeof(float));
    unsigned char* output = malloc(test_size / 4); // 2-bit packing
    float* scales = malloc((test_size / 32) * sizeof(float));
    
    // Initialize test data
    for (size_t i = 0; i < test_size; i++) {
        input[i] = (float)i / test_size - 0.5f;
    }
    
    int result = bitnet_quantize_i2s(input, output, scales, test_size);
    if (result == 0) {
        printf("   ✓ Quantization successful\n");
        printf("   Input range: [%.3f, %.3f]\n", input[0], input[test_size-1]);
        printf("   First scale: %.6f\n", scales[0]);
    } else {
        printf("   ✗ Quantization failed with code: %d\n", result);
    }
    
    // Load model if path provided
    if (argc > 1) {
        printf("\n3. Loading model: %s\n", argv[1]);
        bitnet_handle_t model = bitnet_model_load(argv[1]);
        
        if (model != NULL) {
            int vocab_size = bitnet_model_vocab_size(model);
            printf("   ✓ Model loaded successfully\n");
            printf("   Vocab size: %d\n", vocab_size);
            
            bitnet_model_free(model);
            printf("   ✓ Model freed\n");
        } else {
            printf("   ✗ Failed to load model\n");
        }
    } else {
        printf("\n3. No model path provided. Use: %s <model.gguf>\n", argv[0]);
    }
    
    // Cleanup
    free(input);
    free(output);
    free(scales);
    
    printf("\n✅ FFI example completed successfully!\n");
    return 0;
}