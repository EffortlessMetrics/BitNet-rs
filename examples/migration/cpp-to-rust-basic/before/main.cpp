/**
 * Basic BitNet C++ Example - BEFORE Migration
 * 
 * This is the original C++ implementation that we'll migrate to Rust.
 * It demonstrates typical C++ BitNet usage patterns.
 */

#include <iostream>
#include <cstdlib>
#include <cstring>

// Note: These would be actual BitNet C++ headers
// For this example, we'll use placeholder function signatures
extern "C" {
    typedef struct BitNetModel BitNetModel;
    
    BitNetModel* bitnet_load_model(const char* model_path);
    char* bitnet_generate(BitNetModel* model, const char* prompt, int max_tokens, float temperature);
    void bitnet_free_string(char* str);
    void bitnet_free_model(BitNetModel* model);
    const char* bitnet_get_error();
}

// Placeholder implementations for demonstration
BitNetModel* bitnet_load_model(const char* model_path) {
    std::cout << "Loading model: " << model_path << std::endl;
    // Return dummy pointer for demonstration
    return reinterpret_cast<BitNetModel*>(0x1234);
}

char* bitnet_generate(BitNetModel* model, const char* prompt, int max_tokens, float temperature) {
    std::cout << "Generating with prompt: '" << prompt << "'" << std::endl;
    std::cout << "Max tokens: " << max_tokens << ", Temperature: " << temperature << std::endl;
    
    // Simulate generation - in real implementation this would call actual BitNet
    const char* response = "This is a generated response from the C++ implementation.";
    char* result = new char[strlen(response) + 1];
    strcpy(result, response);
    return result;
}

void bitnet_free_string(char* str) {
    delete[] str;
}

void bitnet_free_model(BitNetModel* model) {
    // In real implementation, this would free the actual model
    std::cout << "Freeing model" << std::endl;
}

const char* bitnet_get_error() {
    return "No error";
}

// Main application
int main() {
    std::cout << "BitNet C++ Example - Basic Usage" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Load model
    const char* model_path = "model.gguf";
    BitNetModel* model = bitnet_load_model(model_path);
    
    if (!model) {
        std::cerr << "Error: Failed to load model: " << bitnet_get_error() << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model loaded successfully" << std::endl;
    
    // Test prompts
    const char* prompts[] = {
        "Hello, world!",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint."
    };
    
    const int num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    
    // Generation parameters
    const int max_tokens = 100;
    const float temperature = 0.7f;
    
    // Process each prompt
    for (int i = 0; i < num_prompts; i++) {
        std::cout << std::endl << "Prompt " << (i + 1) << ": " << prompts[i] << std::endl;
        std::cout << "Response: ";
        
        // Generate response
        char* result = bitnet_generate(model, prompts[i], max_tokens, temperature);
        
        if (!result) {
            std::cerr << "Error: Generation failed: " << bitnet_get_error() << std::endl;
            continue;
        }
        
        std::cout << result << std::endl;
        
        // Important: Must manually free the result
        bitnet_free_string(result);
    }
    
    // Cleanup
    bitnet_free_model(model);
    
    std::cout << std::endl << "✅ Example completed successfully" << std::endl;
    return 0;
}

/**
 * Issues with this C++ implementation:
 * 
 * 1. Manual memory management - easy to forget bitnet_free_string()
 * 2. No RAII - resources not automatically cleaned up
 * 3. Error handling via null pointers and global error state
 * 4. No type safety - easy to pass wrong parameters
 * 5. Complex build system requirements
 * 6. Potential for memory leaks and segfaults
 * 7. No modern C++ features like smart pointers
 * 8. Global error state is not thread-safe
 * 
 * The Rust version addresses all of these issues.
 */