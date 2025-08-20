#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    printf("Testing tokenizer with model: %s\n", model_path);

    // Initialize backend
    llama_backend_init();
    
    // Set up model params
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only for testing
    
    // Load the model
    printf("Loading model...\n");
    struct llama_model *model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        llama_backend_free();
        return 1;
    }
    
    // Get vocab info
    int n_vocab = llama_n_vocab(model);
    printf("Model loaded successfully!\n");
    printf("Vocab size: %d\n", n_vocab);
    
    // Test tokenization
    const char *test_text = "Hello, world!";
    printf("\nTesting tokenization of: '%s'\n", test_text);
    
    // Allocate token buffer
    int max_tokens = 100;
    int32_t *tokens = malloc(sizeof(int32_t) * max_tokens);
    
    // First pass - get number of tokens
    int n_tokens = llama_tokenize(
        model,
        test_text,
        strlen(test_text),
        NULL,
        0,
        true,  // add_special
        false  // parse_special
    );
    
    printf("First pass - estimated tokens: %d\n", n_tokens);
    
    if (n_tokens < 0) {
        fprintf(stderr, "ERROR: Tokenization failed on first pass (returned %d)\n", n_tokens);
        
        // Try to understand why
        printf("\nDiagnostics:\n");
        printf("- Text length: %zu\n", strlen(test_text));
        printf("- Model vocab size: %d\n", n_vocab);
        
        // Check if it's a BPE vs SPM issue
        printf("\nTrying without special tokens...\n");
        n_tokens = llama_tokenize(
            model,
            test_text,
            strlen(test_text),
            NULL,
            0,
            false,  // no special tokens
            false
        );
        printf("Result without special tokens: %d\n", n_tokens);
        
    } else if (n_tokens > 0 && n_tokens < max_tokens) {
        // Second pass - get actual tokens
        int actual_tokens = llama_tokenize(
            model,
            test_text,
            strlen(test_text),
            tokens,
            max_tokens,
            true,
            false
        );
        
        printf("Second pass - actual tokens: %d\n", actual_tokens);
        
        if (actual_tokens > 0) {
            printf("Tokens: [");
            for (int i = 0; i < actual_tokens; i++) {
                printf("%d", tokens[i]);
                if (i < actual_tokens - 1) printf(", ");
            }
            printf("]\n");
            
            // Try to decode back
            printf("Decoded text: ");
            for (int i = 0; i < actual_tokens; i++) {
                const char *token_str = llama_token_get_text(model, tokens[i]);
                if (token_str) {
                    printf("%s", token_str);
                }
            }
            printf("\n");
        }
    }
    
    // Clean up
    free(tokens);
    llama_free_model(model);
    llama_backend_free();
    
    return 0;
}