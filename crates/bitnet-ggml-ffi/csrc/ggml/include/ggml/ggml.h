// Minimal GGML header stub for IQ2_S support
// This is a placeholder until proper vendoring from llama.cpp
#ifndef GGML_H
#define GGML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Minimal type definitions needed for IQ2_S
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // ... other types ...
    GGML_TYPE_IQ2_S = 24,  // Placeholder value
    GGML_TYPE_COUNT,
} ggml_type;

// Quantization block sizes
#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32
#define QK_K 256
#define QK8_K 256

// IQ2_S specific
#define QK_IQ2_S 256

#ifdef __cplusplus
}
#endif

#endif // GGML_H