// Minimal GGML quants header stub for IQ2_S support
#ifndef GGML_QUANTS_H
#define GGML_QUANTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// IQ2_S block structure (placeholder - actual structure from llama.cpp needed)
typedef struct {
    uint8_t qs[QK_IQ2_S/4];      // 2-bit quantized values
    uint8_t scales[QK_IQ2_S/32]; // Scales
    int8_t  shift;                // Shift value
} block_iq2_s;

// Dequantization functions
void dequantize_row_iq2_s(const void * restrict x, float * restrict y, int64_t k);

// Quantization functions (optional for now)
size_t quantize_iq2_s(const float * restrict src, void * restrict dst, int64_t nrows, int64_t n_per_row, const float * quant_weights);

#ifdef __cplusplus
}
#endif

#endif // GGML_QUANTS_H