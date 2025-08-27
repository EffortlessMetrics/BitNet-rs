#ifndef GGML_QUANTS_H
#define GGML_QUANTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// IQ2_S block as used by GGML
#ifndef QK_IQ2_S
#define QK_IQ2_S 256
#endif

typedef uint16_t ggml_fp16_t;

typedef struct {
    ggml_fp16_t d;                 // block scale stored as fp16
    uint8_t     qs[QK_IQ2_S/4];    // 2-bit quants packed four per byte
    uint8_t     qh[QK_IQ2_S/32];   // unused in this simplified implementation
    uint8_t     scales[QK_IQ2_S/32]; // unused scales
} block_iq2_s;

void dequantize_row_iq2_s(const void * restrict x, float * restrict y, int64_t k);
size_t quantize_iq2_s(const float * restrict src, void * restrict dst,
                      int64_t nrows, int64_t n_per_row,
                      const float * quant_weights);

#ifdef __cplusplus
}
#endif

#endif // GGML_QUANTS_H
