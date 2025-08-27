// Minimal public shim to access the reference IQ2_S quantizer/dequantizer.
// We include GGML's quants implementation and forward stable symbol names.

#include "ggml/include/ggml/ggml.h"
#include "ggml/src/ggml-quants.h"

// ggml-quants.c defines the row quantizers and dequantizers for each quant type.
#include "ggml/src/ggml-quants.c"

// Export stable symbol names for Rust to link.
void bitnet_dequantize_row_iq2_s(const void *src, float *dst, int n) {
    dequantize_row_iq2_s(src, dst, (int64_t)n);
}

size_t bitnet_quantize_iq2_s(const float *src, void *dst, int64_t nrow, int64_t n_per_row) {
    return quantize_iq2_s(src, dst, nrow, n_per_row, NULL);
}
