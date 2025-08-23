// Minimal public shim to access the reference IQ2_S dequantizer.
// We include GGML's quants implementation and forward exactly one symbol.

#include "ggml/include/ggml/ggml.h"
#include "ggml/src/ggml-quants.h"

// ggml-quants.c defines the row dequantizers for each quant type.
#include "ggml/src/ggml-quants.c"

// Export a stable symbol name for Rust to link.
void bitnet_dequantize_row_iq2_s(const void *src, float *dst, int n) {
    // GGML dequantizers are row-based: src points to a packed row (array of blocks)
    // n is the number of elements (float weights) to produce
    dequantize_row_iq2_s(src, dst, (int64_t)n);
}