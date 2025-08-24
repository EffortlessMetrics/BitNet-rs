// GGML constants extraction for Rust
// This provides a stable interface to GGML's internal constants
// without Rust needing to parse or guess the layout

#include "ggml/include/ggml/ggml.h"
#include "ggml/src/ggml-quants.h"
#include <stdint.h>

// Export IQ2_S constants for Rust
int32_t bitnet_iq2s_qk(void) {
    return QK_IQ2_S;
}

int32_t bitnet_iq2s_block_size_bytes(void) {
    return (int32_t)sizeof(block_iq2_s);
}

// Helper to check if GGML's dequantize_row_iq2_s requires n to be multiple of QK
// This is critical for tail handling safety
int32_t bitnet_iq2s_requires_qk_multiple(void) {
    // Most GGML dequantizers assume n % QK == 0
    // Return 1 to indicate this requirement (safer assumption)
    return 1;
}