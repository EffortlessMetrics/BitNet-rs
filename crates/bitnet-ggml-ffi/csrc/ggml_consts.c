// ============================================================================
// GGML Constants Extraction - FFI Bridge
// ============================================================================
// llama.cpp API version: See VENDORED_GGML_COMMIT for exact commit hash
// BitNet.rs integration: AC6 FFI build hygiene consolidation (Issue #469)
// Compatible with: BitNet C++ v0.1.0-mvp and later
// Build date: Generated at compile time from csrc/VENDORED_GGML_COMMIT
//
// This shim provides stable access to GGML's internal constants for Rust.
// The vendored GGML commit hash is embedded in the build configuration.
// See build.rs for version detection and build-time configuration.
// ============================================================================

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
    // Derived at build time by scanning GGML's source.
#ifdef BITNET_IQ2S_DEQUANT_NEEDS_QK_MULTIPLE
    return 1;
#else
    return 0;
#endif
}
