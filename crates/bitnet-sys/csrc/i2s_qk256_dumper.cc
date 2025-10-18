// GGML I2_S (QK=256) block dequantization helper
//
// This utility dequantizes a single GGML I2_S block (64 bytes → 256 f32 values)
// using llama.cpp's reference implementation to confirm the exact code→float mapping.
//
// Compile and run this once to lock down the LUT, then delete or move to dev tools.
//
// Usage:
//   g++ -std=c++17 -O2 -I/path/to/llama.cpp i2s_qk256_dumper.cc -o i2s_qk256_dumper
//   ./i2s_qk256_dumper <path_to_gguf_model.gguf>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

// GGML I2_S block structure (from ggml-quants.h)
// NOTE: Adjust this based on actual llama.cpp header
#define QK_K 256

struct block_iq2_s {
    uint8_t qs[QK_K/4]; // 64 bytes of 2-bit packed codes
    // Note: GGML IQ2_S may have additional fields (scales, etc.)
    // Adjust based on actual llama.cpp definition
};

// Placeholder dequantization function
// TODO: Replace with actual llama.cpp function linkage or inline implementation
void dequantize_row_iq2_s_reference(const void* src, float* dst, int k) {
    // This is a PLACEHOLDER!
    // You need to either:
    // 1. Link against llama.cpp's dequant_row_iq2_s
    // 2. Copy the exact implementation from ggml-quants.c
    // 3. Use dlopen/dlsym to load it at runtime

    const block_iq2_s* x = static_cast<const block_iq2_s*>(src);

    // PLACEHOLDER LOGIC (replace with real llama.cpp logic)
    // This assumes simple 2-bit unpacking with {-3, -1, +1, +3} LUT
    const float lut[4] = {-3.0f, -1.0f, 1.0f, 3.0f};

    for (int i = 0; i < 64; i++) {
        uint8_t byte = x->qs[i];
        int base_idx = i * 4;

        dst[base_idx + 0] = lut[byte & 0x03];
        dst[base_idx + 1] = lut[(byte >> 2) & 0x03];
        dst[base_idx + 2] = lut[(byte >> 4) & 0x03];
        dst[base_idx + 3] = lut[(byte >> 6) & 0x03];
    }

    // TODO: Apply any global/per-tensor scale factor if present
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <gguf_model.gguf>\n", argv[0]);
        fprintf(stderr, "\nThis helper dequantizes a few GGML I2_S blocks to confirm the code→float LUT.\n");
        return 1;
    }

    // For now, just demonstrate with synthetic test blocks
    printf("GGML I2_S (QK=256) Block Dequantization Test\n");
    printf("============================================\n\n");

    // Test block 1: All codes = 0
    {
        block_iq2_s blk;
        memset(blk.qs, 0x00, 64); // All codes 0

        float out[QK_K];
        dequantize_row_iq2_s_reference(&blk, out, QK_K);

        printf("Test 1: All codes = 0\n");
        printf("  First 8 values: ");
        for (int i = 0; i < 8; i++) {
            printf("%.2f ", out[i]);
        }
        printf("\n  Expected: all same value (code 0 mapping)\n\n");
    }

    // Test block 2: All codes = 1
    {
        block_iq2_s blk;
        memset(blk.qs, 0x55, 64); // 0b_01_01_01_01 → all codes 1

        float out[QK_K];
        dequantize_row_iq2_s_reference(&blk, out, QK_K);

        printf("Test 2: All codes = 1\n");
        printf("  First 8 values: ");
        for (int i = 0; i < 8; i++) {
            printf("%.2f ", out[i]);
        }
        printf("\n  Expected: all same value (code 1 mapping)\n\n");
    }

    // Test block 3: All codes = 2
    {
        block_iq2_s blk;
        memset(blk.qs, 0xAA, 64); // 0b_10_10_10_10 → all codes 2

        float out[QK_K];
        dequantize_row_iq2_s_reference(&blk, out, QK_K);

        printf("Test 3: All codes = 2\n");
        printf("  First 8 values: ");
        for (int i = 0; i < 8; i++) {
            printf("%.2f ", out[i]);
        }
        printf("\n  Expected: all same value (code 2 mapping)\n\n");
    }

    // Test block 4: All codes = 3
    {
        block_iq2_s blk;
        memset(blk.qs, 0xFF, 64); // 0b_11_11_11_11 → all codes 3

        float out[QK_K];
        dequantize_row_iq2_s_reference(&blk, out, QK_K);

        printf("Test 4: All codes = 3\n");
        printf("  First 8 values: ");
        for (int i = 0; i < 8; i++) {
            printf("%.2f ", out[i]);
        }
        printf("\n  Expected: all same value (code 3 mapping)\n\n");
    }

    // Test block 5: Pattern 0,1,2,3 repeating
    {
        block_iq2_s blk;
        for (int i = 0; i < 64; i++) {
            blk.qs[i] = 0b_11_10_01_00; // codes 0,1,2,3
        }

        float out[QK_K];
        dequantize_row_iq2_s_reference(&blk, out, QK_K);

        printf("Test 5: Pattern 0,1,2,3 repeating\n");
        printf("  First 16 values:\n  ");
        for (int i = 0; i < 16; i++) {
            printf("%.2f ", out[i]);
            if ((i + 1) % 4 == 0) printf("\n  ");
        }
        printf("\n  Expected: code 0, code 1, code 2, code 3 pattern\n\n");
    }

    printf("============================================\n");
    printf("ACTION REQUIRED:\n");
    printf("1. Compare these outputs with Rust kernel's code_to_f32() LUT\n");
    printf("2. If values differ, update crates/bitnet-models/src/quant/i2s_qk256.rs\n");
    printf("3. Check if there's a global/per-tensor scale factor applied\n");
    printf("4. Run unit tests to verify parity\n");

    return 0;
}
