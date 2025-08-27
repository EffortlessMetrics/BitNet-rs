#include "crates/bitnet-ggml-ffi/csrc/ggml/include/ggml/ggml.h"
#include "crates/bitnet-ggml-ffi/csrc/ggml/src/ggml-quants.h"
#include <stdio.h>

int main() {
    printf("QK_IQ2_S = %d\n", QK_IQ2_S);
    printf("sizeof(block_iq2_s) = %lu\n", sizeof(block_iq2_s));
    printf("sizeof(ggml_fp16_t) = %lu\n", sizeof(ggml_fp16_t));
    printf("QK_IQ2_S/4 = %d\n", QK_IQ2_S/4);
    printf("QK_IQ2_S/32 = %d\n", QK_IQ2_S/32);
    return 0;
}