#include "crates/bitnet-ggml-ffi/csrc/ggml_consts.c"
#include <stdio.h>

int main() {
    printf("bitnet_iq2s_qk() = %d\n", bitnet_iq2s_qk());
    printf("bitnet_iq2s_block_size_bytes() = %d\n", bitnet_iq2s_block_size_bytes());
    printf("bitnet_iq2s_requires_qk_multiple() = %d\n", bitnet_iq2s_requires_qk_multiple());
    return 0;
}