/// Ternary matmul kernel — exploits {-1, 0, +1} weight structure.
///
/// For BitNet ternary weights only three operations are needed per element:
///   w == +1 -> accumulate +activation
///   w == -1 -> accumulate -activation
///   w ==  0 -> skip
///
/// Packed representation: two bits per weight in uint8 containers.
///   00 -> 0, 01 -> +1, 11 -> -1  (10 reserved/unused)

/// Ternary matmul: C[M,N] = ternary_W[M,K] * A[K,N]
///
/// Weights are packed 4-per-byte (2 bits each).
/// packed_w layout: row-major, packed_k = ceil(K/4) bytes per row.
__kernel void ternary_matmul(
    __global const uchar* restrict packed_w,
    __global const float* restrict activations,
    __global       float* restrict output,
    const int M,
    const int K,
    const int N,
    const int packed_k)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    const int w_row_offset = row * packed_k;

    for (int byte_idx = 0; byte_idx < packed_k; ++byte_idx) {
        uchar pack = packed_w[w_row_offset + byte_idx];
        int base_k = byte_idx * 4;

        for (int sub = 0; sub < 4; ++sub) {
            int k_idx = base_k + sub;
            if (k_idx >= K) break;

            uchar bits = (pack >> (sub * 2)) & 0x03;
            if (bits == 1) {
                acc += activations[k_idx * N + col];
            } else if (bits == 3) {
                acc -= activations[k_idx * N + col];
            }
        }
    }

    output[row * N + col] = acc;
}

/// POPCOUNT-based ternary inner product.
///
/// Weights split into two bitmasks per row:
///   plus_mask[row, word_k]:  bit j set if w[row, word_k*32+j] == +1
///   minus_mask[row, word_k]: bit j set if w[row, word_k*32+j] == -1
///
/// Inner product = popcount(plus & act_bits) - popcount(minus & act_bits)
/// where act_bits encodes sign(activation) for binary approximation.
__kernel void ternary_popcount_matmul(
    __global const uint*  restrict plus_mask,
    __global const uint*  restrict minus_mask,
    __global const uint*  restrict act_sign_bits,
    __global       int*   restrict output,
    const int M,
    const int K,
    const int N,
    const int words_per_row)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    int acc = 0;
    const int w_offset = row * words_per_row;
    const int a_offset = col * words_per_row;

    for (int w = 0; w < words_per_row; ++w) {
        uint p = plus_mask[w_offset + w];
        uint m = minus_mask[w_offset + w];
        uint a = act_sign_bits[a_offset + w];

        acc += popcount(p & a) - popcount(m & a);
    }

    output[row * N + col] = acc;
}
