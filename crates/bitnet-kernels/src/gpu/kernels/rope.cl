// Rotary Position Embedding (RoPE) kernels for Intel Arc A770
//
// Two variants:
// - rope_apply: computes sin/cos on the fly per position
// - rope_apply_cached: uses a pre-computed frequency table
//
// Supports KV cache continuation via position_offset.

// RoPE with real-time sin/cos computation.
// Applies rotation to pairs (x[2d], x[2d+1]) at each position.
// Grid: (half_head_dim, seq_len, num_heads)
__kernel void rope_apply(
    __global float* x,              // [num_heads, seq_len, head_dim] — modified in place
    const int seq_len,
    const int head_dim,
    const float theta_base,         // base frequency (e.g. 10000.0)
    const int position_offset       // offset for KV cache continuation
) {
    const int d    = get_global_id(0);  // pair index within head [0, head_dim/2)
    const int pos  = get_global_id(1);
    const int head = get_global_id(2);

    const int half_dim = head_dim / 2;
    if (d >= half_dim || pos >= seq_len) return;

    // Frequency for this dimension pair
    float freq = 1.0f / pow(theta_base, (float)(2 * d) / (float)head_dim);
    float angle = (float)(pos + position_offset) * freq;

    float cos_a = cos(angle);
    float sin_a = sin(angle);

    int base_idx = head * seq_len * head_dim + pos * head_dim;
    float x0 = x[base_idx + 2 * d];
    float x1 = x[base_idx + 2 * d + 1];

    // Apply 2D rotation
    x[base_idx + 2 * d]     = x0 * cos_a - x1 * sin_a;
    x[base_idx + 2 * d + 1] = x0 * sin_a + x1 * cos_a;
}

// RoPE with pre-computed frequency table for repeated calls.
// Grid: (half_head_dim, seq_len, num_heads)
__kernel void rope_apply_cached(
    __global float* x,              // [num_heads, seq_len, head_dim] — modified in place
    __global const float* cos_cache, // [max_seq_len, half_head_dim]
    __global const float* sin_cache, // [max_seq_len, half_head_dim]
    const int seq_len,
    const int head_dim,
    const int position_offset       // offset for KV cache continuation
) {
    const int d    = get_global_id(0);
    const int pos  = get_global_id(1);
    const int head = get_global_id(2);

    const int half_dim = head_dim / 2;
    if (d >= half_dim || pos >= seq_len) return;

    int table_pos = pos + position_offset;
    float cos_a = cos_cache[table_pos * half_dim + d];
    float sin_a = sin_cache[table_pos * half_dim + d];

    int base_idx = head * seq_len * head_dim + pos * head_dim;
    float x0 = x[base_idx + 2 * d];
    float x1 = x[base_idx + 2 * d + 1];

    x[base_idx + 2 * d]     = x0 * cos_a - x1 * sin_a;
    x[base_idx + 2 * d + 1] = x0 * sin_a + x1 * cos_a;
}
