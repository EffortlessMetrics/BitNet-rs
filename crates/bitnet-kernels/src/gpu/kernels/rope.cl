// Rotary Position Embedding kernel for Intel Arc A770
// Applies rotation to pairs of dimensions in Q/K vectors
//
// RoPE formula for each pair (x_{2i}, x_{2i+1}) at position pos:
//   freq_i = 1 / (base^(2i/d))
//   angle  = pos * freq_i
//   y_{2i}   = x_{2i}   * cos(angle) - x_{2i+1} * sin(angle)
//   y_{2i+1} = x_{2i}   * sin(angle) + x_{2i+1} * cos(angle)

// Real-time sin/cos computation variant.
// Global work: [head_dim/2, num_heads, seq_len]
__kernel void rope_apply(
    __global float* data,           // [seq_len, num_heads, head_dim] — modified in place
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float theta_base,         // typically 10000.0
    const int position_offset       // for KV cache continuation
) {
    const int pair_idx = get_global_id(0);  // which dimension pair (head_dim/2)
    const int head = get_global_id(1);      // which attention head
    const int pos = get_global_id(2);       // sequence position

    if (pair_idx >= head_dim / 2 || head >= num_heads || pos >= seq_len) return;

    int actual_pos = pos + position_offset;

    // Compute rotation frequency: freq = base^(-2i/d)
    float freq = 1.0f / pow(theta_base, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)actual_pos * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Apply rotation to pair (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
    int base_idx = ((pos * num_heads + head) * head_dim) + 2 * pair_idx;
    float x0 = data[base_idx];
    float x1 = data[base_idx + 1];

    data[base_idx]     = x0 * cos_val - x1 * sin_val;
    data[base_idx + 1] = x0 * sin_val + x1 * cos_val;
}

// Pre-computed frequency table version (faster for repeated calls).
// cos_cache and sin_cache are indexed by [position, pair_idx].
// Global work: [head_dim/2, num_heads, seq_len]
__kernel void rope_apply_cached(
    __global float* data,            // [seq_len, num_heads, head_dim]
    __global const float* cos_cache, // [max_seq, head_dim/2]
    __global const float* sin_cache, // [max_seq, head_dim/2]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int position_offset
) {
    const int pair_idx = get_global_id(0);
    const int head = get_global_id(1);
    const int pos = get_global_id(2);

    if (pair_idx >= head_dim / 2 || head >= num_heads || pos >= seq_len) return;

    int actual_pos = pos + position_offset;
    float cos_val = cos_cache[actual_pos * (head_dim / 2) + pair_idx];
    float sin_val = sin_cache[actual_pos * (head_dim / 2) + pair_idx];

    int base_idx = ((pos * num_heads + head) * head_dim) + 2 * pair_idx;
    float x0 = data[base_idx];
    float x1 = data[base_idx + 1];

    data[base_idx]     = x0 * cos_val - x1 * sin_val;
    data[base_idx + 1] = x0 * sin_val + x1 * cos_val;
}
