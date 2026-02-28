/// Rotary Position Embedding (RoPE) kernel for Intel Arc GPUs.
///
/// Applies rotary positional encoding to Q/K tensors in-place.
/// Each work-item handles one (even, odd) pair in the head dimension.
///
/// Launch with:
///   global_work_size = [half_head_dim, num_heads * seq_len]
///   local_work_size  = driver default or [32, 1]
///
/// The frequency table `freq_cos`/`freq_sin` has shape [max_seq_len × half_head_dim]
/// and is pre-computed on the host:
///   theta_i = 1 / (base ^ (2i / head_dim))
///   freq_cos[pos][i] = cos(pos * theta_i)
///   freq_sin[pos][i] = sin(pos * theta_i)

/// Apply RoPE to a single tensor (Q or K).
///
/// Args:
///   x        — [seq_len × num_heads × head_dim]  (row-major, in/out)
///   freq_cos — [max_seq_len × half_head_dim]
///   freq_sin — [max_seq_len × half_head_dim]
///   seq_len, num_heads, head_dim — tensor dimensions
///   pos_offset — starting position index (for KV-cache continuation)
__kernel void rope_apply(
    __global float* x,
    __global const float* freq_cos,
    __global const float* freq_sin,
    const uint seq_len,
    const uint num_heads,
    const uint head_dim,
    const uint pos_offset
) {
    const uint half_dim = head_dim >> 1;

    // gid(0) = pair index within head  [0 .. half_head_dim)
    // gid(1) = flattened (seq_pos * num_heads + head_idx)
    const uint pair_idx = get_global_id(0);
    const uint sh_idx   = get_global_id(1);

    if (pair_idx >= half_dim) return;
    if (sh_idx >= seq_len * num_heads) return;

    const uint seq_pos  = sh_idx / num_heads;
    const uint head_idx = sh_idx % num_heads;

    // Position in the frequency table
    const uint pos = pos_offset + seq_pos;

    // Indices into x:  x[seq_pos][head_idx][2*pair_idx], x[...][2*pair_idx+1]
    const uint base_idx = (seq_pos * num_heads + head_idx) * head_dim + 2 * pair_idx;

    float x_even = x[base_idx];
    float x_odd  = x[base_idx + 1];

    // Frequency lookup
    const uint freq_idx = pos * half_dim + pair_idx;
    float fc = freq_cos[freq_idx];
    float fs = freq_sin[freq_idx];

    // Rotation:  [x_even']   [cos  -sin] [x_even]
    //            [x_odd' ] = [sin   cos] [x_odd ]
    x[base_idx]     = x_even * fc - x_odd * fs;
    x[base_idx + 1] = x_even * fs + x_odd * fc;
}

/// Fused RoPE for Q and K simultaneously (saves one kernel launch).
///
/// Q and K must have the same head_dim.  num_kv_heads may differ from
/// num_q_heads (GQA / MQA support).
__kernel void rope_apply_qk(
    __global float* q,
    __global float* k,
    __global const float* freq_cos,
    __global const float* freq_sin,
    const uint seq_len,
    const uint num_q_heads,
    const uint num_kv_heads,
    const uint head_dim,
    const uint pos_offset
) {
    const uint half_dim = head_dim >> 1;
    const uint pair_idx = get_global_id(0);
    const uint sh_idx   = get_global_id(1);

    if (pair_idx >= half_dim) return;

    const uint total_heads = num_q_heads + num_kv_heads;
    if (sh_idx >= seq_len * total_heads) return;

    const uint seq_pos   = sh_idx / total_heads;
    const uint head_flat = sh_idx % total_heads;

    const uint pos = pos_offset + seq_pos;
    const uint freq_idx = pos * half_dim + pair_idx;
    float fc = freq_cos[freq_idx];
    float fs = freq_sin[freq_idx];

    if (head_flat < num_q_heads) {
        // Q head
        const uint base_idx =
            (seq_pos * num_q_heads + head_flat) * head_dim + 2 * pair_idx;
        float x_even = q[base_idx];
        float x_odd  = q[base_idx + 1];
        q[base_idx]     = x_even * fc - x_odd * fs;
        q[base_idx + 1] = x_even * fs + x_odd * fc;
    } else {
        // K head
        const uint kv_head = head_flat - num_q_heads;
        const uint base_idx =
            (seq_pos * num_kv_heads + kv_head) * head_dim + 2 * pair_idx;
        float x_even = k[base_idx];
        float x_odd  = k[base_idx + 1];
        k[base_idx]     = x_even * fc - x_odd * fs;
        k[base_idx + 1] = x_even * fs + x_odd * fc;
    }
}
