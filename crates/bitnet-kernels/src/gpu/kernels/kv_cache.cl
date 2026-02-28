/// KV cache management kernels for transformer attention.

__kernel void kv_cache_append(
    __global float* cache,
    __global const float* new_kv,
    __global const uint* positions,
    const uint head_dim,
    const uint page_size,
    const uint max_pages
) {
    const uint gid = get_global_id(0);
    const uint head = gid / head_dim;
    const uint d = gid % head_dim;
    const uint pos = positions[head];
    const uint page_idx = pos / page_size;
    const uint page_offset = pos % page_size;
    if (page_idx >= max_pages) return;
    const uint stride = max_pages * page_size * head_dim;
    const uint cache_idx = head * stride + page_idx * page_size * head_dim + page_offset * head_dim + d;
    cache[cache_idx] = new_kv[head * head_dim + d];
}

__kernel void kv_cache_read(
    __global const float* cache,
    __global float* output,
    __global const uint* positions,
    const uint head_dim,
    const uint seq_len,
    const uint page_size,
    const uint max_pages
) {
    const uint gid = get_global_id(0);
    const uint head = gid / (seq_len * head_dim);
    const uint remainder = gid % (seq_len * head_dim);
    const uint s = remainder / head_dim;
    const uint d = remainder % head_dim;
    if (s >= seq_len) return;
    const uint pos = positions[s];
    const uint page_idx = pos / page_size;
    const uint page_offset = pos % page_size;
    if (page_idx >= max_pages) return;
    const uint stride = max_pages * page_size * head_dim;
    const uint cache_idx = head * stride + page_idx * page_size * head_dim + page_offset * head_dim + d;
    const uint out_idx = head * seq_len * head_dim + s * head_dim + d;
    output[out_idx] = cache[cache_idx];
}
