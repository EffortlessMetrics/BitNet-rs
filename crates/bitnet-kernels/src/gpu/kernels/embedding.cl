// OpenCL embedding lookup kernels for GPU-accelerated transformer inference.

/// Look up a single token embedding from the embedding table.
/// table: [vocab_size * embed_dim], output: [embed_dim]
__kernel void embedding_lookup(
    __global const float* table,
    __global float* output,
    const uint token_id,
    const uint embed_dim)
{
    uint gid = get_global_id(0);
    if (gid < embed_dim) {
        output[gid] = table[(ulong)token_id * embed_dim + gid];
    }
}

/// Batched embedding lookup: fetch embeddings for a sequence of token IDs.
/// table: [vocab_size * embed_dim], tokens: [seq_len], output: [seq_len * embed_dim]
__kernel void embedding_lookup_batched(
    __global const float* table,
    __global const uint* tokens,
    __global float* output,
    const uint embed_dim,
    const uint seq_len)
{
    uint gid = get_global_id(0);
    uint total = seq_len * embed_dim;
    if (gid < total) {
        uint seq_idx = gid / embed_dim;
        uint dim_idx = gid % embed_dim;
        uint tok = tokens[seq_idx];
        output[gid] = table[(ulong)tok * embed_dim + dim_idx];
    }
}
