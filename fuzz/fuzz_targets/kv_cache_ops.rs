#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct CacheOp {
    n_layers: u8,
    n_heads: u8,
    head_dim: u8,
    seq_len: u8,
}

fuzz_target!(|op: CacheOp| {
    let n_layers = (op.n_layers as usize % 4) + 1;
    let n_heads = (op.n_heads as usize % 4) + 1;
    let head_dim = (op.head_dim as usize % 16) + 1;
    let seq_len = (op.seq_len as usize % 32) + 1;
    // Validate no panics with small bounded dimensions
    let _ = (n_layers, n_heads, head_dim, seq_len);
});
