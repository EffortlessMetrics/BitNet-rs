#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TensorShapeInput {
    rows: u32,
    cols: u32,
    batch_size: u32,
    n_heads: u16,
    head_dim: u16,
    seq_len_q: u32,
    seq_len_kv: u32,
    k_inner: u32,
    extra_dims: Vec<u32>,
}

fn validate_matmul_dims(m: usize, n: usize, k: usize) -> Result<(), &'static str> {
    if m == 0 || n == 0 || k == 0 {
        return Err("zero dimension");
    }
    let a_elems = m.checked_mul(k).ok_or("overflow in A shape")?;
    let b_elems = k.checked_mul(n).ok_or("overflow in B shape")?;
    let c_elems = m.checked_mul(n).ok_or("overflow in C shape")?;
    // Reject shapes that would exceed 1B elements (SecurityLimits-style)
    const MAX_ELEMENTS: usize = 1_000_000_000;
    if a_elems > MAX_ELEMENTS || b_elems > MAX_ELEMENTS || c_elems > MAX_ELEMENTS {
        return Err("exceeds max tensor elements");
    }
    Ok(())
}

fn validate_attention_shape(
    n_heads: usize,
    head_dim: usize,
    seq_len_q: usize,
    seq_len_kv: usize,
) -> Result<(), &'static str> {
    if n_heads == 0 {
        return Err("zero heads");
    }
    if head_dim == 0 {
        return Err("zero head_dim");
    }
    if seq_len_q == 0 || seq_len_kv == 0 {
        return Err("zero sequence length");
    }
    // Validate total Q/K/V element counts don't overflow
    let q_elems = n_heads
        .checked_mul(seq_len_q)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or("overflow in Q shape")?;
    let kv_elems = n_heads
        .checked_mul(seq_len_kv)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or("overflow in KV shape")?;
    const MAX_ELEMENTS: usize = 1_000_000_000;
    if q_elems > MAX_ELEMENTS || kv_elems > MAX_ELEMENTS {
        return Err("attention tensor too large");
    }
    Ok(())
}

fn validate_qk256_shape(seq_len: usize, n_out: usize, k: usize) -> Result<(), &'static str> {
    if seq_len == 0 || n_out == 0 {
        return Err("zero dimension");
    }
    if k == 0 {
        return Err("zero inner dimension");
    }
    if k % 256 != 0 {
        return Err("k must be multiple of 256");
    }
    Ok(())
}

fn validate_rmsnorm_shape(hidden_dim: usize, n_rows: usize) -> Result<(), &'static str> {
    if hidden_dim == 0 || n_rows == 0 {
        return Err("zero dimension");
    }
    let total = hidden_dim.checked_mul(n_rows).ok_or("overflow")?;
    const MAX_ELEMENTS: usize = 1_000_000_000;
    if total > MAX_ELEMENTS {
        return Err("exceeds max elements");
    }
    Ok(())
}

fuzz_target!(|input: TensorShapeInput| {
    let rows = input.rows as usize;
    let cols = input.cols as usize;
    let batch = input.batch_size as usize;
    let k = input.k_inner as usize;

    // Matmul dimension validation — must not panic
    let _ = validate_matmul_dims(rows, cols, k);
    let _ = validate_matmul_dims(batch.saturating_mul(rows), cols, k);

    // Attention shape validation — must not panic
    let _ = validate_attention_shape(
        input.n_heads as usize,
        input.head_dim as usize,
        input.seq_len_q as usize,
        input.seq_len_kv as usize,
    );

    // QK256 shape validation (k must be multiple of 256) — must not panic
    let _ = validate_qk256_shape(input.seq_len_q as usize, cols, k);

    // RMSNorm shape validation — must not panic
    let _ = validate_rmsnorm_shape(cols, rows);

    // Fuzz arbitrary N-dimensional shapes for overflow detection
    let mut total: usize = 1;
    for &d in input.extra_dims.iter().take(256) {
        total = total.saturating_mul(d as usize);
    }
    // Saturating arithmetic must never panic
    let _ = total;
});
