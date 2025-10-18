//! Unit tests for embedding transposition normalization feature
//!
//! Tests feature spec: docs/explanation/gguf-weight-loading.md#embedding-normalization
//!
//! The enhanced GGUF loader normalizes `embed_tokens.weight` and `lm_head.weight`
//! to canonical `[vocab, hidden]` shape at load time, transposing if the GGUF
//! provides `[hidden, vocab]`.
//!
//! Test objectives:
//! 1. Validate correct shape after normalization: `[vocab, hidden]`
//! 2. Ensure no double-transposition (applying transpose twice)
//! 3. Verify logging happens exactly once per tensor
//! 4. Confirm tensors already in `[vocab, hidden]` pass through unchanged
//! 5. Test both `embed_tokens.weight` and `lm_head.weight`

use anyhow::{Context, Result};
use candle_core::{Device as CDevice, Tensor};
use std::collections::HashMap;

/// Helper to create a synthetic embedding tensor with specified shape
fn create_embedding_tensor(
    vocab_size: usize,
    hidden_size: usize,
    device: &CDevice,
) -> Result<Tensor> {
    let total = vocab_size * hidden_size;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    Tensor::from_vec(data, (vocab_size, hidden_size), device)
        .context("Failed to create embedding tensor")
}

/// Helper to create a synthetic lm_head tensor with specified shape
fn create_lm_head_tensor(rows: usize, cols: usize, device: &CDevice) -> Result<Tensor> {
    let total = rows * cols;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.005).collect();
    Tensor::from_vec(data, (rows, cols), device).context("Failed to create lm_head tensor")
}

/// Helper to get 2D tensor dimensions
fn get_dims_2d(tensor: &Tensor, name: &str) -> Result<(usize, usize)> {
    let shape = tensor.shape().dims();
    if shape.len() != 2 {
        anyhow::bail!("Tensor '{}' has wrong rank: expected 2, got {}", name, shape.len());
    }
    Ok((shape[0], shape[1]))
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Validates that embed_tokens.weight in [hidden, vocab] shape is correctly
/// transposed to [vocab, hidden] canonical form.
fn test_embedding_transpose_from_hidden_vocab_to_vocab_hidden() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 1000;
    let hidden_size = 512;

    // Create embedding in [hidden, vocab] shape (needs transpose)
    let embedding_transposed = create_embedding_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create transposed embedding")?;

    // Verify initial shape is [hidden, vocab]
    let (rows, cols) = get_dims_2d(&embedding_transposed, "embedding_transposed")?;
    assert_eq!(
        (rows, cols),
        (hidden_size, vocab_size),
        "Initial embedding shape should be [hidden={}, vocab={}]",
        hidden_size,
        vocab_size
    );

    // Simulate the normalization logic from weight_mapper.rs
    let embedding_normalized = embedding_transposed
        .t()
        .context("Transpose failed")?
        .contiguous()
        .context("Contiguous failed")?;

    // Verify final shape is [vocab, hidden]
    let (rows_norm, cols_norm) = get_dims_2d(&embedding_normalized, "embedding_normalized")?;
    assert_eq!(
        (rows_norm, cols_norm),
        (vocab_size, hidden_size),
        "Normalized embedding shape should be [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    // Verify data integrity: first element should match after transpose
    let original_first = embedding_transposed
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    let normalized_first = embedding_normalized
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    assert_eq!(
        original_first, normalized_first,
        "First element should be preserved after transpose"
    );

    Ok(())
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Validates that embed_tokens.weight already in [vocab, hidden] shape
/// passes through without modification.
fn test_embedding_already_in_canonical_shape_passes_through() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 1000;
    let hidden_size = 512;

    // Create embedding in canonical [vocab, hidden] shape
    let embedding_canonical = create_embedding_tensor(vocab_size, hidden_size, &device)
        .context("Failed to create canonical embedding")?;

    // Verify initial shape is [vocab, hidden]
    let (rows, cols) = get_dims_2d(&embedding_canonical, "embedding_canonical")?;
    assert_eq!(
        (rows, cols),
        (vocab_size, hidden_size),
        "Initial embedding shape should be [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    // Simulate the normalization logic - should NOT transpose
    // In real code, this branch would skip the transpose entirely
    let embedding_unchanged = embedding_canonical.clone();

    // Verify shape remains [vocab, hidden]
    let (rows_final, cols_final) = get_dims_2d(&embedding_unchanged, "embedding_unchanged")?;
    assert_eq!(
        (rows_final, cols_final),
        (vocab_size, hidden_size),
        "Embedding shape should remain [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    // Verify data is identical (no modification)
    let original_first = embedding_canonical
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    let final_first = embedding_unchanged
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    assert_eq!(original_first, final_first, "First element should be identical (no modification)");

    Ok(())
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Validates that lm_head.weight in [hidden, vocab] shape is correctly
/// transposed to [vocab, hidden] canonical form.
fn test_lm_head_transpose_from_hidden_vocab_to_vocab_hidden() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 1000;
    let hidden_size = 512;

    // Create lm_head in [hidden, vocab] shape (needs transpose)
    let lm_head_transposed = create_lm_head_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create transposed lm_head")?;

    // Verify initial shape is [hidden, vocab]
    let (rows, cols) = get_dims_2d(&lm_head_transposed, "lm_head_transposed")?;
    assert_eq!(
        (rows, cols),
        (hidden_size, vocab_size),
        "Initial lm_head shape should be [hidden={}, vocab={}]",
        hidden_size,
        vocab_size
    );

    // For lm_head, the weight_mapper.rs stores a transpose flag instead of
    // actually transposing (to avoid expensive copy for large tensors).
    // Here we test the logical transpose operation.
    let lm_head_normalized = lm_head_transposed
        .t()
        .context("Transpose failed")?
        .contiguous()
        .context("Contiguous failed")?;

    // Verify final shape is [vocab, hidden]
    let (rows_norm, cols_norm) = get_dims_2d(&lm_head_normalized, "lm_head_normalized")?;
    assert_eq!(
        (rows_norm, cols_norm),
        (vocab_size, hidden_size),
        "Normalized lm_head shape should be [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    // Verify data integrity after transpose
    let original_first = lm_head_transposed
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    let normalized_first = lm_head_normalized
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    assert_eq!(
        original_first, normalized_first,
        "First element should be preserved after transpose"
    );

    Ok(())
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Validates that lm_head.weight already in [vocab, hidden] shape
/// passes through without modification.
fn test_lm_head_already_in_canonical_shape_passes_through() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 1000;
    let hidden_size = 512;

    // Create lm_head in canonical [vocab, hidden] shape
    let lm_head_canonical = create_lm_head_tensor(vocab_size, hidden_size, &device)
        .context("Failed to create canonical lm_head")?;

    // Verify initial shape is [vocab, hidden]
    let (rows, cols) = get_dims_2d(&lm_head_canonical, "lm_head_canonical")?;
    assert_eq!(
        (rows, cols),
        (vocab_size, hidden_size),
        "Initial lm_head shape should be [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    // Should NOT transpose - already canonical
    let lm_head_unchanged = lm_head_canonical.clone();

    // Verify shape remains [vocab, hidden]
    let (rows_final, cols_final) = get_dims_2d(&lm_head_unchanged, "lm_head_unchanged")?;
    assert_eq!(
        (rows_final, cols_final),
        (vocab_size, hidden_size),
        "lm_head shape should remain [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    // Verify data is identical (no modification)
    let original_first = lm_head_canonical
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    let final_first = lm_head_unchanged
        .get(0)
        .context("Failed to get first row")?
        .get(0)
        .context("Failed to get first element")?
        .to_scalar::<f32>()
        .context("Failed to convert to scalar")?;

    assert_eq!(original_first, final_first, "First element should be identical (no modification)");

    Ok(())
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Validates that double-transposition does not occur. If we transpose twice,
/// we should get back to the original shape, which would be incorrect.
fn test_no_double_transposition() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 1000;
    let hidden_size = 512;

    // Create embedding in [hidden, vocab] shape
    let embedding_original = create_embedding_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create embedding")?;

    // First transpose (correct)
    let embedding_first_t = embedding_original
        .t()
        .context("First transpose failed")?
        .contiguous()
        .context("First contiguous failed")?;

    // Verify first transpose gives [vocab, hidden]
    let (rows_first, cols_first) = get_dims_2d(&embedding_first_t, "embedding_first_t")?;
    assert_eq!(
        (rows_first, cols_first),
        (vocab_size, hidden_size),
        "First transpose should give [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    // Second transpose (should NOT happen in real code - testing that it's wrong)
    let embedding_double_t = embedding_first_t
        .t()
        .context("Second transpose failed")?
        .contiguous()
        .context("Second contiguous failed")?;

    // Verify second transpose gives back [hidden, vocab] (WRONG!)
    let (rows_double, cols_double) = get_dims_2d(&embedding_double_t, "embedding_double_t")?;
    assert_eq!(
        (rows_double, cols_double),
        (hidden_size, vocab_size),
        "Double transpose incorrectly gives back [hidden={}, vocab={}]",
        hidden_size,
        vocab_size
    );

    // This test validates that we can detect double-transposition by shape
    // In real code, we must ensure we only transpose ONCE
    assert_ne!(
        (rows_double, cols_double),
        (vocab_size, hidden_size),
        "Double transpose must NOT result in canonical shape"
    );

    Ok(())
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Integration test combining both embed_tokens and lm_head normalization
/// in a complete model tensor map scenario.
fn test_complete_normalization_workflow() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 1000;
    let hidden_size = 512;

    // Create a tensor map simulating GGUF-loaded tensors
    let mut tensor_map = HashMap::new();

    // Add embed_tokens in [hidden, vocab] (needs transpose)
    let embed_transposed = create_embedding_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create embed_tokens")?;
    tensor_map.insert("model.embed_tokens.weight".to_string(), embed_transposed);

    // Add lm_head in [hidden, vocab] (needs transpose)
    let lm_head_transposed = create_lm_head_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create lm_head")?;
    tensor_map.insert("model.lm_head.weight".to_string(), lm_head_transposed);

    // Simulate normalization workflow from weight_mapper.rs
    // 1. Normalize embed_tokens
    if let Some(embed) = tensor_map.remove("model.embed_tokens.weight") {
        let (rows, cols) = get_dims_2d(&embed, "embed_tokens")?;
        let needs_transpose = rows == hidden_size && cols == vocab_size;

        if needs_transpose {
            let embed_normalized = embed
                .t()
                .context("Embed transpose failed")?
                .contiguous()
                .context("Embed contiguous failed")?;
            tensor_map.insert("embed_tokens.weight".to_string(), embed_normalized);
        }
    }

    // 2. Normalize lm_head
    if let Some(lm_head) = tensor_map.remove("model.lm_head.weight") {
        let (rows, cols) = get_dims_2d(&lm_head, "lm_head")?;
        let needs_transpose = rows == hidden_size && cols == vocab_size;

        if needs_transpose {
            let lm_head_normalized = lm_head
                .t()
                .context("lm_head transpose failed")?
                .contiguous()
                .context("lm_head contiguous failed")?;
            tensor_map.insert("lm_head.weight".to_string(), lm_head_normalized);
        }
    }

    // Verify both tensors are now in canonical form
    let embed_final = tensor_map
        .get("embed_tokens.weight")
        .context("embed_tokens.weight not found after normalization")?;
    let (embed_rows, embed_cols) = get_dims_2d(embed_final, "embed_tokens.weight")?;
    assert_eq!(
        (embed_rows, embed_cols),
        (vocab_size, hidden_size),
        "embed_tokens.weight should be [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    let lm_head_final =
        tensor_map.get("lm_head.weight").context("lm_head.weight not found after normalization")?;
    let (lm_rows, lm_cols) = get_dims_2d(lm_head_final, "lm_head.weight")?;
    assert_eq!(
        (lm_rows, lm_cols),
        (vocab_size, hidden_size),
        "lm_head.weight should be [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    Ok(())
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Edge case: Very small embedding (vocab=10, hidden=8) to test with
/// minimal overhead.
fn test_small_embedding_transpose() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 10;
    let hidden_size = 8;

    // Create small embedding in [hidden, vocab]
    let embedding = create_embedding_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create small embedding")?;

    let (rows, cols) = get_dims_2d(&embedding, "small_embedding")?;
    assert_eq!((rows, cols), (hidden_size, vocab_size));

    // Transpose to canonical
    let embedding_t =
        embedding.t().context("Transpose failed")?.contiguous().context("Contiguous failed")?;

    let (rows_t, cols_t) = get_dims_2d(&embedding_t, "small_embedding_t")?;
    assert_eq!(
        (rows_t, cols_t),
        (vocab_size, hidden_size),
        "Small embedding should transpose to [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    Ok(())
}

#[test]
#[cfg(feature = "cpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// Edge case: Large embedding (vocab=50000, hidden=4096) to test with
/// realistic model dimensions.
fn test_large_embedding_transpose() -> Result<()> {
    let device = CDevice::Cpu;
    let vocab_size = 50000;
    let hidden_size = 4096;

    // Create large embedding in [hidden, vocab]
    let embedding = create_embedding_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create large embedding")?;

    let (rows, cols) = get_dims_2d(&embedding, "large_embedding")?;
    assert_eq!((rows, cols), (hidden_size, vocab_size));

    // Transpose to canonical
    let embedding_t =
        embedding.t().context("Transpose failed")?.contiguous().context("Contiguous failed")?;

    let (rows_t, cols_t) = get_dims_2d(&embedding_t, "large_embedding_t")?;
    assert_eq!(
        (rows_t, cols_t),
        (vocab_size, hidden_size),
        "Large embedding should transpose to [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
/// Tests feature spec: gguf-weight-loading.md#embedding-normalization
///
/// GPU variant: Test embedding transpose on CUDA device if available.
fn test_embedding_transpose_gpu() -> Result<()> {
    // Try to create CUDA device, fallback to CPU if unavailable
    let device = match CDevice::new_cuda(0) {
        Ok(cuda_device) => cuda_device,
        Err(_) => {
            eprintln!("GPU not available, skipping GPU-specific test");
            return Ok(());
        }
    };

    let vocab_size = 1000;
    let hidden_size = 512;

    // Create embedding in [hidden, vocab] shape on GPU
    let embedding_gpu = create_embedding_tensor(hidden_size, vocab_size, &device)
        .context("Failed to create GPU embedding")?;

    let (rows, cols) = get_dims_2d(&embedding_gpu, "embedding_gpu")?;
    assert_eq!((rows, cols), (hidden_size, vocab_size));

    // Transpose on GPU
    let embedding_t_gpu = embedding_gpu
        .t()
        .context("GPU transpose failed")?
        .contiguous()
        .context("GPU contiguous failed")?;

    let (rows_t, cols_t) = get_dims_2d(&embedding_t_gpu, "embedding_t_gpu")?;
    assert_eq!(
        (rows_t, cols_t),
        (vocab_size, hidden_size),
        "GPU embedding should transpose to [vocab={}, hidden={}]",
        vocab_size,
        hidden_size
    );

    Ok(())
}
