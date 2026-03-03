//! Error types for `I2_S` quantization.

/// Errors that can occur during `I2_S` quantization or dequantization.
#[derive(Debug, thiserror::Error)]
pub enum I2SError {
    /// Input length is not a multiple of the block size.
    #[error("input length {len} is not a multiple of block size {block_size}")]
    BlockAlignment { len: usize, block_size: usize },

    /// Packed data length does not match the expected size.
    #[error("packed data length {actual} does not match expected {expected}")]
    PackedLengthMismatch { actual: usize, expected: usize },

    /// Empty input is not allowed.
    #[error("empty input is not allowed")]
    EmptyInput,

    /// Scale value is not finite.
    #[error("scale at block {block_idx} is not finite: {value}")]
    NonFiniteScale { block_idx: usize, value: f32 },
}
