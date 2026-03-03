//! Broadcasting semantics for element-wise operations.
//!
//! Supports three broadcast modes:
//! - **Scalar–tensor**: a single value is applied to every element.
//! - **Vector–tensor**: a 1-D vector is repeated along the outer dimension.
//! - **Tensor–tensor**: both operands have identical length (no broadcast).

use crate::error::{ElementWiseError, Result};

/// Describes how two operands are broadcast against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastShape {
    /// Both operands have the same length; no broadcast needed.
    Same(usize),
    /// The left operand is a scalar broadcast to `len` elements.
    ScalarLeft(usize),
    /// The right operand is a scalar broadcast to `len` elements.
    ScalarRight(usize),
    /// The right operand (length `rhs_len`) is tiled to fill `total_len`.
    VectorRight { total_len: usize, rhs_len: usize },
    /// The left operand (length `lhs_len`) is tiled to fill `total_len`.
    VectorLeft { total_len: usize, lhs_len: usize },
}

impl BroadcastShape {
    /// Resolve the broadcast relationship between two flat lengths.
    ///
    /// # Errors
    ///
    /// Returns [`ElementWiseError::EmptyTensor`] if either length is zero, or
    /// [`ElementWiseError::ShapeMismatch`] when the lengths are incompatible.
    #[must_use = "returns the resolved broadcast shape"]
    pub fn resolve(lhs_len: usize, rhs_len: usize) -> Result<Self> {
        if lhs_len == 0 || rhs_len == 0 {
            return Err(ElementWiseError::EmptyTensor);
        }
        if lhs_len == rhs_len {
            return Ok(Self::Same(lhs_len));
        }
        if lhs_len == 1 {
            return Ok(Self::ScalarLeft(rhs_len));
        }
        if rhs_len == 1 {
            return Ok(Self::ScalarRight(lhs_len));
        }
        // Vector broadcast: the smaller must evenly divide the larger.
        if lhs_len > rhs_len && lhs_len.is_multiple_of(rhs_len) {
            return Ok(Self::VectorRight { total_len: lhs_len, rhs_len });
        }
        if rhs_len > lhs_len && rhs_len.is_multiple_of(lhs_len) {
            return Ok(Self::VectorLeft { total_len: rhs_len, lhs_len });
        }
        Err(ElementWiseError::ShapeMismatch { lhs: vec![lhs_len], rhs: vec![rhs_len] })
    }

    /// The output length after broadcast.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        match *self {
            Self::Same(n) | Self::ScalarLeft(n) | Self::ScalarRight(n) => n,
            Self::VectorRight { total_len, .. } | Self::VectorLeft { total_len, .. } => total_len,
        }
    }
}
