//! Error types for element-wise operations.

use std::fmt;

/// Errors produced by element-wise operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementWiseError {
    /// Tensor shapes are incompatible for the requested broadcast.
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },
    /// A tensor has zero elements, which is not supported.
    EmptyTensor,
    /// Division by zero detected in the divisor tensor.
    DivisionByZero,
    /// The FMA operand lengths are inconsistent.
    FmaLengthMismatch { a_len: usize, b_len: usize, c_len: usize },
}

impl fmt::Display for ElementWiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { lhs, rhs } => {
                write!(f, "shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::EmptyTensor => write!(f, "empty tensor not supported"),
            Self::DivisionByZero => write!(f, "division by zero in divisor"),
            Self::FmaLengthMismatch { a_len, b_len, c_len } => {
                write!(f, "FMA length mismatch: a={a_len}, b={b_len}, c={c_len}")
            }
        }
    }
}

impl std::error::Error for ElementWiseError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, ElementWiseError>;
