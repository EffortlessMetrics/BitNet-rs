//! CUDA reduction operations for neural network inference.
//!
//! This microcrate provides GPU-accelerated reduction kernels (sum, max, min,
//! mean, softmax, argmax) with warp-level shuffle primitives for efficient
//! parallel reductions. All GPU code is gated behind `feature = "gpu"` /
//! `feature = "cuda"`.
//!
//! # Reduction dimensions
//!
//! [`ReductionDim`] selects the axis along which the reduction is applied:
//!
//! * **`Row`** — reduce each row independently (output has one element per row).
//! * **`Column`** — reduce each column independently (output has one element per
//!   column).
//! * **`Full`** — reduce the entire tensor to a single scalar.
//!
//! # Example
//!
//! ```
//! use bitnet_cuda_reduction::{ReductionDim, ReductionKernels, WarpPrimitives};
//!
//! // Row-wise sum of a 2×3 matrix stored in row-major order
//! let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let result = ReductionKernels::sum(&data, 2, 3, ReductionDim::Row);
//! assert_eq!(result, vec![6.0, 15.0]);
//! ```

// Re-export everything from sub-modules at the crate root.
mod reduction;
mod warp;

pub use reduction::{ReductionDim, ReductionKernels};
pub use warp::WarpPrimitives;
