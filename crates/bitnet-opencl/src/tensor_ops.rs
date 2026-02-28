// GPU tensor operations: high-level API with CPU fallback dispatch.

use std::fmt;

use crate::tensor_ops_cpu;

/// Errors from tensor operations.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("empty tensor not supported for this operation")]
    EmptyTensor,
    #[error("invalid dimension {dim} for tensor with {ndim} dimensions")]
    InvalidDimension { dim: usize, ndim: usize },
    #[error("numerical error: {0}")]
    Numerical(String),
}

/// Result alias for tensor operations.
pub type TensorResult<T> = Result<T, TensorError>;

/// Shape of a tensor with up to 4 dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: &[usize]) -> Self {
        Self { dims: dims.to_vec() }
    }

    pub const fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn dim(&self, i: usize) -> TensorResult<usize> {
        self.dims
            .get(i)
            .copied()
            .ok_or_else(|| TensorError::InvalidDimension { dim: i, ndim: self.ndim() })
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Dense tensor stored in row-major order.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: TensorShape,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: TensorShape, data: Vec<f32>) -> TensorResult<Self> {
        let expected = shape.numel();
        if data.len() != expected {
            return Err(TensorError::ShapeMismatch(format!(
                "data length {} doesn't match shape {} (expected {expected})",
                data.len(),
                shape,
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: TensorShape) -> Self {
        let n = shape.numel();
        Self { shape, data: vec![0.0; n] }
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

/// Whether GPU context is available for dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

/// High-level tensor operation API that dispatches to GPU kernels or CPU
/// fallback.
pub trait GpuTensorOps {
    /// Matrix multiplication: `[M, K] × [K, N] → [M, N]`.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;

    /// Element-wise addition.
    fn add(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;

    /// Element-wise multiplication.
    fn mul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;

    /// Softmax along `dim`.
    fn softmax(&self, input: &Tensor, dim: usize) -> TensorResult<Tensor>;

    /// RMS normalization.
    fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> TensorResult<Tensor>;

    /// Rotary position embedding.
    fn rope(&self, input: &Tensor, freqs: &Tensor) -> TensorResult<Tensor>;

    /// `SiLU` activation: `x * sigmoid(x)`.
    fn silu(&self, input: &Tensor) -> TensorResult<Tensor>;

    /// Scaled dot-product attention. `mask` is optional.
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> TensorResult<Tensor>;

    /// Embedding lookup: `input_ids` indexes into `table`.
    fn embedding(&self, input_ids: &[u32], table: &Tensor) -> TensorResult<Tensor>;

    /// Linear projection: `input @ weight^T + bias`.
    fn linear(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> TensorResult<Tensor>;
}

/// Dispatcher that selects GPU or CPU backend at runtime.
pub struct TensorOpsDispatcher {
    backend: Backend,
}

impl TensorOpsDispatcher {
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    /// Create dispatcher with CPU backend.
    pub const fn cpu() -> Self {
        Self::new(Backend::Cpu)
    }

    /// Create dispatcher preferring GPU, falling back to CPU.
    pub const fn auto() -> Self {
        let backend = if gpu_available() { Backend::Gpu } else { Backend::Cpu };
        Self::new(backend)
    }

    pub const fn backend(&self) -> Backend {
        self.backend
    }
}

/// Check GPU availability at runtime (stub — always false for now).
const fn gpu_available() -> bool {
    false
}

// All operations delegate to CPU reference implementations. When a GPU
// backend is wired up the dispatch will branch on `self.backend`.
impl GpuTensorOps for TensorOpsDispatcher {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        tensor_ops_cpu::matmul(a, b)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        tensor_ops_cpu::add(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        tensor_ops_cpu::mul(a, b)
    }

    fn softmax(&self, input: &Tensor, dim: usize) -> TensorResult<Tensor> {
        tensor_ops_cpu::softmax(input, dim)
    }

    fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> TensorResult<Tensor> {
        tensor_ops_cpu::rmsnorm(input, weight, eps)
    }

    fn rope(&self, input: &Tensor, freqs: &Tensor) -> TensorResult<Tensor> {
        tensor_ops_cpu::rope(input, freqs)
    }

    fn silu(&self, input: &Tensor) -> TensorResult<Tensor> {
        tensor_ops_cpu::silu(input)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> TensorResult<Tensor> {
        tensor_ops_cpu::attention(q, k, v, mask)
    }

    fn embedding(&self, input_ids: &[u32], table: &Tensor) -> TensorResult<Tensor> {
        tensor_ops_cpu::embedding(input_ids, table)
    }

    fn linear(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> TensorResult<Tensor> {
        tensor_ops_cpu::linear(input, weight, bias)
    }
}
