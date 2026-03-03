//! Public normalization operation types with runtime AVX2 dispatch.

use crate::scalar;

/// Errors that can occur during normalization operations.
#[derive(Debug, Clone, PartialEq)]
pub enum NormError {
    /// Input and parameter dimensions do not match.
    DimensionMismatch { input_len: usize, param_len: usize, param_name: &'static str },
    /// Epsilon value is non-positive.
    InvalidEpsilon(f32),
    /// Input slice is empty.
    EmptyInput,
}

impl std::fmt::Display for NormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { input_len, param_len, param_name } => write!(
                f,
                "dimension mismatch: input length {input_len} != {param_name} length {param_len}"
            ),
            Self::InvalidEpsilon(eps) => write!(f, "invalid epsilon: {eps} (must be > 0)"),
            Self::EmptyInput => write!(f, "input slice is empty"),
        }
    }
}

impl std::error::Error for NormError {}

/// Result type for normalization operations.
pub type NormResult<T> = Result<T, NormError>;

/// `LayerNorm` operation with gamma (scale) and beta (bias) parameters.
///
/// Computes: `gamma * (x - mean) / sqrt(var + eps) + beta`
#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    epsilon: f32,
}

impl LayerNorm {
    /// Create a new `LayerNorm` with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`NormError::DimensionMismatch`] if `gamma` and `beta` differ in length,
    /// or [`NormError::InvalidEpsilon`] if `epsilon <= 0`.
    #[must_use = "constructing a LayerNorm has no effect without calling forward()"]
    pub fn new(gamma: Vec<f32>, beta: Vec<f32>, epsilon: f32) -> NormResult<Self> {
        if gamma.len() != beta.len() {
            return Err(NormError::DimensionMismatch {
                input_len: gamma.len(),
                param_len: beta.len(),
                param_name: "beta",
            });
        }
        if epsilon <= 0.0 {
            return Err(NormError::InvalidEpsilon(epsilon));
        }
        Ok(Self { gamma, beta, epsilon })
    }

    /// Returns the feature dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.gamma.len()
    }

    /// Returns the epsilon value.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns a reference to the gamma (scale) parameters.
    #[must_use]
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Returns a reference to the beta (bias) parameters.
    #[must_use]
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// Run `LayerNorm` forward pass, writing results into `output`.
    ///
    /// # Errors
    ///
    /// Returns [`NormError::DimensionMismatch`] if `input` or `output` length
    /// does not match the configured dimension.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> NormResult<()> {
        let dim = self.dim();
        if input.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: input.len(),
                param_len: dim,
                param_name: "gamma",
            });
        }
        if output.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: output.len(),
                param_len: dim,
                param_name: "output",
            });
        }
        if dim == 0 {
            return Ok(());
        }

        #[cfg(target_arch = "x86_64")]
        {
            if crate::avx2_available() {
                // SAFETY: AVX2+FMA availability checked above.
                unsafe {
                    crate::avx2::layer_norm_avx2(
                        input,
                        &self.gamma,
                        &self.beta,
                        self.epsilon,
                        output,
                    );
                }
                return Ok(());
            }
        }

        scalar::layer_norm(input, &self.gamma, &self.beta, self.epsilon, output);
        Ok(())
    }

    /// Convenience: allocate output and return it.
    ///
    /// # Errors
    ///
    /// Same as [`forward`](Self::forward).
    pub fn forward_alloc(&self, input: &[f32]) -> NormResult<Vec<f32>> {
        let mut out = vec![0.0; self.dim()];
        self.forward(input, &mut out)?;
        Ok(out)
    }
}

/// `RmsNorm` operation with gamma (scale) parameter.
///
/// Computes: `gamma * x / sqrt(mean(x^2) + eps)`
#[derive(Debug, Clone)]
pub struct RmsNorm {
    gamma: Vec<f32>,
    epsilon: f32,
}

impl RmsNorm {
    /// Create a new `RmsNorm` with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`NormError::InvalidEpsilon`] if `epsilon <= 0`.
    #[must_use = "constructing an RmsNorm has no effect without calling forward()"]
    pub fn new(gamma: Vec<f32>, epsilon: f32) -> NormResult<Self> {
        if epsilon <= 0.0 {
            return Err(NormError::InvalidEpsilon(epsilon));
        }
        Ok(Self { gamma, epsilon })
    }

    /// Returns the feature dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.gamma.len()
    }

    /// Returns the epsilon value.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns a reference to the gamma (scale) parameters.
    #[must_use]
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Run `RmsNorm` forward pass, writing results into `output`.
    ///
    /// # Errors
    ///
    /// Returns [`NormError`] on dimension mismatch.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> NormResult<()> {
        let dim = self.dim();
        if input.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: input.len(),
                param_len: dim,
                param_name: "gamma",
            });
        }
        if output.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: output.len(),
                param_len: dim,
                param_name: "output",
            });
        }
        if dim == 0 {
            return Ok(());
        }

        #[cfg(target_arch = "x86_64")]
        {
            if crate::avx2_available() {
                unsafe {
                    crate::avx2::rms_norm_avx2(input, &self.gamma, self.epsilon, output);
                }
                return Ok(());
            }
        }

        scalar::rms_norm(input, &self.gamma, self.epsilon, output);
        Ok(())
    }

    /// Convenience: allocate output and return it.
    ///
    /// # Errors
    ///
    /// Same as [`forward`](Self::forward).
    pub fn forward_alloc(&self, input: &[f32]) -> NormResult<Vec<f32>> {
        let mut out = vec![0.0; self.dim()];
        self.forward(input, &mut out)?;
        Ok(out)
    }
}

/// Parameters for [`BatchNorm`] construction.
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    /// Scale (gamma) per feature.
    pub gamma: Vec<f32>,
    /// Bias (beta) per feature.
    pub beta: Vec<f32>,
    /// Running mean per feature.
    pub running_mean: Vec<f32>,
    /// Running variance per feature.
    pub running_var: Vec<f32>,
    /// Small constant for numerical stability.
    pub epsilon: f32,
}

/// `BatchNorm` operation using precomputed running statistics.
///
/// Computes: `gamma * (x - running_mean) / sqrt(running_var + eps) + beta`
#[derive(Debug, Clone)]
pub struct BatchNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    epsilon: f32,
}

impl BatchNorm {
    /// Create a new `BatchNorm` from the given parameters.
    ///
    /// # Errors
    ///
    /// Returns [`NormError::DimensionMismatch`] if parameter lengths differ,
    /// or [`NormError::InvalidEpsilon`] if `epsilon <= 0`.
    #[must_use = "constructing a BatchNorm has no effect without calling forward()"]
    pub fn new(params: BatchNormParams) -> NormResult<Self> {
        let dim = params.gamma.len();
        if params.beta.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: dim,
                param_len: params.beta.len(),
                param_name: "beta",
            });
        }
        if params.running_mean.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: dim,
                param_len: params.running_mean.len(),
                param_name: "running_mean",
            });
        }
        if params.running_var.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: dim,
                param_len: params.running_var.len(),
                param_name: "running_var",
            });
        }
        if params.epsilon <= 0.0 {
            return Err(NormError::InvalidEpsilon(params.epsilon));
        }
        Ok(Self {
            gamma: params.gamma,
            beta: params.beta,
            running_mean: params.running_mean,
            running_var: params.running_var,
            epsilon: params.epsilon,
        })
    }

    /// Returns the feature dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.gamma.len()
    }

    /// Returns the epsilon value.
    #[must_use]
    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Run `BatchNorm` forward pass, writing results into `output`.
    ///
    /// # Errors
    ///
    /// Returns [`NormError`] on dimension mismatch.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> NormResult<()> {
        let dim = self.dim();
        if input.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: input.len(),
                param_len: dim,
                param_name: "gamma",
            });
        }
        if output.len() != dim {
            return Err(NormError::DimensionMismatch {
                input_len: output.len(),
                param_len: dim,
                param_name: "output",
            });
        }
        if dim == 0 {
            return Ok(());
        }

        #[cfg(target_arch = "x86_64")]
        {
            if crate::avx2_available() {
                unsafe {
                    crate::avx2::batch_norm_avx2(
                        input,
                        &self.gamma,
                        &self.beta,
                        &self.running_mean,
                        &self.running_var,
                        self.epsilon,
                        output,
                    );
                }
                return Ok(());
            }
        }

        scalar::batch_norm(
            input,
            &self.gamma,
            &self.beta,
            &self.running_mean,
            &self.running_var,
            self.epsilon,
            output,
        );
        Ok(())
    }

    /// Convenience: allocate output and return it.
    ///
    /// # Errors
    ///
    /// Same as [`forward`](Self::forward).
    pub fn forward_alloc(&self, input: &[f32]) -> NormResult<Vec<f32>> {
        let mut out = vec![0.0; self.dim()];
        self.forward(input, &mut out)?;
        Ok(out)
    }
}
