//! Module stub - implementation pending merge from feature branch
//! Dynamic shape management for GPU inference pipelines.
//!
//! Provides symbolic shape specifications, shape inference, constraint
//! validation, bucketed memory allocation, and variable-length sequence
//! packing — all with CPU reference implementations.

// Pedantic/nursery lints that conflict with the public-facing API style here.
#![allow(
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::return_self_not_must_use
)]

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// DynamicDim
// ---------------------------------------------------------------------------

/// A single dimension that is either statically known or resolved at runtime.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum DynamicDim {
    /// Compile-time constant.
    Static(usize),
    /// Named symbolic dimension resolved at runtime.
    Named(String),
    /// Inferred from another dimension via an expression.
    Derived { base: String, scale: usize, offset: isize },
}

impl DynamicDim {
    pub fn fixed(value: usize) -> Self {
        Self::Static(value)
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self::Named(name.into())
    }

    pub fn derived(base: impl Into<String>, scale: usize, offset: isize) -> Self {
        Self::Derived { base: base.into(), scale, offset }
    }

    /// Resolve the dimension given a binding map.
    /// Returns `None` when a referenced name is not bound.
    pub fn resolve(&self, bindings: &HashMap<String, usize>) -> Option<usize> {
        match self {
            Self::Static(v) => Some(*v),
            Self::Named(n) => bindings.get(n).copied(),
            Self::Derived { base, scale, offset } => {
                let base_val = bindings.get(base).copied()? as isize;
                let result = base_val * (*scale as isize) + offset;
                if result < 0 { None } else { Some(result as usize) }
            }
        }
    }

    /// Returns `true` when the dimension is statically known.
    pub fn is_static(&self) -> bool {
        matches!(self, Self::Static(_))
    }

    /// Returns the static value if known.
    pub fn static_value(&self) -> Option<usize> {
        if let Self::Static(v) = self { Some(*v) } else { None }
    }

    /// Collect all symbolic names referenced by this dimension.
    pub fn referenced_names(&self) -> Vec<&str> {
        match self {
            Self::Static(_) => vec![],
            Self::Named(n) => vec![n.as_str()],
            Self::Derived { base, .. } => vec![base.as_str()],
        }
    }
}

impl fmt::Debug for DynamicDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static(v) => write!(f, "{v}"),
            Self::Named(n) => write!(f, "{n}"),
            Self::Derived { base, scale, offset } => {
                write!(f, "{base}*{scale}{offset:+}")
            }
        }
    }
}

impl fmt::Display for DynamicDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// ShapeSpec
// ---------------------------------------------------------------------------

/// Symbolic tensor-shape specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeSpec {
    dims: Vec<DynamicDim>,
}

impl ShapeSpec {
    pub fn new(dims: Vec<DynamicDim>) -> Self {
        Self { dims }
    }

    /// Convenience: fully static shape.
    pub fn static_shape(dims: &[usize]) -> Self {
        Self { dims: dims.iter().map(|&d| DynamicDim::Static(d)).collect() }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[DynamicDim] {
        &self.dims
    }

    /// Resolve every dimension. Returns `None` if any dimension cannot resolve.
    pub fn resolve(&self, bindings: &HashMap<String, usize>) -> Option<Vec<usize>> {
        self.dims.iter().map(|d| d.resolve(bindings)).collect()
    }

    /// Compute the total element count (product of resolved dims).
    pub fn resolve_numel(&self, bindings: &HashMap<String, usize>) -> Option<usize> {
        self.resolve(bindings).map(|v| v.iter().product())
    }

    /// Returns `true` when all dimensions are static.
    pub fn is_fully_static(&self) -> bool {
        self.dims.iter().all(DynamicDim::is_static)
    }

    /// Collect all symbolic names across the shape.
    pub fn symbolic_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.dims.iter().flat_map(|d| d.referenced_names()).collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Broadcast two shapes following NumPy-style rules.
    pub fn broadcast(a: &Self, b: &Self) -> Result<Self, ShapeError> {
        let max_rank = a.rank().max(b.rank());
        let mut result = Vec::with_capacity(max_rank);
        let unit = DynamicDim::Static(1);
        for i in 0..max_rank {
            let da =
                if i < max_rank - a.rank() { &unit } else { &a.dims[i - (max_rank - a.rank())] };
            let db =
                if i < max_rank - b.rank() { &unit } else { &b.dims[i - (max_rank - b.rank())] };
            match (da, db) {
                (DynamicDim::Static(1), other) | (other, DynamicDim::Static(1)) => {
                    result.push(other.clone());
                }
                (DynamicDim::Static(x), DynamicDim::Static(y)) if x == y => {
                    result.push(DynamicDim::Static(*x));
                }
                (DynamicDim::Static(x), DynamicDim::Static(y)) => {
                    return Err(ShapeError::BroadcastIncompatible { dim_a: *x, dim_b: *y });
                }
                // When either side is symbolic, keep the non-unit side.
                (sym, _) => result.push(sym.clone()),
            }
        }
        Ok(Self::new(result))
    }
}

// ---------------------------------------------------------------------------
// ShapeError
// ---------------------------------------------------------------------------

/// Errors produced by shape operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    UnresolvedDimension(String),
    BroadcastIncompatible { dim_a: usize, dim_b: usize },
    RankMismatch { expected: usize, got: usize },
    ConstraintViolation(String),
    InvalidPadding(String),
    BucketExhausted { requested: usize, max_bucket: usize },
    ValidationFailed(String),
    PackingError(String),
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnresolvedDimension(n) => write!(f, "unresolved dim: {n}"),
            Self::BroadcastIncompatible { dim_a, dim_b } => {
                write!(f, "cannot broadcast {dim_a} with {dim_b}")
            }
            Self::RankMismatch { expected, got } => {
                write!(f, "rank mismatch: expected {expected}, got {got}")
            }
            Self::ConstraintViolation(msg) => write!(f, "constraint: {msg}"),
            Self::InvalidPadding(msg) => write!(f, "padding: {msg}"),
            Self::BucketExhausted { requested, max_bucket } => {
                write!(f, "bucket exhausted: need {requested}, max {max_bucket}")
            }
            Self::ValidationFailed(msg) => write!(f, "validation: {msg}"),
            Self::PackingError(msg) => write!(f, "packing: {msg}"),
        }
    }
}

impl std::error::Error for ShapeError {}

// ---------------------------------------------------------------------------
// ShapeConstraint
// ---------------------------------------------------------------------------

/// A constraint between symbolic or concrete dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeConstraint {
    /// Two dimensions must be equal.
    Equal(String, String),
    /// Dimension must be a multiple of a given factor.
    MultipleOf(String, usize),
    /// Dimension must lie within `[min, max]` inclusive.
    Bounded { name: String, min: usize, max: usize },
    /// Product of a set of dims must not exceed a limit.
    MaxProduct { names: Vec<String>, limit: usize },
}

impl ShapeConstraint {
    /// Check whether the constraint holds given concrete bindings.
    pub fn check(&self, bindings: &HashMap<String, usize>) -> Result<(), ShapeError> {
        match self {
            Self::Equal(a, b) => {
                let va =
                    bindings.get(a).ok_or_else(|| ShapeError::UnresolvedDimension(a.clone()))?;
                let vb =
                    bindings.get(b).ok_or_else(|| ShapeError::UnresolvedDimension(b.clone()))?;
                if va != vb {
                    return Err(ShapeError::ConstraintViolation(format!("{a}={va} != {b}={vb}")));
                }
                Ok(())
            }
            Self::MultipleOf(name, factor) => {
                let v = bindings
                    .get(name)
                    .ok_or_else(|| ShapeError::UnresolvedDimension(name.clone()))?;
                if v % factor != 0 {
                    return Err(ShapeError::ConstraintViolation(format!(
                        "{name}={v} not multiple of {factor}"
                    )));
                }
                Ok(())
            }
            Self::Bounded { name, min, max } => {
                let v = bindings
                    .get(name)
                    .ok_or_else(|| ShapeError::UnresolvedDimension(name.clone()))?;
                if v < min || v > max {
                    return Err(ShapeError::ConstraintViolation(format!(
                        "{name}={v} not in [{min}, {max}]"
                    )));
                }
                Ok(())
            }
            Self::MaxProduct { names, limit } => {
                let product: usize = names
                    .iter()
                    .map(|n| {
                        bindings
                            .get(n)
                            .copied()
                            .ok_or_else(|| ShapeError::UnresolvedDimension(n.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .product();
                if product > *limit {
                    return Err(ShapeError::ConstraintViolation(format!(
                        "product {product} exceeds limit {limit}"
                    )));
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ShapeInference
// ---------------------------------------------------------------------------

/// Infers output shapes from operation descriptors and input shapes.
#[derive(Debug, Clone)]
pub struct ShapeInference {
    rules: Vec<InferenceRule>,
}

/// An operation-level rule that maps input shapes to output shapes.
#[derive(Debug, Clone)]
pub struct InferenceRule {
    pub op_name: String,
    pub input_ranks: Vec<usize>,
    pub output_rank: usize,
    /// Index-based mapping: output dim `i` comes from input `source.0`, dim `source.1`.
    pub dim_map: Vec<DimSource>,
}

/// Where an output dimension comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimSource {
    /// Copy from input tensor at (`input_index`, `dim_index`).
    Input(usize, usize),
    /// A fixed constant.
    Constant(usize),
    /// Product of two input dims.
    Product(usize, usize, usize, usize),
    /// Ceiling division of an input dim by a constant.
    CeilDiv(usize, usize, usize),
}

impl ShapeInference {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Register an inference rule.
    pub fn add_rule(&mut self, rule: InferenceRule) {
        self.rules.push(rule);
    }

    /// Register a built-in matmul rule: `[M, K] x [K, N] -> [M, N]`.
    pub fn add_matmul_rule(&mut self) {
        self.add_rule(InferenceRule {
            op_name: "matmul".into(),
            input_ranks: vec![2, 2],
            output_rank: 2,
            dim_map: vec![DimSource::Input(0, 0), DimSource::Input(1, 1)],
        });
    }

    /// Register a built-in batch-matmul rule: `[B, M, K] x [B, K, N] -> [B, M, N]`.
    pub fn add_batch_matmul_rule(&mut self) {
        self.add_rule(InferenceRule {
            op_name: "batch_matmul".into(),
            input_ranks: vec![3, 3],
            output_rank: 3,
            dim_map: vec![DimSource::Input(0, 0), DimSource::Input(0, 1), DimSource::Input(1, 2)],
        });
    }

    /// Register a concat rule along a given axis for `n_inputs` inputs of rank `rank`.
    pub fn add_concat_rule(&mut self, rank: usize, axis: usize, n_inputs: usize) {
        // All non-axis dims come from input 0; axis dim is sum (approximated
        // by product for symbolic ease — callers use concrete resolve).
        let mut dim_map: Vec<DimSource> = Vec::with_capacity(rank);
        for d in 0..rank {
            if d == axis {
                // For concat, output axis = sum of all input axis dims.
                // We encode only 2-input sum via Product(placeholder) — the
                // concrete path handles arbitrary n via `infer_concat`.
                dim_map.push(DimSource::Constant(0));
            } else {
                dim_map.push(DimSource::Input(0, d));
            }
        }
        self.add_rule(InferenceRule {
            op_name: format!("concat_axis{axis}_n{n_inputs}"),
            input_ranks: vec![rank; n_inputs],
            output_rank: rank,
            dim_map,
        });
    }

    /// Infer the output shape of a named operation given concrete input shapes.
    pub fn infer(&self, op_name: &str, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let rule =
            self.rules.iter().find(|r| r.op_name == op_name).ok_or_else(|| {
                ShapeError::ValidationFailed(format!("no rule for op `{op_name}`"))
            })?;
        if inputs.len() != rule.input_ranks.len() {
            return Err(ShapeError::RankMismatch {
                expected: rule.input_ranks.len(),
                got: inputs.len(),
            });
        }
        for (i, (&expected_rank, input)) in rule.input_ranks.iter().zip(inputs.iter()).enumerate() {
            if input.len() != expected_rank {
                return Err(ShapeError::RankMismatch { expected: expected_rank, got: input.len() });
            }
            let _ = i;
        }
        let mut out = Vec::with_capacity(rule.output_rank);
        for src in &rule.dim_map {
            let val = match src {
                DimSource::Input(inp, dim) => inputs[*inp][*dim],
                DimSource::Constant(c) => *c,
                DimSource::Product(i0, d0, i1, d1) => inputs[*i0][*d0] * inputs[*i1][*d1],
                DimSource::CeilDiv(inp, dim, divisor) => inputs[*inp][*dim].div_ceil(*divisor),
            };
            out.push(val);
        }
        Ok(out)
    }

    /// Infer output shape for a concat along `axis` with arbitrary inputs.
    pub fn infer_concat(&self, axis: usize, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        if inputs.is_empty() {
            return Err(ShapeError::ValidationFailed("concat requires ≥1 input".into()));
        }
        let rank = inputs[0].len();
        for (i, inp) in inputs.iter().enumerate().skip(1) {
            if inp.len() != rank {
                return Err(ShapeError::RankMismatch { expected: rank, got: inp.len() });
            }
            for d in 0..rank {
                if d != axis && inp[d] != inputs[0][d] {
                    return Err(ShapeError::ConstraintViolation(format!(
                        "concat dim {d}: input 0 has {}, input {i} has {}",
                        inputs[0][d], inp[d]
                    )));
                }
            }
        }
        let mut out = inputs[0].to_vec();
        out[axis] = inputs.iter().map(|s| s[axis]).sum();
        Ok(out)
    }
}

impl Default for ShapeInference {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PaddingStrategy
// ---------------------------------------------------------------------------

/// Strategy for padding variable-length sequences to uniform length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad to the maximum length in the batch.
    PadToMax,
    /// Pad to the next multiple of `alignment`.
    PadToMultiple(usize),
    /// Pad to a fixed length; error if any sequence exceeds it.
    PadToFixed(usize),
    /// No padding — each sequence keeps its own length.
    NoPadding,
}

impl PaddingStrategy {
    /// Compute the padded length for a batch of sequence lengths.
    pub fn padded_length(&self, lengths: &[usize]) -> Result<usize, ShapeError> {
        if lengths.is_empty() {
            return Err(ShapeError::InvalidPadding("empty batch".into()));
        }
        let max_len = *lengths.iter().max().unwrap();
        match self {
            Self::PadToMax | Self::NoPadding => Ok(max_len),
            Self::PadToMultiple(align) => {
                if *align == 0 {
                    return Err(ShapeError::InvalidPadding("alignment must be > 0".into()));
                }
                Ok(max_len.div_ceil(*align) * align)
            }
            Self::PadToFixed(fixed) => {
                if max_len > *fixed {
                    return Err(ShapeError::InvalidPadding(format!(
                        "sequence length {max_len} exceeds fixed {fixed}"
                    )));
                }
                Ok(*fixed)
            }
        }
    }

    /// Build a padding mask (1 = real token, 0 = pad) for each sequence.
    pub fn build_mask(&self, lengths: &[usize]) -> Result<Vec<Vec<u8>>, ShapeError> {
        let padded = self.padded_length(lengths)?;
        Ok(lengths
            .iter()
            .map(|&len| {
                let mut mask = vec![1u8; len];
                mask.resize(padded, 0);
                mask
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// BucketAllocator
// ---------------------------------------------------------------------------

/// Allocates from a set of pre-defined bucket sizes to reduce allocation churn.
#[derive(Debug, Clone)]
pub struct BucketAllocator {
    /// Sorted bucket sizes (ascending).
    buckets: Vec<usize>,
    /// Count of active allocations per bucket index.
    active: Vec<usize>,
    /// Total allocations served per bucket index (lifetime).
    total_allocs: Vec<usize>,
}

impl BucketAllocator {
    /// Create with the given bucket sizes. They will be sorted internally.
    pub fn new(mut buckets: Vec<usize>) -> Self {
        buckets.sort_unstable();
        buckets.dedup();
        let n = buckets.len();
        Self { buckets, active: vec![0; n], total_allocs: vec![0; n] }
    }

    /// Create buckets at powers of two from `min_pow2` to `max_pow2` inclusive.
    pub fn power_of_two(min_pow2: u32, max_pow2: u32) -> Self {
        let buckets: Vec<usize> = (min_pow2..=max_pow2).map(|p| 1usize << p).collect();
        Self::new(buckets)
    }

    /// Find the smallest bucket that fits `size`.
    pub fn bucket_for(&self, size: usize) -> Result<usize, ShapeError> {
        self.buckets.iter().copied().find(|&b| b >= size).ok_or_else(|| {
            ShapeError::BucketExhausted {
                requested: size,
                max_bucket: self.buckets.last().copied().unwrap_or(0),
            }
        })
    }

    /// Allocate memory for `size` elements, returning the bucket size used.
    pub fn allocate(&mut self, size: usize) -> Result<usize, ShapeError> {
        let bucket = self.bucket_for(size)?;
        let idx = self.buckets.iter().position(|&b| b == bucket).unwrap();
        self.active[idx] += 1;
        self.total_allocs[idx] += 1;
        Ok(bucket)
    }

    /// Release an allocation of the given bucket size.
    pub fn release(&mut self, bucket_size: usize) -> bool {
        if let Some(idx) =
            self.buckets.iter().position(|&b| b == bucket_size).filter(|&idx| self.active[idx] > 0)
        {
            self.active[idx] -= 1;
            return true;
        }
        false
    }

    /// Bucket sizes in sorted order.
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// Fragmentation ratio: wasted / total allocated across all active allocs.
    /// Returns 0.0 when nothing is allocated.
    pub fn fragmentation(&self, requests: &[usize]) -> f64 {
        if requests.is_empty() {
            return 0.0;
        }
        let mut total_alloc = 0usize;
        let mut total_request = 0usize;
        for &r in requests {
            if let Ok(b) = self.bucket_for(r) {
                total_alloc += b;
                total_request += r;
            }
        }
        if total_alloc == 0 {
            return 0.0;
        }
        (total_alloc - total_request) as f64 / total_alloc as f64
    }

    /// Number of currently active allocations.
    pub fn active_count(&self) -> usize {
        self.active.iter().sum()
    }

    /// Total allocations served over the allocator's lifetime.
    pub fn total_allocations(&self) -> usize {
        self.total_allocs.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// ShapeValidator
// ---------------------------------------------------------------------------

/// Validates resolved shapes against a set of constraints and operation rules.
#[derive(Debug, Clone)]
pub struct ShapeValidator {
    constraints: Vec<ShapeConstraint>,
    max_rank: usize,
    max_numel: usize,
}

impl ShapeValidator {
    pub fn new(max_rank: usize, max_numel: usize) -> Self {
        Self { constraints: Vec::new(), max_rank, max_numel }
    }

    pub fn add_constraint(&mut self, c: ShapeConstraint) {
        self.constraints.push(c);
    }

    /// Validate that concrete `dims` are within global limits.
    pub fn validate_shape(&self, dims: &[usize]) -> Result<(), ShapeError> {
        if dims.len() > self.max_rank {
            return Err(ShapeError::ValidationFailed(format!(
                "rank {} exceeds max {}",
                dims.len(),
                self.max_rank
            )));
        }
        let numel: usize = dims.iter().product();
        if numel > self.max_numel {
            return Err(ShapeError::ValidationFailed(format!(
                "numel {numel} exceeds max {}",
                self.max_numel
            )));
        }
        if dims.contains(&0) {
            return Err(ShapeError::ValidationFailed("zero-sized dimension".into()));
        }
        Ok(())
    }

    /// Validate all registered constraints against bindings.
    pub fn validate_constraints(
        &self,
        bindings: &HashMap<String, usize>,
    ) -> Result<(), ShapeError> {
        for c in &self.constraints {
            c.check(bindings)?;
        }
        Ok(())
    }

    /// Combined validation: shape limits + constraints.
    pub fn validate_all(
        &self,
        dims: &[usize],
        bindings: &HashMap<String, usize>,
    ) -> Result<(), ShapeError> {
        self.validate_shape(dims)?;
        self.validate_constraints(bindings)
    }

    pub fn max_rank(&self) -> usize {
        self.max_rank
    }

    pub fn max_numel(&self) -> usize {
        self.max_numel
    }

    pub fn constraints(&self) -> &[ShapeConstraint] {
        &self.constraints
    }
}

// ---------------------------------------------------------------------------
// ShapeOptimizer
// ---------------------------------------------------------------------------

/// Optimizes memory layout for a set of shape patterns.
#[derive(Debug, Clone)]
pub struct ShapeOptimizer {
    /// Alignment requirement in elements (e.g., 64 for cache-line).
    alignment: usize,
    /// Whether to prefer contiguous (row-major) layout.
    prefer_contiguous: bool,
}

impl ShapeOptimizer {
    pub fn new(alignment: usize, prefer_contiguous: bool) -> Self {
        Self { alignment, prefer_contiguous }
    }

    /// Compute row-major strides for a given shape.
    pub fn row_major_strides(dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![1usize; dims.len()];
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    /// Compute column-major strides for a given shape.
    pub fn col_major_strides(dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return vec![];
        }
        let mut strides = vec![1usize; dims.len()];
        for i in 1..dims.len() {
            strides[i] = strides[i - 1] * dims[i - 1];
        }
        strides
    }

    /// Align the innermost stride to `self.alignment`.
    pub fn aligned_strides(&self, dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return vec![];
        }
        let base = if self.prefer_contiguous {
            Self::row_major_strides(dims)
        } else {
            Self::col_major_strides(dims)
        };
        if self.alignment <= 1 {
            return base;
        }
        // Align the innermost dimension's stride.
        let mut strides = base;
        if self.prefer_contiguous {
            // For row-major, the last stride is 1, but we pad the second-to-last.
            if dims.len() >= 2 {
                let inner = dims[dims.len() - 1];
                let aligned_inner = inner.div_ceil(self.alignment) * self.alignment;
                // Recompute strides with aligned inner dim.
                strides[dims.len() - 1] = 1;
                for i in (0..dims.len() - 1).rev() {
                    let dim_below = if i == dims.len() - 2 { aligned_inner } else { dims[i + 1] };
                    strides[i] = strides[i + 1] * dim_below;
                }
            }
        } else {
            // For column-major, the first stride is 1, pad the second.
            if dims.len() >= 2 {
                let first = dims[0];
                let aligned_first = first.div_ceil(self.alignment) * self.alignment;
                strides[0] = 1;
                for i in 1..dims.len() {
                    let dim_below = if i == 1 { aligned_first } else { dims[i - 1] };
                    strides[i] = strides[i - 1] * dim_below;
                }
            }
        }
        strides
    }

    /// Total allocation size (in elements) for aligned strides.
    pub fn alloc_size(&self, dims: &[usize]) -> usize {
        if dims.is_empty() {
            return 0;
        }
        let strides = self.aligned_strides(dims);
        // Allocation = max(stride[i] * dim[i])
        strides.iter().zip(dims.iter()).map(|(&s, &d)| s * d).max().unwrap_or(0)
    }

    /// Suggest the best layout (row-major or column-major) for a reduction on `axis`.
    pub fn suggest_layout_for_reduction(&self, dims: &[usize], axis: usize) -> &'static str {
        if axis == dims.len() - 1 {
            "row_major"
        } else if axis == 0 {
            "col_major"
        } else {
            "row_major"
        }
    }

    /// Transpose shape dimensions (swap two axes).
    pub fn transpose_shape(dims: &[usize], axis_a: usize, axis_b: usize) -> Vec<usize> {
        let mut out = dims.to_vec();
        out.swap(axis_a, axis_b);
        out
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

// ---------------------------------------------------------------------------
// SequencePacker
// ---------------------------------------------------------------------------

/// Packs variable-length sequences into a contiguous buffer for batched processing.
#[derive(Debug, Clone)]
pub struct SequencePacker {
    padding_strategy: PaddingStrategy,
    pad_value: f32,
}

/// Result of packing multiple sequences.
#[derive(Debug, Clone)]
pub struct PackedSequences {
    /// Flat buffer of packed data (`batch_size` × `padded_len`).
    pub data: Vec<f32>,
    /// Original lengths of each sequence.
    pub lengths: Vec<usize>,
    /// Padded length per sequence (uniform).
    pub padded_length: usize,
    /// Number of sequences.
    pub batch_size: usize,
    /// Cumulative offsets into `data` for each sequence.
    pub offsets: Vec<usize>,
    /// Attention mask (1.0 = real, 0.0 = pad).
    pub mask: Vec<f32>,
}

impl SequencePacker {
    pub fn new(padding_strategy: PaddingStrategy, pad_value: f32) -> Self {
        Self { padding_strategy, pad_value }
    }

    /// Pack a batch of variable-length sequences.
    pub fn pack(&self, sequences: &[&[f32]]) -> Result<PackedSequences, ShapeError> {
        if sequences.is_empty() {
            return Err(ShapeError::PackingError("empty batch".into()));
        }
        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let padded_length = self.padding_strategy.padded_length(&lengths)?;
        let batch_size = sequences.len();
        let total = batch_size * padded_length;
        let mut data = vec![self.pad_value; total];
        let mut mask = vec![0.0f32; total];
        let mut offsets = Vec::with_capacity(batch_size);

        for (i, seq) in sequences.iter().enumerate() {
            let offset = i * padded_length;
            offsets.push(offset);
            data[offset..offset + seq.len()].copy_from_slice(seq);
            for j in 0..seq.len() {
                mask[offset + j] = 1.0;
            }
        }

        Ok(PackedSequences { data, lengths, padded_length, batch_size, offsets, mask })
    }

    /// Unpack sequences from packed representation back to individual vectors.
    pub fn unpack(&self, packed: &PackedSequences) -> Vec<Vec<f32>> {
        packed
            .lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                let start = packed.offsets[i];
                packed.data[start..start + len].to_vec()
            })
            .collect()
    }

    /// Pack sequences sorted by length (longest first) for efficiency.
    pub fn pack_sorted(
        &self,
        sequences: &[&[f32]],
    ) -> Result<(PackedSequences, Vec<usize>), ShapeError> {
        if sequences.is_empty() {
            return Err(ShapeError::PackingError("empty batch".into()));
        }
        let mut indexed: Vec<(usize, &&[f32])> = sequences.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        let sort_order: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let sorted_seqs: Vec<&[f32]> = indexed.iter().map(|(_, s)| **s).collect();
        let packed = self.pack(&sorted_seqs)?;
        Ok((packed, sort_order))
    }

    pub fn padding_strategy(&self) -> &PaddingStrategy {
        &self.padding_strategy
    }

    pub fn pad_value(&self) -> f32 {
        self.pad_value
    }
}

// ---------------------------------------------------------------------------
// DynamicShapeEngine
// ---------------------------------------------------------------------------

/// Unified shape management for inference pipelines.
///
/// Combines shape inference, constraint validation, bucket allocation,
/// sequence packing, and memory optimization into one facade.
#[derive(Debug, Clone)]
pub struct DynamicShapeEngine {
    inference: ShapeInference,
    validator: ShapeValidator,
    allocator: BucketAllocator,
    optimizer: ShapeOptimizer,
    packer: SequencePacker,
    bindings: HashMap<String, usize>,
}

impl DynamicShapeEngine {
    pub fn new(
        inference: ShapeInference,
        validator: ShapeValidator,
        allocator: BucketAllocator,
        optimizer: ShapeOptimizer,
        packer: SequencePacker,
    ) -> Self {
        Self { inference, validator, allocator, optimizer, packer, bindings: HashMap::new() }
    }

    /// Create with sensible defaults for typical LLM inference.
    pub fn default_for_llm(max_seq_len: usize, _hidden_dim: usize) -> Self {
        let mut inference = ShapeInference::new();
        inference.add_matmul_rule();
        inference.add_batch_matmul_rule();

        let mut validator = ShapeValidator::new(6, 1 << 30);
        validator.add_constraint(ShapeConstraint::Bounded {
            name: "seq_len".into(),
            min: 1,
            max: max_seq_len,
        });
        validator.add_constraint(ShapeConstraint::MultipleOf("hidden_dim".into(), 64));

        let allocator = BucketAllocator::power_of_two(8, 24);
        let optimizer = ShapeOptimizer::new(64, true);
        let packer = SequencePacker::new(PaddingStrategy::PadToMultiple(64), 0.0);

        Self::new(inference, validator, allocator, optimizer, packer)
    }

    /// Bind a symbolic name to a concrete value.
    pub fn bind(&mut self, name: impl Into<String>, value: usize) {
        self.bindings.insert(name.into(), value);
    }

    /// Remove a binding.
    pub fn unbind(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// Resolve a `ShapeSpec` using current bindings.
    pub fn resolve(&self, spec: &ShapeSpec) -> Result<Vec<usize>, ShapeError> {
        spec.resolve(&self.bindings).ok_or_else(|| {
            let missing: Vec<_> = spec
                .symbolic_names()
                .into_iter()
                .filter(|n| !self.bindings.contains_key(*n))
                .map(String::from)
                .collect();
            ShapeError::UnresolvedDimension(missing.join(", "))
        })
    }

    /// Resolve and validate a shape spec.
    pub fn resolve_and_validate(&self, spec: &ShapeSpec) -> Result<Vec<usize>, ShapeError> {
        let dims = self.resolve(spec)?;
        self.validator.validate_all(&dims, &self.bindings)?;
        Ok(dims)
    }

    /// Infer an operation output shape and validate.
    pub fn infer_and_validate(
        &self,
        op_name: &str,
        inputs: &[&[usize]],
    ) -> Result<Vec<usize>, ShapeError> {
        let out = self.inference.infer(op_name, inputs)?;
        self.validator.validate_shape(&out)?;
        Ok(out)
    }

    /// Allocate a bucket for the given number of elements.
    pub fn allocate(&mut self, numel: usize) -> Result<usize, ShapeError> {
        self.allocator.allocate(numel)
    }

    /// Release a bucket allocation.
    pub fn release(&mut self, bucket: usize) -> bool {
        self.allocator.release(bucket)
    }

    /// Pack variable-length sequences.
    pub fn pack_sequences(&self, sequences: &[&[f32]]) -> Result<PackedSequences, ShapeError> {
        self.packer.pack(sequences)
    }

    /// Unpack sequences from packed form.
    pub fn unpack_sequences(&self, packed: &PackedSequences) -> Vec<Vec<f32>> {
        self.packer.unpack(packed)
    }

    /// Compute optimized strides for a resolved shape.
    pub fn optimized_strides(&self, dims: &[usize]) -> Vec<usize> {
        self.optimizer.aligned_strides(dims)
    }

    pub fn bindings(&self) -> &HashMap<String, usize> {
        &self.bindings
    }

    pub fn inference(&self) -> &ShapeInference {
        &self.inference
    }

    pub fn validator(&self) -> &ShapeValidator {
        &self.validator
    }

    pub fn allocator(&self) -> &BucketAllocator {
        &self.allocator
    }

    pub fn optimizer(&self) -> &ShapeOptimizer {
        &self.optimizer
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bindings(pairs: &[(&str, usize)]) -> HashMap<String, usize> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // -----------------------------------------------------------------------
    // DynamicDim
    // -----------------------------------------------------------------------

    #[test]
    fn dim_static_resolve() {
        let d = DynamicDim::fixed(42);
        assert_eq!(d.resolve(&HashMap::new()), Some(42));
    }

    #[test]
    fn dim_named_resolve() {
        let d = DynamicDim::named("batch");
        assert_eq!(d.resolve(&bindings(&[("batch", 8)])), Some(8));
    }

    #[test]
    fn dim_named_unresolved() {
        let d = DynamicDim::named("batch");
        assert_eq!(d.resolve(&HashMap::new()), None);
    }

    #[test]
    fn dim_derived_resolve() {
        let d = DynamicDim::derived("seq_len", 2, 1);
        assert_eq!(d.resolve(&bindings(&[("seq_len", 128)])), Some(257));
    }

    #[test]
    fn dim_derived_negative_result() {
        let d = DynamicDim::derived("x", 1, -100);
        assert_eq!(d.resolve(&bindings(&[("x", 10)])), None);
    }

    #[test]
    fn dim_is_static() {
        assert!(DynamicDim::fixed(1).is_static());
        assert!(!DynamicDim::named("x").is_static());
    }

    #[test]
    fn dim_static_value() {
        assert_eq!(DynamicDim::fixed(7).static_value(), Some(7));
        assert_eq!(DynamicDim::named("x").static_value(), None);
    }

    #[test]
    fn dim_referenced_names_static() {
        assert!(DynamicDim::fixed(1).referenced_names().is_empty());
    }

    #[test]
    fn dim_referenced_names_named() {
        assert_eq!(DynamicDim::named("seq").referenced_names(), vec!["seq"]);
    }

    #[test]
    fn dim_referenced_names_derived() {
        let d = DynamicDim::derived("base", 2, 0);
        assert_eq!(d.referenced_names(), vec!["base"]);
    }

    #[test]
    fn dim_debug_display() {
        assert_eq!(format!("{:?}", DynamicDim::fixed(5)), "5");
        assert_eq!(format!("{}", DynamicDim::named("B")), "B");
        assert_eq!(format!("{}", DynamicDim::derived("S", 4, -1)), "S*4-1");
    }

    #[test]
    fn dim_clone_eq() {
        let a = DynamicDim::named("x");
        let b = a.clone();
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // ShapeSpec
    // -----------------------------------------------------------------------

    #[test]
    fn shape_static_construction() {
        let s = ShapeSpec::static_shape(&[2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert!(s.is_fully_static());
    }

    #[test]
    fn shape_resolve_static() {
        let s = ShapeSpec::static_shape(&[2, 3]);
        assert_eq!(s.resolve(&HashMap::new()), Some(vec![2, 3]));
    }

    #[test]
    fn shape_resolve_dynamic() {
        let s = ShapeSpec::new(vec![DynamicDim::named("B"), DynamicDim::fixed(768)]);
        let b = bindings(&[("B", 4)]);
        assert_eq!(s.resolve(&b), Some(vec![4, 768]));
    }

    #[test]
    fn shape_resolve_missing_binding() {
        let s = ShapeSpec::new(vec![DynamicDim::named("B")]);
        assert_eq!(s.resolve(&HashMap::new()), None);
    }

    #[test]
    fn shape_numel() {
        let s = ShapeSpec::static_shape(&[2, 3, 4]);
        assert_eq!(s.resolve_numel(&HashMap::new()), Some(24));
    }

    #[test]
    fn shape_symbolic_names() {
        let s = ShapeSpec::new(vec![
            DynamicDim::named("B"),
            DynamicDim::named("S"),
            DynamicDim::fixed(768),
        ]);
        let names = s.symbolic_names();
        assert_eq!(names, vec!["B", "S"]);
    }

    #[test]
    fn shape_broadcast_same() {
        let a = ShapeSpec::static_shape(&[3, 4]);
        let b = ShapeSpec::static_shape(&[3, 4]);
        let c = ShapeSpec::broadcast(&a, &b).unwrap();
        assert_eq!(c.resolve(&HashMap::new()), Some(vec![3, 4]));
    }

    #[test]
    fn shape_broadcast_unit_expand() {
        let a = ShapeSpec::static_shape(&[1, 4]);
        let b = ShapeSpec::static_shape(&[3, 1]);
        let c = ShapeSpec::broadcast(&a, &b).unwrap();
        assert_eq!(c.resolve(&HashMap::new()), Some(vec![3, 4]));
    }

    #[test]
    fn shape_broadcast_rank_expand() {
        let a = ShapeSpec::static_shape(&[4]);
        let b = ShapeSpec::static_shape(&[3, 4]);
        let c = ShapeSpec::broadcast(&a, &b).unwrap();
        assert_eq!(c.resolve(&HashMap::new()), Some(vec![3, 4]));
    }

    #[test]
    fn shape_broadcast_incompatible() {
        let a = ShapeSpec::static_shape(&[3]);
        let b = ShapeSpec::static_shape(&[4]);
        assert!(ShapeSpec::broadcast(&a, &b).is_err());
    }

    #[test]
    fn shape_not_fully_static() {
        let s = ShapeSpec::new(vec![DynamicDim::named("B"), DynamicDim::fixed(8)]);
        assert!(!s.is_fully_static());
    }

    #[test]
    fn shape_empty() {
        let s = ShapeSpec::static_shape(&[]);
        assert_eq!(s.rank(), 0);
        assert!(s.is_fully_static());
        assert_eq!(s.resolve(&HashMap::new()), Some(vec![]));
    }

    // -----------------------------------------------------------------------
    // ShapeConstraint
    // -----------------------------------------------------------------------

    #[test]
    fn constraint_equal_pass() {
        let c = ShapeConstraint::Equal("A".into(), "B".into());
        assert!(c.check(&bindings(&[("A", 8), ("B", 8)])).is_ok());
    }

    #[test]
    fn constraint_equal_fail() {
        let c = ShapeConstraint::Equal("A".into(), "B".into());
        assert!(c.check(&bindings(&[("A", 8), ("B", 16)])).is_err());
    }

    #[test]
    fn constraint_equal_unresolved() {
        let c = ShapeConstraint::Equal("A".into(), "B".into());
        assert!(c.check(&bindings(&[("A", 8)])).is_err());
    }

    #[test]
    fn constraint_multiple_of_pass() {
        let c = ShapeConstraint::MultipleOf("D".into(), 64);
        assert!(c.check(&bindings(&[("D", 768)])).is_ok());
    }

    #[test]
    fn constraint_multiple_of_fail() {
        let c = ShapeConstraint::MultipleOf("D".into(), 64);
        assert!(c.check(&bindings(&[("D", 100)])).is_err());
    }

    #[test]
    fn constraint_bounded_pass() {
        let c = ShapeConstraint::Bounded { name: "S".into(), min: 1, max: 2048 };
        assert!(c.check(&bindings(&[("S", 512)])).is_ok());
    }

    #[test]
    fn constraint_bounded_below_min() {
        let c = ShapeConstraint::Bounded { name: "S".into(), min: 1, max: 2048 };
        assert!(c.check(&bindings(&[("S", 0)])).is_err());
    }

    #[test]
    fn constraint_bounded_above_max() {
        let c = ShapeConstraint::Bounded { name: "S".into(), min: 1, max: 2048 };
        assert!(c.check(&bindings(&[("S", 4096)])).is_err());
    }

    #[test]
    fn constraint_bounded_at_boundary() {
        let c = ShapeConstraint::Bounded { name: "S".into(), min: 1, max: 2048 };
        assert!(c.check(&bindings(&[("S", 1)])).is_ok());
        assert!(c.check(&bindings(&[("S", 2048)])).is_ok());
    }

    #[test]
    fn constraint_max_product_pass() {
        let c = ShapeConstraint::MaxProduct { names: vec!["B".into(), "S".into()], limit: 65536 };
        assert!(c.check(&bindings(&[("B", 8), ("S", 1024)])).is_ok());
    }

    #[test]
    fn constraint_max_product_fail() {
        let c = ShapeConstraint::MaxProduct { names: vec!["B".into(), "S".into()], limit: 65536 };
        assert!(c.check(&bindings(&[("B", 128), ("S", 1024)])).is_err());
    }

    #[test]
    fn constraint_max_product_unresolved() {
        let c = ShapeConstraint::MaxProduct { names: vec!["B".into(), "S".into()], limit: 65536 };
        assert!(c.check(&bindings(&[("B", 8)])).is_err());
    }

    // -----------------------------------------------------------------------
    // ShapeInference
    // -----------------------------------------------------------------------

    #[test]
    fn infer_matmul() {
        let mut si = ShapeInference::new();
        si.add_matmul_rule();
        let out = si.infer("matmul", &[&[4, 8], &[8, 16]]).unwrap();
        assert_eq!(out, vec![4, 16]);
    }

    #[test]
    fn infer_batch_matmul() {
        let mut si = ShapeInference::new();
        si.add_batch_matmul_rule();
        let out = si.infer("batch_matmul", &[&[2, 4, 8], &[2, 8, 16]]).unwrap();
        assert_eq!(out, vec![2, 4, 16]);
    }

    #[test]
    fn infer_wrong_op() {
        let si = ShapeInference::new();
        assert!(si.infer("unknown", &[&[1]]).is_err());
    }

    #[test]
    fn infer_wrong_num_inputs() {
        let mut si = ShapeInference::new();
        si.add_matmul_rule();
        assert!(si.infer("matmul", &[&[4, 8]]).is_err());
    }

    #[test]
    fn infer_wrong_rank() {
        let mut si = ShapeInference::new();
        si.add_matmul_rule();
        assert!(si.infer("matmul", &[&[4, 8, 2], &[8, 16]]).is_err());
    }

    #[test]
    fn infer_concat_basic() {
        let si = ShapeInference::new();
        let out = si.infer_concat(0, &[&[3, 4], &[5, 4]]).unwrap();
        assert_eq!(out, vec![8, 4]);
    }

    #[test]
    fn infer_concat_axis1() {
        let si = ShapeInference::new();
        let out = si.infer_concat(1, &[&[3, 4], &[3, 6]]).unwrap();
        assert_eq!(out, vec![3, 10]);
    }

    #[test]
    fn infer_concat_multi() {
        let si = ShapeInference::new();
        let out = si.infer_concat(0, &[&[1, 4], &[2, 4], &[3, 4]]).unwrap();
        assert_eq!(out, vec![6, 4]);
    }

    #[test]
    fn infer_concat_empty() {
        let si = ShapeInference::new();
        assert!(si.infer_concat(0, &[]).is_err());
    }

    #[test]
    fn infer_concat_rank_mismatch() {
        let si = ShapeInference::new();
        assert!(si.infer_concat(0, &[&[3, 4], &[5, 4, 1]]).is_err());
    }

    #[test]
    fn infer_concat_non_axis_mismatch() {
        let si = ShapeInference::new();
        assert!(si.infer_concat(0, &[&[3, 4], &[5, 7]]).is_err());
    }

    #[test]
    fn infer_custom_rule_ceil_div() {
        let mut si = ShapeInference::new();
        si.add_rule(InferenceRule {
            op_name: "pool2x".into(),
            input_ranks: vec![2],
            output_rank: 2,
            dim_map: vec![DimSource::CeilDiv(0, 0, 2), DimSource::CeilDiv(0, 1, 2)],
        });
        let out = si.infer("pool2x", &[&[7, 10]]).unwrap();
        assert_eq!(out, vec![4, 5]);
    }

    #[test]
    fn infer_custom_rule_product() {
        let mut si = ShapeInference::new();
        si.add_rule(InferenceRule {
            op_name: "outer".into(),
            input_ranks: vec![1, 1],
            output_rank: 1,
            dim_map: vec![DimSource::Product(0, 0, 1, 0)],
        });
        let out = si.infer("outer", &[&[3], &[4]]).unwrap();
        assert_eq!(out, vec![12]);
    }

    #[test]
    fn infer_default_construction() {
        let si = ShapeInference::default();
        assert!(si.infer("nope", &[]).is_err());
    }

    // -----------------------------------------------------------------------
    // PaddingStrategy
    // -----------------------------------------------------------------------

    #[test]
    fn pad_to_max() {
        let p = PaddingStrategy::PadToMax;
        assert_eq!(p.padded_length(&[3, 5, 7]).unwrap(), 7);
    }

    #[test]
    fn pad_to_multiple() {
        let p = PaddingStrategy::PadToMultiple(8);
        assert_eq!(p.padded_length(&[3, 5, 7]).unwrap(), 8);
    }

    #[test]
    fn pad_to_multiple_exact() {
        let p = PaddingStrategy::PadToMultiple(4);
        assert_eq!(p.padded_length(&[4]).unwrap(), 4);
    }

    #[test]
    fn pad_to_multiple_zero_align() {
        let p = PaddingStrategy::PadToMultiple(0);
        assert!(p.padded_length(&[1]).is_err());
    }

    #[test]
    fn pad_to_fixed() {
        let p = PaddingStrategy::PadToFixed(16);
        assert_eq!(p.padded_length(&[3, 5]).unwrap(), 16);
    }

    #[test]
    fn pad_to_fixed_exceeded() {
        let p = PaddingStrategy::PadToFixed(4);
        assert!(p.padded_length(&[5]).is_err());
    }

    #[test]
    fn pad_no_padding() {
        let p = PaddingStrategy::NoPadding;
        assert_eq!(p.padded_length(&[3, 5, 7]).unwrap(), 7);
    }

    #[test]
    fn pad_empty_batch() {
        let p = PaddingStrategy::PadToMax;
        assert!(p.padded_length(&[]).is_err());
    }

    #[test]
    fn pad_mask_basic() {
        let p = PaddingStrategy::PadToMax;
        let masks = p.build_mask(&[2, 4]).unwrap();
        assert_eq!(masks[0], vec![1, 1, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 1]);
    }

    #[test]
    fn pad_mask_fixed() {
        let p = PaddingStrategy::PadToFixed(6);
        let masks = p.build_mask(&[3]).unwrap();
        assert_eq!(masks[0], vec![1, 1, 1, 0, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // BucketAllocator
    // -----------------------------------------------------------------------

    #[test]
    fn bucket_basic_fit() {
        let alloc = BucketAllocator::new(vec![64, 128, 256, 512]);
        assert_eq!(alloc.bucket_for(100).unwrap(), 128);
    }

    #[test]
    fn bucket_exact_fit() {
        let alloc = BucketAllocator::new(vec![64, 128, 256]);
        assert_eq!(alloc.bucket_for(128).unwrap(), 128);
    }

    #[test]
    fn bucket_too_large() {
        let alloc = BucketAllocator::new(vec![64, 128]);
        assert!(alloc.bucket_for(200).is_err());
    }

    #[test]
    fn bucket_allocate_release() {
        let mut alloc = BucketAllocator::new(vec![64, 128, 256]);
        let b = alloc.allocate(100).unwrap();
        assert_eq!(b, 128);
        assert_eq!(alloc.active_count(), 1);
        assert!(alloc.release(128));
        assert_eq!(alloc.active_count(), 0);
    }

    #[test]
    fn bucket_release_wrong_size() {
        let mut alloc = BucketAllocator::new(vec![64, 128]);
        assert!(!alloc.release(999));
    }

    #[test]
    fn bucket_release_when_empty() {
        let mut alloc = BucketAllocator::new(vec![64]);
        assert!(!alloc.release(64));
    }

    #[test]
    fn bucket_power_of_two() {
        let alloc = BucketAllocator::power_of_two(6, 10);
        assert_eq!(alloc.buckets(), &[64, 128, 256, 512, 1024]);
    }

    #[test]
    fn bucket_fragmentation_zero() {
        let alloc = BucketAllocator::new(vec![64]);
        assert_eq!(alloc.fragmentation(&[64]), 0.0);
    }

    #[test]
    fn bucket_fragmentation_positive() {
        let alloc = BucketAllocator::new(vec![128]);
        let frag = alloc.fragmentation(&[100]);
        assert!(frag > 0.0 && frag < 1.0);
    }

    #[test]
    fn bucket_fragmentation_empty() {
        let alloc = BucketAllocator::new(vec![64]);
        assert_eq!(alloc.fragmentation(&[]), 0.0);
    }

    #[test]
    fn bucket_total_allocations() {
        let mut alloc = BucketAllocator::new(vec![64, 128]);
        alloc.allocate(50).unwrap();
        alloc.allocate(100).unwrap();
        alloc.allocate(50).unwrap();
        assert_eq!(alloc.total_allocations(), 3);
    }

    #[test]
    fn bucket_dedup_and_sort() {
        let alloc = BucketAllocator::new(vec![256, 64, 128, 64]);
        assert_eq!(alloc.buckets(), &[64, 128, 256]);
    }

    #[test]
    fn bucket_smallest_first() {
        let alloc = BucketAllocator::new(vec![64, 128, 256, 512]);
        assert_eq!(alloc.bucket_for(1).unwrap(), 64);
    }

    // -----------------------------------------------------------------------
    // ShapeValidator
    // -----------------------------------------------------------------------

    #[test]
    fn validator_pass() {
        let v = ShapeValidator::new(4, 1_000_000);
        assert!(v.validate_shape(&[2, 3, 4]).is_ok());
    }

    #[test]
    fn validator_rank_exceeded() {
        let v = ShapeValidator::new(2, 1_000_000);
        assert!(v.validate_shape(&[1, 2, 3]).is_err());
    }

    #[test]
    fn validator_numel_exceeded() {
        let v = ShapeValidator::new(4, 100);
        assert!(v.validate_shape(&[20, 20]).is_err());
    }

    #[test]
    fn validator_zero_dim() {
        let v = ShapeValidator::new(4, 1_000_000);
        assert!(v.validate_shape(&[2, 0, 4]).is_err());
    }

    #[test]
    fn validator_constraints() {
        let mut v = ShapeValidator::new(4, 1_000_000);
        v.add_constraint(ShapeConstraint::MultipleOf("D".into(), 8));
        assert!(v.validate_constraints(&bindings(&[("D", 64)])).is_ok());
        assert!(v.validate_constraints(&bindings(&[("D", 65)])).is_err());
    }

    #[test]
    fn validator_all_combined() {
        let mut v = ShapeValidator::new(4, 1_000_000);
        v.add_constraint(ShapeConstraint::Bounded { name: "S".into(), min: 1, max: 512 });
        let b = bindings(&[("S", 128)]);
        assert!(v.validate_all(&[2, 128], &b).is_ok());
    }

    #[test]
    fn validator_accessors() {
        let v = ShapeValidator::new(5, 999);
        assert_eq!(v.max_rank(), 5);
        assert_eq!(v.max_numel(), 999);
        assert!(v.constraints().is_empty());
    }

    // -----------------------------------------------------------------------
    // ShapeOptimizer
    // -----------------------------------------------------------------------

    #[test]
    fn optimizer_row_major_strides() {
        assert_eq!(ShapeOptimizer::row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn optimizer_col_major_strides() {
        assert_eq!(ShapeOptimizer::col_major_strides(&[2, 3, 4]), vec![1, 2, 6]);
    }

    #[test]
    fn optimizer_empty_strides() {
        assert!(ShapeOptimizer::row_major_strides(&[]).is_empty());
        assert!(ShapeOptimizer::col_major_strides(&[]).is_empty());
    }

    #[test]
    fn optimizer_aligned_strides_no_align() {
        let opt = ShapeOptimizer::new(1, true);
        assert_eq!(opt.aligned_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn optimizer_aligned_strides_with_align() {
        let opt = ShapeOptimizer::new(8, true);
        let strides = opt.aligned_strides(&[2, 3, 5]);
        // Inner dim 5 -> aligned to 8. Strides: [24, 8, 1].
        assert_eq!(strides[2], 1);
        assert!(strides[1] >= 5);
        assert_eq!(strides[1] % 8, 0);
    }

    #[test]
    fn optimizer_alloc_size() {
        let opt = ShapeOptimizer::new(1, true);
        assert_eq!(opt.alloc_size(&[2, 3, 4]), 24);
    }

    #[test]
    fn optimizer_alloc_size_aligned() {
        let opt = ShapeOptimizer::new(8, true);
        let size = opt.alloc_size(&[2, 3, 5]);
        assert!(size >= 30); // at least 2*3*5
    }

    #[test]
    fn optimizer_alloc_size_empty() {
        let opt = ShapeOptimizer::new(8, true);
        assert_eq!(opt.alloc_size(&[]), 0);
    }

    #[test]
    fn optimizer_suggest_reduction_last() {
        let opt = ShapeOptimizer::new(1, true);
        assert_eq!(opt.suggest_layout_for_reduction(&[2, 3, 4], 2), "row_major");
    }

    #[test]
    fn optimizer_suggest_reduction_first() {
        let opt = ShapeOptimizer::new(1, true);
        assert_eq!(opt.suggest_layout_for_reduction(&[2, 3, 4], 0), "col_major");
    }

    #[test]
    fn optimizer_transpose() {
        assert_eq!(ShapeOptimizer::transpose_shape(&[2, 3, 4], 0, 2), vec![4, 3, 2]);
    }

    #[test]
    fn optimizer_alignment_accessor() {
        let opt = ShapeOptimizer::new(64, true);
        assert_eq!(opt.alignment(), 64);
    }

    #[test]
    fn optimizer_col_major_aligned() {
        let opt = ShapeOptimizer::new(8, false);
        let strides = opt.aligned_strides(&[5, 3, 4]);
        assert_eq!(strides[0], 1);
        // second stride should be aligned
        assert!(strides[1] >= 5);
        assert_eq!(strides[1] % 8, 0);
    }

    // -----------------------------------------------------------------------
    // SequencePacker
    // -----------------------------------------------------------------------

    #[test]
    fn packer_basic_roundtrip() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
        let s1 = [1.0, 2.0, 3.0];
        let s2 = [4.0, 5.0];
        let packed = packer.pack(&[&s1, &s2]).unwrap();
        let unpacked = packer.unpack(&packed);
        assert_eq!(unpacked[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(unpacked[1], vec![4.0, 5.0]);
    }

    #[test]
    fn packer_padded_length() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMultiple(8), 0.0);
        let s1 = [1.0; 5];
        let packed = packer.pack(&[&s1[..]]).unwrap();
        assert_eq!(packed.padded_length, 8);
    }

    #[test]
    fn packer_pad_value() {
        let packer = SequencePacker::new(PaddingStrategy::PadToFixed(4), -1.0);
        let s = [1.0, 2.0];
        let packed = packer.pack(&[&s]).unwrap();
        assert_eq!(packed.data[2], -1.0);
        assert_eq!(packed.data[3], -1.0);
    }

    #[test]
    fn packer_mask() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
        let s1 = [1.0, 2.0];
        let s2 = [3.0, 4.0, 5.0];
        let packed = packer.pack(&[&s1, &s2]).unwrap();
        assert_eq!(&packed.mask[0..3], &[1.0, 1.0, 0.0]);
        assert_eq!(&packed.mask[3..6], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn packer_empty_error() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
        assert!(packer.pack(&[]).is_err());
    }

    #[test]
    fn packer_sorted() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
        let s1 = [1.0];
        let s2 = [2.0, 3.0, 4.0];
        let s3 = [5.0, 6.0];
        let (packed, order) = packer.pack_sorted(&[&s1, &s2, &s3]).unwrap();
        // Longest first: s2, s3, s1 → order = [1, 2, 0]
        assert_eq!(order, vec![1, 2, 0]);
        assert_eq!(packed.batch_size, 3);
    }

    #[test]
    fn packer_sorted_empty() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
        assert!(packer.pack_sorted(&[]).is_err());
    }

    #[test]
    fn packer_offsets() {
        let packer = SequencePacker::new(PaddingStrategy::PadToFixed(4), 0.0);
        let s1 = [1.0, 2.0];
        let s2 = [3.0];
        let packed = packer.pack(&[&s1, &s2]).unwrap();
        assert_eq!(packed.offsets, vec![0, 4]);
    }

    #[test]
    fn packer_accessors() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMax, -1.0);
        assert_eq!(*packer.padding_strategy(), PaddingStrategy::PadToMax);
        assert_eq!(packer.pad_value(), -1.0);
    }

    #[test]
    fn packer_single_element_sequences() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMax, 0.0);
        let s1 = [1.0f32];
        let s2 = [2.0f32];
        let packed = packer.pack(&[&s1, &s2]).unwrap();
        assert_eq!(packed.padded_length, 1);
        let unpacked = packer.unpack(&packed);
        assert_eq!(unpacked, vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn packer_large_batch() {
        let packer = SequencePacker::new(PaddingStrategy::PadToMultiple(16), 0.0);
        let seqs: Vec<Vec<f32>> = (1..=32).map(|i| vec![i as f32; i]).collect();
        let refs: Vec<&[f32]> = seqs.iter().map(|s| s.as_slice()).collect();
        let packed = packer.pack(&refs).unwrap();
        assert_eq!(packed.batch_size, 32);
        assert_eq!(packed.padded_length, 32); // 32 is already mult of 16
        let unpacked = packer.unpack(&packed);
        for (i, seq) in unpacked.iter().enumerate() {
            assert_eq!(seq.len(), i + 1);
        }
    }

    // -----------------------------------------------------------------------
    // DynamicShapeEngine
    // -----------------------------------------------------------------------

    #[test]
    fn engine_bind_resolve() {
        let mut engine = DynamicShapeEngine::default_for_llm(2048, 768);
        engine.bind("B", 4);
        engine.bind("S", 128);
        let spec = ShapeSpec::new(vec![
            DynamicDim::named("B"),
            DynamicDim::named("S"),
            DynamicDim::fixed(768),
        ]);
        let dims = engine.resolve(&spec).unwrap();
        assert_eq!(dims, vec![4, 128, 768]);
    }

    #[test]
    fn engine_resolve_unresolved() {
        let engine = DynamicShapeEngine::default_for_llm(2048, 768);
        let spec = ShapeSpec::new(vec![DynamicDim::named("missing")]);
        assert!(engine.resolve(&spec).is_err());
    }

    #[test]
    fn engine_resolve_and_validate() {
        let mut engine = DynamicShapeEngine::default_for_llm(2048, 768);
        engine.bind("seq_len", 512);
        engine.bind("hidden_dim", 768);
        let spec = ShapeSpec::new(vec![
            DynamicDim::fixed(1),
            DynamicDim::named("seq_len"),
            DynamicDim::named("hidden_dim"),
        ]);
        let dims = engine.resolve_and_validate(&spec).unwrap();
        assert_eq!(dims, vec![1, 512, 768]);
    }

    #[test]
    fn engine_infer_matmul() {
        let engine = DynamicShapeEngine::default_for_llm(2048, 768);
        let out = engine.infer_and_validate("matmul", &[&[4, 768], &[768, 256]]).unwrap();
        assert_eq!(out, vec![4, 256]);
    }

    #[test]
    fn engine_allocate_release() {
        let mut engine = DynamicShapeEngine::default_for_llm(2048, 768);
        let bucket = engine.allocate(1000).unwrap();
        assert!(bucket >= 1000);
        assert!(engine.release(bucket));
    }

    #[test]
    fn engine_pack_unpack() {
        let engine = DynamicShapeEngine::default_for_llm(2048, 768);
        let s1 = [1.0, 2.0, 3.0];
        let s2 = [4.0, 5.0];
        let packed = engine.pack_sequences(&[&s1, &s2]).unwrap();
        let unpacked = engine.unpack_sequences(&packed);
        assert_eq!(unpacked[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(unpacked[1], vec![4.0, 5.0]);
    }

    #[test]
    fn engine_optimized_strides() {
        let engine = DynamicShapeEngine::default_for_llm(2048, 768);
        let strides = engine.optimized_strides(&[4, 128, 768]);
        assert_eq!(strides.len(), 3);
        assert_eq!(strides[2], 1);
    }

    #[test]
    fn engine_unbind() {
        let mut engine = DynamicShapeEngine::default_for_llm(2048, 768);
        engine.bind("x", 10);
        engine.unbind("x");
        assert!(!engine.bindings().contains_key("x"));
    }

    #[test]
    fn engine_accessors() {
        let engine = DynamicShapeEngine::default_for_llm(2048, 768);
        assert!(engine.bindings().is_empty());
        assert_eq!(engine.validator().max_rank(), 6);
        assert_eq!(engine.optimizer().alignment(), 64);
    }

    #[test]
    fn engine_validate_constraint_violation() {
        let mut engine = DynamicShapeEngine::default_for_llm(2048, 768);
        engine.bind("seq_len", 9999); // exceeds max 2048
        engine.bind("hidden_dim", 768);
        let spec = ShapeSpec::new(vec![DynamicDim::fixed(1), DynamicDim::named("seq_len")]);
        assert!(engine.resolve_and_validate(&spec).is_err());
    }

    // -----------------------------------------------------------------------
    // ShapeError
    // -----------------------------------------------------------------------

    #[test]
    fn error_display() {
        let e = ShapeError::UnresolvedDimension("x".into());
        assert!(e.to_string().contains("unresolved"));
    }

    #[test]
    fn error_broadcast_display() {
        let e = ShapeError::BroadcastIncompatible { dim_a: 3, dim_b: 4 };
        assert!(e.to_string().contains("broadcast"));
    }

    #[test]
    fn error_rank_mismatch_display() {
        let e = ShapeError::RankMismatch { expected: 2, got: 3 };
        assert!(e.to_string().contains("rank"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ShapeError::ValidationFailed("test".into()));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn error_bucket_display() {
        let e = ShapeError::BucketExhausted { requested: 100, max_bucket: 64 };
        assert!(e.to_string().contains("bucket"));
    }

    #[test]
    fn error_packing_display() {
        let e = ShapeError::PackingError("oops".into());
        assert!(e.to_string().contains("packing"));
    }

    #[test]
    fn error_padding_display() {
        let e = ShapeError::InvalidPadding("bad".into());
        assert!(e.to_string().contains("padding"));
    }

    #[test]
    fn error_constraint_display() {
        let e = ShapeError::ConstraintViolation("nope".into());
        assert!(e.to_string().contains("constraint"));
    }
}
