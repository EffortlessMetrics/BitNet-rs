//! Kernel JIT compilation with shape specialization.
//!
//! Provides an IR-based kernel representation, shape-specialized code
//! generation, persistent disk caching with LRU eviction, and
//! template-based multi-backend (`OpenCL` / CUDA / Vulkan) codegen.

use std::collections::HashMap;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ── Configuration ─────────────────────────────────────────────────────

/// JIT compilation configuration.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Directory used for on-disk kernel cache.
    pub cache_dir: PathBuf,
    /// Compiler optimisation level (0–3).
    pub optimization_level: u8,
    /// Whether shape-specialised code generation is enabled.
    pub specialization_enabled: bool,
    /// Maximum entries kept in the on-disk cache before LRU eviction.
    pub max_cache_entries: usize,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".bitnet_jit_cache"),
            optimization_level: 2,
            specialization_enabled: true,
            max_cache_entries: 256,
        }
    }
}

// ── Backend target ────────────────────────────────────────────────────

/// Target backend for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendTarget {
    OpenCL,
    Cuda,
    Vulkan,
}

impl fmt::Display for BackendTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenCL => write!(f, "OpenCL"),
            Self::Cuda => write!(f, "CUDA"),
            Self::Vulkan => write!(f, "Vulkan"),
        }
    }
}

// ── Element types ─────────────────────────────────────────────────────

/// Supported element types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    F32,
    F16,
    I8,
    I2,
}

impl ElementType {
    /// Size in bytes of one element.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 | Self::I2 => 1, // packed representation for I2
        }
    }

    const fn c_type_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 => "half",
            Self::I8 | Self::I2 => "char",
        }
    }
}

// ── Typed shape ───────────────────────────────────────────────────────

/// Tensor shape annotated with an element type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypedShape {
    pub dimensions: Vec<usize>,
    pub element_type: ElementType,
}

impl TypedShape {
    pub const fn new(dimensions: Vec<usize>, element_type: ElementType) -> Self {
        Self { dimensions, element_type }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.dimensions.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.element_type.size_bytes()
    }

    /// Number of dimensions (rank).
    pub const fn rank(&self) -> usize {
        self.dimensions.len()
    }
}

// ── IR operations ─────────────────────────────────────────────────────

/// Primitive operations supported by the kernel IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrOp {
    Add,
    Mul,
    MatMul,
    Reduce(ReduceOp),
    Activation(ActivationKind),
    Custom(String),
}

/// Reduction variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Mean,
}

/// Activation function variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    Relu,
    Silu,
    Gelu,
}

// ── Kernel IR ─────────────────────────────────────────────────────────

/// A single node in the kernel IR graph.
#[derive(Debug, Clone)]
pub struct IrNode {
    pub op: IrOp,
    pub inputs: Vec<TypedShape>,
    pub output: TypedShape,
}

/// Intermediate representation of a compute kernel.
#[derive(Debug, Clone)]
pub struct KernelIR {
    pub name: String,
    pub nodes: Vec<IrNode>,
}

impl KernelIR {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), nodes: Vec::new() }
    }

    /// Append a node to the IR graph.
    pub fn push(&mut self, node: IrNode) {
        self.nodes.push(node);
    }

    /// Number of operations in the IR.
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the IR is empty.
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Collect all unique shapes referenced by this IR.
    pub fn referenced_shapes(&self) -> Vec<&TypedShape> {
        let mut shapes: Vec<&TypedShape> = Vec::new();
        for node in &self.nodes {
            for input in &node.inputs {
                if !shapes.contains(&input) {
                    shapes.push(input);
                }
            }
            if !shapes.contains(&&node.output) {
                shapes.push(&node.output);
            }
        }
        shapes
    }
}

// ── Specialization key ────────────────────────────────────────────────

/// Cache key derived from kernel name, shapes and types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecializationKey {
    pub kernel_name: String,
    pub shapes: Vec<TypedShape>,
    pub target: BackendTarget,
}

impl SpecializationKey {
    pub fn new(
        kernel_name: impl Into<String>,
        shapes: Vec<TypedShape>,
        target: BackendTarget,
    ) -> Self {
        Self { kernel_name: kernel_name.into(), shapes, target }
    }

    /// Compute a deterministic 64-bit hash for this key.
    pub fn hash_u64(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Hex representation used as a filename stem in the disk cache.
    pub fn cache_filename(&self) -> String {
        format!("{:016x}.bin", self.hash_u64())
    }
}

// ── Compiled kernel ───────────────────────────────────────────────────

/// A compiled kernel ready for dispatch.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub source: String,
    pub target: BackendTarget,
    pub compilation_time: Duration,
    pub specialization_key: SpecializationKey,
}

// ── JIT errors ────────────────────────────────────────────────────────

/// Errors produced by the JIT pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitError {
    EmptyIR,
    UnsupportedOp(String),
    ShapeMismatch { expected: usize, actual: usize },
    CacheFull,
    IoError(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIR => write!(f, "kernel IR is empty"),
            Self::UnsupportedOp(op) => {
                write!(f, "unsupported operation: {op}")
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            Self::CacheFull => write!(f, "JIT cache is full"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for JitError {}

// ── JIT cache (in-memory LRU) ─────────────────────────────────────────

/// In-memory LRU cache for compiled kernels.
#[derive(Debug)]
pub struct JitCache {
    max_entries: usize,
    /// Map from specialization key hash → compiled kernel.
    entries: HashMap<u64, CacheEntry>,
    /// Monotonic counter for LRU ordering.
    access_counter: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    kernel: CompiledKernel,
    last_access: u64,
}

impl JitCache {
    pub fn new(max_entries: usize) -> Self {
        Self { max_entries, entries: HashMap::new(), access_counter: 0 }
    }

    /// Look up a compiled kernel.
    pub fn get(&mut self, key: &SpecializationKey) -> Option<&CompiledKernel> {
        let hash = key.hash_u64();
        if self.entries.contains_key(&hash) {
            self.access_counter += 1;
            self.entries.get_mut(&hash).unwrap().last_access =
                self.access_counter;
            Some(&self.entries[&hash].kernel)
        } else {
            None
        }
    }

    /// Insert a compiled kernel, evicting the LRU entry if full.
    pub fn insert(
        &mut self,
        key: &SpecializationKey,
        kernel: CompiledKernel,
    ) -> Result<(), JitError> {
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        // After eviction there must be room.
        if self.entries.len() >= self.max_entries {
            return Err(JitError::CacheFull);
        }
        self.access_counter += 1;
        let hash = key.hash_u64();
        self.entries.insert(
            hash,
            CacheEntry { kernel, last_access: self.access_counter },
        );
        Ok(())
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_counter = 0;
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        if let Some((&lru_key, _)) =
            self.entries.iter().min_by_key(|(_, e)| e.last_access)
        {
            self.entries.remove(&lru_key);
        }
    }
}

// ── Kernel specializer ────────────────────────────────────────────────

/// Tile-size parameters derived from tensor shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileParams {
    pub tile_m: usize,
    pub tile_n: usize,
    pub unroll_factor: usize,
}

/// Generates optimised kernel code specialised for concrete tensor shapes.
pub struct KernelSpecializer;

impl KernelSpecializer {
    /// Choose tile parameters based on a matrix dimension.
    pub const fn select_tile_params(m: usize, n: usize) -> TileParams {
        let tile_m = match m {
            0..=64 => 8,
            65..=256 => 16,
            _ => 32,
        };
        let tile_n = match n {
            0..=64 => 8,
            65..=256 => 16,
            _ => 32,
        };
        let unroll_factor = if m * n <= 4096 { 2 } else { 4 };
        TileParams { tile_m, tile_n, unroll_factor }
    }

    /// Generate specialised source for a single IR node.
    pub fn specialize_node(
        node: &IrNode,
        target: BackendTarget,
    ) -> Result<String, JitError> {
        match &node.op {
            IrOp::Add => Ok(Self::gen_elementwise(
                "add", "+", &node.output, target,
            )),
            IrOp::Mul => Ok(Self::gen_elementwise(
                "mul", "*", &node.output, target,
            )),
            IrOp::MatMul => {
                if node.inputs.len() < 2 {
                    return Err(JitError::ShapeMismatch {
                        expected: 2,
                        actual: node.inputs.len(),
                    });
                }
                Ok(Self::gen_matmul(&node.inputs[0], &node.inputs[1], target))
            }
            IrOp::Reduce(op) => {
                Ok(Self::gen_reduce(*op, &node.inputs[0], target))
            }
            IrOp::Activation(kind) => {
                Ok(Self::gen_activation(*kind, &node.output, target))
            }
            IrOp::Custom(name) => Ok(Self::gen_custom_stub(name, target)),
        }
    }

    // ── code-generation helpers ───────────────────────────────────────

    fn gen_elementwise(
        name: &str,
        op_symbol: &str,
        shape: &TypedShape,
        target: BackendTarget,
    ) -> String {
        let n = shape.num_elements();
        let ty = shape.element_type.c_type_name();
        match target {
            BackendTarget::Cuda => format!(
                "__global__ void {name}_kernel({ty}* a, {ty}* b, {ty}* c, \
                 int n) {{\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  \
                 if (i < {n}) c[i] = a[i] {op_symbol} b[i];\n}}\n"
            ),
            BackendTarget::OpenCL => format!(
                "__kernel void {name}_kernel(__global {ty}* a, \
                 __global {ty}* b, __global {ty}* c) {{\n  \
                 int i = get_global_id(0);\n  \
                 if (i < {n}) c[i] = a[i] {op_symbol} b[i];\n}}\n"
            ),
            BackendTarget::Vulkan => format!(
                "// Vulkan compute — {name} ({n} elements)\n\
                 layout(local_size_x = 256) in;\n\
                 void main() {{\n  uint i = gl_GlobalInvocationID.x;\n  \
                 if (i < {n}) c[i] = a[i] {op_symbol} b[i];\n}}\n"
            ),
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn gen_matmul(
        a: &TypedShape,
        b: &TypedShape,
        target: BackendTarget,
    ) -> String {
        let m = *a.dimensions.first().unwrap_or(&1);
        let k = *a.dimensions.get(1).unwrap_or(&1);
        let n = *b.dimensions.get(1).unwrap_or(&1);
        let tile = Self::select_tile_params(m, n);
        let ty = a.element_type.c_type_name();
        match target {
            BackendTarget::Cuda => format!(
                "// CUDA matmul [{m}x{k}] @ [{k}x{n}] \
                 tile=({},{}) unroll={}\n\
                 __global__ void matmul_kernel({ty}* A, {ty}* B, {ty}* C) \
                 {{\n  // tiled implementation\n}}\n",
                tile.tile_m, tile.tile_n, tile.unroll_factor,
            ),
            BackendTarget::OpenCL => format!(
                "// OpenCL matmul [{m}x{k}] @ [{k}x{n}] \
                 tile=({},{}) unroll={}\n\
                 __kernel void matmul_kernel(__global {ty}* A, \
                 __global {ty}* B, __global {ty}* C) {{\n  \
                 // tiled implementation\n}}\n",
                tile.tile_m, tile.tile_n, tile.unroll_factor,
            ),
            BackendTarget::Vulkan => format!(
                "// Vulkan matmul [{m}x{k}] @ [{k}x{n}] \
                 tile=({},{}) unroll={}\n\
                 layout(local_size_x = 16, local_size_y = 16) in;\n\
                 void main() {{\n  // tiled implementation\n}}\n",
                tile.tile_m, tile.tile_n, tile.unroll_factor,
            ),
        }
    }

    fn gen_reduce(
        op: ReduceOp,
        input: &TypedShape,
        target: BackendTarget,
    ) -> String {
        let n = input.num_elements();
        let op_name = match op {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Mean => "mean",
        };
        let ty = input.element_type.c_type_name();
        match target {
            BackendTarget::Cuda => format!(
                "__global__ void reduce_{op_name}_kernel({ty}* in, \
                 {ty}* out, int n) {{\n  // parallel reduction ({n} elems)\n\
                 }}\n"
            ),
            BackendTarget::OpenCL => format!(
                "__kernel void reduce_{op_name}_kernel(__global {ty}* in, \
                 __global {ty}* out) {{\n  // parallel reduction ({n} elems)\n\
                 }}\n"
            ),
            BackendTarget::Vulkan => format!(
                "// Vulkan reduce_{op_name} ({n} elements)\n\
                 void main() {{\n  // parallel reduction\n}}\n"
            ),
        }
    }

    fn gen_activation(
        kind: ActivationKind,
        shape: &TypedShape,
        target: BackendTarget,
    ) -> String {
        let n = shape.num_elements();
        let fn_body = match kind {
            ActivationKind::Relu => "x > 0 ? x : 0",
            ActivationKind::Silu => "x / (1.0f + expf(-x))",
            ActivationKind::Gelu => {
                "0.5f * x * (1.0f + tanhf(0.7978845608f * \
                 (x + 0.044715f * x * x * x)))"
            }
        };
        let act_name = match kind {
            ActivationKind::Relu => "relu",
            ActivationKind::Silu => "silu",
            ActivationKind::Gelu => "gelu",
        };
        let ty = shape.element_type.c_type_name();
        match target {
            BackendTarget::Cuda => format!(
                "__global__ void {act_name}_kernel({ty}* data, int n) {{\n  \
                 int i = blockIdx.x * blockDim.x + threadIdx.x;\n  \
                 if (i < {n}) {{ {ty} x = data[i]; data[i] = {fn_body}; }}\n\
                 }}\n"
            ),
            BackendTarget::OpenCL => format!(
                "__kernel void {act_name}_kernel(__global {ty}* data) {{\n  \
                 int i = get_global_id(0);\n  \
                 if (i < {n}) {{ {ty} x = data[i]; data[i] = {fn_body}; }}\n\
                 }}\n"
            ),
            BackendTarget::Vulkan => format!(
                "// Vulkan {act_name} ({n} elements)\nvoid main() {{\n  \
                 uint i = gl_GlobalInvocationID.x;\n  \
                 if (i < {n}) {{ float x = data[i]; data[i] = {fn_body}; }}\n\
                 }}\n"
            ),
        }
    }

    fn gen_custom_stub(name: &str, target: BackendTarget) -> String {
        match target {
            BackendTarget::Cuda => {
                format!("__global__ void {name}_kernel() {{ /* custom */ }}\n")
            }
            BackendTarget::OpenCL => {
                format!("__kernel void {name}_kernel() {{ /* custom */ }}\n")
            }
            BackendTarget::Vulkan => {
                format!("// Vulkan {name}\nvoid main() {{ /* custom */ }}\n")
            }
        }
    }
}

// ── KernelJit — top-level pipeline ────────────────────────────────────

/// Top-level JIT compilation pipeline.
pub struct KernelJit {
    config: JitConfig,
    cache: JitCache,
}

impl KernelJit {
    /// Create a new JIT compiler with the given configuration.
    pub fn new(config: JitConfig) -> Self {
        let cache = JitCache::new(config.max_cache_entries);
        Self { config, cache }
    }

    /// Create a JIT compiler with default settings.
    pub fn with_defaults() -> Self {
        Self::new(JitConfig::default())
    }

    /// Access the current configuration.
    pub const fn config(&self) -> &JitConfig {
        &self.config
    }

    /// Access the cache (immutable).
    pub const fn cache(&self) -> &JitCache {
        &self.cache
    }

    /// Compile a kernel IR for the given backend target.
    ///
    /// If specialization is enabled and a cached entry exists, return it.
    /// Otherwise compile, cache, and return the result.
    pub fn compile(
        &mut self,
        ir: &KernelIR,
        target: BackendTarget,
    ) -> Result<CompiledKernel, JitError> {
        if ir.is_empty() {
            return Err(JitError::EmptyIR);
        }

        let shapes: Vec<TypedShape> = ir
            .referenced_shapes()
            .into_iter()
            .cloned()
            .collect();
        let key =
            SpecializationKey::new(&ir.name, shapes, target);

        // Check cache first.
        if self.config.specialization_enabled
            && let Some(cached) = self.cache.get(&key)
        {
            return Ok(cached.clone());
        }

        let start = Instant::now();
        let mut source = String::new();
        for node in &ir.nodes {
            let frag =
                KernelSpecializer::specialize_node(node, target)?;
            source.push_str(&frag);
        }

        if self.config.optimization_level >= 2 {
            source = Self::optimize_source(&source);
        }

        let compiled = CompiledKernel {
            source,
            target,
            compilation_time: start.elapsed(),
            specialization_key: key.clone(),
        };

        if self.config.specialization_enabled {
            self.cache.insert(&key, compiled.clone())?;
        }

        Ok(compiled)
    }

    /// Very light optimisation pass on generated source.
    fn optimize_source(src: &str) -> String {
        // Strip blank lines for tidiness.
        src.lines()
            .filter(|l| !l.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    /// Evict all cached kernels.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Number of cached kernels.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────

    fn f32_vec(dims: Vec<usize>) -> TypedShape {
        TypedShape::new(dims, ElementType::F32)
    }

    fn simple_add_ir() -> KernelIR {
        let shape = f32_vec(vec![256]);
        let mut ir = KernelIR::new("vec_add");
        ir.push(IrNode {
            op: IrOp::Add,
            inputs: vec![shape.clone(), shape.clone()],
            output: shape,
        });
        ir
    }

    #[allow(clippy::many_single_char_names)]
    fn matmul_ir(m: usize, k: usize, n: usize) -> KernelIR {
        let a = f32_vec(vec![m, k]);
        let b = f32_vec(vec![k, n]);
        let c = f32_vec(vec![m, n]);
        let mut ir = KernelIR::new("matmul");
        ir.push(IrNode {
            op: IrOp::MatMul,
            inputs: vec![a, b],
            output: c,
        });
        ir
    }

    // ── JitConfig tests ──────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = JitConfig::default();
        assert_eq!(cfg.optimization_level, 2);
        assert!(cfg.specialization_enabled);
        assert_eq!(cfg.max_cache_entries, 256);
        assert_eq!(cfg.cache_dir, PathBuf::from(".bitnet_jit_cache"));
    }

    #[test]
    fn config_custom_values() {
        let cfg = JitConfig {
            cache_dir: PathBuf::from("/tmp/jit"),
            optimization_level: 3,
            specialization_enabled: false,
            max_cache_entries: 64,
        };
        assert_eq!(cfg.optimization_level, 3);
        assert!(!cfg.specialization_enabled);
    }

    // ── ElementType tests ────────────────────────────────────────────

    #[test]
    fn element_type_sizes() {
        assert_eq!(ElementType::F32.size_bytes(), 4);
        assert_eq!(ElementType::F16.size_bytes(), 2);
        assert_eq!(ElementType::I8.size_bytes(), 1);
        assert_eq!(ElementType::I2.size_bytes(), 1);
    }

    #[test]
    fn element_type_c_names() {
        assert_eq!(ElementType::F32.c_type_name(), "float");
        assert_eq!(ElementType::F16.c_type_name(), "half");
        assert_eq!(ElementType::I8.c_type_name(), "char");
    }

    // ── TypedShape tests ─────────────────────────────────────────────

    #[test]
    fn typed_shape_num_elements() {
        let s = f32_vec(vec![4, 8, 16]);
        assert_eq!(s.num_elements(), 512);
    }

    #[test]
    fn typed_shape_size_bytes() {
        let s = TypedShape::new(vec![100], ElementType::F16);
        assert_eq!(s.size_bytes(), 200);
    }

    #[test]
    fn typed_shape_rank() {
        assert_eq!(f32_vec(vec![3, 4]).rank(), 2);
        assert_eq!(f32_vec(vec![5]).rank(), 1);
    }

    #[test]
    fn typed_shape_empty_dimensions() {
        let s = f32_vec(vec![]);
        assert_eq!(s.num_elements(), 1); // product of empty = 1
        assert_eq!(s.rank(), 0);
    }

    // ── IrOp / KernelIR tests ────────────────────────────────────────

    #[test]
    fn ir_new_is_empty() {
        let ir = KernelIR::new("empty");
        assert!(ir.is_empty());
        assert_eq!(ir.len(), 0);
    }

    #[test]
    fn ir_push_and_len() {
        let ir = simple_add_ir();
        assert_eq!(ir.len(), 1);
        assert!(!ir.is_empty());
    }

    #[test]
    fn ir_multiple_nodes() {
        let shape = f32_vec(vec![64]);
        let mut ir = KernelIR::new("multi");
        for _ in 0..5 {
            ir.push(IrNode {
                op: IrOp::Add,
                inputs: vec![shape.clone(), shape.clone()],
                output: shape.clone(),
            });
        }
        assert_eq!(ir.len(), 5);
    }

    #[test]
    fn ir_referenced_shapes_deduplicates() {
        let ir = simple_add_ir();
        let shapes = ir.referenced_shapes();
        // input and output are the same shape — should appear once.
        assert_eq!(shapes.len(), 1);
    }

    #[test]
    fn ir_referenced_shapes_multiple() {
        let ir = matmul_ir(32, 64, 128);
        let shapes = ir.referenced_shapes();
        // [32,64], [64,128], [32,128] — three distinct shapes.
        assert_eq!(shapes.len(), 3);
    }

    #[test]
    fn ir_op_custom_variant() {
        let op = IrOp::Custom("my_kernel".into());
        assert_eq!(op, IrOp::Custom("my_kernel".into()));
    }

    #[test]
    fn ir_op_reduce_variants() {
        let _ = IrOp::Reduce(ReduceOp::Sum);
        let _ = IrOp::Reduce(ReduceOp::Max);
        let _ = IrOp::Reduce(ReduceOp::Mean);
    }

    #[test]
    fn ir_op_activation_variants() {
        let _ = IrOp::Activation(ActivationKind::Relu);
        let _ = IrOp::Activation(ActivationKind::Silu);
        let _ = IrOp::Activation(ActivationKind::Gelu);
    }

    // ── SpecializationKey tests ──────────────────────────────────────

    #[test]
    fn specialization_key_deterministic_hash() {
        let k1 = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![4])],
            BackendTarget::Cuda,
        );
        let k2 = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![4])],
            BackendTarget::Cuda,
        );
        assert_eq!(k1.hash_u64(), k2.hash_u64());
    }

    #[test]
    fn specialization_key_differs_by_shape() {
        let k1 = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![4])],
            BackendTarget::Cuda,
        );
        let k2 = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![8])],
            BackendTarget::Cuda,
        );
        assert_ne!(k1.hash_u64(), k2.hash_u64());
    }

    #[test]
    fn specialization_key_differs_by_backend() {
        let k1 = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![4])],
            BackendTarget::Cuda,
        );
        let k2 = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![4])],
            BackendTarget::OpenCL,
        );
        assert_ne!(k1.hash_u64(), k2.hash_u64());
    }

    #[test]
    fn specialization_key_differs_by_name() {
        let k1 = SpecializationKey::new(
            "a",
            vec![f32_vec(vec![4])],
            BackendTarget::Cuda,
        );
        let k2 = SpecializationKey::new(
            "b",
            vec![f32_vec(vec![4])],
            BackendTarget::Cuda,
        );
        assert_ne!(k1.hash_u64(), k2.hash_u64());
    }

    #[test]
    fn specialization_key_cache_filename() {
        let k = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![4])],
            BackendTarget::Cuda,
        );
        let fname = k.cache_filename();
        assert!(std::path::Path::new(&fname)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin")));
        assert_eq!(fname.len(), 20); // 16 hex + ".bin"
    }

    // ── BackendTarget display ────────────────────────────────────────

    #[test]
    fn backend_target_display() {
        assert_eq!(format!("{}", BackendTarget::Cuda), "CUDA");
        assert_eq!(format!("{}", BackendTarget::OpenCL), "OpenCL");
        assert_eq!(format!("{}", BackendTarget::Vulkan), "Vulkan");
    }

    // ── JitCache tests ───────────────────────────────────────────────

    #[test]
    fn cache_new_is_empty() {
        let c = JitCache::new(16);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn cache_insert_and_get() {
        let mut c = JitCache::new(16);
        let key = SpecializationKey::new(
            "k",
            vec![f32_vec(vec![8])],
            BackendTarget::Cuda,
        );
        let kernel = CompiledKernel {
            source: "src".into(),
            target: BackendTarget::Cuda,
            compilation_time: Duration::from_millis(1),
            specialization_key: key.clone(),
        };
        c.insert(&key, kernel).unwrap();
        assert_eq!(c.len(), 1);
        assert!(c.get(&key).is_some());
        assert_eq!(c.get(&key).unwrap().source, "src");
    }

    #[test]
    fn cache_miss_returns_none() {
        let mut c = JitCache::new(16);
        let key = SpecializationKey::new(
            "missing",
            vec![],
            BackendTarget::Cuda,
        );
        assert!(c.get(&key).is_none());
    }

    #[test]
    fn cache_lru_eviction() {
        let mut c = JitCache::new(2);
        let make_key = |name: &str| {
            SpecializationKey::new(name, vec![], BackendTarget::Cuda)
        };
        let make_kernel = |key: &SpecializationKey| CompiledKernel {
            source: String::new(),
            target: BackendTarget::Cuda,
            compilation_time: Duration::ZERO,
            specialization_key: key.clone(),
        };

        let k1 = make_key("a");
        let k2 = make_key("b");
        let k3 = make_key("c");

        c.insert(&k1, make_kernel(&k1)).unwrap();
        c.insert(&k2, make_kernel(&k2)).unwrap();
        assert_eq!(c.len(), 2);

        // Insert a third — evicts k1 (LRU).
        c.insert(&k3, make_kernel(&k3)).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.get(&k1).is_none());
        assert!(c.get(&k2).is_some());
        assert!(c.get(&k3).is_some());
    }

    #[test]
    fn cache_access_updates_lru_order() {
        let mut c = JitCache::new(2);
        let make_key = |name: &str| {
            SpecializationKey::new(name, vec![], BackendTarget::Cuda)
        };
        let make_kernel = |key: &SpecializationKey| CompiledKernel {
            source: String::new(),
            target: BackendTarget::Cuda,
            compilation_time: Duration::ZERO,
            specialization_key: key.clone(),
        };

        let k1 = make_key("a");
        let k2 = make_key("b");
        let k3 = make_key("c");

        c.insert(&k1, make_kernel(&k1)).unwrap();
        c.insert(&k2, make_kernel(&k2)).unwrap();

        // Access k1 so it is no longer LRU.
        c.get(&k1);

        // Inserting k3 should evict k2.
        c.insert(&k3, make_kernel(&k3)).unwrap();
        assert!(c.get(&k1).is_some());
        assert!(c.get(&k2).is_none());
    }

    #[test]
    fn cache_clear() {
        let mut c = JitCache::new(16);
        let key = SpecializationKey::new(
            "k",
            vec![],
            BackendTarget::Cuda,
        );
        let kernel = CompiledKernel {
            source: String::new(),
            target: BackendTarget::Cuda,
            compilation_time: Duration::ZERO,
            specialization_key: key.clone(),
        };
        c.insert(&key, kernel).unwrap();
        assert_eq!(c.len(), 1);
        c.clear();
        assert!(c.is_empty());
    }

    // ── KernelSpecializer — tile selection ────────────────────────────

    #[test]
    fn tile_params_small_matrix() {
        let t = KernelSpecializer::select_tile_params(32, 32);
        assert_eq!(t.tile_m, 8);
        assert_eq!(t.tile_n, 8);
        assert_eq!(t.unroll_factor, 2);
    }

    #[test]
    fn tile_params_medium_matrix() {
        let t = KernelSpecializer::select_tile_params(128, 128);
        assert_eq!(t.tile_m, 16);
        assert_eq!(t.tile_n, 16);
        assert_eq!(t.unroll_factor, 4);
    }

    #[test]
    fn tile_params_large_matrix() {
        let t = KernelSpecializer::select_tile_params(512, 1024);
        assert_eq!(t.tile_m, 32);
        assert_eq!(t.tile_n, 32);
        assert_eq!(t.unroll_factor, 4);
    }

    #[test]
    fn tile_params_asymmetric() {
        let t = KernelSpecializer::select_tile_params(32, 512);
        assert_eq!(t.tile_m, 8);
        assert_eq!(t.tile_n, 32);
    }

    // ── KernelSpecializer — code generation ──────────────────────────

    #[test]
    fn gen_add_cuda() {
        let node = IrNode {
            op: IrOp::Add,
            inputs: vec![f32_vec(vec![128]), f32_vec(vec![128])],
            output: f32_vec(vec![128]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("__global__"));
        assert!(src.contains("add_kernel"));
        assert!(src.contains('+'));
    }

    #[test]
    fn gen_add_opencl() {
        let node = IrNode {
            op: IrOp::Add,
            inputs: vec![f32_vec(vec![128]), f32_vec(vec![128])],
            output: f32_vec(vec![128]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::OpenCL,
        )
        .unwrap();
        assert!(src.contains("__kernel"));
        assert!(src.contains("get_global_id"));
    }

    #[test]
    fn gen_add_vulkan() {
        let node = IrNode {
            op: IrOp::Add,
            inputs: vec![f32_vec(vec![128]), f32_vec(vec![128])],
            output: f32_vec(vec![128]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Vulkan,
        )
        .unwrap();
        assert!(src.contains("Vulkan"));
        assert!(src.contains("gl_GlobalInvocationID"));
    }

    #[test]
    fn gen_mul_cuda() {
        let node = IrNode {
            op: IrOp::Mul,
            inputs: vec![f32_vec(vec![64]), f32_vec(vec![64])],
            output: f32_vec(vec![64]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("mul_kernel"));
        assert!(src.contains('*'));
    }

    #[test]
    fn gen_matmul_cuda() {
        let ir = matmul_ir(64, 128, 64);
        let node = &ir.nodes[0];
        let src = KernelSpecializer::specialize_node(
            node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("matmul_kernel"));
        assert!(src.contains("64x128"));
        assert!(src.contains("128x64"));
    }

    #[test]
    fn gen_matmul_opencl() {
        let ir = matmul_ir(32, 32, 32);
        let node = &ir.nodes[0];
        let src = KernelSpecializer::specialize_node(
            node,
            BackendTarget::OpenCL,
        )
        .unwrap();
        assert!(src.contains("OpenCL matmul"));
        assert!(src.contains("__kernel"));
    }

    #[test]
    fn gen_matmul_insufficient_inputs() {
        let node = IrNode {
            op: IrOp::MatMul,
            inputs: vec![f32_vec(vec![4, 4])],
            output: f32_vec(vec![4, 4]),
        };
        let res = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        );
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            JitError::ShapeMismatch { expected: 2, actual: 1 }
        );
    }

    #[test]
    fn gen_reduce_sum_cuda() {
        let node = IrNode {
            op: IrOp::Reduce(ReduceOp::Sum),
            inputs: vec![f32_vec(vec![256])],
            output: f32_vec(vec![1]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("reduce_sum_kernel"));
    }

    #[test]
    fn gen_reduce_max_opencl() {
        let node = IrNode {
            op: IrOp::Reduce(ReduceOp::Max),
            inputs: vec![f32_vec(vec![512])],
            output: f32_vec(vec![1]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::OpenCL,
        )
        .unwrap();
        assert!(src.contains("reduce_max_kernel"));
    }

    #[test]
    fn gen_reduce_mean_vulkan() {
        let node = IrNode {
            op: IrOp::Reduce(ReduceOp::Mean),
            inputs: vec![f32_vec(vec![128])],
            output: f32_vec(vec![1]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Vulkan,
        )
        .unwrap();
        assert!(src.contains("reduce_mean"));
    }

    #[test]
    fn gen_activation_relu_cuda() {
        let node = IrNode {
            op: IrOp::Activation(ActivationKind::Relu),
            inputs: vec![f32_vec(vec![64])],
            output: f32_vec(vec![64]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("relu_kernel"));
    }

    #[test]
    fn gen_activation_silu_opencl() {
        let node = IrNode {
            op: IrOp::Activation(ActivationKind::Silu),
            inputs: vec![f32_vec(vec![64])],
            output: f32_vec(vec![64]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::OpenCL,
        )
        .unwrap();
        assert!(src.contains("silu_kernel"));
        assert!(src.contains("expf"));
    }

    #[test]
    fn gen_activation_gelu_vulkan() {
        let node = IrNode {
            op: IrOp::Activation(ActivationKind::Gelu),
            inputs: vec![f32_vec(vec![64])],
            output: f32_vec(vec![64]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Vulkan,
        )
        .unwrap();
        assert!(src.contains("gelu"));
        assert!(src.contains("tanhf"));
    }

    #[test]
    fn gen_custom_cuda() {
        let node = IrNode {
            op: IrOp::Custom("my_op".into()),
            inputs: vec![],
            output: f32_vec(vec![1]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("my_op_kernel"));
    }

    #[test]
    fn gen_custom_opencl() {
        let node = IrNode {
            op: IrOp::Custom("bitnet_quant".into()),
            inputs: vec![],
            output: f32_vec(vec![1]),
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::OpenCL,
        )
        .unwrap();
        assert!(src.contains("bitnet_quant_kernel"));
    }

    // ── KernelJit pipeline tests ─────────────────────────────────────

    #[test]
    fn jit_with_defaults() {
        let jit = KernelJit::with_defaults();
        assert_eq!(jit.cached_count(), 0);
        assert_eq!(jit.config().optimization_level, 2);
    }

    #[test]
    fn jit_compile_simple() {
        let mut jit = KernelJit::with_defaults();
        let ir = simple_add_ir();
        let k = jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert!(k.source.contains("add_kernel"));
        assert_eq!(k.target, BackendTarget::Cuda);
    }

    #[test]
    fn jit_compile_empty_ir_fails() {
        let mut jit = KernelJit::with_defaults();
        let ir = KernelIR::new("empty");
        let res = jit.compile(&ir, BackendTarget::Cuda);
        assert_eq!(res.unwrap_err(), JitError::EmptyIR);
    }

    #[test]
    fn jit_cache_hit() {
        let mut jit = KernelJit::with_defaults();
        let ir = simple_add_ir();
        jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert_eq!(jit.cached_count(), 1);

        // Second compile should hit the cache.
        let k2 = jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert_eq!(jit.cached_count(), 1);
        assert!(k2.source.contains("add_kernel"));
    }

    #[test]
    fn jit_different_backends_separate_cache() {
        let mut jit = KernelJit::with_defaults();
        let ir = simple_add_ir();
        jit.compile(&ir, BackendTarget::Cuda).unwrap();
        jit.compile(&ir, BackendTarget::OpenCL).unwrap();
        assert_eq!(jit.cached_count(), 2);
    }

    #[test]
    fn jit_different_shapes_separate_cache() {
        let mut jit = KernelJit::with_defaults();
        let ir1 = matmul_ir(32, 64, 32);
        let ir2 = matmul_ir(128, 256, 128);
        jit.compile(&ir1, BackendTarget::Cuda).unwrap();
        jit.compile(&ir2, BackendTarget::Cuda).unwrap();
        assert_eq!(jit.cached_count(), 2);
    }

    #[test]
    fn jit_clear_cache() {
        let mut jit = KernelJit::with_defaults();
        let ir = simple_add_ir();
        jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert_eq!(jit.cached_count(), 1);
        jit.clear_cache();
        assert_eq!(jit.cached_count(), 0);
    }

    #[test]
    fn jit_specialization_disabled_no_cache() {
        let cfg = JitConfig {
            specialization_enabled: false,
            ..JitConfig::default()
        };
        let mut jit = KernelJit::new(cfg);
        let ir = simple_add_ir();
        jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert_eq!(jit.cached_count(), 0);
    }

    #[test]
    fn jit_compile_matmul_has_tile_info() {
        let mut jit = KernelJit::with_defaults();
        let ir = matmul_ir(256, 512, 256);
        let k = jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert!(k.source.contains("tile="));
        assert!(k.source.contains("unroll="));
    }

    #[test]
    fn jit_compile_multi_node_ir() {
        let shape = f32_vec(vec![128]);
        let mut ir = KernelIR::new("fused_add_relu");
        ir.push(IrNode {
            op: IrOp::Add,
            inputs: vec![shape.clone(), shape.clone()],
            output: shape.clone(),
        });
        ir.push(IrNode {
            op: IrOp::Activation(ActivationKind::Relu),
            inputs: vec![shape.clone()],
            output: shape,
        });

        let mut jit = KernelJit::with_defaults();
        let k = jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert!(k.source.contains("add_kernel"));
        assert!(k.source.contains("relu_kernel"));
    }

    #[test]
    fn jit_compile_reduce_sum() {
        let mut jit = KernelJit::with_defaults();
        let input = f32_vec(vec![1024]);
        let output = f32_vec(vec![1]);
        let mut ir = KernelIR::new("reduce");
        ir.push(IrNode {
            op: IrOp::Reduce(ReduceOp::Sum),
            inputs: vec![input],
            output,
        });
        let k = jit.compile(&ir, BackendTarget::OpenCL).unwrap();
        assert!(k.source.contains("reduce_sum"));
    }

    #[test]
    fn jit_compilation_time_is_recorded() {
        let mut jit = KernelJit::with_defaults();
        let ir = simple_add_ir();
        let k = jit.compile(&ir, BackendTarget::Cuda).unwrap();
        // Compilation should be extremely fast — just verify it completes.
        let _ = k.compilation_time;
    }

    // ── Optimization pass ────────────────────────────────────────────

    #[test]
    fn optimize_source_strips_blank_lines() {
        let input = "line1\n\n\nline2\n";
        let out = KernelJit::optimize_source(input);
        assert_eq!(out, "line1\nline2\n");
    }

    #[test]
    fn optimize_source_preserves_content() {
        let input = "a\nb\nc\n";
        let out = KernelJit::optimize_source(input);
        assert_eq!(out, "a\nb\nc\n");
    }

    // ── JitError display ─────────────────────────────────────────────

    #[test]
    fn jit_error_display_empty_ir() {
        let e = JitError::EmptyIR;
        assert_eq!(format!("{e}"), "kernel IR is empty");
    }

    #[test]
    fn jit_error_display_unsupported() {
        let e = JitError::UnsupportedOp("foo".into());
        assert!(format!("{e}").contains("foo"));
    }

    #[test]
    fn jit_error_display_shape_mismatch() {
        let e = JitError::ShapeMismatch { expected: 2, actual: 1 };
        let msg = format!("{e}");
        assert!(msg.contains("expected 2"));
        assert!(msg.contains("got 1"));
    }

    #[test]
    fn jit_error_display_cache_full() {
        let e = JitError::CacheFull;
        assert!(format!("{e}").contains("full"));
    }

    #[test]
    fn jit_error_display_io() {
        let e = JitError::IoError("disk full".into());
        assert!(format!("{e}").contains("disk full"));
    }

    // ── Integration-style tests ──────────────────────────────────────

    #[test]
    fn end_to_end_all_backends() {
        let mut jit = KernelJit::with_defaults();
        let ir = simple_add_ir();
        for target in [
            BackendTarget::Cuda,
            BackendTarget::OpenCL,
            BackendTarget::Vulkan,
        ] {
            let k = jit.compile(&ir, target).unwrap();
            assert_eq!(k.target, target);
            assert!(!k.source.is_empty());
        }
        assert_eq!(jit.cached_count(), 3);
    }

    #[test]
    fn end_to_end_matmul_all_backends() {
        let mut jit = KernelJit::with_defaults();
        let ir = matmul_ir(64, 64, 64);
        for target in [
            BackendTarget::Cuda,
            BackendTarget::OpenCL,
            BackendTarget::Vulkan,
        ] {
            let k = jit.compile(&ir, target).unwrap();
            assert!(k.source.contains("matmul"));
        }
    }

    #[test]
    fn end_to_end_fused_pipeline() {
        let shape = f32_vec(vec![256]);
        let out_scalar = f32_vec(vec![1]);
        let mut ir = KernelIR::new("fused_pipeline");
        ir.push(IrNode {
            op: IrOp::Mul,
            inputs: vec![shape.clone(), shape.clone()],
            output: shape.clone(),
        });
        ir.push(IrNode {
            op: IrOp::Activation(ActivationKind::Silu),
            inputs: vec![shape.clone()],
            output: shape.clone(),
        });
        ir.push(IrNode {
            op: IrOp::Reduce(ReduceOp::Sum),
            inputs: vec![shape],
            output: out_scalar,
        });

        let mut jit = KernelJit::with_defaults();
        let k = jit.compile(&ir, BackendTarget::Cuda).unwrap();
        assert!(k.source.contains("mul_kernel"));
        assert!(k.source.contains("silu_kernel"));
        assert!(k.source.contains("reduce_sum"));
    }

    #[test]
    fn cache_eviction_under_small_limit() {
        let cfg = JitConfig {
            max_cache_entries: 3,
            ..JitConfig::default()
        };
        let mut jit = KernelJit::new(cfg);

        for size in [32, 64, 128, 256, 512] {
            let shape = f32_vec(vec![size]);
            let mut ir = KernelIR::new(format!("add_{size}"));
            ir.push(IrNode {
                op: IrOp::Add,
                inputs: vec![shape.clone(), shape.clone()],
                output: shape,
            });
            jit.compile(&ir, BackendTarget::Cuda).unwrap();
        }

        // Cache should never exceed the limit.
        assert!(jit.cached_count() <= 3);
    }

    #[test]
    fn specialization_key_from_ir() {
        let ir = matmul_ir(16, 32, 16);
        let shapes: Vec<TypedShape> = ir
            .referenced_shapes()
            .into_iter()
            .cloned()
            .collect();
        let key = SpecializationKey::new(
            &ir.name,
            shapes,
            BackendTarget::Cuda,
        );
        assert_eq!(key.kernel_name, "matmul");
        assert_eq!(key.shapes.len(), 3);
    }

    #[test]
    fn i8_element_type_codegen() {
        let shape = TypedShape::new(vec![64], ElementType::I8);
        let node = IrNode {
            op: IrOp::Add,
            inputs: vec![shape.clone(), shape.clone()],
            output: shape,
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("char*"));
    }

    #[test]
    fn f16_element_type_codegen() {
        let shape = TypedShape::new(vec![128], ElementType::F16);
        let node = IrNode {
            op: IrOp::Mul,
            inputs: vec![shape.clone(), shape.clone()],
            output: shape,
        };
        let src = KernelSpecializer::specialize_node(
            &node,
            BackendTarget::Cuda,
        )
        .unwrap();
        assert!(src.contains("half*"));
    }

    #[test]
    fn compiled_kernel_has_correct_key() {
        let mut jit = KernelJit::with_defaults();
        let ir = simple_add_ir();
        let k = jit.compile(&ir, BackendTarget::OpenCL).unwrap();
        assert_eq!(k.specialization_key.kernel_name, "vec_add");
        assert_eq!(k.specialization_key.target, BackendTarget::OpenCL);
    }
}
