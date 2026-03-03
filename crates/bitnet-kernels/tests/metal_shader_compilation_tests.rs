#![cfg(feature = "cpu")]

//! Metal shader compilation and validation tests for Apple Silicon.
//!
//! Validates MSL code generation, shader compilation pipeline, argument
//! buffers, feature levels, optimisation heuristics, and PSO caching
//! using pure Rust mocks — no Metal SDK or GPU hardware required.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

// ════════════════════════════════════════════════════════════════════
// Shared helpers
// ════════════════════════════════════════════════════════════════════

fn align_up(n: usize, align: usize) -> usize {
    assert!(align > 0);
    (n + align - 1) / align * align
}

/// Simple FNV-1a–style hash for cache keys.
fn fnv1a(data: &[u8]) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001b3;
    data.iter().fold(BASIS, |h, &b| (h ^ b as u64).wrapping_mul(PRIME))
}

// ════════════════════════════════════════════════════════════════════
//  §1  MSL Code Generation
// ════════════════════════════════════════════════════════════════════

/// Kernel category for MSL codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KernelKind {
    Matmul,
    Softmax,
    LayerNorm,
    RmsNorm,
    Attention,
    ElementWise,
    Reduction,
    Embedding,
    RoPE,
    Quantize,
    Dequantize,
    Transpose,
    Concat,
    GeLU,
    SiLU,
    TopK,
}

/// Generated MSL source with metadata.
#[derive(Debug, Clone)]
struct MslSource {
    kind: KernelKind,
    source: String,
    entry_point: String,
    threadgroup_memory_bytes: usize,
    max_threads_per_tg: u32,
}

/// Required MSL keywords every kernel must contain.
const MSL_REQUIRED_KEYWORDS: &[&str] = &["kernel", "device", "threadgroup"];

fn validate_msl_keywords(source: &str) -> Vec<&'static str> {
    MSL_REQUIRED_KEYWORDS.iter().copied().filter(|kw| !source.contains(kw)).collect()
}

fn generate_matmul_msl(m: u32, n: u32, k: u32) -> MslSource {
    let tg = 256u32;
    let shared = tg as usize * 4; // one f32 per thread
    let src = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void matmul_{m}x{n}x{k}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]]
) {{
    // {m}×{k} * {k}×{n} -> {m}×{n}
}}"#
    );
    MslSource {
        kind: KernelKind::Matmul,
        source: src,
        entry_point: format!("matmul_{m}x{n}x{k}"),
        threadgroup_memory_bytes: shared,
        max_threads_per_tg: tg,
    }
}

fn generate_softmax_msl(dim: u32) -> MslSource {
    let tg = dim.min(1024);
    let shared = tg as usize * 4;
    let src = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void softmax_{dim}(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    threadgroup float* shared  [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]]
) {{
    // stable softmax over {dim} elements
}}"#
    );
    MslSource {
        kind: KernelKind::Softmax,
        source: src,
        entry_point: format!("softmax_{dim}"),
        threadgroup_memory_bytes: shared,
        max_threads_per_tg: tg,
    }
}

fn generate_layernorm_msl(hidden: u32) -> MslSource {
    let tg = hidden.min(1024);
    let shared = tg as usize * 4 * 2; // mean + variance
    let src = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void layernorm_{hidden}(
    device const float* input  [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device const float* beta   [[buffer(2)]],
    device float* output       [[buffer(3)]],
    threadgroup float* shared  [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]]
) {{
    // layer normalisation over {hidden} dimensions
}}"#
    );
    MslSource {
        kind: KernelKind::LayerNorm,
        source: src,
        entry_point: format!("layernorm_{hidden}"),
        threadgroup_memory_bytes: shared,
        max_threads_per_tg: tg,
    }
}

fn generate_attention_msl(head_dim: u32, num_heads: u32) -> MslSource {
    let tg = head_dim.min(1024);
    let shared = tg as usize * 4;
    let src = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void attention_h{num_heads}_d{head_dim}(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output  [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]]
) {{
    // multi-head attention: {num_heads} heads, dim={head_dim}
}}"#
    );
    MslSource {
        kind: KernelKind::Attention,
        source: src,
        entry_point: format!("attention_h{num_heads}_d{head_dim}"),
        threadgroup_memory_bytes: shared,
        max_threads_per_tg: tg,
    }
}

fn generate_elementwise_msl(op: &str) -> MslSource {
    let src = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void ewise_{op}(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    threadgroup float* tg [[threadgroup(0)]]
) {{
    // element-wise {op}
}}"#
    );
    MslSource {
        kind: KernelKind::ElementWise,
        source: src,
        entry_point: format!("ewise_{op}"),
        threadgroup_memory_bytes: 0,
        max_threads_per_tg: 1024,
    }
}

fn generate_reduction_msl(dim: u32) -> MslSource {
    let tg = dim.min(1024);
    let shared = tg as usize * 4;
    let src = format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void reduce_sum_{dim}(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    threadgroup float* shared  [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]]
) {{
    // parallel tree reduction over {dim} elements
}}"#
    );
    MslSource {
        kind: KernelKind::Reduction,
        source: src,
        entry_point: format!("reduce_sum_{dim}"),
        threadgroup_memory_bytes: shared,
        max_threads_per_tg: tg,
    }
}

// ════════════════════════════════════════════════════════════════════
//  §2  Shader Compilation Pipeline Mocks
// ════════════════════════════════════════════════════════════════════

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MetalLanguageVersion {
    V2_4,
    V3_0,
    V3_1,
}

#[derive(Debug, Clone)]
struct CompileOptions {
    language_version: MetalLanguageVersion,
    fast_math: bool,
    preserve_invariance: bool,
    preprocessor_macros: HashMap<String, String>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            language_version: MetalLanguageVersion::V3_0,
            fast_math: true,
            preserve_invariance: false,
            preprocessor_macros: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct SpecializationConstant {
    index: u32,
    name: String,
    value: SpecConstValue,
}

#[derive(Debug, Clone, PartialEq)]
enum SpecConstValue {
    Bool(bool),
    Int(i32),
    Float(f32),
}

#[derive(Debug, Clone)]
struct MockFunction {
    name: String,
    #[allow(dead_code)]
    specialization_constants: Vec<SpecializationConstant>,
}

#[derive(Debug)]
struct MockLibrary {
    id: u64,
    functions: Vec<MockFunction>,
    source_hash: u64,
}

impl MockLibrary {
    fn from_source(source: &str, opts: &CompileOptions) -> Result<Self, String> {
        if source.is_empty() {
            return Err("empty source".into());
        }
        if !source.contains("kernel") {
            return Err("source contains no kernel functions".into());
        }
        // Reject Metal 3.1 features when targeting < 3.1.
        if source.contains("mesh") && opts.language_version != MetalLanguageVersion::V3_1 {
            return Err("mesh shaders require Metal 3.1".into());
        }
        // Extract entry points by scanning for `kernel void <name>`.
        let functions: Vec<MockFunction> = source
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if trimmed.starts_with("kernel void ") {
                    let rest = &trimmed["kernel void ".len()..];
                    let name = rest.split('(').next().unwrap_or("").trim();
                    if !name.is_empty() {
                        return Some(MockFunction {
                            name: name.to_string(),
                            specialization_constants: Vec::new(),
                        });
                    }
                }
                None
            })
            .collect();

        if functions.is_empty() {
            return Err("no kernel entry points found".into());
        }

        Ok(Self { id: next_id(), functions, source_hash: fnv1a(source.as_bytes()) })
    }

    fn function_names(&self) -> Vec<&str> {
        self.functions.iter().map(|f| f.name.as_str()).collect()
    }

    fn find_function(&self, name: &str) -> Option<&MockFunction> {
        self.functions.iter().find(|f| f.name == name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineStatus {
    Ready,
    Compiling,
    Error,
}

#[derive(Debug)]
struct MockComputePipelineState {
    id: u64,
    function_name: String,
    max_total_threads: u32,
    threadgroup_memory_length: usize,
    status: PipelineStatus,
    #[allow(dead_code)]
    spec_constants: Vec<SpecializationConstant>,
}

#[derive(Debug)]
struct MockDevice {
    name: String,
    max_threads_per_threadgroup: u32,
    max_threadgroup_memory: usize,
    max_buffer_length: usize,
    libraries: Vec<u64>,
}

impl MockDevice {
    fn apple_m_series(name: &str) -> Self {
        Self {
            name: name.to_string(),
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory: 32 * 1024,
            max_buffer_length: 256 * 1024 * 1024,
            libraries: Vec::new(),
        }
    }

    fn compile_library(
        &mut self,
        source: &str,
        opts: &CompileOptions,
    ) -> Result<MockLibrary, String> {
        let lib = MockLibrary::from_source(source, opts)?;
        self.libraries.push(lib.id);
        Ok(lib)
    }

    fn create_pipeline(
        &self,
        lib: &MockLibrary,
        fn_name: &str,
        tg_mem: usize,
        spec_constants: Vec<SpecializationConstant>,
    ) -> Result<MockComputePipelineState, String> {
        let func =
            lib.find_function(fn_name).ok_or_else(|| format!("function '{fn_name}' not found"))?;

        if tg_mem > self.max_threadgroup_memory {
            return Err(format!(
                "threadgroup memory {tg_mem} exceeds \
                 device limit {}",
                self.max_threadgroup_memory
            ));
        }

        Ok(MockComputePipelineState {
            id: next_id(),
            function_name: func.name.clone(),
            max_total_threads: self.max_threads_per_threadgroup,
            threadgroup_memory_length: tg_mem,
            status: PipelineStatus::Ready,
            spec_constants,
        })
    }
}

// ════════════════════════════════════════════════════════════════════
//  §3  Argument Buffer Mocks
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ResourceUsage {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone)]
struct BufferReference {
    binding_index: u32,
    offset: usize,
    size: usize,
    usage: ResourceUsage,
    label: String,
}

#[derive(Debug)]
struct ArgumentEncoder {
    encoded_length: usize,
    alignment: usize,
    entries: Vec<BufferReference>,
    resources_used: HashSet<u32>,
}

impl ArgumentEncoder {
    fn new(alignment: usize) -> Self {
        Self { encoded_length: 0, alignment, entries: Vec::new(), resources_used: HashSet::new() }
    }

    fn bind_buffer(
        &mut self,
        index: u32,
        size: usize,
        usage: ResourceUsage,
        label: &str,
    ) -> Result<(), String> {
        if self.resources_used.contains(&index) {
            return Err(format!("binding index {index} already used"));
        }
        let aligned_size = align_up(size, self.alignment);
        self.entries.push(BufferReference {
            binding_index: index,
            offset: self.encoded_length,
            size: aligned_size,
            usage,
            label: label.to_string(),
        });
        self.encoded_length += aligned_size;
        self.resources_used.insert(index);
        Ok(())
    }

    fn total_size(&self) -> usize {
        self.encoded_length
    }

    fn binding_count(&self) -> usize {
        self.entries.len()
    }

    fn read_resources(&self) -> Vec<&BufferReference> {
        self.entries
            .iter()
            .filter(|e| matches!(e.usage, ResourceUsage::Read | ResourceUsage::ReadWrite))
            .collect()
    }

    fn write_resources(&self) -> Vec<&BufferReference> {
        self.entries
            .iter()
            .filter(|e| matches!(e.usage, ResourceUsage::Write | ResourceUsage::ReadWrite))
            .collect()
    }
}

/// Indirect command buffer entry.
#[derive(Debug, Clone)]
struct IndirectDispatch {
    pipeline_id: u64,
    threadgroups: [u32; 3],
    threads_per_tg: [u32; 3],
}

#[derive(Debug)]
struct IndirectCommandBuffer {
    commands: Vec<IndirectDispatch>,
    max_commands: usize,
}

impl IndirectCommandBuffer {
    fn new(capacity: usize) -> Self {
        Self { commands: Vec::new(), max_commands: capacity }
    }

    fn push(&mut self, cmd: IndirectDispatch) -> Result<(), String> {
        if self.commands.len() >= self.max_commands {
            return Err("indirect command buffer full".into());
        }
        Ok(self.commands.push(cmd))
    }

    fn len(&self) -> usize {
        self.commands.len()
    }
}

// ════════════════════════════════════════════════════════════════════
//  §4  GPU Family / Feature Level Mocks
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum AppleGpuFamily {
    Apple7, // M1
    Apple8, // M2
    Apple9, // M3
}

#[derive(Debug, Clone)]
struct GpuFeatures {
    family: AppleGpuFamily,
    metal_version: MetalLanguageVersion,
    has_raytracing: bool,
    has_mesh_shaders: bool,
    has_dynamic_libraries: bool,
    max_threadgroup_memory: usize,
    max_threads_per_threadgroup: u32,
    simd_width: u32,
    max_buffer_length: usize,
    supports_bfloat16: bool,
}

impl GpuFeatures {
    fn for_family(family: AppleGpuFamily) -> Self {
        match family {
            AppleGpuFamily::Apple7 => Self {
                family,
                metal_version: MetalLanguageVersion::V2_4,
                has_raytracing: false,
                has_mesh_shaders: false,
                has_dynamic_libraries: false,
                max_threadgroup_memory: 32 * 1024,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_buffer_length: 256 * 1024 * 1024,
                supports_bfloat16: false,
            },
            AppleGpuFamily::Apple8 => Self {
                family,
                metal_version: MetalLanguageVersion::V3_0,
                has_raytracing: true,
                has_mesh_shaders: false,
                has_dynamic_libraries: true,
                max_threadgroup_memory: 32 * 1024,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_buffer_length: 256 * 1024 * 1024,
                supports_bfloat16: true,
            },
            AppleGpuFamily::Apple9 => Self {
                family,
                metal_version: MetalLanguageVersion::V3_1,
                has_raytracing: true,
                has_mesh_shaders: true,
                has_dynamic_libraries: true,
                max_threadgroup_memory: 32 * 1024,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_buffer_length: 256 * 1024 * 1024,
                supports_bfloat16: true,
            },
        }
    }

    fn supports_kernel(&self, kind: KernelKind) -> bool {
        match kind {
            // All families support basic compute.
            KernelKind::Matmul
            | KernelKind::Softmax
            | KernelKind::LayerNorm
            | KernelKind::RmsNorm
            | KernelKind::Attention
            | KernelKind::ElementWise
            | KernelKind::Reduction
            | KernelKind::Embedding
            | KernelKind::RoPE
            | KernelKind::Quantize
            | KernelKind::Dequantize
            | KernelKind::Transpose
            | KernelKind::Concat
            | KernelKind::GeLU
            | KernelKind::SiLU
            | KernelKind::TopK => true,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  §5  Shader Optimisation Heuristics
// ════════════════════════════════════════════════════════════════════

/// Estimated register pressure for a kernel.
#[derive(Debug, Clone)]
struct RegisterEstimate {
    scalar_regs: u32,
    vector_regs: u32,
    total: u32,
}

fn estimate_registers(kind: KernelKind) -> RegisterEstimate {
    let (s, v) = match kind {
        KernelKind::Matmul => (16, 32),
        KernelKind::Softmax => (8, 16),
        KernelKind::LayerNorm | KernelKind::RmsNorm => (12, 24),
        KernelKind::Attention => (20, 48),
        KernelKind::Reduction => (8, 16),
        _ => (8, 8),
    };
    RegisterEstimate { scalar_regs: s, vector_regs: v, total: s + v }
}

/// Occupancy as a fraction [0, 1].
fn estimate_occupancy(threads_per_tg: u32, tg_memory: usize, regs: &RegisterEstimate) -> f64 {
    let max_threads = 1024u32;
    let max_mem = 32 * 1024usize;
    let max_regs = 128u32;

    let thread_ratio = threads_per_tg as f64 / max_threads as f64;
    let mem_ratio = 1.0 - (tg_memory as f64 / max_mem as f64);
    let reg_ratio = 1.0 - (regs.total as f64 / max_regs as f64);

    (thread_ratio * mem_ratio * reg_ratio).clamp(0.0, 1.0)
}

/// Instruction count estimate (very rough).
fn estimate_instruction_count(kind: KernelKind, n: u32) -> u64 {
    match kind {
        KernelKind::Matmul => {
            // ~2N³ FLOPs for N×N×N
            2 * (n as u64).pow(3)
        }
        KernelKind::Softmax => {
            // 5N: max, sub, exp, sum, div
            5 * n as u64
        }
        KernelKind::LayerNorm | KernelKind::RmsNorm => {
            // 7N: sum, mean, sub, sq, var, normalise, scale
            7 * n as u64
        }
        KernelKind::Attention => {
            // Q·K^T + softmax + ·V ≈ 4N²
            4 * (n as u64).pow(2)
        }
        KernelKind::Reduction => 2 * n as u64,
        _ => n as u64,
    }
}

/// Should the compiler unroll a loop?
fn should_unroll(trip_count: u32, body_instructions: u32) -> bool {
    let total = trip_count as u64 * body_instructions as u64;
    // Unroll if total instructions ≤ 256.
    total <= 256
}

/// Validate memory access pattern for coalescing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessPattern {
    Sequential,
    Strided(u32),
    Random,
}

fn is_coalesced(pattern: AccessPattern, simd_width: u32) -> bool {
    match pattern {
        AccessPattern::Sequential => true,
        AccessPattern::Strided(s) => s == 1 || s.is_multiple_of(simd_width),
        AccessPattern::Random => false,
    }
}

// ════════════════════════════════════════════════════════════════════
//  §6  Shader Cache Mocks
// ════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct PsoCacheEntry {
    source_hash: u64,
    function_name: String,
    pipeline_id: u64,
    version: u32,
    created_at: u64,
}

#[derive(Debug)]
struct PsoCache {
    entries: HashMap<u64, PsoCacheEntry>,
    version: u32,
    max_entries: usize,
    hits: u64,
    misses: u64,
}

impl PsoCache {
    fn new(max_entries: usize, version: u32) -> Self {
        Self { entries: HashMap::new(), version, max_entries, hits: 0, misses: 0 }
    }

    fn lookup(&mut self, hash: u64) -> Option<&PsoCacheEntry> {
        if let Some(e) = self.entries.get(&hash) {
            if e.version == self.version {
                self.hits += 1;
                return Some(e);
            }
        }
        self.misses += 1;
        None
    }

    fn insert(&mut self, entry: PsoCacheEntry) -> bool {
        if self.entries.len() >= self.max_entries {
            return false;
        }
        self.entries.insert(entry.source_hash, entry);
        true
    }

    fn invalidate_version(&mut self, old_version: u32) {
        self.entries.retain(|_, e| e.version != old_version);
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// Binary archive: serialisable representation of compiled shaders.
#[derive(Debug, Clone)]
struct BinaryArchive {
    entries: BTreeMap<String, Vec<u8>>,
    total_bytes: usize,
}

impl BinaryArchive {
    fn new() -> Self {
        Self { entries: BTreeMap::new(), total_bytes: 0 }
    }

    fn add(&mut self, name: &str, data: Vec<u8>) -> Result<(), String> {
        if data.is_empty() {
            return Err("empty binary data".into());
        }
        if self.entries.contains_key(name) {
            return Err(format!("duplicate entry '{name}'"));
        }
        self.total_bytes += data.len();
        self.entries.insert(name.to_string(), data);
        Ok(())
    }

    fn get(&self, name: &str) -> Option<&[u8]> {
        self.entries.get(name).map(|v| v.as_slice())
    }

    fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// ════════════════════════════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── §1 MSL Code Generation ────────────────────────────────────

    #[test]
    fn msl_matmul_kernel_has_required_keywords() {
        let msl = generate_matmul_msl(128, 128, 128);
        let missing = validate_msl_keywords(&msl.source);
        assert!(missing.is_empty(), "missing keywords: {missing:?}");
        assert_eq!(msl.kind, KernelKind::Matmul);
    }

    #[test]
    fn msl_matmul_entry_point_encodes_dimensions() {
        let msl = generate_matmul_msl(64, 32, 16);
        assert_eq!(msl.entry_point, "matmul_64x32x16");
        assert!(msl.source.contains("matmul_64x32x16"));
    }

    #[test]
    fn msl_softmax_kernel_has_required_keywords() {
        let msl = generate_softmax_msl(512);
        let missing = validate_msl_keywords(&msl.source);
        assert!(missing.is_empty());
        assert_eq!(msl.kind, KernelKind::Softmax);
    }

    #[test]
    fn msl_softmax_threadgroup_memory_scales_with_dim() {
        let s256 = generate_softmax_msl(256);
        let s1024 = generate_softmax_msl(1024);
        assert!(s1024.threadgroup_memory_bytes >= s256.threadgroup_memory_bytes);
    }

    #[test]
    fn msl_layernorm_kernel_has_required_keywords() {
        let msl = generate_layernorm_msl(768);
        let missing = validate_msl_keywords(&msl.source);
        assert!(missing.is_empty());
        assert_eq!(msl.kind, KernelKind::LayerNorm);
    }

    #[test]
    fn msl_layernorm_uses_four_buffers() {
        let msl = generate_layernorm_msl(256);
        // input, gamma, beta, output
        let buf_count = msl.source.matches("[[buffer(").count();
        assert_eq!(buf_count, 4);
    }

    #[test]
    fn msl_attention_kernel_encodes_heads_and_dim() {
        let msl = generate_attention_msl(64, 12);
        assert_eq!(msl.entry_point, "attention_h12_d64");
        assert!(msl.source.contains("attention_h12_d64"));
    }

    #[test]
    fn msl_attention_kernel_has_required_keywords() {
        let msl = generate_attention_msl(128, 8);
        let missing = validate_msl_keywords(&msl.source);
        assert!(missing.is_empty());
    }

    #[test]
    fn msl_elementwise_ops_generate_distinct_kernels() {
        let ops = ["add", "mul", "sub", "div", "max", "min"];
        let names: HashSet<String> =
            ops.iter().map(|op| generate_elementwise_msl(op).entry_point).collect();
        assert_eq!(names.len(), ops.len());
    }

    #[test]
    fn msl_elementwise_has_required_keywords() {
        let msl = generate_elementwise_msl("add");
        let missing = validate_msl_keywords(&msl.source);
        assert!(missing.is_empty());
    }

    #[test]
    fn msl_reduction_kernel_has_required_keywords() {
        let msl = generate_reduction_msl(1024);
        let missing = validate_msl_keywords(&msl.source);
        assert!(missing.is_empty());
        assert_eq!(msl.kind, KernelKind::Reduction);
    }

    #[test]
    fn msl_reduction_shared_mem_within_limit() {
        let msl = generate_reduction_msl(1024);
        assert!(msl.threadgroup_memory_bytes <= 32 * 1024);
    }

    #[test]
    fn msl_threadgroup_capped_at_1024() {
        let msl = generate_softmax_msl(4096);
        assert!(msl.max_threads_per_tg <= 1024);
    }

    #[test]
    fn msl_all_kernel_kinds_generate_valid_source() {
        let sources = [
            generate_matmul_msl(32, 32, 32),
            generate_softmax_msl(256),
            generate_layernorm_msl(512),
            generate_attention_msl(64, 8),
            generate_elementwise_msl("add"),
            generate_reduction_msl(256),
        ];
        for msl in &sources {
            let missing = validate_msl_keywords(&msl.source);
            assert!(missing.is_empty(), "{:?} missing: {missing:?}", msl.kind);
            assert!(!msl.entry_point.is_empty(), "{:?} has empty entry point", msl.kind);
        }
    }

    #[test]
    fn msl_matmul_shared_mem_proportional_to_threads() {
        let msl = generate_matmul_msl(256, 256, 256);
        assert_eq!(msl.threadgroup_memory_bytes, msl.max_threads_per_tg as usize * 4);
    }

    // ── §2 Shader Compilation Pipeline ────────────────────────────

    #[test]
    fn compile_matmul_library_succeeds() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let msl = generate_matmul_msl(128, 128, 128);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts);
        assert!(lib.is_ok());
    }

    #[test]
    fn compile_empty_source_fails() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let opts = CompileOptions::default();
        let lib = dev.compile_library("", &opts);
        assert_eq!(lib.unwrap_err(), "empty source");
    }

    #[test]
    fn compile_no_kernel_fails() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let opts = CompileOptions::default();
        let lib = dev.compile_library("void foo() {}", &opts);
        assert_eq!(lib.unwrap_err(), "source contains no kernel functions");
    }

    #[test]
    fn library_extracts_function_names() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let msl = generate_matmul_msl(64, 64, 64);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        let names = lib.function_names();
        assert_eq!(names, vec!["matmul_64x64x64"]);
    }

    #[test]
    fn library_find_function_returns_none_for_missing() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let msl = generate_matmul_msl(32, 32, 32);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        assert!(lib.find_function("nonexistent").is_none());
    }

    #[test]
    fn create_pipeline_succeeds() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let msl = generate_matmul_msl(64, 64, 64);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        let pso = dev.create_pipeline(&lib, "matmul_64x64x64", 1024, vec![]);
        assert!(pso.is_ok());
        let pso = pso.unwrap();
        assert_eq!(pso.status, PipelineStatus::Ready);
        assert_eq!(pso.function_name, "matmul_64x64x64");
    }

    #[test]
    fn create_pipeline_missing_function_fails() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let msl = generate_matmul_msl(32, 32, 32);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        let pso = dev.create_pipeline(&lib, "nope", 0, vec![]);
        assert!(pso.is_err());
    }

    #[test]
    fn create_pipeline_excess_tg_memory_fails() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let msl = generate_matmul_msl(32, 32, 32);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        let pso = dev.create_pipeline(
            &lib,
            "matmul_32x32x32",
            64 * 1024, // exceeds 32 KB
            vec![],
        );
        assert!(pso.is_err());
    }

    #[test]
    fn specialization_constants_attached_to_pipeline() {
        let mut dev = MockDevice::apple_m_series("Apple M2");
        let msl = generate_softmax_msl(512);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        let specs = vec![
            SpecializationConstant {
                index: 0,
                name: "USE_FAST_MATH".into(),
                value: SpecConstValue::Bool(true),
            },
            SpecializationConstant {
                index: 1,
                name: "TILE_SIZE".into(),
                value: SpecConstValue::Int(16),
            },
        ];
        let pso = dev.create_pipeline(&lib, &msl.entry_point, msl.threadgroup_memory_bytes, specs);
        assert!(pso.is_ok());
    }

    #[test]
    fn compile_options_fast_math_default() {
        let opts = CompileOptions::default();
        assert!(opts.fast_math);
        assert!(!opts.preserve_invariance);
        assert_eq!(opts.language_version, MetalLanguageVersion::V3_0);
    }

    #[test]
    fn compile_with_preprocessor_macros() {
        let mut opts = CompileOptions::default();
        opts.preprocessor_macros.insert("TILE_M".into(), "8".into());
        opts.preprocessor_macros.insert("TILE_N".into(), "8".into());
        assert_eq!(opts.preprocessor_macros.len(), 2);
    }

    #[test]
    fn library_has_unique_id() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let opts = CompileOptions::default();
        let lib1 = dev.compile_library(&generate_matmul_msl(32, 32, 32).source, &opts).unwrap();
        let lib2 = dev.compile_library(&generate_softmax_msl(256).source, &opts).unwrap();
        assert_ne!(lib1.id, lib2.id);
    }

    #[test]
    fn device_tracks_compiled_libraries() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let opts = CompileOptions::default();
        assert!(dev.libraries.is_empty());
        let _ = dev.compile_library(&generate_matmul_msl(32, 32, 32).source, &opts).unwrap();
        assert_eq!(dev.libraries.len(), 1);
    }

    #[test]
    fn mesh_shader_requires_metal_3_1() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void mesh_shader(
    device float* out [[buffer(0)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]]
) {
    // mesh dispatch
}"#;
        let opts_v3 =
            CompileOptions { language_version: MetalLanguageVersion::V3_0, ..Default::default() };
        assert!(dev.compile_library(source, &opts_v3).is_err());

        let opts_v31 =
            CompileOptions { language_version: MetalLanguageVersion::V3_1, ..Default::default() };
        assert!(dev.compile_library(source, &opts_v31).is_ok());
    }

    #[test]
    fn pipeline_records_threadgroup_memory_length() {
        let mut dev = MockDevice::apple_m_series("Apple M1");
        let msl = generate_layernorm_msl(256);
        let opts = CompileOptions::default();
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        let pso = dev
            .create_pipeline(&lib, &msl.entry_point, msl.threadgroup_memory_bytes, vec![])
            .unwrap();
        assert_eq!(pso.threadgroup_memory_length, msl.threadgroup_memory_bytes);
    }

    // ── §3 Argument Buffers ───────────────────────────────────────

    #[test]
    fn argument_encoder_bind_single_buffer() {
        let mut enc = ArgumentEncoder::new(16);
        enc.bind_buffer(0, 64, ResourceUsage::Read, "weights").unwrap();
        assert_eq!(enc.binding_count(), 1);
        assert_eq!(enc.total_size(), 64);
    }

    #[test]
    fn argument_encoder_bind_multiple_buffers() {
        let mut enc = ArgumentEncoder::new(16);
        enc.bind_buffer(0, 64, ResourceUsage::Read, "weights").unwrap();
        enc.bind_buffer(1, 32, ResourceUsage::Read, "input").unwrap();
        enc.bind_buffer(2, 128, ResourceUsage::Write, "output").unwrap();
        assert_eq!(enc.binding_count(), 3);
        assert_eq!(enc.total_size(), 64 + 32 + 128);
    }

    #[test]
    fn argument_encoder_rejects_duplicate_binding() {
        let mut enc = ArgumentEncoder::new(16);
        enc.bind_buffer(0, 64, ResourceUsage::Read, "a").unwrap();
        let err = enc.bind_buffer(0, 32, ResourceUsage::Read, "b");
        assert!(err.is_err());
    }

    #[test]
    fn argument_encoder_aligns_buffer_sizes() {
        let mut enc = ArgumentEncoder::new(16);
        enc.bind_buffer(0, 10, ResourceUsage::Read, "small").unwrap();
        // 10 bytes aligned to 16 → 16
        assert_eq!(enc.total_size(), 16);
    }

    #[test]
    fn argument_encoder_tracks_read_resources() {
        let mut enc = ArgumentEncoder::new(8);
        enc.bind_buffer(0, 64, ResourceUsage::Read, "weights").unwrap();
        enc.bind_buffer(1, 32, ResourceUsage::Write, "output").unwrap();
        enc.bind_buffer(2, 16, ResourceUsage::ReadWrite, "scratch").unwrap();
        let reads = enc.read_resources();
        assert_eq!(reads.len(), 2); // weights + scratch
    }

    #[test]
    fn argument_encoder_tracks_write_resources() {
        let mut enc = ArgumentEncoder::new(8);
        enc.bind_buffer(0, 64, ResourceUsage::Read, "weights").unwrap();
        enc.bind_buffer(1, 32, ResourceUsage::Write, "output").unwrap();
        enc.bind_buffer(2, 16, ResourceUsage::ReadWrite, "scratch").unwrap();
        let writes = enc.write_resources();
        assert_eq!(writes.len(), 2); // output + scratch
    }

    #[test]
    fn argument_encoder_offsets_are_sequential() {
        let mut enc = ArgumentEncoder::new(8);
        enc.bind_buffer(0, 16, ResourceUsage::Read, "a").unwrap();
        enc.bind_buffer(1, 24, ResourceUsage::Read, "b").unwrap();
        enc.bind_buffer(2, 8, ResourceUsage::Write, "c").unwrap();
        let offsets: Vec<usize> = enc.entries.iter().map(|e| e.offset).collect();
        assert_eq!(offsets, vec![0, 16, 40]);
    }

    #[test]
    fn argument_encoder_empty_has_zero_size() {
        let enc = ArgumentEncoder::new(16);
        assert_eq!(enc.total_size(), 0);
        assert_eq!(enc.binding_count(), 0);
    }

    #[test]
    fn indirect_command_buffer_push_within_capacity() {
        let mut icb = IndirectCommandBuffer::new(4);
        let cmd = IndirectDispatch {
            pipeline_id: 1,
            threadgroups: [8, 1, 1],
            threads_per_tg: [256, 1, 1],
        };
        assert!(icb.push(cmd).is_ok());
        assert_eq!(icb.len(), 1);
    }

    #[test]
    fn indirect_command_buffer_rejects_overflow() {
        let mut icb = IndirectCommandBuffer::new(1);
        let cmd = IndirectDispatch {
            pipeline_id: 1,
            threadgroups: [1, 1, 1],
            threads_per_tg: [32, 1, 1],
        };
        assert!(icb.push(cmd.clone()).is_ok());
        assert!(icb.push(cmd).is_err());
    }

    #[test]
    fn indirect_command_buffer_multiple_dispatches() {
        let mut icb = IndirectCommandBuffer::new(64);
        let dispatches: Vec<IndirectDispatch> = (0..16u64)
            .map(|i| IndirectDispatch {
                pipeline_id: i,
                threadgroups: [4, 1, 1],
                threads_per_tg: [256, 1, 1],
            })
            .collect();
        for d in dispatches {
            icb.push(d).unwrap();
        }
        assert_eq!(icb.len(), 16);
    }

    #[test]
    fn argument_encoder_readwrite_is_in_both_sets() {
        let mut enc = ArgumentEncoder::new(8);
        enc.bind_buffer(0, 64, ResourceUsage::ReadWrite, "rw_buf").unwrap();
        assert_eq!(enc.read_resources().len(), 1);
        assert_eq!(enc.write_resources().len(), 1);
    }

    #[test]
    fn argument_encoder_large_alignment() {
        let mut enc = ArgumentEncoder::new(256);
        enc.bind_buffer(0, 100, ResourceUsage::Read, "data").unwrap();
        assert_eq!(enc.total_size(), 256);
    }

    #[test]
    fn argument_encoder_labels_preserved() {
        let mut enc = ArgumentEncoder::new(8);
        enc.bind_buffer(0, 8, ResourceUsage::Read, "my_weights").unwrap();
        assert_eq!(enc.entries[0].label, "my_weights");
    }

    #[test]
    fn indirect_dispatch_stores_grid_dims() {
        let d = IndirectDispatch {
            pipeline_id: 42,
            threadgroups: [8, 4, 2],
            threads_per_tg: [32, 16, 1],
        };
        assert_eq!(d.threadgroups[0], 8);
        assert_eq!(d.threads_per_tg[1], 16);
    }

    // ── §4 GPU Feature Levels ─────────────────────────────────────

    #[test]
    fn apple7_is_metal_2_4() {
        let f = GpuFeatures::for_family(AppleGpuFamily::Apple7);
        assert_eq!(f.metal_version, MetalLanguageVersion::V2_4);
        assert!(!f.has_raytracing);
        assert!(!f.has_mesh_shaders);
    }

    #[test]
    fn apple8_supports_raytracing() {
        let f = GpuFeatures::for_family(AppleGpuFamily::Apple8);
        assert!(f.has_raytracing);
        assert!(!f.has_mesh_shaders);
    }

    #[test]
    fn apple9_supports_mesh_shaders() {
        let f = GpuFeatures::for_family(AppleGpuFamily::Apple9);
        assert!(f.has_mesh_shaders);
        assert!(f.has_raytracing);
        assert_eq!(f.metal_version, MetalLanguageVersion::V3_1);
    }

    #[test]
    fn all_families_support_inference_kernels() {
        let families = [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9];
        let kinds = [
            KernelKind::Matmul,
            KernelKind::Softmax,
            KernelKind::LayerNorm,
            KernelKind::Attention,
            KernelKind::ElementWise,
            KernelKind::Reduction,
        ];
        for &fam in &families {
            let feats = GpuFeatures::for_family(fam);
            for &k in &kinds {
                assert!(feats.supports_kernel(k), "{fam:?} should support {k:?}");
            }
        }
    }

    #[test]
    fn simd_width_is_32_for_all_apple_families() {
        let families = [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9];
        for &fam in &families {
            let f = GpuFeatures::for_family(fam);
            assert_eq!(f.simd_width, 32);
        }
    }

    #[test]
    fn bfloat16_requires_apple8_or_later() {
        let a7 = GpuFeatures::for_family(AppleGpuFamily::Apple7);
        let a8 = GpuFeatures::for_family(AppleGpuFamily::Apple8);
        let a9 = GpuFeatures::for_family(AppleGpuFamily::Apple9);
        assert!(!a7.supports_bfloat16);
        assert!(a8.supports_bfloat16);
        assert!(a9.supports_bfloat16);
    }

    #[test]
    fn dynamic_libraries_require_apple8() {
        let a7 = GpuFeatures::for_family(AppleGpuFamily::Apple7);
        let a8 = GpuFeatures::for_family(AppleGpuFamily::Apple8);
        assert!(!a7.has_dynamic_libraries);
        assert!(a8.has_dynamic_libraries);
    }

    #[test]
    fn max_threadgroup_memory_is_32k() {
        let families = [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9];
        for &fam in &families {
            let f = GpuFeatures::for_family(fam);
            assert_eq!(f.max_threadgroup_memory, 32 * 1024);
        }
    }

    #[test]
    fn gpu_family_ordering() {
        assert!(AppleGpuFamily::Apple7 < AppleGpuFamily::Apple8);
        assert!(AppleGpuFamily::Apple8 < AppleGpuFamily::Apple9);
    }

    #[test]
    fn all_additional_kernels_supported() {
        let f = GpuFeatures::for_family(AppleGpuFamily::Apple7);
        let extra = [
            KernelKind::RmsNorm,
            KernelKind::Embedding,
            KernelKind::RoPE,
            KernelKind::Quantize,
            KernelKind::Dequantize,
            KernelKind::Transpose,
            KernelKind::Concat,
            KernelKind::GeLU,
            KernelKind::SiLU,
            KernelKind::TopK,
        ];
        for &k in &extra {
            assert!(f.supports_kernel(k), "Apple7 should support {k:?}");
        }
    }

    #[test]
    fn max_buffer_length_consistent() {
        let families = [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9];
        for &fam in &families {
            let f = GpuFeatures::for_family(fam);
            assert_eq!(f.max_buffer_length, 256 * 1024 * 1024);
        }
    }

    #[test]
    fn apple9_metal_version_is_3_1() {
        let f = GpuFeatures::for_family(AppleGpuFamily::Apple9);
        assert_eq!(f.metal_version, MetalLanguageVersion::V3_1);
    }

    #[test]
    fn apple8_metal_version_is_3_0() {
        let f = GpuFeatures::for_family(AppleGpuFamily::Apple8);
        assert_eq!(f.metal_version, MetalLanguageVersion::V3_0);
    }

    #[test]
    fn max_threads_per_tg_is_1024_all_families() {
        let families = [AppleGpuFamily::Apple7, AppleGpuFamily::Apple8, AppleGpuFamily::Apple9];
        for &fam in &families {
            let f = GpuFeatures::for_family(fam);
            assert_eq!(f.max_threads_per_threadgroup, 1024);
        }
    }

    // ── §5 Shader Optimisation ────────────────────────────────────

    #[test]
    fn matmul_register_estimate() {
        let est = estimate_registers(KernelKind::Matmul);
        assert_eq!(est.scalar_regs, 16);
        assert_eq!(est.vector_regs, 32);
        assert_eq!(est.total, 48);
    }

    #[test]
    fn attention_has_highest_register_pressure() {
        let kinds = [
            KernelKind::Matmul,
            KernelKind::Softmax,
            KernelKind::LayerNorm,
            KernelKind::Attention,
            KernelKind::Reduction,
            KernelKind::ElementWise,
        ];
        let max_kind = kinds.iter().max_by_key(|&&k| estimate_registers(k).total).unwrap();
        assert_eq!(*max_kind, KernelKind::Attention);
    }

    #[test]
    fn occupancy_decreases_with_more_shared_memory() {
        let regs = estimate_registers(KernelKind::Matmul);
        let occ_low = estimate_occupancy(256, 1024, &regs);
        let occ_high = estimate_occupancy(256, 16 * 1024, &regs);
        assert!(occ_low > occ_high, "more shared mem → lower occupancy");
    }

    #[test]
    fn occupancy_clamped_to_unit_interval() {
        let regs = RegisterEstimate { scalar_regs: 0, vector_regs: 0, total: 0 };
        let occ = estimate_occupancy(1024, 0, &regs);
        assert!((0.0..=1.0).contains(&occ));
    }

    #[test]
    fn matmul_instruction_count_cubic() {
        let n32 = estimate_instruction_count(KernelKind::Matmul, 32);
        let n64 = estimate_instruction_count(KernelKind::Matmul, 64);
        // 64³ / 32³ = 8×
        assert_eq!(n64, n32 * 8);
    }

    #[test]
    fn softmax_instruction_count_linear() {
        let n256 = estimate_instruction_count(KernelKind::Softmax, 256);
        let n512 = estimate_instruction_count(KernelKind::Softmax, 512);
        assert_eq!(n512, n256 * 2);
    }

    #[test]
    fn attention_instruction_count_quadratic() {
        let n64 = estimate_instruction_count(KernelKind::Attention, 64);
        let n128 = estimate_instruction_count(KernelKind::Attention, 128);
        assert_eq!(n128, n64 * 4);
    }

    #[test]
    fn unroll_small_loop() {
        assert!(should_unroll(8, 4)); // 32 insns
        assert!(should_unroll(16, 16)); // 256 insns
    }

    #[test]
    fn no_unroll_large_loop() {
        assert!(!should_unroll(64, 16)); // 1024 insns
        assert!(!should_unroll(1024, 4)); // 4096 insns
    }

    #[test]
    fn sequential_access_is_coalesced() {
        assert!(is_coalesced(AccessPattern::Sequential, 32));
    }

    #[test]
    fn stride_1_is_coalesced() {
        assert!(is_coalesced(AccessPattern::Strided(1), 32));
    }

    #[test]
    fn stride_matching_simd_width_is_coalesced() {
        assert!(is_coalesced(AccessPattern::Strided(32), 32));
        assert!(is_coalesced(AccessPattern::Strided(64), 32));
    }

    #[test]
    fn random_access_is_not_coalesced() {
        assert!(!is_coalesced(AccessPattern::Random, 32));
    }

    #[test]
    fn non_aligned_stride_is_not_coalesced() {
        assert!(!is_coalesced(AccessPattern::Strided(7), 32));
    }

    #[test]
    fn occupancy_increases_with_more_threads() {
        let regs = estimate_registers(KernelKind::Softmax);
        let occ_low = estimate_occupancy(64, 1024, &regs);
        let occ_high = estimate_occupancy(512, 1024, &regs);
        assert!(occ_high > occ_low);
    }

    #[test]
    fn layernorm_instruction_count_linear() {
        let n256 = estimate_instruction_count(KernelKind::LayerNorm, 256);
        let n512 = estimate_instruction_count(KernelKind::LayerNorm, 512);
        assert_eq!(n512, n256 * 2);
    }

    // ── §6 Shader Caching ────────────────────────────────────────

    #[test]
    fn pso_cache_miss_on_empty() {
        let mut cache = PsoCache::new(16, 1);
        assert!(cache.lookup(42).is_none());
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn pso_cache_hit_after_insert() {
        let mut cache = PsoCache::new(16, 1);
        let entry = PsoCacheEntry {
            source_hash: 42,
            function_name: "matmul".into(),
            pipeline_id: 1,
            version: 1,
            created_at: 0,
        };
        assert!(cache.insert(entry));
        let result = cache.lookup(42);
        assert!(result.is_some());
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn pso_cache_version_mismatch_is_miss() {
        let mut cache = PsoCache::new(16, 2);
        let entry = PsoCacheEntry {
            source_hash: 99,
            function_name: "softmax".into(),
            pipeline_id: 2,
            version: 1, // old version
            created_at: 0,
        };
        cache.insert(entry);
        assert!(cache.lookup(99).is_none());
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn pso_cache_respects_capacity() {
        let mut cache = PsoCache::new(2, 1);
        let mk = |h| PsoCacheEntry {
            source_hash: h,
            function_name: "f".into(),
            pipeline_id: h,
            version: 1,
            created_at: 0,
        };
        assert!(cache.insert(mk(1)));
        assert!(cache.insert(mk(2)));
        assert!(!cache.insert(mk(3))); // full
    }

    #[test]
    fn pso_cache_invalidation_by_version() {
        let mut cache = PsoCache::new(16, 1);
        let mk = |h, v| PsoCacheEntry {
            source_hash: h,
            function_name: "f".into(),
            pipeline_id: h,
            version: v,
            created_at: 0,
        };
        cache.insert(mk(1, 1));
        cache.insert(mk(2, 1));
        cache.insert(mk(3, 2));
        cache.invalidate_version(1);
        assert_eq!(cache.entries.len(), 1);
        assert!(cache.entries.contains_key(&3));
    }

    #[test]
    fn pso_cache_hit_rate_calculation() {
        let mut cache = PsoCache::new(16, 1);
        let entry = PsoCacheEntry {
            source_hash: 10,
            function_name: "f".into(),
            pipeline_id: 1,
            version: 1,
            created_at: 0,
        };
        cache.insert(entry);
        cache.lookup(10); // hit
        cache.lookup(10); // hit
        cache.lookup(99); // miss
        let rate = cache.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn binary_archive_add_and_retrieve() {
        let mut archive = BinaryArchive::new();
        archive.add("matmul_v1", vec![0xCA, 0xFE]).unwrap();
        let data = archive.get("matmul_v1").unwrap();
        assert_eq!(data, &[0xCA, 0xFE]);
    }

    #[test]
    fn binary_archive_rejects_empty_data() {
        let mut archive = BinaryArchive::new();
        assert!(archive.add("bad", vec![]).is_err());
    }

    #[test]
    fn binary_archive_rejects_duplicate() {
        let mut archive = BinaryArchive::new();
        archive.add("x", vec![1]).unwrap();
        assert!(archive.add("x", vec![2]).is_err());
    }

    #[test]
    fn binary_archive_tracks_total_bytes() {
        let mut archive = BinaryArchive::new();
        archive.add("a", vec![1, 2, 3]).unwrap();
        archive.add("b", vec![4, 5]).unwrap();
        assert_eq!(archive.total_bytes, 5);
    }

    #[test]
    fn binary_archive_entry_count() {
        let mut archive = BinaryArchive::new();
        archive.add("x", vec![1]).unwrap();
        archive.add("y", vec![2]).unwrap();
        archive.add("z", vec![3]).unwrap();
        assert_eq!(archive.entry_count(), 3);
    }

    #[test]
    fn binary_archive_get_missing_returns_none() {
        let archive = BinaryArchive::new();
        assert!(archive.get("missing").is_none());
    }

    #[test]
    fn pso_cache_empty_hit_rate_is_zero() {
        let cache = PsoCache::new(16, 1);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn pso_cache_source_hash_distinguishes_kernels() {
        let mut cache = PsoCache::new(16, 1);
        let mk = |h, name: &str| PsoCacheEntry {
            source_hash: h,
            function_name: name.into(),
            pipeline_id: h,
            version: 1,
            created_at: 0,
        };
        cache.insert(mk(100, "matmul"));
        cache.insert(mk(200, "softmax"));
        assert!(cache.lookup(100).is_some());
        assert!(cache.lookup(200).is_some());
    }

    #[test]
    fn fnv1a_produces_distinct_hashes() {
        let h1 = fnv1a(b"matmul_v1");
        let h2 = fnv1a(b"matmul_v2");
        let h3 = fnv1a(b"softmax_v1");
        let hashes: HashSet<u64> = [h1, h2, h3].iter().copied().collect();
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn pipeline_end_to_end_compile_cache_roundtrip() {
        let mut dev = MockDevice::apple_m_series("Apple M2");
        let opts = CompileOptions::default();
        let msl = generate_matmul_msl(128, 128, 128);
        let lib = dev.compile_library(&msl.source, &opts).unwrap();
        let pso = dev
            .create_pipeline(&lib, &msl.entry_point, msl.threadgroup_memory_bytes, vec![])
            .unwrap();

        // Cache the PSO.
        let mut cache = PsoCache::new(16, 1);
        let entry = PsoCacheEntry {
            source_hash: lib.source_hash,
            function_name: pso.function_name.clone(),
            pipeline_id: pso.id,
            version: 1,
            created_at: 0,
        };
        assert!(cache.insert(entry));
        assert!(cache.lookup(lib.source_hash).is_some());

        // Archive the binary.
        let mut archive = BinaryArchive::new();
        archive.add(&pso.function_name, vec![0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        assert!(archive.get(&pso.function_name).is_some());
    }
}
