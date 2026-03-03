#![cfg(feature = "cpu")]
#![allow(dead_code, clippy::manual_div_ceil)]

//! Comprehensive tests for Metal shader compilation patterns.
//!
//! Validates MSL source analysis, pipeline state object management,
//! compilation error handling, shader variant generation, function
//! library linkage, and performance configuration — all in pure Rust
//! without requiring a GPU.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

// =========================================================================
// Domain types modelling Metal shader compilation
// =========================================================================

/// Represents a Metal Shading Language source unit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MslSource {
    code: String,
    entry_point: String,
}

/// Compilation options that affect code generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CompilationOptions {
    fast_math: bool,
    language_version: (u32, u32),
    preprocessor_macros: Vec<(String, String)>,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self { fast_math: false, language_version: (3, 0), preprocessor_macros: Vec::new() }
    }
}

/// Cache key combining source hash and options hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    source_hash: u64,
    options_hash: u64,
}

impl PipelineKey {
    fn new(source: &MslSource, opts: &CompilationOptions) -> Self {
        Self { source_hash: hash_of(source), options_hash: hash_of(opts) }
    }
}

fn hash_of<T: Hash>(val: &T) -> u64 {
    let mut h = DefaultHasher::new();
    val.hash(&mut h);
    h.finish()
}

/// Result of compiling a shader source.
#[derive(Debug, Clone)]
struct CompilationResult {
    success: bool,
    errors: Vec<String>,
    warnings: Vec<String>,
    /// Opaque pipeline id when compilation succeeds.
    pipeline_id: Option<u64>,
}

impl CompilationResult {
    fn ok(pipeline_id: u64) -> Self {
        Self {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            pipeline_id: Some(pipeline_id),
        }
    }

    fn fail(errors: Vec<String>) -> Self {
        Self { success: false, errors, warnings: Vec::new(), pipeline_id: None }
    }

    fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }
}

/// Detected MSL function qualifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FunctionQualifier {
    Vertex,
    Fragment,
    Kernel,
    None,
}

/// Address space of a buffer parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressSpace {
    Device,
    Constant,
    Threadgroup,
    ThreadgroupImageblock,
    Thread,
}

/// Describes a single buffer parameter parsed from MSL source.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BufferParam {
    name: String,
    address_space: AddressSpace,
    binding_index: u32,
    is_pointer: bool,
}

/// Pipeline specialization constant.
#[derive(Debug, Clone, PartialEq)]
struct SpecializationConstant {
    name: String,
    index: u32,
    default_value: f64,
}

/// A compiled function inside a library.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LibraryFunction {
    name: String,
    qualifier: FunctionQualifier,
}

/// Shader variant descriptor for template generation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShaderVariant {
    base_name: String,
    dtype: String,
    workgroup_size: (u32, u32, u32),
    defines: Vec<(String, String)>,
}

/// Performance tuning for a compute pipeline.
#[derive(Debug, Clone, PartialEq)]
struct PerformanceConfig {
    thread_execution_width: u32,
    max_total_threads_per_threadgroup: u32,
    threadgroup_memory_bytes: u32,
    simd_group_size: u32,
}

// =========================================================================
// MSL source analysis helpers
// =========================================================================

/// Detect the function qualifier for a given entry point in MSL source.
fn detect_function_qualifier(source: &str, entry: &str) -> FunctionQualifier {
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.contains(entry) {
            if trimmed.starts_with("vertex ") {
                return FunctionQualifier::Vertex;
            }
            if trimmed.starts_with("fragment ") {
                return FunctionQualifier::Fragment;
            }
            if trimmed.starts_with("kernel ") {
                return FunctionQualifier::Kernel;
            }
        }
    }
    FunctionQualifier::None
}

/// Check whether the source declares threadgroup memory via `threadgroup`.
fn has_threadgroup_memory(source: &str) -> bool {
    source.lines().any(|l| {
        let t = l.trim();
        t.contains("threadgroup ") && !t.starts_with("//")
    })
}

/// Parse buffer parameters from a kernel signature (simplified).
fn parse_buffer_params(source: &str) -> Vec<BufferParam> {
    let mut params = Vec::new();
    for line in source.lines() {
        let t = line.trim();
        // Look for `[[buffer(N)]]`
        if let Some(idx_start) = t.find("[[buffer(") {
            let after = &t[idx_start + 9..];
            if let Some(idx_end) = after.find(")]]") {
                if let Ok(idx) = after[..idx_end].parse::<u32>() {
                    let addr = if t.contains("device ") {
                        AddressSpace::Device
                    } else if t.contains("constant ") {
                        AddressSpace::Constant
                    } else if t.contains("threadgroup ") {
                        AddressSpace::Threadgroup
                    } else {
                        AddressSpace::Thread
                    };
                    let is_ptr = t.contains('*');
                    let name = format!("param_{idx}");
                    params.push(BufferParam {
                        name,
                        address_space: addr,
                        binding_index: idx,
                        is_pointer: is_ptr,
                    });
                }
            }
        }
    }
    params
}

/// Validate that array sizes in source are within a max bound.
fn validate_array_sizes(source: &str, max_elements: usize) -> Vec<String> {
    let mut errors = Vec::new();
    for (i, line) in source.lines().enumerate() {
        if let Some(start) = line.find('[') {
            if let Some(end) = line[start + 1..].find(']') {
                let inner = line[start + 1..start + 1 + end].trim();
                if let Ok(n) = inner.parse::<usize>() {
                    if n > max_elements {
                        errors.push(format!(
                            "line {}: array size {n} exceeds max {max_elements}",
                            i + 1
                        ));
                    }
                }
            }
        }
    }
    errors
}

/// Detect address spaces used in source.
fn detect_address_spaces(source: &str) -> Vec<AddressSpace> {
    let mut found = Vec::new();
    if source.contains("device ") {
        found.push(AddressSpace::Device);
    }
    if source.contains("constant ") {
        found.push(AddressSpace::Constant);
    }
    if source.contains("threadgroup ") {
        found.push(AddressSpace::Threadgroup);
    }
    if source.contains("threadgroup_imageblock ") {
        found.push(AddressSpace::ThreadgroupImageblock);
    }
    if source.contains("thread ") {
        found.push(AddressSpace::Thread);
    }
    found
}

/// Validate that required keywords are present in a kernel source.
fn validate_kernel_source(source: &str) -> Vec<String> {
    let mut issues = Vec::new();
    if !source.contains("kernel ") {
        issues.push("missing 'kernel' qualifier".into());
    }
    if !source.contains("[[thread_position_in_grid]]")
        && !source.contains("[[thread_position_in_threadgroup]]")
        && !source.contains("[[threadgroup_position_in_grid]]")
    {
        issues.push("missing thread position attribute".into());
    }
    issues
}

// =========================================================================
// Pipeline cache
// =========================================================================

struct PipelineCache {
    entries: HashMap<PipelineKey, CompilationResult>,
    hits: u64,
    misses: u64,
}

impl PipelineCache {
    fn new() -> Self {
        Self { entries: HashMap::new(), hits: 0, misses: 0 }
    }

    fn get_or_compile(
        &mut self,
        source: &MslSource,
        opts: &CompilationOptions,
        compiler: impl Fn(&MslSource, &CompilationOptions) -> CompilationResult,
    ) -> &CompilationResult {
        let key = PipelineKey::new(source, opts);
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            self.misses += 1;
            let result = compiler(source, opts);
            self.entries.insert(key, result);
        }
        self.entries.get(&key).unwrap()
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

// =========================================================================
// Compilation logic
// =========================================================================

/// Simplified MSL compiler that validates source structure.
fn compile_msl(source: &MslSource, opts: &CompilationOptions) -> CompilationResult {
    let code = &source.code;
    let entry = &source.entry_point;

    // Syntax: balanced braces
    let opens = code.chars().filter(|&c| c == '{').count();
    let closes = code.chars().filter(|&c| c == '}').count();
    if opens != closes {
        return CompilationResult::fail(vec![format!(
            "unbalanced braces: {opens} open vs {closes} close"
        )]);
    }

    // Entry point must be present
    if !code.contains(entry) {
        return CompilationResult::fail(vec![format!("undefined function '{entry}'")]);
    }

    // Type checking: mismatched float/half casts
    let mut warnings = Vec::new();
    if code.contains("float(") && code.contains("half(") {
        warnings.push("mixed float/half casts — verify precision".into());
    }

    if opts.fast_math && code.contains("precise ") {
        warnings.push("fast_math enabled but source uses 'precise' qualifier".into());
    }

    // Language version check
    if opts.language_version < (2, 0) {
        return CompilationResult::fail(vec!["Metal language version < 2.0 not supported".into()]);
    }

    let pid = hash_of(&(code, entry, &opts.language_version));
    CompilationResult::ok(pid).with_warnings(warnings)
}

// =========================================================================
// Shader variant generation
// =========================================================================

fn generate_variant_source(template: &str, variant: &ShaderVariant) -> String {
    let mut out = String::new();
    // Emit defines
    for (k, v) in &variant.defines {
        out.push_str(&format!("#define {k} {v}\n"));
    }
    out.push_str(&format!(
        "#define WORKGROUP_X {}\n#define WORKGROUP_Y {}\n#define WORKGROUP_Z {}\n",
        variant.workgroup_size.0, variant.workgroup_size.1, variant.workgroup_size.2,
    ));
    out.push_str(&format!("#define DTYPE {}\n", variant.dtype));
    out.push_str(template);
    out
}

fn preprocess_conditionals(source: &str, defined: &[&str]) -> String {
    let mut out = Vec::new();
    let mut active_stack: Vec<bool> = Vec::new();

    for line in source.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("#ifdef ") {
            let sym = rest.trim();
            active_stack.push(defined.contains(&sym));
        } else if let Some(rest) = t.strip_prefix("#if ") {
            let sym = rest.trim();
            active_stack.push(defined.contains(&sym));
        } else if t == "#else" {
            if let Some(last) = active_stack.last_mut() {
                *last = !*last;
            }
        } else if t == "#endif" {
            active_stack.pop();
        } else if active_stack.iter().all(|&a| a) {
            out.push(line.to_string());
        }
    }
    out.join("\n")
}

// =========================================================================
// Function library
// =========================================================================

struct FunctionLibrary {
    functions: Vec<LibraryFunction>,
    source_units: Vec<MslSource>,
}

impl FunctionLibrary {
    fn new() -> Self {
        Self { functions: Vec::new(), source_units: Vec::new() }
    }

    fn add_source(&mut self, src: MslSource) {
        let qualifier = detect_function_qualifier(&src.code, &src.entry_point);
        self.functions.push(LibraryFunction { name: src.entry_point.clone(), qualifier });
        self.source_units.push(src);
    }

    fn lookup(&self, name: &str) -> Option<&LibraryFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    fn kernels(&self) -> Vec<&LibraryFunction> {
        self.functions.iter().filter(|f| f.qualifier == FunctionQualifier::Kernel).collect()
    }

    fn link_check(&self, caller: &str, callee: &str) -> bool {
        self.lookup(caller).is_some() && self.lookup(callee).is_some()
    }
}

// =========================================================================
// Performance helpers
// =========================================================================

const METAL_MAX_THREADGROUP_MEMORY: u32 = 32 * 1024;
const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;
const METAL_SIMD_WIDTH: u32 = 32;

fn validate_perf_config(cfg: &PerformanceConfig) -> Vec<String> {
    let mut errors = Vec::new();
    if cfg.thread_execution_width == 0
        || (cfg.thread_execution_width & (cfg.thread_execution_width - 1)) != 0
    {
        errors.push("thread_execution_width must be a power of 2".into());
    }
    if cfg.max_total_threads_per_threadgroup > METAL_MAX_THREADS_PER_THREADGROUP {
        errors.push(format!(
            "threads {} exceeds max {}",
            cfg.max_total_threads_per_threadgroup, METAL_MAX_THREADS_PER_THREADGROUP
        ));
    }
    if cfg.threadgroup_memory_bytes > METAL_MAX_THREADGROUP_MEMORY {
        errors.push(format!(
            "threadgroup memory {} exceeds max {}",
            cfg.threadgroup_memory_bytes, METAL_MAX_THREADGROUP_MEMORY
        ));
    }
    if cfg.simd_group_size != 0 && cfg.max_total_threads_per_threadgroup % cfg.simd_group_size != 0
    {
        errors.push("threads not divisible by SIMD group size".into());
    }
    errors
}

fn calculate_occupancy(threads_per_group: u32, threadgroup_mem: u32, max_groups: u32) -> f64 {
    if threads_per_group == 0 || max_groups == 0 {
        return 0.0;
    }
    let mem_limited_groups = if threadgroup_mem > 0 {
        METAL_MAX_THREADGROUP_MEMORY / threadgroup_mem
    } else {
        max_groups
    };
    let effective = mem_limited_groups.min(max_groups);
    let total_threads = effective * threads_per_group;
    let max_threads = max_groups * METAL_MAX_THREADS_PER_THREADGROUP;
    (total_threads as f64 / max_threads as f64).min(1.0)
}

fn optimal_thread_execution_width(data_elements: u32) -> u32 {
    // Prefer SIMD width, then fall back to smaller pow2.
    if data_elements >= METAL_SIMD_WIDTH {
        METAL_SIMD_WIDTH
    } else {
        data_elements.next_power_of_two().min(METAL_SIMD_WIDTH)
    }
}

// =========================================================================
// Tests — MSL Source Validation (21 tests)
// =========================================================================

mod msl_source_validation {
    use super::*;

    #[test]
    fn detect_kernel_qualifier() {
        let src = "kernel void my_kernel(device float* x [[buffer(0)]]) {}";
        assert_eq!(detect_function_qualifier(src, "my_kernel"), FunctionQualifier::Kernel);
    }

    #[test]
    fn detect_vertex_qualifier() {
        let src = "vertex float4 my_vs(uint vid [[vertex_id]]) { return float4(0); }";
        assert_eq!(detect_function_qualifier(src, "my_vs"), FunctionQualifier::Vertex);
    }

    #[test]
    fn detect_fragment_qualifier() {
        let src = "fragment float4 my_fs() { return float4(1); }";
        assert_eq!(detect_function_qualifier(src, "my_fs"), FunctionQualifier::Fragment);
    }

    #[test]
    fn detect_no_qualifier() {
        let src = "void helper_fn(float x) { }";
        assert_eq!(detect_function_qualifier(src, "helper_fn"), FunctionQualifier::None);
    }

    #[test]
    fn detect_qualifier_missing_entry() {
        let src = "kernel void other_fn() {}";
        assert_eq!(detect_function_qualifier(src, "not_here"), FunctionQualifier::None);
    }

    #[test]
    fn threadgroup_memory_present() {
        let src = "threadgroup float shared[256];";
        assert!(has_threadgroup_memory(src));
    }

    #[test]
    fn threadgroup_memory_absent() {
        let src = "device float* buf [[buffer(0)]];";
        assert!(!has_threadgroup_memory(src));
    }

    #[test]
    fn threadgroup_memory_in_comment_ignored() {
        let src = "// threadgroup float shared[256];";
        assert!(!has_threadgroup_memory(src));
    }

    #[test]
    fn parse_single_buffer_param() {
        let src = "device float* x [[buffer(0)]]";
        let params = parse_buffer_params(src);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].binding_index, 0);
        assert_eq!(params[0].address_space, AddressSpace::Device);
        assert!(params[0].is_pointer);
    }

    #[test]
    fn parse_constant_buffer_param() {
        let src = "constant Params& p [[buffer(3)]]";
        let params = parse_buffer_params(src);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].address_space, AddressSpace::Constant);
        assert_eq!(params[0].binding_index, 3);
    }

    #[test]
    fn parse_multiple_buffer_params() {
        let src = "\
            device float* input [[buffer(0)]],\n\
            device float* output [[buffer(1)]],\n\
            constant Params& p [[buffer(2)]]";
        let params = parse_buffer_params(src);
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].binding_index, 0);
        assert_eq!(params[1].binding_index, 1);
        assert_eq!(params[2].binding_index, 2);
    }

    #[test]
    fn detect_device_address_space() {
        let src = "device float* buf;";
        let spaces = detect_address_spaces(src);
        assert!(spaces.contains(&AddressSpace::Device));
    }

    #[test]
    fn detect_constant_address_space() {
        let src = "constant float4 val;";
        let spaces = detect_address_spaces(src);
        assert!(spaces.contains(&AddressSpace::Constant));
    }

    #[test]
    fn detect_multiple_address_spaces() {
        let src = "device float* a; constant int b; threadgroup float c;";
        let spaces = detect_address_spaces(src);
        assert!(spaces.contains(&AddressSpace::Device));
        assert!(spaces.contains(&AddressSpace::Constant));
        assert!(spaces.contains(&AddressSpace::Threadgroup));
    }

    #[test]
    fn array_size_within_bounds() {
        let src = "float data[128];";
        let errors = validate_array_sizes(src, 1024);
        assert!(errors.is_empty());
    }

    #[test]
    fn array_size_exceeds_bounds() {
        let src = "float data[2048];";
        let errors = validate_array_sizes(src, 1024);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("2048"));
    }

    #[test]
    fn multiple_array_sizes_mixed() {
        let src = "float a[100];\nfloat b[5000];\nfloat c[50];";
        let errors = validate_array_sizes(src, 1024);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("5000"));
    }

    #[test]
    fn validate_valid_kernel_source() {
        let src = "kernel void k(uint gid [[thread_position_in_grid]]) {}";
        let issues = validate_kernel_source(src);
        assert!(issues.is_empty());
    }

    #[test]
    fn validate_missing_kernel_qualifier() {
        let src = "void k(uint gid [[thread_position_in_grid]]) {}";
        let issues = validate_kernel_source(src);
        assert!(issues.iter().any(|i| i.contains("kernel")));
    }

    #[test]
    fn validate_missing_thread_position() {
        let src = "kernel void k() {}";
        let issues = validate_kernel_source(src);
        assert!(issues.iter().any(|i| i.contains("thread position")));
    }

    #[test]
    fn validate_threadgroup_position_accepted() {
        let src = "kernel void k(uint gid [[threadgroup_position_in_grid]]) {}";
        let issues = validate_kernel_source(src);
        assert!(!issues.iter().any(|i| i.contains("thread position")));
    }
}

// =========================================================================
// Tests — Pipeline State Object (22 tests)
// =========================================================================

mod pipeline_state_object {
    use super::*;

    fn sample_source(entry: &str) -> MslSource {
        MslSource {
            code: format!(
                "kernel void {entry}(\
                 device float* x [[buffer(0)]], \
                 uint gid [[thread_position_in_grid]]) {{ \
                 x[gid] = x[gid] * 2.0; }}"
            ),
            entry_point: entry.to_string(),
        }
    }

    #[test]
    fn compile_valid_source_succeeds() {
        let src = sample_source("scale");
        let result = compile_msl(&src, &CompilationOptions::default());
        assert!(result.success);
        assert!(result.pipeline_id.is_some());
    }

    #[test]
    fn compile_invalid_braces_fails() {
        let src = MslSource { code: "kernel void bad() {".into(), entry_point: "bad".into() };
        let result = compile_msl(&src, &CompilationOptions::default());
        assert!(!result.success);
        assert!(result.errors[0].contains("braces"));
    }

    #[test]
    fn compile_missing_entry_fails() {
        let src =
            MslSource { code: "kernel void other() {}".into(), entry_point: "missing_fn".into() };
        let result = compile_msl(&src, &CompilationOptions::default());
        assert!(!result.success);
        assert!(result.errors[0].contains("undefined"));
    }

    #[test]
    fn compile_old_language_version_fails() {
        let src = sample_source("k");
        let opts = CompilationOptions { language_version: (1, 2), ..Default::default() };
        let result = compile_msl(&src, &opts);
        assert!(!result.success);
        assert!(result.errors[0].contains("version"));
    }

    #[test]
    fn compile_v2_0_succeeds() {
        let src = sample_source("k");
        let opts = CompilationOptions { language_version: (2, 0), ..Default::default() };
        assert!(compile_msl(&src, &opts).success);
    }

    #[test]
    fn compile_v3_1_succeeds() {
        let src = sample_source("k");
        let opts = CompilationOptions { language_version: (3, 1), ..Default::default() };
        assert!(compile_msl(&src, &opts).success);
    }

    #[test]
    fn fast_math_with_precise_warns() {
        let src = MslSource {
            code: "kernel void k(uint gid [[thread_position_in_grid]]) \
                   { precise float x = 1.0; }"
                .into(),
            entry_point: "k".into(),
        };
        let opts = CompilationOptions { fast_math: true, ..Default::default() };
        let result = compile_msl(&src, &opts);
        assert!(result.success);
        assert!(!result.warnings.is_empty());
        assert!(result.warnings[0].contains("fast_math"));
    }

    #[test]
    fn mixed_float_half_warns() {
        let src = MslSource {
            code: "kernel void k(uint gid [[thread_position_in_grid]]) \
                   { float a = float(1); half b = half(a); }"
                .into(),
            entry_point: "k".into(),
        };
        let result = compile_msl(&src, &CompilationOptions::default());
        assert!(result.success);
        assert!(result.warnings.iter().any(|w| w.contains("precision")));
    }

    #[test]
    fn pipeline_key_deterministic() {
        let src = sample_source("k");
        let opts = CompilationOptions::default();
        let k1 = PipelineKey::new(&src, &opts);
        let k2 = PipelineKey::new(&src, &opts);
        assert_eq!(k1, k2);
    }

    #[test]
    fn pipeline_key_differs_for_different_source() {
        let s1 = sample_source("k1");
        let s2 = sample_source("k2");
        let opts = CompilationOptions::default();
        assert_ne!(PipelineKey::new(&s1, &opts), PipelineKey::new(&s2, &opts));
    }

    #[test]
    fn pipeline_key_differs_for_different_options() {
        let src = sample_source("k");
        let o1 = CompilationOptions::default();
        let o2 = CompilationOptions { fast_math: true, ..Default::default() };
        assert_ne!(PipelineKey::new(&src, &o1), PipelineKey::new(&src, &o2));
    }

    #[test]
    fn cache_miss_then_hit() {
        let mut cache = PipelineCache::new();
        let src = sample_source("k");
        let opts = CompilationOptions::default();
        cache.get_or_compile(&src, &opts, compile_msl);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);

        cache.get_or_compile(&src, &opts, compile_msl);
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn cache_hit_rate_zero_initially() {
        let cache = PipelineCache::new();
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn cache_hit_rate_correct_after_mixed() {
        let mut cache = PipelineCache::new();
        let src = sample_source("k");
        let opts = CompilationOptions::default();
        // 1 miss
        cache.get_or_compile(&src, &opts, compile_msl);
        // 3 hits
        for _ in 0..3 {
            cache.get_or_compile(&src, &opts, compile_msl);
        }
        assert!((cache.hit_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn cache_different_sources_separate() {
        let mut cache = PipelineCache::new();
        let opts = CompilationOptions::default();
        let s1 = sample_source("k1");
        let s2 = sample_source("k2");
        cache.get_or_compile(&s1, &opts, compile_msl);
        cache.get_or_compile(&s2, &opts, compile_msl);
        assert_eq!(cache.misses, 2);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn cache_same_source_different_opts_separate() {
        let mut cache = PipelineCache::new();
        let src = sample_source("k");
        let o1 = CompilationOptions::default();
        let o2 = CompilationOptions { fast_math: true, ..Default::default() };
        cache.get_or_compile(&src, &o1, compile_msl);
        cache.get_or_compile(&src, &o2, compile_msl);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn pipeline_id_stable_across_compilations() {
        let src = sample_source("k");
        let opts = CompilationOptions::default();
        let r1 = compile_msl(&src, &opts);
        let r2 = compile_msl(&src, &opts);
        assert_eq!(r1.pipeline_id, r2.pipeline_id);
    }

    #[test]
    fn pipeline_id_differs_for_different_source() {
        let s1 = sample_source("k1");
        let s2 = sample_source("k2");
        let opts = CompilationOptions::default();
        let r1 = compile_msl(&s1, &opts);
        let r2 = compile_msl(&s2, &opts);
        assert_ne!(r1.pipeline_id, r2.pipeline_id);
    }

    #[test]
    fn specialization_constant_defaults() {
        let c = SpecializationConstant { name: "TILE_SIZE".into(), index: 0, default_value: 8.0 };
        assert_eq!(c.default_value, 8.0);
    }

    #[test]
    fn specialization_constants_unique_indices() {
        let constants = vec![
            SpecializationConstant { name: "TILE_M".into(), index: 0, default_value: 8.0 },
            SpecializationConstant { name: "TILE_N".into(), index: 1, default_value: 8.0 },
            SpecializationConstant { name: "TILE_K".into(), index: 2, default_value: 16.0 },
        ];
        let mut indices: Vec<_> = constants.iter().map(|c| c.index).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), constants.len());
    }

    #[test]
    fn compile_result_no_errors_on_success() {
        let r = CompilationResult::ok(42);
        assert!(r.errors.is_empty());
        assert!(r.success);
    }

    #[test]
    fn compile_result_no_pipeline_on_failure() {
        let r = CompilationResult::fail(vec!["error".into()]);
        assert!(r.pipeline_id.is_none());
        assert!(!r.success);
    }
}

// =========================================================================
// Tests — Compilation Error Handling (17 tests)
// =========================================================================

mod compilation_error_handling {
    use super::*;

    #[test]
    fn syntax_error_unbalanced_open() {
        let src = MslSource { code: "kernel void k() { { }".into(), entry_point: "k".into() };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(!r.success);
        assert!(r.errors[0].contains("braces"));
    }

    #[test]
    fn syntax_error_unbalanced_close() {
        let src = MslSource { code: "kernel void k() { } }".into(), entry_point: "k".into() };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(!r.success);
    }

    #[test]
    fn syntax_error_empty_source() {
        let src = MslSource { code: String::new(), entry_point: "k".into() };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(!r.success);
        assert!(r.errors[0].contains("undefined"));
    }

    #[test]
    fn undefined_function_reference() {
        let src = MslSource {
            code: "kernel void real_fn() {}".into(),
            entry_point: "nonexistent".into(),
        };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(!r.success);
        assert!(r.errors[0].contains("undefined"));
        assert!(r.errors[0].contains("nonexistent"));
    }

    #[test]
    fn type_mismatch_warning_detected() {
        let src = MslSource {
            code: "kernel void k(uint gid [[thread_position_in_grid]]) \
                   { float a = float(1); half b = half(a); }"
                .into(),
            entry_point: "k".into(),
        };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(r.success);
        assert!(!r.warnings.is_empty());
    }

    #[test]
    fn graceful_degradation_preserves_error_list() {
        let src = MslSource { code: "kernel void k() {".into(), entry_point: "k".into() };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(!r.success);
        assert!(!r.errors.is_empty());
        assert!(r.pipeline_id.is_none());
    }

    #[test]
    fn multiple_errors_possible() {
        // Missing entry AND unbalanced
        let src = MslSource { code: "void other() {".into(), entry_point: "k".into() };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(!r.success);
        // At least one error (unbalanced braces is checked first)
        assert!(!r.errors.is_empty());
    }

    #[test]
    fn error_message_includes_counts() {
        let src = MslSource { code: "kernel void k() { { {".into(), entry_point: "k".into() };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(r.errors[0].contains("3") || r.errors[0].contains("open"));
    }

    #[test]
    fn warnings_preserved_on_success() {
        let r = CompilationResult::ok(1).with_warnings(vec!["warn1".into(), "warn2".into()]);
        assert!(r.success);
        assert_eq!(r.warnings.len(), 2);
    }

    #[test]
    fn no_warnings_by_default_on_simple_source() {
        let src = MslSource {
            code: "kernel void k(uint gid [[thread_position_in_grid]]) \
                   { }"
            .into(),
            entry_point: "k".into(),
        };
        let r = compile_msl(&src, &CompilationOptions::default());
        assert!(r.success);
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn fast_math_no_warning_without_precise() {
        let src = MslSource {
            code: "kernel void k(uint gid [[thread_position_in_grid]]) \
                   { float x = 1.0; }"
                .into(),
            entry_point: "k".into(),
        };
        let opts = CompilationOptions { fast_math: true, ..Default::default() };
        let r = compile_msl(&src, &opts);
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn version_boundary_1_9_fails() {
        let src = MslSource { code: "kernel void k() {}".into(), entry_point: "k".into() };
        let opts = CompilationOptions { language_version: (1, 9), ..Default::default() };
        assert!(!compile_msl(&src, &opts).success);
    }

    #[test]
    fn version_boundary_2_0_passes() {
        let src = MslSource { code: "kernel void k() {}".into(), entry_point: "k".into() };
        let opts = CompilationOptions { language_version: (2, 0), ..Default::default() };
        assert!(compile_msl(&src, &opts).success);
    }

    #[test]
    fn preprocessor_macros_in_options() {
        let opts = CompilationOptions {
            preprocessor_macros: vec![
                ("TILE_SIZE".into(), "16".into()),
                ("USE_FP16".into(), "1".into()),
            ],
            ..Default::default()
        };
        assert_eq!(opts.preprocessor_macros.len(), 2);
    }

    #[test]
    fn compilation_result_fail_has_no_pipeline() {
        let r = CompilationResult::fail(vec!["err".into()]);
        assert!(r.pipeline_id.is_none());
    }

    #[test]
    fn compilation_result_ok_has_pipeline() {
        let r = CompilationResult::ok(99);
        assert_eq!(r.pipeline_id, Some(99));
    }

    #[test]
    fn deeply_nested_braces_balanced() {
        let src =
            MslSource { code: "kernel void k() { { { { } } } }".into(), entry_point: "k".into() };
        assert!(compile_msl(&src, &CompilationOptions::default()).success);
    }
}

// =========================================================================
// Tests — Shader Variant Generation (22 tests)
// =========================================================================

mod shader_variant_generation {
    use super::*;

    const TEMPLATE: &str = "\
kernel void matmul(
    device DTYPE* a [[buffer(0)]],
    device DTYPE* b [[buffer(1)]],
    device DTYPE* c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // WORKGROUP_X, WORKGROUP_Y used
}
";

    #[test]
    fn generate_f32_variant() {
        let v = ShaderVariant {
            base_name: "matmul".into(),
            dtype: "float".into(),
            workgroup_size: (8, 8, 1),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("#define DTYPE float"));
    }

    #[test]
    fn generate_f16_variant() {
        let v = ShaderVariant {
            base_name: "matmul".into(),
            dtype: "half".into(),
            workgroup_size: (16, 16, 1),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("#define DTYPE half"));
    }

    #[test]
    fn generate_i8_variant() {
        let v = ShaderVariant {
            base_name: "matmul".into(),
            dtype: "char".into(),
            workgroup_size: (32, 1, 1),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("#define DTYPE char"));
    }

    #[test]
    fn generate_i2_variant() {
        let v = ShaderVariant {
            base_name: "matmul".into(),
            dtype: "uchar".into(),
            workgroup_size: (64, 1, 1),
            defines: vec![("BITS_PER_WEIGHT".into(), "2".into())],
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("#define BITS_PER_WEIGHT 2"));
        assert!(out.contains("#define DTYPE uchar"));
    }

    #[test]
    fn workgroup_size_in_output() {
        let v = ShaderVariant {
            base_name: "k".into(),
            dtype: "float".into(),
            workgroup_size: (4, 8, 2),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("#define WORKGROUP_X 4"));
        assert!(out.contains("#define WORKGROUP_Y 8"));
        assert!(out.contains("#define WORKGROUP_Z 2"));
    }

    #[test]
    fn custom_defines_prepended() {
        let v = ShaderVariant {
            base_name: "k".into(),
            dtype: "float".into(),
            workgroup_size: (1, 1, 1),
            defines: vec![("MY_FLAG".into(), "42".into())],
        };
        let out = generate_variant_source(TEMPLATE, &v);
        let flag_pos = out.find("#define MY_FLAG").unwrap();
        let template_pos = out.find("kernel void").unwrap();
        assert!(flag_pos < template_pos);
    }

    #[test]
    fn preprocess_ifdef_defined() {
        let src = "#ifdef USE_FP16\nhalf x;\n#endif";
        let out = preprocess_conditionals(src, &["USE_FP16"]);
        assert!(out.contains("half x;"));
    }

    #[test]
    fn preprocess_ifdef_not_defined() {
        let src = "#ifdef USE_FP16\nhalf x;\n#endif";
        let out = preprocess_conditionals(src, &[]);
        assert!(!out.contains("half x;"));
    }

    #[test]
    fn preprocess_ifdef_else_branch() {
        let src = "#ifdef USE_FP16\nhalf x;\n#else\nfloat x;\n#endif";
        let out = preprocess_conditionals(src, &[]);
        assert!(!out.contains("half x;"));
        assert!(out.contains("float x;"));
    }

    #[test]
    fn preprocess_ifdef_else_defined() {
        let src = "#ifdef USE_FP16\nhalf x;\n#else\nfloat x;\n#endif";
        let out = preprocess_conditionals(src, &["USE_FP16"]);
        assert!(out.contains("half x;"));
        assert!(!out.contains("float x;"));
    }

    #[test]
    fn preprocess_nested_ifdef() {
        let src = "\
#ifdef OUTER
#ifdef INNER
int both;
#endif
int outer_only;
#endif";
        let out = preprocess_conditionals(src, &["OUTER", "INNER"]);
        assert!(out.contains("int both;"));
        assert!(out.contains("int outer_only;"));
    }

    #[test]
    fn preprocess_nested_inner_undefined() {
        let src = "\
#ifdef OUTER
#ifdef INNER
int both;
#endif
int outer_only;
#endif";
        let out = preprocess_conditionals(src, &["OUTER"]);
        assert!(!out.contains("int both;"));
        assert!(out.contains("int outer_only;"));
    }

    #[test]
    fn preprocess_if_directive() {
        let src = "#if ENABLE_SIMD\nsimdgroup_float8x8 tile;\n#endif";
        let out = preprocess_conditionals(src, &["ENABLE_SIMD"]);
        assert!(out.contains("simdgroup_float8x8"));
    }

    #[test]
    fn preprocess_if_not_defined() {
        let src = "#if ENABLE_SIMD\nsimdgroup_float8x8 tile;\n#endif";
        let out = preprocess_conditionals(src, &[]);
        assert!(!out.contains("simdgroup_float8x8"));
    }

    #[test]
    fn multiple_dtype_variants() {
        let dtypes = ["float", "half", "char", "uchar"];
        for dtype in &dtypes {
            let v = ShaderVariant {
                base_name: "k".into(),
                dtype: dtype.to_string(),
                workgroup_size: (1, 1, 1),
                defines: Vec::new(),
            };
            let out = generate_variant_source(TEMPLATE, &v);
            assert!(out.contains(&format!("#define DTYPE {dtype}")), "variant for {dtype} missing");
        }
    }

    #[test]
    fn variant_preserves_template_body() {
        let v = ShaderVariant {
            base_name: "matmul".into(),
            dtype: "float".into(),
            workgroup_size: (1, 1, 1),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("kernel void matmul"));
        assert!(out.contains("[[buffer(0)]]"));
    }

    #[test]
    fn workgroup_size_1d() {
        let v = ShaderVariant {
            base_name: "k".into(),
            dtype: "float".into(),
            workgroup_size: (256, 1, 1),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("WORKGROUP_X 256"));
    }

    #[test]
    fn workgroup_size_3d() {
        let v = ShaderVariant {
            base_name: "k".into(),
            dtype: "float".into(),
            workgroup_size: (4, 4, 4),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("WORKGROUP_X 4"));
        assert!(out.contains("WORKGROUP_Y 4"));
        assert!(out.contains("WORKGROUP_Z 4"));
    }

    #[test]
    fn multiple_custom_defines() {
        let v = ShaderVariant {
            base_name: "k".into(),
            dtype: "float".into(),
            workgroup_size: (1, 1, 1),
            defines: vec![
                ("A".into(), "1".into()),
                ("B".into(), "2".into()),
                ("C".into(), "3".into()),
            ],
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("#define A 1"));
        assert!(out.contains("#define B 2"));
        assert!(out.contains("#define C 3"));
    }

    #[test]
    fn template_substitution_empty_defines() {
        let v = ShaderVariant {
            base_name: "k".into(),
            dtype: "float".into(),
            workgroup_size: (8, 8, 1),
            defines: Vec::new(),
        };
        let out = generate_variant_source(TEMPLATE, &v);
        // Must not have extraneous defines beyond workgroup/dtype
        let define_count = out.lines().filter(|l| l.starts_with("#define")).count();
        assert_eq!(define_count, 4); // X, Y, Z, DTYPE
    }

    #[test]
    fn preprocess_preserves_non_directive_lines() {
        let src = "int a;\n#ifdef X\nint b;\n#endif\nint c;";
        let out = preprocess_conditionals(src, &[]);
        assert!(out.contains("int a;"));
        assert!(out.contains("int c;"));
        assert!(!out.contains("int b;"));
    }

    #[test]
    fn generate_variant_for_bf16() {
        let v = ShaderVariant {
            base_name: "k".into(),
            dtype: "bfloat".into(),
            workgroup_size: (32, 1, 1),
            defines: vec![("USE_BF16".into(), "1".into())],
        };
        let out = generate_variant_source(TEMPLATE, &v);
        assert!(out.contains("#define DTYPE bfloat"));
        assert!(out.contains("#define USE_BF16 1"));
    }
}

// =========================================================================
// Tests — Function Library (16 tests)
// =========================================================================

mod function_library_tests {
    use super::*;

    fn make_kernel(name: &str) -> MslSource {
        MslSource {
            code: format!("kernel void {name}(uint gid [[thread_position_in_grid]]) {{}}"),
            entry_point: name.to_string(),
        }
    }

    fn make_vertex(name: &str) -> MslSource {
        MslSource {
            code: format!(
                "vertex float4 {name}(uint vid [[vertex_id]]) {{ \
                 return float4(0); }}"
            ),
            entry_point: name.to_string(),
        }
    }

    fn make_helper(name: &str) -> MslSource {
        MslSource {
            code: format!("float {name}(float x) {{ return x * x; }}"),
            entry_point: name.to_string(),
        }
    }

    #[test]
    fn empty_library() {
        let lib = FunctionLibrary::new();
        assert!(lib.functions.is_empty());
        assert!(lib.source_units.is_empty());
    }

    #[test]
    fn add_single_kernel() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("matmul"));
        assert_eq!(lib.functions.len(), 1);
        assert_eq!(lib.functions[0].qualifier, FunctionQualifier::Kernel);
    }

    #[test]
    fn add_vertex_function() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_vertex("vs_main"));
        assert_eq!(lib.functions[0].qualifier, FunctionQualifier::Vertex);
    }

    #[test]
    fn add_helper_has_no_qualifier() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_helper("square"));
        assert_eq!(lib.functions[0].qualifier, FunctionQualifier::None);
    }

    #[test]
    fn lookup_existing_function() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("k"));
        assert!(lib.lookup("k").is_some());
    }

    #[test]
    fn lookup_missing_function() {
        let lib = FunctionLibrary::new();
        assert!(lib.lookup("nonexistent").is_none());
    }

    #[test]
    fn kernels_filter() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("k1"));
        lib.add_source(make_vertex("vs"));
        lib.add_source(make_kernel("k2"));
        let kernels = lib.kernels();
        assert_eq!(kernels.len(), 2);
        assert!(kernels.iter().all(|f| f.qualifier == FunctionQualifier::Kernel));
    }

    #[test]
    fn link_check_both_present() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("caller"));
        lib.add_source(make_helper("callee"));
        assert!(lib.link_check("caller", "callee"));
    }

    #[test]
    fn link_check_caller_missing() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_helper("callee"));
        assert!(!lib.link_check("missing", "callee"));
    }

    #[test]
    fn link_check_callee_missing() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("caller"));
        assert!(!lib.link_check("caller", "missing"));
    }

    #[test]
    fn multiple_source_units_tracked() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("a"));
        lib.add_source(make_kernel("b"));
        lib.add_source(make_kernel("c"));
        assert_eq!(lib.source_units.len(), 3);
    }

    #[test]
    fn library_preserves_insertion_order() {
        let mut lib = FunctionLibrary::new();
        let names = ["alpha", "beta", "gamma"];
        for &n in &names {
            lib.add_source(make_kernel(n));
        }
        for (i, &expected) in names.iter().enumerate() {
            assert_eq!(lib.functions[i].name, expected);
        }
    }

    #[test]
    fn shared_utility_extraction() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_helper("relu"));
        lib.add_source(make_helper("gelu"));
        lib.add_source(make_kernel("forward"));
        assert!(lib.link_check("forward", "relu"));
        assert!(lib.link_check("forward", "gelu"));
    }

    #[test]
    fn function_name_preserved() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("my_special_kernel_v2"));
        let f = lib.lookup("my_special_kernel_v2").unwrap();
        assert_eq!(f.name, "my_special_kernel_v2");
    }

    #[test]
    fn library_with_mixed_qualifiers() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("compute"));
        lib.add_source(make_vertex("vert"));
        lib.add_source(make_helper("util"));
        let qualifiers: Vec<_> = lib.functions.iter().map(|f| f.qualifier).collect();
        assert_eq!(
            qualifiers,
            vec![FunctionQualifier::Kernel, FunctionQualifier::Vertex, FunctionQualifier::None,]
        );
    }

    #[test]
    fn link_check_self_reference() {
        let mut lib = FunctionLibrary::new();
        lib.add_source(make_kernel("k"));
        assert!(lib.link_check("k", "k"));
    }
}

// =========================================================================
// Tests — Performance Configuration (17 tests)
// =========================================================================

mod performance_configuration {
    use super::*;

    #[test]
    fn valid_config_no_errors() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 256,
            threadgroup_memory_bytes: 4096,
            simd_group_size: 32,
        };
        assert!(validate_perf_config(&cfg).is_empty());
    }

    #[test]
    fn zero_execution_width_rejected() {
        let cfg = PerformanceConfig {
            thread_execution_width: 0,
            max_total_threads_per_threadgroup: 256,
            threadgroup_memory_bytes: 0,
            simd_group_size: 0,
        };
        let errors = validate_perf_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("power of 2")));
    }

    #[test]
    fn non_power_of_two_width_rejected() {
        let cfg = PerformanceConfig {
            thread_execution_width: 48,
            max_total_threads_per_threadgroup: 256,
            threadgroup_memory_bytes: 0,
            simd_group_size: 0,
        };
        let errors = validate_perf_config(&cfg);
        assert!(!errors.is_empty());
    }

    #[test]
    fn power_of_two_widths_accepted() {
        for w in [1, 2, 4, 8, 16, 32, 64] {
            let cfg = PerformanceConfig {
                thread_execution_width: w,
                max_total_threads_per_threadgroup: 256,
                threadgroup_memory_bytes: 0,
                simd_group_size: 0,
            };
            assert!(
                validate_perf_config(&cfg).iter().all(|e| !e.contains("power of 2")),
                "width {w} should be accepted"
            );
        }
    }

    #[test]
    fn exceeding_max_threads_rejected() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 2048,
            threadgroup_memory_bytes: 0,
            simd_group_size: 0,
        };
        let errors = validate_perf_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("exceeds max")));
    }

    #[test]
    fn max_threads_boundary_accepted() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 1024,
            threadgroup_memory_bytes: 0,
            simd_group_size: 0,
        };
        assert!(!validate_perf_config(&cfg).iter().any(|e| e.contains("exceeds max")));
    }

    #[test]
    fn exceeding_threadgroup_memory_rejected() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 256,
            threadgroup_memory_bytes: 64 * 1024,
            simd_group_size: 0,
        };
        let errors = validate_perf_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("threadgroup memory")));
    }

    #[test]
    fn threadgroup_memory_boundary_accepted() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 256,
            threadgroup_memory_bytes: 32 * 1024,
            simd_group_size: 0,
        };
        assert!(!validate_perf_config(&cfg).iter().any(|e| e.contains("threadgroup memory")));
    }

    #[test]
    fn simd_group_indivisible_rejected() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 100,
            threadgroup_memory_bytes: 0,
            simd_group_size: 32,
        };
        let errors = validate_perf_config(&cfg);
        assert!(errors.iter().any(|e| e.contains("SIMD group size")));
    }

    #[test]
    fn simd_group_divisible_accepted() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 128,
            threadgroup_memory_bytes: 0,
            simd_group_size: 32,
        };
        assert!(!validate_perf_config(&cfg).iter().any(|e| e.contains("SIMD")));
    }

    #[test]
    fn simd_group_zero_skips_check() {
        let cfg = PerformanceConfig {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 100,
            threadgroup_memory_bytes: 0,
            simd_group_size: 0,
        };
        assert!(!validate_perf_config(&cfg).iter().any(|e| e.contains("SIMD")));
    }

    #[test]
    fn occupancy_zero_threads() {
        assert_eq!(calculate_occupancy(0, 0, 4), 0.0);
    }

    #[test]
    fn occupancy_zero_max_groups() {
        assert_eq!(calculate_occupancy(256, 0, 0), 0.0);
    }

    #[test]
    fn occupancy_no_memory_limit() {
        let occ = calculate_occupancy(256, 0, 4);
        assert!(occ > 0.0 && occ <= 1.0);
    }

    #[test]
    fn occupancy_memory_limited() {
        // 16 KB per group → max 2 groups out of 4
        let occ = calculate_occupancy(256, 16 * 1024, 4);
        assert!(occ > 0.0);
        assert!(occ < 1.0);
    }

    #[test]
    fn optimal_width_for_large_data() {
        assert_eq!(optimal_thread_execution_width(1024), METAL_SIMD_WIDTH);
    }

    #[test]
    fn optimal_width_for_small_data() {
        assert_eq!(optimal_thread_execution_width(3), 4);
        assert_eq!(optimal_thread_execution_width(1), 1);
        assert_eq!(optimal_thread_execution_width(7), 8);
        assert_eq!(optimal_thread_execution_width(16), 16);
        assert_eq!(optimal_thread_execution_width(32), 32);
    }
}
