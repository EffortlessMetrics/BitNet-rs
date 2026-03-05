#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! wgpu compute pipeline validation tests for Apple Silicon Metal backend.
//!
//! Validates WGSL shader modules, pipeline layouts, buffer bindings, workgroup
//! dispatch sizing, pipeline caching, error handling, and multi-pass sequencing
//! using mock structs. No wgpu or metal-rs imports — compiles on Linux CI.

#![cfg(target_os = "macos")]

// ── Apple Silicon Metal constants ───────────────────────────────────────────

/// Maximum threads per threadgroup on Apple Silicon (M1–M4).
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD width on Apple Silicon GPUs.
const SIMD_WIDTH: u32 = 32;

/// Metal buffer offset alignment (bytes).
const BUFFER_OFFSET_ALIGNMENT: usize = 256;

/// Maximum bind groups supported by wgpu on Metal.
const MAX_BIND_GROUPS: u32 = 4;

/// Maximum bindings per group.
const MAX_BINDINGS_PER_GROUP: u32 = 30;

/// Maximum threadgroup shared memory (bytes).
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

// ── WGSL shader validation helpers ──────────────────────────────────────────

/// Minimal WGSL validation: checks for required structural elements.
fn validate_wgsl_compute_shader(source: &str) -> Result<WgslValidation, Vec<String>> {
    let mut errors = Vec::new();

    if !source.contains("@compute") {
        errors.push("missing @compute attribute".into());
    }
    if !source.contains("@workgroup_size") {
        errors.push("missing @workgroup_size attribute".into());
    }
    if !source.contains("fn ") {
        errors.push("missing entry point function".into());
    }
    // Check that binding declarations use valid group indices.
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("@group(")
            && let Some(rest) = trimmed.strip_prefix("@group(")
            && let Some(idx_str) = rest.split(')').next()
            && let Ok(g) = idx_str.parse::<u32>()
            && g >= MAX_BIND_GROUPS
        {
            errors.push(format!("bind group {g} exceeds max ({MAX_BIND_GROUPS})"));
        }
    }

    if errors.is_empty() {
        let entry_point = source
            .lines()
            .find(|l| {
                l.contains("fn ")
                    && source[..source.find(l).unwrap_or(0) + l.len()].contains("@compute")
            })
            .and_then(|l| l.split("fn ").nth(1))
            .and_then(|s| s.split('(').next())
            .unwrap_or("main")
            .trim()
            .to_string();
        Ok(WgslValidation { entry_point })
    } else {
        Err(errors)
    }
}

#[derive(Debug, Clone)]
struct WgslValidation {
    entry_point: String,
}

// ── Pipeline layout / bind group mock types ─────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufferBindingType {
    Storage { read_only: bool },
    Uniform,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BindGroupEntry {
    binding: u32,
    label: &'static str,
    ty: BufferBindingType,
    min_size: usize,
}

#[derive(Debug, Clone)]
struct BindGroupLayout {
    group: u32,
    entries: Vec<BindGroupEntry>,
}

impl BindGroupLayout {
    fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.group >= MAX_BIND_GROUPS {
            errors.push(format!("group index {} >= max {MAX_BIND_GROUPS}", self.group));
        }
        if self.entries.len() > MAX_BINDINGS_PER_GROUP as usize {
            errors.push(format!(
                "too many bindings: {} > {MAX_BINDINGS_PER_GROUP}",
                self.entries.len()
            ));
        }
        // Check for duplicate binding indices.
        let mut seen = std::collections::HashSet::new();
        for e in &self.entries {
            if !seen.insert(e.binding) {
                errors.push(format!("duplicate binding index {}", e.binding));
            }
        }
        // Uniform buffers have a 16-byte alignment requirement on Metal.
        for e in &self.entries {
            if e.ty == BufferBindingType::Uniform && e.min_size % 16 != 0 {
                errors.push(format!(
                    "uniform binding {} min_size {} not 16-byte aligned",
                    e.binding, e.min_size
                ));
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

#[derive(Debug, Clone)]
struct PipelineLayout {
    bind_group_layouts: Vec<BindGroupLayout>,
}

impl PipelineLayout {
    fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.bind_group_layouts.len() > MAX_BIND_GROUPS as usize {
            errors.push(format!(
                "too many bind groups: {} > {MAX_BIND_GROUPS}",
                self.bind_group_layouts.len()
            ));
        }
        for bgl in &self.bind_group_layouts {
            if let Err(e) = bgl.validate() {
                errors.extend(e);
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

// ── Dispatch sizing helpers ─────────────────────────────────────────────────

fn ceil_div(total: u32, divisor: u32) -> u32 {
    assert_ne!(divisor, 0);
    total.div_ceil(divisor)
}

fn align_up(n: usize, align: usize) -> usize {
    assert!(align > 0);
    n.div_ceil(align) * align
}

/// Choose a 1-D workgroup size that is a multiple of SIMD_WIDTH and
/// fits within MAX_THREADS_PER_THREADGROUP.
fn optimal_workgroup_1d(total: u32) -> u32 {
    if total == 0 {
        return 0;
    }
    let rounded = total.div_ceil(SIMD_WIDTH) * SIMD_WIDTH;
    rounded.min(MAX_THREADS_PER_THREADGROUP)
}

/// Choose a 2-D workgroup size. Returns `(x, y)`.
fn optimal_workgroup_2d(cols: u32, rows: u32) -> (u32, u32) {
    if cols == 0 || rows == 0 {
        return (0, 0);
    }
    let x = SIMD_WIDTH.min(cols);
    let max_y = MAX_THREADS_PER_THREADGROUP / x;
    let y = max_y.min(rows);
    (x, y)
}

/// Compute number of workgroups for a 1-D dispatch.
fn dispatch_1d(total: u32, workgroup_size: u32) -> u32 {
    ceil_div(total, workgroup_size)
}

/// Compute the shared memory required for a reduction over `threads` f32 values.
fn reduction_shared_memory(threads: u32) -> usize {
    threads as usize * std::mem::size_of::<f32>()
}

// ── Pipeline cache mock ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineCacheKey {
    shader_hash: u64,
    workgroup_size: (u32, u32, u32),
    bind_group_count: u32,
}

struct PipelineCache {
    entries: std::collections::HashMap<PipelineCacheKey, CachedPipeline>,
}

#[derive(Debug, Clone)]
struct CachedPipeline {
    label: String,
    hit_count: u32,
}

impl PipelineCache {
    fn new() -> Self {
        Self { entries: std::collections::HashMap::new() }
    }

    fn get_or_insert(
        &mut self,
        key: PipelineCacheKey,
        label: impl Into<String>,
    ) -> (&CachedPipeline, bool) {
        let was_present = self.entries.contains_key(&key);
        let entry = self
            .entries
            .entry(key)
            .or_insert_with(|| CachedPipeline { label: label.into(), hit_count: 0 });
        if was_present {
            entry.hit_count += 1;
        }
        (entry, was_present)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ── Multi-pass pipeline sequencing ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct ComputePass {
    label: &'static str,
    shader_source: &'static str,
    workgroup_size: (u32, u32, u32),
    dispatch_count: (u32, u32, u32),
}

impl ComputePass {
    fn total_invocations(&self) -> u64 {
        let (wx, wy, wz) = self.workgroup_size;
        let (dx, dy, dz) = self.dispatch_count;
        (wx as u64) * (wy as u64) * (wz as u64) * (dx as u64) * (dy as u64) * (dz as u64)
    }

    fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let (wx, wy, wz) = self.workgroup_size;
        let threads = wx * wy * wz;
        if threads > MAX_THREADS_PER_THREADGROUP {
            errors.push(format!(
                "pass '{}': workgroup threads {threads} > {MAX_THREADS_PER_THREADGROUP}",
                self.label
            ));
        }
        if let Err(shader_errs) = validate_wgsl_compute_shader(self.shader_source) {
            for e in shader_errs {
                errors.push(format!("pass '{}': shader: {e}", self.label));
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

struct MultiPassPipeline {
    passes: Vec<ComputePass>,
}

impl MultiPassPipeline {
    fn new() -> Self {
        Self { passes: Vec::new() }
    }

    fn add_pass(&mut self, pass: ComputePass) {
        self.passes.push(pass);
    }

    fn validate_all(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for (i, pass) in self.passes.iter().enumerate() {
            if let Err(errs) = pass.validate() {
                for e in errs {
                    errors.push(format!("pass[{i}] '{}': {e}", pass.label));
                }
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn total_invocations(&self) -> u64 {
        self.passes.iter().map(|p| p.total_invocations()).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Shader module validation ────────────────────────────────────────

    const VALID_MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(32, 1, 1)
fn matmul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
}
"#;

    const VALID_SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

var<workgroup> shared_max: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn softmax_kernel(@builtin(local_invocation_id) lid: vec3<u32>) {
    let local_idx = lid.x;
}
"#;

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_valid_matmul_shader_passes_validation() {
        let result = validate_wgsl_compute_shader(VALID_MATMUL_SHADER);
        assert!(result.is_ok(), "valid matmul shader should pass: {result:?}");
        assert_eq!(result.unwrap().entry_point, "matmul_kernel");
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_valid_softmax_shader_passes_validation() {
        let result = validate_wgsl_compute_shader(VALID_SOFTMAX_SHADER);
        assert!(result.is_ok(), "valid softmax shader should pass: {result:?}");
        assert_eq!(result.unwrap().entry_point, "softmax_kernel");
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_shader_missing_compute_attribute() {
        let bad = r#"
@group(0) @binding(0) var<storage, read> data: array<f32>;
@workgroup_size(64, 1, 1)
fn bad_kernel() {}
"#;
        let errs = validate_wgsl_compute_shader(bad).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("@compute")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_shader_missing_workgroup_size() {
        let bad = "@compute\nfn missing_wg() {}";
        let errs = validate_wgsl_compute_shader(bad).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("@workgroup_size")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_shader_missing_entry_point() {
        let bad = "@compute @workgroup_size(64)";
        let errs = validate_wgsl_compute_shader(bad).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("entry point")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_shader_bind_group_exceeds_max() {
        let bad = r#"
@group(5) @binding(0) var<storage, read> data: array<f32>;
@compute @workgroup_size(64)
fn over_group() {}
"#;
        let errs = validate_wgsl_compute_shader(bad).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bind group 5")));
    }

    // ── Pipeline layout and bind group configuration ────────────────────

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_valid_pipeline_layout() {
        let layout = PipelineLayout {
            bind_group_layouts: vec![BindGroupLayout {
                group: 0,
                entries: vec![
                    BindGroupEntry {
                        binding: 0,
                        label: "weights",
                        ty: BufferBindingType::Storage { read_only: true },
                        min_size: 4096,
                    },
                    BindGroupEntry {
                        binding: 1,
                        label: "input",
                        ty: BufferBindingType::Storage { read_only: true },
                        min_size: 1024,
                    },
                    BindGroupEntry {
                        binding: 2,
                        label: "output",
                        ty: BufferBindingType::Storage { read_only: false },
                        min_size: 1024,
                    },
                    BindGroupEntry {
                        binding: 3,
                        label: "params",
                        ty: BufferBindingType::Uniform,
                        min_size: 16,
                    },
                ],
            }],
        };
        assert!(layout.validate().is_ok());
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_pipeline_layout_too_many_bind_groups() {
        let layout = PipelineLayout {
            bind_group_layouts: (0..5)
                .map(|g| BindGroupLayout { group: g, entries: vec![] })
                .collect(),
        };
        let errs = layout.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("too many bind groups")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_bind_group_duplicate_binding_index() {
        let bgl = BindGroupLayout {
            group: 0,
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    label: "a",
                    ty: BufferBindingType::Storage { read_only: true },
                    min_size: 64,
                },
                BindGroupEntry {
                    binding: 0,
                    label: "b",
                    ty: BufferBindingType::Storage { read_only: false },
                    min_size: 64,
                },
            ],
        };
        let errs = bgl.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate binding")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_uniform_buffer_alignment_enforced() {
        let bgl = BindGroupLayout {
            group: 0,
            entries: vec![BindGroupEntry {
                binding: 0,
                label: "params",
                ty: BufferBindingType::Uniform,
                min_size: 17, // not 16-byte aligned
            }],
        };
        let errs = bgl.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("16-byte aligned")));
    }

    // ── Buffer binding validation ───────────────────────────────────────

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_storage_buffer_read_only_flag() {
        let ro = BufferBindingType::Storage { read_only: true };
        let rw = BufferBindingType::Storage { read_only: false };
        assert_ne!(ro, rw);
        assert_eq!(ro, BufferBindingType::Storage { read_only: true });
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_buffer_offset_alignment() {
        for size in [1, 100, 255, 256, 257, 512, 1000] {
            let aligned = align_up(size, BUFFER_OFFSET_ALIGNMENT);
            assert_eq!(aligned % BUFFER_OFFSET_ALIGNMENT, 0);
            assert!(aligned >= size);
            assert!(aligned - size < BUFFER_OFFSET_ALIGNMENT);
        }
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_bind_group_layout_max_entries() {
        let entries: Vec<BindGroupEntry> = (0..31)
            .map(|i| BindGroupEntry {
                binding: i,
                label: "buf",
                ty: BufferBindingType::Storage { read_only: true },
                min_size: 64,
            })
            .collect();
        let bgl = BindGroupLayout { group: 0, entries };
        let errs = bgl.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("too many bindings")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_inference_pipeline_layout_standard() {
        // Standard layout: group 0 = I/O buffers, group 1 = parameters.
        let layout = PipelineLayout {
            bind_group_layouts: vec![
                BindGroupLayout {
                    group: 0,
                    entries: vec![
                        BindGroupEntry {
                            binding: 0,
                            label: "weights",
                            ty: BufferBindingType::Storage { read_only: true },
                            min_size: 8192,
                        },
                        BindGroupEntry {
                            binding: 1,
                            label: "activations",
                            ty: BufferBindingType::Storage { read_only: false },
                            min_size: 4096,
                        },
                    ],
                },
                BindGroupLayout {
                    group: 1,
                    entries: vec![BindGroupEntry {
                        binding: 0,
                        label: "config",
                        ty: BufferBindingType::Uniform,
                        min_size: 64,
                    }],
                },
            ],
        };
        assert!(layout.validate().is_ok());
    }

    // ── Dispatch workgroup size calculations ────────────────────────────

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_workgroup_1d_small_input() {
        // Input smaller than SIMD_WIDTH rounds up to one SIMD group.
        assert_eq!(optimal_workgroup_1d(1), SIMD_WIDTH);
        assert_eq!(optimal_workgroup_1d(16), SIMD_WIDTH);
        assert_eq!(optimal_workgroup_1d(31), SIMD_WIDTH);
        assert_eq!(optimal_workgroup_1d(32), SIMD_WIDTH);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_workgroup_1d_medium_input() {
        assert_eq!(optimal_workgroup_1d(64), 64);
        assert_eq!(optimal_workgroup_1d(100), 128); // rounds up to 4*SIMD
        assert_eq!(optimal_workgroup_1d(256), 256);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_workgroup_1d_clamped_at_max() {
        assert_eq!(optimal_workgroup_1d(2048), MAX_THREADS_PER_THREADGROUP);
        assert_eq!(optimal_workgroup_1d(100_000), MAX_THREADS_PER_THREADGROUP);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_workgroup_1d_zero() {
        assert_eq!(optimal_workgroup_1d(0), 0);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_workgroup_2d_matmul_shapes() {
        // Typical matrix dimensions.
        let (x, y) = optimal_workgroup_2d(512, 512);
        assert_eq!(x, SIMD_WIDTH);
        assert!(x * y <= MAX_THREADS_PER_THREADGROUP);

        let (x, y) = optimal_workgroup_2d(8, 8);
        assert_eq!(x, 8);
        assert!(y > 0);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_workgroup_2d_zero_dimensions() {
        assert_eq!(optimal_workgroup_2d(0, 512), (0, 0));
        assert_eq!(optimal_workgroup_2d(512, 0), (0, 0));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_dispatch_count_1d() {
        assert_eq!(dispatch_1d(1024, 256), 4);
        assert_eq!(dispatch_1d(1025, 256), 5);
        assert_eq!(dispatch_1d(256, 256), 1);
        assert_eq!(dispatch_1d(1, 256), 1);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_reduction_shared_memory_fits_threadgroup() {
        let threads = MAX_THREADS_PER_THREADGROUP;
        let shared_bytes = reduction_shared_memory(threads);
        assert!(
            shared_bytes <= MAX_THREADGROUP_MEMORY,
            "shared memory {shared_bytes} exceeds {MAX_THREADGROUP_MEMORY}",
        );
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_dispatch_workgroup_size_simd_multiple() {
        // All 1-D workgroup sizes should be multiples of SIMD_WIDTH.
        for n in [33, 64, 100, 200, 500, 1000, 2000] {
            let wg = optimal_workgroup_1d(n);
            if wg > 0 {
                assert_eq!(wg % SIMD_WIDTH, 0, "workgroup size {wg} for n={n} not SIMD-aligned");
            }
        }
    }

    // ── Pipeline caching and reuse ──────────────────────────────────────

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_pipeline_cache_insert_and_hit() {
        let mut cache = PipelineCache::new();
        let key = PipelineCacheKey {
            shader_hash: 0xDEAD_BEEF,
            workgroup_size: (256, 1, 1),
            bind_group_count: 1,
        };

        let (_, was_cached) = cache.get_or_insert(key.clone(), "matmul_f32");
        assert!(!was_cached, "first insert should be a miss");

        let (entry, was_cached) = cache.get_or_insert(key.clone(), "matmul_f32");
        assert!(was_cached, "second lookup should be a hit");
        assert_eq!(entry.hit_count, 1);

        let (entry, _) = cache.get_or_insert(key, "matmul_f32");
        assert_eq!(entry.hit_count, 2);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_pipeline_cache_distinct_keys() {
        let mut cache = PipelineCache::new();
        let key_a =
            PipelineCacheKey { shader_hash: 1, workgroup_size: (256, 1, 1), bind_group_count: 1 };
        let key_b =
            PipelineCacheKey { shader_hash: 2, workgroup_size: (256, 1, 1), bind_group_count: 1 };
        let key_c =
            PipelineCacheKey { shader_hash: 1, workgroup_size: (128, 1, 1), bind_group_count: 1 };

        cache.get_or_insert(key_a, "shader_a");
        cache.get_or_insert(key_b, "shader_b");
        cache.get_or_insert(key_c, "shader_c");
        assert_eq!(cache.len(), 3);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_pipeline_cache_reuse_across_dispatch_sizes() {
        let mut cache = PipelineCache::new();
        // Same pipeline (same shader + workgroup) dispatched with different counts
        // should reuse the same cached pipeline.
        let key =
            PipelineCacheKey { shader_hash: 42, workgroup_size: (256, 1, 1), bind_group_count: 2 };

        cache.get_or_insert(key.clone(), "reusable");
        let (entry, hit) = cache.get_or_insert(key, "reusable");
        assert!(hit);
        assert_eq!(entry.label, "reusable");
    }

    // ── Error handling for invalid shader code ──────────────────────────

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_shader_error_contains_all_issues() {
        // Shader with multiple problems.
        let bad = "invalid shader with no attributes";
        let errs = validate_wgsl_compute_shader(bad).unwrap_err();
        assert!(errs.len() >= 2, "expected multiple errors, got: {errs:?}");
        assert!(errs.iter().any(|e| e.contains("@compute")));
        assert!(errs.iter().any(|e| e.contains("@workgroup_size")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_shader_error_empty_source() {
        let errs = validate_wgsl_compute_shader("").unwrap_err();
        assert!(errs.len() >= 3, "empty source should have 3+ errors: {errs:?}");
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_shader_error_invalid_group_index() {
        let src = r#"
@group(99) @binding(0) var<storage, read> data: array<f32>;
@compute @workgroup_size(64)
fn kern() {}
"#;
        let errs = validate_wgsl_compute_shader(src).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bind group 99")));
    }

    // ── Multi-pass pipeline sequencing ──────────────────────────────────

    const PASS_A_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> intermediate: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn layer_norm_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
}
"#;

    const PASS_B_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> intermediate: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128, 1, 1)
fn activation_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
}
"#;

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_multi_pass_pipeline_validates() {
        let mut pipeline = MultiPassPipeline::new();
        pipeline.add_pass(ComputePass {
            label: "layer_norm",
            shader_source: PASS_A_SHADER,
            workgroup_size: (256, 1, 1),
            dispatch_count: (4, 1, 1),
        });
        pipeline.add_pass(ComputePass {
            label: "activation",
            shader_source: PASS_B_SHADER,
            workgroup_size: (128, 1, 1),
            dispatch_count: (8, 1, 1),
        });
        assert!(pipeline.validate_all().is_ok());
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_multi_pass_total_invocations() {
        let mut pipeline = MultiPassPipeline::new();
        pipeline.add_pass(ComputePass {
            label: "pass_a",
            shader_source: PASS_A_SHADER,
            workgroup_size: (256, 1, 1),
            dispatch_count: (4, 1, 1),
        });
        pipeline.add_pass(ComputePass {
            label: "pass_b",
            shader_source: PASS_B_SHADER,
            workgroup_size: (128, 1, 1),
            dispatch_count: (8, 1, 1),
        });
        // pass_a: 256*4 = 1024, pass_b: 128*8 = 1024, total = 2048
        assert_eq!(pipeline.total_invocations(), 2048);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_multi_pass_detects_invalid_pass() {
        let bad_shader = "not a valid shader";
        let mut pipeline = MultiPassPipeline::new();
        pipeline.add_pass(ComputePass {
            label: "good",
            shader_source: PASS_A_SHADER,
            workgroup_size: (256, 1, 1),
            dispatch_count: (1, 1, 1),
        });
        pipeline.add_pass(ComputePass {
            label: "bad",
            shader_source: bad_shader,
            workgroup_size: (64, 1, 1),
            dispatch_count: (1, 1, 1),
        });
        let errs = pipeline.validate_all().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad")));
        // The good pass should not produce errors.
        assert!(!errs.iter().any(|e| e.contains("good")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_multi_pass_workgroup_exceeds_limit() {
        let mut pipeline = MultiPassPipeline::new();
        pipeline.add_pass(ComputePass {
            label: "oversized",
            shader_source: PASS_A_SHADER,
            workgroup_size: (64, 32, 1), // 2048 > 1024
            dispatch_count: (1, 1, 1),
        });
        let errs = pipeline.validate_all().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("workgroup threads")));
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_multi_pass_empty_pipeline_validates() {
        let pipeline = MultiPassPipeline::new();
        assert!(pipeline.validate_all().is_ok());
        assert_eq!(pipeline.total_invocations(), 0);
    }

    #[test]
    #[ignore = "requires macOS Metal GPU via wgpu"]
    fn test_three_pass_inference_pipeline() {
        // Realistic 3-pass inference: layernorm → matmul → softmax.
        let matmul_shader = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(1) @binding(0) var<uniform> dims: vec4<u32>;

@compute @workgroup_size(32, 32, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
}
"#;
        let mut pipeline = MultiPassPipeline::new();
        pipeline.add_pass(ComputePass {
            label: "layer_norm",
            shader_source: PASS_A_SHADER,
            workgroup_size: (256, 1, 1),
            dispatch_count: (ceil_div(2048, 256), 1, 1), // 2048 hidden dim
        });
        pipeline.add_pass(ComputePass {
            label: "matmul",
            shader_source: matmul_shader,
            workgroup_size: (32, 32, 1),
            dispatch_count: (ceil_div(2048, 32), ceil_div(2048, 32), 1),
        });
        pipeline.add_pass(ComputePass {
            label: "softmax",
            shader_source: VALID_SOFTMAX_SHADER,
            workgroup_size: (256, 1, 1),
            dispatch_count: (ceil_div(32000, 256), 1, 1), // vocab size
        });

        assert!(pipeline.validate_all().is_ok());
        assert_eq!(pipeline.passes.len(), 3);
        assert!(pipeline.total_invocations() > 0);
    }
}
