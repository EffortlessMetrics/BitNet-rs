//! Comprehensive Metal shader validation tests.
//!
//! These tests validate MSL shader correctness via source-level analysis —
//! no GPU hardware required. They verify syntax patterns, kernel signatures,
//! buffer bindings, thread dispatch constraints, numerical precision, and
//! alignment requirements for all 6 kernel types.

use bitnet_metal::{MetalKernelSource, kernel_function_names, kernel_source};

// ── Helper utilities ────────────────────────────────────────────────

/// Maximum threads per threadgroup on Apple Silicon (M1–M4).
const APPLE_SILICON_MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Required Metal buffer alignment in bytes.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Extract the full function signature (up to the opening brace) for a
/// `kernel void` function by name.
fn extract_kernel_signature<'a>(source: &'a str, fn_name: &str) -> &'a str {
    let pattern = format!("kernel void {fn_name}(");
    let start = source
        .find(&pattern)
        .unwrap_or_else(|| panic!("kernel void {fn_name}( not found in source"));
    let rest = &source[start..];
    let brace = rest.find('{').unwrap_or(rest.len());
    &rest[..brace]
}

/// Extract the function body (between the first `{` and its matching `}`)
/// for a `kernel void` function by name.
fn extract_kernel_body<'a>(source: &'a str, fn_name: &str) -> &'a str {
    let pattern = format!("kernel void {fn_name}(");
    let start = source
        .find(&pattern)
        .unwrap_or_else(|| panic!("kernel void {fn_name}( not found in source"));
    let rest = &source[start..];
    let brace_open = rest.find('{').expect("missing opening brace") + 1;
    let body = &rest[brace_open..];
    // Find the matching closing brace (depth tracking).
    let mut depth: u32 = 1;
    let mut end = 0;
    for (i, ch) in body.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    &body[..end]
}

/// Collect `[[buffer(N)]]` indices from a signature string.
fn collect_buffer_indices(signature: &str) -> Vec<u32> {
    let mut indices = Vec::new();
    for part in signature.split("[[buffer(") {
        if let Some(end) = part.find(")]]") {
            if let Ok(idx) = part[..end].parse::<u32>() {
                indices.push(idx);
            }
        }
    }
    indices
}

/// Parse a `constant uint … = <value>;` declaration and return the value.
fn parse_constant_uint(source: &str, name: &str) -> Option<u32> {
    let needle = format!("constant uint {name} = ");
    if let Some(start) = source.find(&needle) {
        let rest = &source[start + needle.len()..];
        if let Some(end) = rest.find(';') {
            return rest[..end].trim().parse().ok();
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════
// 1. Shader Syntax Validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn syntax_all_kernels_have_kernel_void_qualifier() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        for name in kernel_function_names(*kernel) {
            let pattern = format!("kernel void {name}(");
            assert!(src.contains(&pattern), "{kernel:?}/{name}: missing `kernel void` qualifier");
        }
    }
}

#[test]
fn syntax_all_kernels_include_metal_stdlib() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("#include <metal_stdlib>"),
            "{kernel:?}: missing #include <metal_stdlib>"
        );
    }
}

#[test]
fn syntax_all_kernels_use_metal_namespace() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("using namespace metal;"),
            "{kernel:?}: missing `using namespace metal;`"
        );
    }
}

#[test]
fn syntax_no_unbalanced_braces() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        let opens = src.chars().filter(|&c| c == '{').count();
        let closes = src.chars().filter(|&c| c == '}').count();
        assert_eq!(opens, closes, "{kernel:?}: unbalanced braces (open={opens}, close={closes})");
    }
}

#[test]
fn syntax_no_unbalanced_parentheses() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        let opens = src.chars().filter(|&c| c == '(').count();
        let closes = src.chars().filter(|&c| c == ')').count();
        assert_eq!(
            opens, closes,
            "{kernel:?}: unbalanced parentheses (open={opens}, close={closes})"
        );
    }
}

#[test]
fn syntax_all_kernels_have_thread_position_attribute() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("thread_position_in_grid") || src.contains("thread_index_in_threadgroup"),
            "{kernel:?}: missing thread position attribute"
        );
    }
}

#[test]
fn syntax_matmul_has_both_naive_and_tiled() {
    let src = kernel_source(MetalKernelSource::Matmul);
    assert!(src.contains("kernel void matmul("), "missing naive matmul");
    assert!(src.contains("kernel void matmul_tiled("), "missing tiled matmul");
}

#[test]
fn syntax_elementwise_has_all_six_ops() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let ops = ["add", "mul", "silu", "gelu", "silu_mul", "scalar_mul"];
    for op in &ops {
        let pattern = format!("kernel void {op}(");
        assert!(src.contains(&pattern), "elementwise: missing kernel `{op}`");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Kernel Argument Consistency (buffer bindings + threadgroup params)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bindings_sequential_from_zero_all_kernels() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        for name in kernel_function_names(*kernel) {
            let sig = extract_kernel_signature(src, name);
            let indices = collect_buffer_indices(sig);
            assert!(!indices.is_empty(), "{kernel:?}/{name}: no buffer bindings");
            for (i, idx) in indices.iter().enumerate() {
                assert_eq!(
                    *idx, i as u32,
                    "{kernel:?}/{name}: buffer({i}) expected, got buffer({idx})"
                );
            }
        }
    }
}

#[test]
fn bindings_matmul_naive_has_four_buffers() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let sig = extract_kernel_signature(src, "matmul");
    let indices = collect_buffer_indices(sig);
    assert_eq!(indices.len(), 4, "matmul: expected 4 buffers (a, b, c, dims)");
}

#[test]
fn bindings_matmul_tiled_has_four_buffers() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let sig = extract_kernel_signature(src, "matmul_tiled");
    let indices = collect_buffer_indices(sig);
    assert_eq!(indices.len(), 4, "matmul_tiled: expected 4 buffers (a, b, c, dims)");
}

#[test]
fn bindings_softmax_has_three_buffers() {
    let src = kernel_source(MetalKernelSource::Softmax);
    let sig = extract_kernel_signature(src, "softmax");
    let indices = collect_buffer_indices(sig);
    assert_eq!(indices.len(), 3, "softmax: expected 3 buffers (input, output, dims)");
}

#[test]
fn bindings_rmsnorm_has_five_buffers() {
    let src = kernel_source(MetalKernelSource::RmsNorm);
    let sig = extract_kernel_signature(src, "rmsnorm");
    let indices = collect_buffer_indices(sig);
    assert_eq!(indices.len(), 5, "rmsnorm: expected 5 buffers (input, weight, output, dims, eps)");
}

#[test]
fn bindings_rope_has_five_buffers() {
    let src = kernel_source(MetalKernelSource::Rope);
    let sig = extract_kernel_signature(src, "rope");
    let indices = collect_buffer_indices(sig);
    assert_eq!(indices.len(), 5, "rope: expected 5 buffers (input, cos, sin, dims, offset)");
}

#[test]
fn bindings_rope_build_tables_has_four_buffers() {
    let src = kernel_source(MetalKernelSource::Rope);
    let sig = extract_kernel_signature(src, "rope_build_tables");
    let indices = collect_buffer_indices(sig);
    assert_eq!(indices.len(), 4, "rope_build_tables: expected 4 buffers (cos, sin, dims, theta)");
}

#[test]
fn bindings_attention_scores_has_six_buffers() {
    let src = kernel_source(MetalKernelSource::Attention);
    let sig = extract_kernel_signature(src, "attention_scores");
    let indices = collect_buffer_indices(sig);
    assert_eq!(
        indices.len(),
        6,
        "attention_scores: expected 6 buffers (q, k, scores, dims, head_dim, causal)"
    );
}

#[test]
fn bindings_attention_weighted_sum_has_five_buffers() {
    let src = kernel_source(MetalKernelSource::Attention);
    let sig = extract_kernel_signature(src, "attention_weighted_sum");
    let indices = collect_buffer_indices(sig);
    assert_eq!(
        indices.len(),
        5,
        "attention_weighted_sum: expected 5 buffers (scores, v, output, dims, head_dim)"
    );
}

#[test]
fn bindings_elementwise_binary_ops_have_four_buffers() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    for name in &["add", "mul", "silu_mul"] {
        let sig = extract_kernel_signature(src, name);
        let indices = collect_buffer_indices(sig);
        assert_eq!(indices.len(), 4, "elementwise/{name}: expected 4 buffers");
    }
}

#[test]
fn bindings_elementwise_unary_ops_have_three_buffers() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    for name in &["silu", "gelu"] {
        let sig = extract_kernel_signature(src, name);
        let indices = collect_buffer_indices(sig);
        assert_eq!(
            indices.len(),
            3,
            "elementwise/{name}: expected 3 buffers (input, output, count)"
        );
    }
}

#[test]
fn bindings_scalar_mul_has_four_buffers() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let sig = extract_kernel_signature(src, "scalar_mul");
    let indices = collect_buffer_indices(sig);
    assert_eq!(indices.len(), 4, "scalar_mul: expected 4 buffers (input, output, scalar, count)");
}

#[test]
fn threadgroup_params_tiled_matmul_uses_threadgroup_position() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let sig = extract_kernel_signature(src, "matmul_tiled");
    assert!(
        sig.contains("thread_position_in_threadgroup"),
        "matmul_tiled: needs thread_position_in_threadgroup for tiling"
    );
    assert!(
        sig.contains("threadgroup_position_in_grid"),
        "matmul_tiled: needs threadgroup_position_in_grid for tiling"
    );
}

#[test]
fn threadgroup_params_reduction_kernels_use_thread_index() {
    for kernel in &[MetalKernelSource::Softmax, MetalKernelSource::RmsNorm] {
        let src = kernel_source(*kernel);
        assert!(
            src.contains("thread_index_in_threadgroup"),
            "{kernel:?}: reduction kernel needs thread_index_in_threadgroup"
        );
        assert!(
            src.contains("threads_per_threadgroup"),
            "{kernel:?}: reduction kernel needs threads_per_threadgroup"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Numerical Precision Patterns (fp16/fp32 handling)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn precision_matmul_accumulates_in_fp32() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let body = extract_kernel_body(src, "matmul");
    assert!(body.contains("float sum"), "matmul: should accumulate in fp32 for precision");
}

#[test]
fn precision_matmul_tiled_accumulates_in_fp32() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let body = extract_kernel_body(src, "matmul_tiled");
    assert!(body.contains("float sum"), "matmul_tiled: should accumulate in fp32 for precision");
}

#[test]
fn precision_softmax_uses_fp32_accumulator() {
    let src = kernel_source(MetalKernelSource::Softmax);
    let body = extract_kernel_body(src, "softmax");
    assert!(
        body.contains("float local_max") && body.contains("float local_sum"),
        "softmax: reductions should use fp32 accumulators"
    );
}

#[test]
fn precision_softmax_subtracts_max_for_stability() {
    let src = kernel_source(MetalKernelSource::Softmax);
    let body = extract_kernel_body(src, "softmax");
    assert!(body.contains("- row_max"), "softmax: must subtract max for numerical stability");
}

#[test]
fn precision_softmax_initialises_max_to_neg_infinity() {
    let src = kernel_source(MetalKernelSource::Softmax);
    let body = extract_kernel_body(src, "softmax");
    assert!(body.contains("-INFINITY"), "softmax: local_max should init to -INFINITY");
}

#[test]
fn precision_rmsnorm_uses_fp32_accumulator() {
    let src = kernel_source(MetalKernelSource::RmsNorm);
    let body = extract_kernel_body(src, "rmsnorm");
    assert!(body.contains("float local_sum_sq"), "rmsnorm: sum of squares should use fp32");
}

#[test]
fn precision_rmsnorm_uses_rsqrt() {
    let src = kernel_source(MetalKernelSource::RmsNorm);
    let body = extract_kernel_body(src, "rmsnorm");
    assert!(body.contains("rsqrt("), "rmsnorm: should use rsqrt for inverse square root");
}

#[test]
fn precision_rmsnorm_adds_epsilon() {
    let src = kernel_source(MetalKernelSource::RmsNorm);
    let body = extract_kernel_body(src, "rmsnorm");
    assert!(body.contains("+ eps"), "rmsnorm: must add epsilon to prevent division by zero");
}

#[test]
fn precision_attention_uses_rsqrt_scaling() {
    let src = kernel_source(MetalKernelSource::Attention);
    let body = extract_kernel_body(src, "attention_scores");
    assert!(
        body.contains("rsqrt(float(head_dim))"),
        "attention_scores: should scale by 1/sqrt(head_dim)"
    );
}

#[test]
fn precision_attention_dot_product_in_fp32() {
    let src = kernel_source(MetalKernelSource::Attention);
    let body = extract_kernel_body(src, "attention_scores");
    assert!(body.contains("float dot"), "attention_scores: dot product should accumulate in fp32");
}

#[test]
fn precision_gelu_uses_correct_constants() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let body = extract_kernel_body(src, "gelu");
    assert!(body.contains("0.7978845608"), "gelu: sqrt(2/pi) constant should be ~0.7978845608");
    assert!(body.contains("0.044715"), "gelu: cubic coefficient should be 0.044715");
}

#[test]
fn precision_silu_sigmoid_formula() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let body = extract_kernel_body(src, "silu");
    assert!(body.contains("1.0f + exp(-x)"), "silu: should use x / (1 + exp(-x)) formulation");
}

#[test]
fn precision_rope_uses_cos_sin_pair() {
    let src = kernel_source(MetalKernelSource::Rope);
    let body = extract_kernel_body(src, "rope");
    assert!(
        body.contains("cos_val") && body.contains("sin_val"),
        "rope: must use both cos and sin for rotation"
    );
}

#[test]
fn precision_rope_build_tables_uses_pow_for_frequencies() {
    let src = kernel_source(MetalKernelSource::Rope);
    let body = extract_kernel_body(src, "rope_build_tables");
    assert!(
        body.contains("pow(theta_base"),
        "rope_build_tables: should use pow(theta_base, ...) for frequencies"
    );
}

#[test]
fn precision_float_literals_use_f_suffix() {
    // Metal best practice: fp32 literals should use `f` suffix to avoid
    // implicit double promotion.
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        if src.contains("0.0f") || src.contains("1.0f") {
            // At least some literals use the suffix — good.
            continue;
        }
        // Kernels that have numeric literals should use the f suffix.
        // Elementwise uses constants, so every kernel should have at least one.
        if src.contains("= 0.0;") || src.contains("= 1.0;") {
            panic!(
                "{kernel:?}: float literals should use `f` suffix (e.g. `0.0f`) \
                 to avoid implicit double promotion"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Thread Dispatch Validation (workgroup size constraints)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn dispatch_threadgroup_sizes_within_apple_silicon_limit() {
    // All declared constant threadgroup sizes must be ≤ 1024.
    let threadgroup_constants = [
        (MetalKernelSource::Matmul, "TILE_SIZE"),
        (MetalKernelSource::Softmax, "SOFTMAX_THREADGROUP_SIZE"),
        (MetalKernelSource::RmsNorm, "RMSNORM_THREADGROUP_SIZE"),
        (MetalKernelSource::Attention, "ATTN_THREADGROUP_SIZE"),
    ];

    for (kernel, constant_name) in &threadgroup_constants {
        let src = kernel_source(*kernel);
        if let Some(value) = parse_constant_uint(src, constant_name) {
            assert!(
                value <= APPLE_SILICON_MAX_THREADS_PER_THREADGROUP,
                "{kernel:?}/{constant_name} = {value} exceeds Apple Silicon max ({APPLE_SILICON_MAX_THREADS_PER_THREADGROUP})"
            );
        }
    }
}

#[test]
fn dispatch_tile_size_is_power_of_two() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let tile_size =
        parse_constant_uint(src, "TILE_SIZE").expect("TILE_SIZE constant not found in matmul");
    assert!(
        tile_size.is_power_of_two(),
        "TILE_SIZE ({tile_size}) should be a power of 2 for efficient tiling"
    );
}

#[test]
fn dispatch_tile_size_squared_within_limit() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let tile_size =
        parse_constant_uint(src, "TILE_SIZE").expect("TILE_SIZE constant not found in matmul");
    let threads = tile_size * tile_size;
    assert!(
        threads <= APPLE_SILICON_MAX_THREADS_PER_THREADGROUP,
        "TILE_SIZE²={threads} exceeds Apple Silicon max ({APPLE_SILICON_MAX_THREADS_PER_THREADGROUP})"
    );
}

#[test]
fn dispatch_reduction_threadgroup_sizes_are_power_of_two() {
    let cases = [
        (MetalKernelSource::Softmax, "SOFTMAX_THREADGROUP_SIZE"),
        (MetalKernelSource::RmsNorm, "RMSNORM_THREADGROUP_SIZE"),
        (MetalKernelSource::Attention, "ATTN_THREADGROUP_SIZE"),
    ];
    for (kernel, name) in &cases {
        let src = kernel_source(*kernel);
        if let Some(value) = parse_constant_uint(src, name) {
            assert!(
                value.is_power_of_two(),
                "{kernel:?}/{name} = {value}: must be power-of-2 for parallel reduction"
            );
        }
    }
}

#[test]
fn dispatch_all_kernels_have_bounds_check() {
    // Every kernel function should guard against out-of-bounds threads,
    // either via early return or conditional write.
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        for name in kernel_function_names(*kernel) {
            let body = extract_kernel_body(src, name);
            let has_early_return = body.contains("return;");
            let has_conditional_write = body.contains("if (row < M") || body.contains("if (col <");
            assert!(
                has_early_return || has_conditional_write,
                "{kernel:?}/{name}: should have bounds check (early return or conditional write)"
            );
        }
    }
}

#[test]
fn dispatch_matmul_naive_uses_2d_grid() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let sig = extract_kernel_signature(src, "matmul");
    assert!(sig.contains("uint2 gid"), "matmul: should use uint2 for 2D thread dispatch");
}

#[test]
fn dispatch_rope_uses_3d_grid() {
    let src = kernel_source(MetalKernelSource::Rope);
    let sig = extract_kernel_signature(src, "rope");
    assert!(
        sig.contains("uint3 gid"),
        "rope: should use uint3 for 3D (batch × seq × dim) dispatch"
    );
}

#[test]
fn dispatch_attention_scores_uses_3d_grid() {
    let src = kernel_source(MetalKernelSource::Attention);
    let sig = extract_kernel_signature(src, "attention_scores");
    assert!(sig.contains("uint3 gid"), "attention_scores: should use uint3 for multi-dim dispatch");
}

#[test]
fn dispatch_elementwise_uses_1d_grid() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    for name in &["add", "mul", "silu", "gelu", "silu_mul", "scalar_mul"] {
        let sig = extract_kernel_signature(src, name);
        assert!(
            sig.contains("uint gid") && !sig.contains("uint2 gid") && !sig.contains("uint3 gid"),
            "elementwise/{name}: should use scalar uint for 1D dispatch"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Buffer Alignment Checks (256-byte alignment for Metal)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn alignment_metal_buffer_alignment_constant_correct() {
    // Verify our test constant matches the Metal spec.
    assert_eq!(METAL_BUFFER_ALIGNMENT, 256, "Metal buffer alignment should be 256 bytes");
}

#[test]
fn alignment_threadgroup_shared_memory_sizes_are_aligned() {
    // Shared memory arrays declared inside kernels should have sizes that
    // are multiples of the element type width (4 bytes for float).
    // We check that declared `threadgroup float name[N]` sizes are
    // powers of two (common best practice for reductions).
    for kernel in
        &[MetalKernelSource::Matmul, MetalKernelSource::Softmax, MetalKernelSource::RmsNorm]
    {
        let src = kernel_source(*kernel);
        for segment in src.split("threadgroup float") {
            // Look for array declarations like `name[256]` or `name[16 * 16]`.
            if let Some(bracket_start) = segment.find('[') {
                if let Some(bracket_end) = segment[bracket_start..].find(']') {
                    let inner = &segment[bracket_start + 1..bracket_start + bracket_end];
                    // Evaluate simple expressions: plain number or N * N.
                    let value = if inner.contains('*') {
                        let parts: Vec<&str> = inner.split('*').collect();
                        if parts.len() == 2 {
                            let a = parts[0].trim().parse::<u32>().unwrap_or(0);
                            let b = parts[1].trim().parse::<u32>().unwrap_or(0);
                            a * b
                        } else {
                            0
                        }
                    } else {
                        inner.trim().parse::<u32>().unwrap_or(0)
                    };
                    if value > 0 {
                        let byte_size = value as usize * std::mem::size_of::<f32>();
                        assert!(
                            byte_size % 16 == 0,
                            "{kernel:?}: threadgroup array of {value} floats ({byte_size} bytes) \
                             should be 16-byte aligned"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn alignment_tiled_matmul_tile_buffers_match_tile_size() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let tile_size = parse_constant_uint(src, "TILE_SIZE").expect("TILE_SIZE not found");

    // The tiled matmul should declare shared arrays of TILE_SIZE * TILE_SIZE.
    let body = extract_kernel_body(src, "matmul_tiled");
    let expected_size = tile_size * tile_size;
    let pattern = format!("[{} * {}]", tile_size, tile_size);
    let alt_pattern = format!("[{}]", expected_size);
    assert!(
        body.contains(&pattern) || body.contains(&alt_pattern),
        "matmul_tiled: threadgroup arrays should be [{tile_size} * {tile_size}] = [{expected_size}]"
    );
}

#[test]
fn alignment_constant_buffer_types_are_reference() {
    // Metal constant buffers should use `constant T&` (reference) for scalar
    // uniforms to ensure proper alignment and avoid copies.
    let cases = [
        (MetalKernelSource::Matmul, "matmul", "constant uint3&"),
        (MetalKernelSource::Softmax, "softmax", "constant uint2&"),
        (MetalKernelSource::RmsNorm, "rmsnorm", "constant uint2&"),
        (MetalKernelSource::RmsNorm, "rmsnorm", "constant float&"),
        (MetalKernelSource::Rope, "rope", "constant uint3&"),
        (MetalKernelSource::Attention, "attention_scores", "constant uint4&"),
    ];
    for (kernel, fn_name, expected_ref) in &cases {
        let src = kernel_source(*kernel);
        let sig = extract_kernel_signature(src, fn_name);
        assert!(
            sig.contains(expected_ref),
            "{kernel:?}/{fn_name}: should use `{expected_ref}` for constant buffer binding"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Per-Kernel Structure and Correctness
// ═══════════════════════════════════════════════════════════════════

// ── matmul ──────────────────────────────────────────────────────────

#[test]
fn matmul_naive_iterates_over_k_dimension() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let body = extract_kernel_body(src, "matmul");
    assert!(body.contains("i < K"), "matmul: inner loop should iterate over K dimension");
}

#[test]
fn matmul_tiled_uses_threadgroup_barrier() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let body = extract_kernel_body(src, "matmul_tiled");
    let barrier_count = body.matches("threadgroup_barrier").count();
    assert!(
        barrier_count >= 2,
        "matmul_tiled: needs ≥2 barriers (after load, after compute); found {barrier_count}"
    );
}

#[test]
fn matmul_tiled_uses_mem_threadgroup_flag() {
    let src = kernel_source(MetalKernelSource::Matmul);
    let body = extract_kernel_body(src, "matmul_tiled");
    assert!(
        body.contains("mem_flags::mem_threadgroup"),
        "matmul_tiled: barrier should specify mem_flags::mem_threadgroup"
    );
}

// ── softmax ─────────────────────────────────────────────────────────

#[test]
fn softmax_has_three_phase_structure() {
    let src = kernel_source(MetalKernelSource::Softmax);
    let body = extract_kernel_body(src, "softmax");
    // Phase 1: max, Phase 2: sum, Phase 3: normalize
    assert!(body.contains("row_max"), "softmax: missing phase 1 (max)");
    assert!(body.contains("row_sum"), "softmax: missing phase 2 (sum)");
    assert!(body.contains("inv_sum"), "softmax: missing phase 3 (normalize)");
}

#[test]
fn softmax_uses_multiplicative_inverse() {
    let src = kernel_source(MetalKernelSource::Softmax);
    let body = extract_kernel_body(src, "softmax");
    assert!(
        body.contains("1.0f / row_sum"),
        "softmax: should use multiplicative inverse for division"
    );
}

// ── rmsnorm ─────────────────────────────────────────────────────────

#[test]
fn rmsnorm_applies_weight_scaling() {
    let src = kernel_source(MetalKernelSource::RmsNorm);
    let body = extract_kernel_body(src, "rmsnorm");
    assert!(body.contains("* weight["), "rmsnorm: output should be scaled by weight vector");
}

#[test]
fn rmsnorm_divides_by_hidden_dim() {
    let src = kernel_source(MetalKernelSource::RmsNorm);
    let body = extract_kernel_body(src, "rmsnorm");
    assert!(
        body.contains("/ float(hidden_dim)"),
        "rmsnorm: mean of squares needs division by hidden_dim"
    );
}

// ── rope ────────────────────────────────────────────────────────────

#[test]
fn rope_applies_rotation_matrix() {
    let src = kernel_source(MetalKernelSource::Rope);
    let body = extract_kernel_body(src, "rope");
    // Rotation: x_re' = x_re*cos - x_im*sin, x_im' = x_re*sin + x_im*cos
    assert!(
        body.contains("cos_val - x_im") || body.contains("cos_val -x_im"),
        "rope: should apply cos component of rotation"
    );
    assert!(
        body.contains("sin_val + x_im") || body.contains("sin_val +x_im"),
        "rope: should apply sin component of rotation"
    );
}

#[test]
fn rope_operates_on_paired_elements() {
    let src = kernel_source(MetalKernelSource::Rope);
    let body = extract_kernel_body(src, "rope");
    assert!(body.contains("half_dim"), "rope: should split head_dim in half for paired rotation");
}

#[test]
fn rope_build_tables_computes_frequencies() {
    let src = kernel_source(MetalKernelSource::Rope);
    let body = extract_kernel_body(src, "rope_build_tables");
    assert!(
        body.contains("cos(angle)") && body.contains("sin(angle)"),
        "rope_build_tables: should compute cos and sin of angles"
    );
}

// ── attention ───────────────────────────────────────────────────────

#[test]
fn attention_scores_applies_causal_mask() {
    let src = kernel_source(MetalKernelSource::Attention);
    let body = extract_kernel_body(src, "attention_scores");
    assert!(
        body.contains("causal") && body.contains("-INFINITY"),
        "attention_scores: causal mask should set future positions to -INFINITY"
    );
}

#[test]
fn attention_scores_uses_dot_product() {
    let src = kernel_source(MetalKernelSource::Attention);
    let body = extract_kernel_body(src, "attention_scores");
    assert!(
        body.contains("dot +=") || body.contains("dot+="),
        "attention_scores: should accumulate dot product"
    );
}

#[test]
fn attention_weighted_sum_accumulates_over_seq_k() {
    let src = kernel_source(MetalKernelSource::Attention);
    let body = extract_kernel_body(src, "attention_weighted_sum");
    assert!(
        body.contains("k < seq_k"),
        "attention_weighted_sum: should iterate over seq_k dimension"
    );
}

// ── elementwise ─────────────────────────────────────────────────────

#[test]
fn elementwise_add_formula() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let body = extract_kernel_body(src, "add");
    assert!(body.contains("a[gid] + b[gid]"), "add: output = a + b");
}

#[test]
fn elementwise_mul_formula() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let body = extract_kernel_body(src, "mul");
    assert!(body.contains("a[gid] * b[gid]"), "mul: output = a * b");
}

#[test]
fn elementwise_silu_mul_is_fused() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let body = extract_kernel_body(src, "silu_mul");
    // silu_mul should compute silu(gate) * up in a single kernel.
    assert!(
        body.contains("exp(-g)") || body.contains("exp(- g)"),
        "silu_mul: should compute sigmoid of gate"
    );
    assert!(body.contains("up[gid]"), "silu_mul: should multiply by up projection");
}

#[test]
fn elementwise_gelu_uses_tanh_approximation() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let body = extract_kernel_body(src, "gelu");
    assert!(body.contains("tanh("), "gelu: should use tanh approximation");
}

#[test]
fn elementwise_scalar_mul_uses_scalar_parameter() {
    let src = kernel_source(MetalKernelSource::Elementwise);
    let sig = extract_kernel_signature(src, "scalar_mul");
    assert!(
        sig.contains("constant float& scalar"),
        "scalar_mul: should take scalar as constant float& parameter"
    );
}

// ═══════════════════════════════════════════════════════════════════
// 7. Cross-Cutting Validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn all_six_kernel_types_covered() {
    assert_eq!(MetalKernelSource::ALL.len(), 6, "Expected exactly 6 kernel types");
}

#[test]
fn no_kernel_source_contains_debug_prints() {
    // Metal printf is expensive; shader sources should not contain debug prints.
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        assert!(!src.contains("printf("), "{kernel:?}: should not contain printf debug prints");
    }
}

#[test]
fn no_kernel_source_contains_todo_markers() {
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        let lower = src.to_lowercase();
        assert!(
            !lower.contains("todo!") && !lower.contains("fixme!"),
            "{kernel:?}: should not contain TODO!/FIXME! markers"
        );
    }
}

#[test]
fn all_output_buffers_are_device_writable() {
    // Output buffers should use `device T*` (not `device const T*`).
    for kernel in MetalKernelSource::ALL {
        let src = kernel_source(*kernel);
        for name in kernel_function_names(*kernel) {
            let sig = extract_kernel_signature(src, name);
            // Find the output buffer (typically the last `device` pointer before
            // `constant` parameters). We check that at least one non-const device
            // pointer exists.
            assert!(
                sig.contains("device float*") || sig.contains("device half*"),
                "{kernel:?}/{name}: should have at least one writable device buffer"
            );
        }
    }
}

#[test]
fn reduction_kernels_shared_memory_matches_threadgroup_size() {
    // For softmax and rmsnorm, the shared memory array size should match
    // the declared threadgroup size constant.
    let cases = [
        (MetalKernelSource::Softmax, "SOFTMAX_THREADGROUP_SIZE"),
        (MetalKernelSource::RmsNorm, "RMSNORM_THREADGROUP_SIZE"),
    ];
    for (kernel, constant_name) in &cases {
        let src = kernel_source(*kernel);
        if let Some(tg_size) = parse_constant_uint(src, constant_name) {
            let pattern = format!("shared_data[{tg_size}]");
            assert!(
                src.contains(&pattern),
                "{kernel:?}: shared_data size should match {constant_name} = {tg_size}"
            );
        }
    }
}
