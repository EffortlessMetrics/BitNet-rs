//! Runtime kernel fusion engine for Intel A770 OpenCL dispatch optimization.
//!
//! Dynamically fuses compatible element-wise kernels into single dispatches
//! to reduce launch overhead and improve data locality.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Operations eligible for fusion.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusableOp {
    Add,
    Mul,
    SiLU,
    GELU,
    ReLU,
    RmsNorm,
    Softmax,
    Scale,
    Bias,
    Transpose,
}

impl fmt::Display for FusableOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::SiLU => "silu",
            Self::GELU => "gelu",
            Self::ReLU => "relu",
            Self::RmsNorm => "rms_norm",
            Self::Softmax => "softmax",
            Self::Scale => "scale",
            Self::Bias => "bias",
            Self::Transpose => "transpose",
        };
        write!(f, "{name}")
    }
}

/// A candidate fusion discovered by pattern matching.
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    pub ops: Vec<FusableOp>,
    pub fused_name: String,
    pub estimated_speedup: f32,
}

/// A rule mapping a pattern of ops to a fused replacement.
#[derive(Debug, Clone)]
pub struct FusionRule {
    pub pattern: Vec<FusableOp>,
    pub replacement: String,
    pub kernel_source: String,
}

/// Accumulated statistics for the fusion engine.
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    pub fusions_applied: u64,
    pub fusions_rejected: u64,
    pub total_ops_fused: u64,
    pub estimated_speedup_total: f64,
}

/// Result of applying fusion to a sequence of ops.
#[derive(Debug, Clone)]
pub struct FusionResult {
    pub original_ops: usize,
    pub fused_ops: usize,
    pub fused_source: String,
    pub speedup: f32,
}

/// Errors that can occur during fusion.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    IncompatibleOps(String),
    PatternNotFound,
    MaxFusionDepthExceeded(usize),
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleOps(msg) => write!(f, "incompatible ops: {msg}"),
            Self::PatternNotFound => write!(f, "no matching fusion pattern"),
            Self::MaxFusionDepthExceeded(d) => {
                write!(f, "max fusion depth exceeded: {d}")
            }
        }
    }
}

impl std::error::Error for FusionError {}

/// The main fusion engine holding rules, cache and statistics.
#[derive(Debug, Clone)]
pub struct FusionEngine {
    pub rules: Vec<FusionRule>,
    pub cache: HashMap<String, String>,
    pub stats: FusionStats,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a cache key from a slice of ops.
fn ops_key(ops: &[FusableOp]) -> String {
    ops.iter().map(|o| format!("{o}")).collect::<Vec<_>>().join("+")
}

/// Whether an op is element-wise (and therefore trivially fusable).
fn is_elementwise(op: &FusableOp) -> bool {
    matches!(
        op,
        FusableOp::Add
            | FusableOp::Mul
            | FusableOp::SiLU
            | FusableOp::GELU
            | FusableOp::ReLU
            | FusableOp::Scale
            | FusableOp::Bias
    )
}

/// Generate an OpenCL expression for a single op applied to variable `x`.
fn op_expr(op: &FusableOp) -> &'static str {
    match op {
        FusableOp::Add => "x = x + y",
        FusableOp::Mul => "x = x * y",
        FusableOp::SiLU => "x = x * native_recip(1.0f + native_exp(-x))",
        FusableOp::GELU => {
            "x = 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"
        }
        FusableOp::ReLU => "x = fmax(x, 0.0f)",
        FusableOp::RmsNorm => "x = x * native_rsqrt(rms + 1e-6f)",
        FusableOp::Softmax => "x = native_exp(x - max_val)",
        FusableOp::Scale => "x = x * scale",
        FusableOp::Bias => "x = x + bias[gid]",
        FusableOp::Transpose => "/* transpose is a memory reorder, not element-wise */",
    }
}

// ---------------------------------------------------------------------------
// Predefined A770 fusion rules
// ---------------------------------------------------------------------------

fn predefined_rules() -> Vec<FusionRule> {
    vec![
        FusionRule {
            pattern: vec![FusableOp::Add, FusableOp::SiLU],
            replacement: "FusedAddSiLU".into(),
            kernel_source: gen_kernel_source("fused_add_silu", &[FusableOp::Add, FusableOp::SiLU]),
        },
        FusionRule {
            pattern: vec![FusableOp::Mul, FusableOp::Add],
            replacement: "FusedMulAdd".into(),
            kernel_source: gen_kernel_source("fused_mul_add", &[FusableOp::Mul, FusableOp::Add]),
        },
        FusionRule {
            pattern: vec![FusableOp::RmsNorm, FusableOp::Scale],
            replacement: "FusedRmsNormScale".into(),
            kernel_source: gen_kernel_source(
                "fused_rms_norm_scale",
                &[FusableOp::RmsNorm, FusableOp::Scale],
            ),
        },
        FusionRule {
            pattern: vec![FusableOp::SiLU, FusableOp::Mul],
            replacement: "FusedSwiGLU".into(),
            kernel_source: gen_kernel_source("fused_swiglu", &[FusableOp::SiLU, FusableOp::Mul]),
        },
        FusionRule {
            pattern: vec![FusableOp::Softmax, FusableOp::Scale],
            replacement: "FusedScaledSoftmax".into(),
            kernel_source: gen_kernel_source(
                "fused_scaled_softmax",
                &[FusableOp::Softmax, FusableOp::Scale],
            ),
        },
        FusionRule {
            pattern: vec![FusableOp::Add, FusableOp::ReLU],
            replacement: "FusedAddReLU".into(),
            kernel_source: gen_kernel_source("fused_add_relu", &[FusableOp::Add, FusableOp::ReLU]),
        },
        FusionRule {
            pattern: vec![FusableOp::Bias, FusableOp::GELU],
            replacement: "FusedBiasGELU".into(),
            kernel_source: gen_kernel_source(
                "fused_bias_gelu",
                &[FusableOp::Bias, FusableOp::GELU],
            ),
        },
    ]
}

fn gen_kernel_source(name: &str, ops: &[FusableOp]) -> String {
    let mut body = String::new();
    body.push_str(&format!(
        "__kernel void {name}(__global float* out, __global const float* in, uint n) {{\n"
    ));
    body.push_str("    uint gid = get_global_id(0);\n");
    body.push_str("    if (gid >= n) return;\n");
    body.push_str("    float x = in[gid];\n");
    for op in ops {
        body.push_str(&format!("    {};  // {op}\n", op_expr(op)));
    }
    body.push_str("    out[gid] = x;\n");
    body.push_str("}\n");
    body
}

// ---------------------------------------------------------------------------
// Public CPU-reference API
// ---------------------------------------------------------------------------

/// Create a new `FusionEngine` pre-loaded with A770-optimised rules.
pub fn create_fusion_engine() -> FusionEngine {
    FusionEngine {
        rules: predefined_rules(),
        cache: HashMap::new(),
        stats: FusionStats::default(),
    }
}

/// Register a custom fusion rule.
pub fn cpu_add_rule(
    engine: &mut FusionEngine,
    pattern: Vec<FusableOp>,
    replacement: &str,
    source: &str,
) {
    engine.rules.push(FusionRule {
        pattern,
        replacement: replacement.to_string(),
        kernel_source: source.to_string(),
    });
}

/// Scan `ops` for subsequences that match any rule and return candidates.
pub fn cpu_find_fusion_candidates(
    engine: &FusionEngine,
    ops: &[FusableOp],
) -> Vec<FusionCandidate> {
    let mut candidates = Vec::new();
    for rule in &engine.rules {
        let plen = rule.pattern.len();
        if plen > ops.len() {
            continue;
        }
        for start in 0..=ops.len() - plen {
            if ops[start..start + plen] == *rule.pattern {
                candidates.push(FusionCandidate {
                    ops: rule.pattern.clone(),
                    fused_name: rule.replacement.clone(),
                    estimated_speedup: cpu_estimate_fusion_speedup(&rule.pattern),
                });
            }
        }
    }
    candidates
}

/// Try to fuse the entire `ops` sequence according to the first matching rule.
pub fn cpu_apply_fusion(
    engine: &mut FusionEngine,
    ops: &[FusableOp],
) -> Result<FusionResult, FusionError> {
    const MAX_DEPTH: usize = 32;
    if ops.len() > MAX_DEPTH {
        engine.stats.fusions_rejected += 1;
        return Err(FusionError::MaxFusionDepthExceeded(ops.len()));
    }
    if ops.len() < 2 {
        engine.stats.fusions_rejected += 1;
        return Err(FusionError::PatternNotFound);
    }

    // Check pairwise compatibility.
    for w in ops.windows(2) {
        if !cpu_is_fusable(&w[0], &w[1]) {
            engine.stats.fusions_rejected += 1;
            return Err(FusionError::IncompatibleOps(format!(
                "{} and {} cannot be fused",
                w[0], w[1]
            )));
        }
    }

    let key = ops_key(ops);

    // Try cache first.
    if let Some(source) = engine.cache.get(&key) {
        let speedup = cpu_estimate_fusion_speedup(ops);
        engine.stats.fusions_applied += 1;
        engine.stats.total_ops_fused += ops.len() as u64;
        engine.stats.estimated_speedup_total += f64::from(speedup);
        return Ok(FusionResult {
            original_ops: ops.len(),
            fused_ops: 1,
            fused_source: source.clone(),
            speedup,
        });
    }

    // Try matching a rule exactly.
    for rule in &engine.rules {
        if rule.pattern == ops {
            let speedup = cpu_estimate_fusion_speedup(ops);
            engine.cache.insert(key, rule.kernel_source.clone());
            engine.stats.fusions_applied += 1;
            engine.stats.total_ops_fused += ops.len() as u64;
            engine.stats.estimated_speedup_total += f64::from(speedup);
            return Ok(FusionResult {
                original_ops: ops.len(),
                fused_ops: 1,
                fused_source: rule.kernel_source.clone(),
                speedup,
            });
        }
    }

    // Fall back to generic generation for compatible ops.
    let source = cpu_generate_fused_kernel(ops);
    let speedup = cpu_estimate_fusion_speedup(ops);
    engine.cache.insert(key, source.clone());
    engine.stats.fusions_applied += 1;
    engine.stats.total_ops_fused += ops.len() as u64;
    engine.stats.estimated_speedup_total += f64::from(speedup);
    Ok(FusionResult { original_ops: ops.len(), fused_ops: 1, fused_source: source, speedup })
}

/// Generate an OpenCL kernel source for an arbitrary fusable op sequence.
pub fn cpu_generate_fused_kernel(ops: &[FusableOp]) -> String {
    let name = ops.iter().map(|o| format!("{o}")).collect::<Vec<_>>().join("_");
    gen_kernel_source(&format!("fused_{name}"), ops)
}

/// Estimate speedup from fusing `ops` into one dispatch (≥ 1.0).
pub fn cpu_estimate_fusion_speedup(ops: &[FusableOp]) -> f32 {
    if ops.len() <= 1 {
        return 1.0;
    }
    // Each eliminated dispatch saves ~5 µs on A770; base op cost ~10 µs.
    let dispatch_savings = (ops.len() - 1) as f32 * 0.5;
    // Locality bonus for element-wise chains.
    let locality_bonus =
        ops.iter().filter(|o| is_elementwise(o)).count() as f32 * 0.05;
    1.0 + dispatch_savings + locality_bonus
}

/// Returns `true` if two adjacent ops can be fused (both element-wise, or
/// known compatible pairs like RmsNorm+Scale, Softmax+Scale).
pub fn cpu_is_fusable(a: &FusableOp, b: &FusableOp) -> bool {
    // Transpose changes memory layout — never fusable.
    if *a == FusableOp::Transpose || *b == FusableOp::Transpose {
        return false;
    }
    // Two element-wise ops are always fusable.
    if is_elementwise(a) && is_elementwise(b) {
        return true;
    }
    // Special pairs.
    matches!(
        (a, b),
        (FusableOp::RmsNorm, FusableOp::Scale)
            | (FusableOp::Softmax, FusableOp::Scale)
            | (FusableOp::RmsNorm, FusableOp::Mul)
            | (FusableOp::Softmax, FusableOp::Mul)
    )
}

/// Greedily fuse longest-matching patterns left-to-right.
pub fn cpu_chain_fusions(
    engine: &mut FusionEngine,
    ops: &[FusableOp],
) -> Vec<FusionResult> {
    let mut results = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        let mut best_len = 0usize;
        let mut best_rule: Option<&FusionRule> = None;
        // Longest match first.
        for rule in &engine.rules {
            let plen = rule.pattern.len();
            if plen > best_len && i + plen <= ops.len() && ops[i..i + plen] == *rule.pattern {
                best_len = plen;
                best_rule = Some(rule);
            }
        }
        if let Some(rule) = best_rule {
            let speedup = cpu_estimate_fusion_speedup(&rule.pattern);
            let key = ops_key(&rule.pattern);
            engine.cache.insert(key, rule.kernel_source.clone());
            engine.stats.fusions_applied += 1;
            engine.stats.total_ops_fused += best_len as u64;
            engine.stats.estimated_speedup_total += f64::from(speedup);
            results.push(FusionResult {
                original_ops: best_len,
                fused_ops: 1,
                fused_source: rule.kernel_source.clone(),
                speedup,
            });
            i += best_len;
        } else {
            // Emit single unfused op.
            results.push(FusionResult {
                original_ops: 1,
                fused_ops: 1,
                fused_source: cpu_generate_fused_kernel(&ops[i..i + 1]),
                speedup: 1.0,
            });
            i += 1;
        }
    }
    results
}

/// CPU reference implementation: execute fused ops sequentially on `input`.
pub fn cpu_execute_fused_reference(ops: &[FusableOp], input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&v| {
            let mut x = v;
            for op in ops {
                x = apply_scalar_op(op, x);
            }
            x
        })
        .collect()
}

fn apply_scalar_op(op: &FusableOp, x: f32) -> f32 {
    match op {
        FusableOp::Add => x + 1.0,
        FusableOp::Mul => x * 2.0,
        FusableOp::SiLU => x * (1.0 / (1.0 + (-x).exp())),
        FusableOp::GELU => {
            0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044715 * x * x * x)).tanh())
        }
        FusableOp::ReLU => x.max(0.0),
        FusableOp::RmsNorm => x / (x * x + 1e-6_f32).sqrt(),
        FusableOp::Softmax => x.exp(),
        FusableOp::Scale => x * 0.5,
        FusableOp::Bias => x + 0.1,
        FusableOp::Transpose => x, // identity for scalar ref
    }
}

/// Validate that `fused` output matches `original` within `tolerance`.
pub fn cpu_validate_fusion(original: &[f32], fused: &[f32], tolerance: f32) -> bool {
    if original.len() != fused.len() {
        return false;
    }
    original.iter().zip(fused.iter()).all(|(a, b)| (a - b).abs() <= tolerance)
}

/// Return a snapshot of the engine's statistics.
pub fn cpu_get_stats(engine: &FusionEngine) -> FusionStats {
    engine.stats.clone()
}

/// Pretty-print a `FusionResult`.
pub fn format_fusion_result(result: &FusionResult) -> String {
    format!(
        "Fusion: {} ops → {} dispatch(es), {:.2}× speedup, source {} bytes",
        result.original_ops,
        result.fused_ops,
        result.speedup,
        result.fused_source.len(),
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- engine creation --------------------------------------------------

    #[test]
    fn test_create_engine_has_predefined_rules() {
        let engine = create_fusion_engine();
        assert!(engine.rules.len() >= 7, "expected at least 7 predefined rules");
    }

    #[test]
    fn test_create_engine_empty_cache() {
        let engine = create_fusion_engine();
        assert!(engine.cache.is_empty());
    }

    #[test]
    fn test_create_engine_zero_stats() {
        let engine = create_fusion_engine();
        assert_eq!(engine.stats.fusions_applied, 0);
        assert_eq!(engine.stats.fusions_rejected, 0);
    }

    // ----- custom rules -----------------------------------------------------

    #[test]
    fn test_add_custom_rule() {
        let mut engine = create_fusion_engine();
        let before = engine.rules.len();
        cpu_add_rule(&mut engine, vec![FusableOp::ReLU, FusableOp::Scale], "FusedReluScale", "/**/");
        assert_eq!(engine.rules.len(), before + 1);
    }

    #[test]
    fn test_custom_rule_fields() {
        let mut engine = create_fusion_engine();
        cpu_add_rule(&mut engine, vec![FusableOp::ReLU], "R", "src");
        let last = engine.rules.last().unwrap();
        assert_eq!(last.replacement, "R");
        assert_eq!(last.kernel_source, "src");
    }

    // ----- find candidates --------------------------------------------------

    #[test]
    fn test_find_candidates_swiglu() {
        let engine = create_fusion_engine();
        let ops = vec![FusableOp::SiLU, FusableOp::Mul];
        let c = cpu_find_fusion_candidates(&engine, &ops);
        assert!(!c.is_empty());
        assert_eq!(c[0].fused_name, "FusedSwiGLU");
    }

    #[test]
    fn test_find_candidates_no_match() {
        let engine = create_fusion_engine();
        let ops = vec![FusableOp::Transpose];
        let c = cpu_find_fusion_candidates(&engine, &ops);
        assert!(c.is_empty());
    }

    #[test]
    fn test_find_candidates_fma() {
        let engine = create_fusion_engine();
        let ops = vec![FusableOp::Mul, FusableOp::Add];
        let c = cpu_find_fusion_candidates(&engine, &ops);
        assert!(c.iter().any(|c| c.fused_name == "FusedMulAdd"));
    }

    #[test]
    fn test_find_candidates_add_silu() {
        let engine = create_fusion_engine();
        let ops = vec![FusableOp::Add, FusableOp::SiLU];
        let c = cpu_find_fusion_candidates(&engine, &ops);
        assert!(c.iter().any(|c| c.fused_name == "FusedAddSiLU"));
    }

    #[test]
    fn test_find_candidates_in_longer_sequence() {
        let engine = create_fusion_engine();
        let ops = vec![FusableOp::ReLU, FusableOp::SiLU, FusableOp::Mul, FusableOp::Add];
        let c = cpu_find_fusion_candidates(&engine, &ops);
        assert!(c.iter().any(|c| c.fused_name == "FusedSwiGLU"));
        assert!(c.iter().any(|c| c.fused_name == "FusedMulAdd"));
    }

    // ----- apply fusion -----------------------------------------------------

    #[test]
    fn test_apply_fusion_swiglu() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::SiLU, FusableOp::Mul];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert_eq!(r.original_ops, 2);
        assert_eq!(r.fused_ops, 1);
    }

    #[test]
    fn test_apply_fusion_fma() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Mul, FusableOp::Add];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert_eq!(r.fused_ops, 1);
        assert!(r.speedup > 1.0);
    }

    #[test]
    fn test_apply_fusion_add_relu() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Add, FusableOp::ReLU];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert_eq!(r.fused_ops, 1);
    }

    #[test]
    fn test_apply_fusion_bias_gelu() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Bias, FusableOp::GELU];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert!(r.fused_source.contains("fused"));
    }

    #[test]
    fn test_apply_fusion_rms_norm_scale() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::RmsNorm, FusableOp::Scale];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert_eq!(r.fused_ops, 1);
    }

    #[test]
    fn test_apply_fusion_incompatible() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Add, FusableOp::Transpose];
        let r = cpu_apply_fusion(&mut engine, &ops);
        assert!(matches!(r, Err(FusionError::IncompatibleOps(_))));
    }

    // ----- kernel source generation -----------------------------------------

    #[test]
    fn test_generate_kernel_contains_ops() {
        let src = cpu_generate_fused_kernel(&[FusableOp::Add, FusableOp::SiLU]);
        assert!(src.contains("__kernel"));
        assert!(src.contains("add"));
        assert!(src.contains("silu"));
    }

    #[test]
    fn test_generate_kernel_has_gid() {
        let src = cpu_generate_fused_kernel(&[FusableOp::Mul]);
        assert!(src.contains("get_global_id"));
    }

    // ----- speedup estimation -----------------------------------------------

    #[test]
    fn test_speedup_single_op_is_1() {
        assert_eq!(cpu_estimate_fusion_speedup(&[FusableOp::Add]), 1.0);
    }

    #[test]
    fn test_speedup_two_ops_gt_1() {
        let s = cpu_estimate_fusion_speedup(&[FusableOp::Add, FusableOp::Mul]);
        assert!(s > 1.0);
    }

    #[test]
    fn test_speedup_grows_with_chain() {
        let s2 = cpu_estimate_fusion_speedup(&[FusableOp::Add, FusableOp::Mul]);
        let s3 =
            cpu_estimate_fusion_speedup(&[FusableOp::Add, FusableOp::Mul, FusableOp::ReLU]);
        assert!(s3 > s2);
    }

    // ----- is_fusable -------------------------------------------------------

    #[test]
    fn test_is_fusable_elementwise_pair() {
        assert!(cpu_is_fusable(&FusableOp::Add, &FusableOp::Mul));
    }

    #[test]
    fn test_is_fusable_rmsnorm_scale() {
        assert!(cpu_is_fusable(&FusableOp::RmsNorm, &FusableOp::Scale));
    }

    #[test]
    fn test_is_not_fusable_transpose() {
        assert!(!cpu_is_fusable(&FusableOp::Add, &FusableOp::Transpose));
    }

    #[test]
    fn test_is_fusable_softmax_scale() {
        assert!(cpu_is_fusable(&FusableOp::Softmax, &FusableOp::Scale));
    }

    // ----- chain fusions ----------------------------------------------------

    #[test]
    fn test_chain_fusions_multiple() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::SiLU, FusableOp::Mul, FusableOp::Add, FusableOp::ReLU];
        let results = cpu_chain_fusions(&mut engine, &ops);
        // SiLU+Mul → fused, Add+ReLU → fused
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.fused_ops == 1));
    }

    #[test]
    fn test_chain_fusions_no_match_emits_singles() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Transpose];
        let results = cpu_chain_fusions(&mut engine, &ops);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].original_ops, 1);
    }

    #[test]
    fn test_chain_fusions_partial_match() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::SiLU, FusableOp::Mul, FusableOp::Transpose];
        let results = cpu_chain_fusions(&mut engine, &ops);
        assert_eq!(results.len(), 2); // fused + single
    }

    // ----- execute fused reference ------------------------------------------

    #[test]
    fn test_execute_fused_add_silu() {
        let input = vec![0.0, 1.0, -1.0];
        let out = cpu_execute_fused_reference(&[FusableOp::Add, FusableOp::SiLU], &input);
        assert_eq!(out.len(), 3);
        // Add(0)+1=1, SiLU(1)=1/(1+e^-1)≈0.731
        assert!((out[0] - 0.7310586).abs() < 1e-4);
    }

    #[test]
    fn test_execute_fused_mul_add() {
        let input = vec![3.0];
        let out = cpu_execute_fused_reference(&[FusableOp::Mul, FusableOp::Add], &input);
        // Mul(3)*2=6, Add(6)+1=7
        assert!((out[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_execute_fused_relu() {
        let input = vec![-5.0, 0.0, 5.0];
        let out = cpu_execute_fused_reference(&[FusableOp::ReLU], &input);
        assert_eq!(out, vec![0.0, 0.0, 5.0]);
    }

    // ----- validate fusion --------------------------------------------------

    #[test]
    fn test_validate_matching() {
        assert!(cpu_validate_fusion(&[1.0, 2.0], &[1.0, 2.0], 1e-6));
    }

    #[test]
    fn test_validate_divergent() {
        assert!(!cpu_validate_fusion(&[1.0, 2.0], &[1.0, 999.0], 1e-6));
    }

    #[test]
    fn test_validate_length_mismatch() {
        assert!(!cpu_validate_fusion(&[1.0], &[1.0, 2.0], 1e-6));
    }

    #[test]
    fn test_validate_within_tolerance() {
        assert!(cpu_validate_fusion(&[1.0], &[1.0001], 0.001));
    }

    // ----- stats ------------------------------------------------------------

    #[test]
    fn test_stats_after_fusions() {
        let mut engine = create_fusion_engine();
        let _ = cpu_apply_fusion(&mut engine, &[FusableOp::Add, FusableOp::SiLU]);
        let _ = cpu_apply_fusion(&mut engine, &[FusableOp::Mul, FusableOp::Add]);
        let s = cpu_get_stats(&engine);
        assert_eq!(s.fusions_applied, 2);
        assert_eq!(s.total_ops_fused, 4);
    }

    #[test]
    fn test_stats_rejected() {
        let mut engine = create_fusion_engine();
        let _ = cpu_apply_fusion(&mut engine, &[FusableOp::Add, FusableOp::Transpose]);
        let s = cpu_get_stats(&engine);
        assert_eq!(s.fusions_rejected, 1);
    }

    // ----- edge cases -------------------------------------------------------

    #[test]
    fn test_single_op_no_fusion() {
        let mut engine = create_fusion_engine();
        let r = cpu_apply_fusion(&mut engine, &[FusableOp::Add]);
        assert!(matches!(r, Err(FusionError::PatternNotFound)));
    }

    #[test]
    fn test_all_same_op() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Add, FusableOp::Add, FusableOp::Add];
        let r = cpu_apply_fusion(&mut engine, &ops);
        assert!(r.is_ok()); // all element-wise, generic fusion applies
    }

    #[test]
    fn test_max_depth_exceeded() {
        let mut engine = create_fusion_engine();
        let ops: Vec<FusableOp> = (0..33).map(|_| FusableOp::Add).collect();
        let r = cpu_apply_fusion(&mut engine, &ops);
        assert!(matches!(r, Err(FusionError::MaxFusionDepthExceeded(33))));
    }

    // ----- properties -------------------------------------------------------

    #[test]
    fn test_fused_ops_le_original() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::SiLU, FusableOp::Mul];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert!(r.fused_ops <= r.original_ops);
    }

    #[test]
    fn test_speedup_ge_1_for_valid() {
        let mut engine = create_fusion_engine();
        for rule in predefined_rules() {
            if let Ok(r) = cpu_apply_fusion(&mut engine, &rule.pattern) {
                assert!(r.speedup >= 1.0, "speedup < 1 for {:?}", rule.pattern);
            }
        }
    }

    // ----- BitNet-specific --------------------------------------------------

    #[test]
    fn test_bitnet_ternary_matmul_bias_activation() {
        // Simulates post-matmul pipeline: Bias → GELU (as in BitNet FFN).
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Bias, FusableOp::GELU];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert_eq!(r.fused_ops, 1);
        // Verify reference execution
        let input = vec![0.5, -0.5];
        let out = cpu_execute_fused_reference(&ops, &input);
        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.0); // bias(0.5)+0.1=0.6 → GELU(0.6)>0
    }

    #[test]
    fn test_bitnet_ffn_swiglu_pattern() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::SiLU, FusableOp::Mul];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert!(r.fused_source.contains("fused_swiglu"));
    }

    #[test]
    fn test_bitnet_attention_scaled_softmax() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Softmax, FusableOp::Scale];
        let r = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert!(r.fused_source.contains("fused_scaled_softmax"));
    }

    // ----- format -----------------------------------------------------------

    #[test]
    fn test_format_fusion_result() {
        let r = FusionResult {
            original_ops: 3,
            fused_ops: 1,
            fused_source: "x".into(),
            speedup: 2.5,
        };
        let s = format_fusion_result(&r);
        assert!(s.contains("3 ops"));
        assert!(s.contains("2.50×"));
    }

    // ----- cache ------------------------------------------------------------

    #[test]
    fn test_cache_populated_after_apply() {
        let mut engine = create_fusion_engine();
        assert!(engine.cache.is_empty());
        let _ = cpu_apply_fusion(&mut engine, &[FusableOp::Add, FusableOp::SiLU]);
        assert!(!engine.cache.is_empty());
    }

    #[test]
    fn test_cache_hit_returns_same_source() {
        let mut engine = create_fusion_engine();
        let ops = vec![FusableOp::Add, FusableOp::ReLU];
        let r1 = cpu_apply_fusion(&mut engine, &ops).unwrap();
        let r2 = cpu_apply_fusion(&mut engine, &ops).unwrap();
        assert_eq!(r1.fused_source, r2.fused_source);
    }

    // ----- error display ----------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = FusionError::IncompatibleOps("test".into());
        assert!(e.to_string().contains("test"));
        assert!(FusionError::PatternNotFound.to_string().contains("pattern"));
        assert!(FusionError::MaxFusionDepthExceeded(5).to_string().contains('5'));
    }

    #[test]
    fn test_fusable_op_display() {
        assert_eq!(format!("{}", FusableOp::SiLU), "silu");
        assert_eq!(format!("{}", FusableOp::RmsNorm), "rms_norm");
    }
}
