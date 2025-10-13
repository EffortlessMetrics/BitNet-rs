// SPDX-License-Identifier: MIT OR Apache-2.0
//! Architecture-aware LayerNorm and projection weight validation rules.
//!
//! This module provides pattern-based threshold validation for:
//! - LayerNorm gamma weights (RMS statistics)
//! - Projection weight RMS envelopes
//!
//! Supports auto-detection from GGUF metadata and extensible YAML policies.

use anyhow::{Result, anyhow};
use regex::Regex;
use serde::Deserialize;

/// A threshold rule with regex pattern matching
#[derive(Debug, Clone)]
pub struct Threshold {
    pub pattern: Regex,
    pub min: f32,
    pub max: f32,
}

/// Complete validation ruleset for an architecture
#[derive(Debug, Clone, Default)]
pub struct Ruleset {
    pub ln: Vec<Threshold>,
    pub proj_weight_rms_min: Option<f32>,
    pub proj_weight_rms_max: Option<f32>,
    pub name: String,
}

impl Ruleset {
    /// Check if LayerNorm weight RMS is within acceptable envelope
    pub fn check_ln(&self, name: &str, rms: f32) -> bool {
        for th in &self.ln {
            if th.pattern.is_match(name) {
                return rms >= th.min && rms <= th.max;
            }
        }
        // No match => best-effort generic envelope
        rms >= 0.50 && rms <= 2.0
    }

    /// Check if projection weight RMS is within acceptable envelope
    pub fn check_proj_rms(&self, rms: f32) -> bool {
        match (self.proj_weight_rms_min, self.proj_weight_rms_max) {
            (Some(min), Some(max)) => rms >= min && rms <= max,
            _ => true, // no opinion
        }
    }
}

// ---------- Built-in rulesets ----------

fn re(s: &str) -> Regex {
    Regex::new(s).unwrap()
}

/// BitNet b1.58, **F16** export (st2gguf output)
///
/// These patterns are derived from empirical analysis of clean F16 exports:
/// - ffn_layernorm: often has low RMS (~0.05-0.10) legitimately
/// - post_attention_layernorm: typically 0.25-1.0
/// - input_layernorm: typically 0.35-1.0
/// - final_norm: should be close to 1.0 (0.5-2.0 envelope)
pub fn rules_bitnet_b158_f16() -> Ruleset {
    Ruleset {
        ln: vec![
            Threshold { pattern: re(r"ffn_layernorm\.weight$"), min: 0.05, max: 2.0 },
            Threshold { pattern: re(r"post_attention_layernorm\.weight$"), min: 0.25, max: 2.0 },
            Threshold { pattern: re(r"input_layernorm\.weight$"), min: 0.35, max: 2.0 },
            Threshold { pattern: re(r"final_(layer)?norm\.weight$"), min: 0.50, max: 2.0 },
            Threshold { pattern: re(r"(attn|ffn|rms).*norm\.weight$"), min: 0.50, max: 2.0 },
            Threshold { pattern: re(r".*norm\.weight$"), min: 0.50, max: 2.0 },
        ],
        // Weight RMS envelope for projections in F16 (empirical ~0.01..0.25)
        proj_weight_rms_min: Some(0.01),
        proj_weight_rms_max: Some(0.40),
        name: "bitnet-b1.58:f16".into(),
    }
}

/// BitNet b1.58, **I2_S** quantized GGUF (e.g., `ggml-model-i2_s.gguf`)
///
/// Many attn_norm weights sit ≈ 0.01..0.02 legitimately after I2_S quantization.
/// So we loosen the LN gate significantly.
pub fn rules_bitnet_b158_i2s() -> Ruleset {
    Ruleset {
        ln: vec![
            Threshold { pattern: re(r"attn_norm\.weight$"), min: 0.01, max: 2.0 },
            Threshold { pattern: re(r"ffn_norm\.weight$"), min: 0.50, max: 2.0 },
            Threshold { pattern: re(r"final_(layer)?norm\.weight$"), min: 0.50, max: 2.0 },
            Threshold { pattern: re(r".*norm\.weight$"), min: 0.25, max: 2.0 },
        ],
        // Weight RMS after I2_S dequant tends to be small but non-zero
        proj_weight_rms_min: Some(0.002),
        proj_weight_rms_max: Some(0.20),
        name: "bitnet-b1.58:i2_s".into(),
    }
}

/// Generic (LLaMA-ish/RMSNorm) fallback
///
/// Assumes standard RMSNorm with gamma weights near 1.0
pub fn rules_generic() -> Ruleset {
    Ruleset {
        ln: vec![Threshold { pattern: re(r".*norm\.weight$"), min: 0.80, max: 1.20 }],
        proj_weight_rms_min: None,
        proj_weight_rms_max: None,
        name: "generic".into(),
    }
}

// ---------- Auto-detection ----------

/// Detect appropriate ruleset from GGUF metadata
///
/// # Arguments
/// * `arch` - Value of "general.architecture" from GGUF
/// * `file_type` - Value of "general.file_type" from GGUF (1=F16, others=quantized)
pub fn detect_rules(arch: &str, file_type: u32) -> Ruleset {
    let arch_l = arch.to_ascii_lowercase();
    if arch_l.contains("bitnet") || arch_l.contains("b1.58") {
        match file_type {
            1 => rules_bitnet_b158_f16(), // F16 clean export
            _ => rules_bitnet_b158_i2s(), // assume quantized
        }
    } else {
        rules_generic()
    }
}

// ---------- YAML policy (optional) ----------

#[derive(Debug, Deserialize)]
struct YamlRule {
    pattern: String,
    min: f32,
    max: f32,
}

#[derive(Debug, Deserialize)]
struct YamlRuleset {
    name: Option<String>,
    ln: Vec<YamlRule>,
    proj_weight_rms_min: Option<f32>,
    proj_weight_rms_max: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct Policy {
    #[allow(dead_code)]
    version: u32,
    rules: std::collections::HashMap<String, YamlRuleset>,
}

/// Load validation rules from YAML policy file
///
/// # Arguments
/// * `path` - Path to YAML policy file
/// * `key` - Key in the policy file (e.g., "bitnet-b1.58:f16")
pub fn load_policy(path: &std::path::Path, key: &str) -> Result<Ruleset> {
    let text = std::fs::read_to_string(path)?;
    let pol: Policy = serde_yaml::from_str(&text)?;
    let rs = pol.rules.get(key).ok_or_else(|| anyhow!("policy key not found: {}", key))?;

    Ok(Ruleset {
        ln: rs
            .ln
            .iter()
            .map(|r| Threshold { pattern: re(&r.pattern), min: r.min, max: r.max })
            .collect(),
        proj_weight_rms_min: rs.proj_weight_rms_min,
        proj_weight_rms_max: rs.proj_weight_rms_max,
        name: rs.name.clone().unwrap_or_else(|| format!("policy:{}", key)),
    })
}
