// SPDX-License-Identifier: MIT OR Apache-2.0
//! Architecture-aware LayerNorm and projection weight validation rules.
//!
//! This module provides pattern-based threshold validation for:
//! - LayerNorm gamma weights (RMS statistics)
//! - Projection weight RMS envelopes
//!
//! Supports auto-detection from GGUF metadata and extensible YAML policies.

use anyhow::{Result, anyhow};
use once_cell::sync::Lazy;
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
        (0.50..=2.0).contains(&rms)
    }

    /// Check if projection weight RMS is within acceptable envelope
    pub fn check_proj_rms(&self, rms: f32) -> bool {
        // Enforce each bound independently (single-sided envelopes work)
        if let Some(min) = self.proj_weight_rms_min
            && rms < min
        {
            return false;
        }
        if let Some(max) = self.proj_weight_rms_max
            && rms > max
        {
            return false;
        }
        true
    }
}

// ---------- Built-in rulesets ----------

// Keep this only for built-ins (never called on user input).
fn re(s: &str) -> Regex {
    // Built-ins are trusted. If this ever panics in dev, it needs fixing.
    Regex::new(s).expect("internal built-in regex must compile")
}

/// BitNet b1.58, **F16** export (st2gguf output)
///
/// These patterns are derived from empirical analysis of clean F16 exports:
/// - ffn_layernorm: often has low RMS (~0.05-0.10) legitimately
/// - post_attention_layernorm: typically 0.25-1.0
/// - input_layernorm: typically 0.35-1.0
/// - final_norm: should be close to 1.0 (0.5-2.0 envelope)
static BITNET_B158_F16: Lazy<Ruleset> = Lazy::new(|| Ruleset {
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
});

pub fn rules_bitnet_b158_f16() -> Ruleset {
    BITNET_B158_F16.clone()
}

/// BitNet b1.58, **I2_S** quantized GGUF (e.g., `ggml-model-i2_s.gguf`)
///
/// Many attn_norm weights sit ≈ 0.01..0.02 legitimately after I2_S quantization.
/// So we loosen the LN gate significantly.
static BITNET_B158_I2S: Lazy<Ruleset> = Lazy::new(|| Ruleset {
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
});

pub fn rules_bitnet_b158_i2s() -> Ruleset {
    BITNET_B158_I2S.clone()
}

/// Generic (LLaMA-ish/RMSNorm) fallback
///
/// Assumes standard RMSNorm with gamma weights near 1.0
static GENERIC: Lazy<Ruleset> = Lazy::new(|| Ruleset {
    ln: vec![Threshold { pattern: re(r".*norm\.weight$"), min: 0.80, max: 1.20 }],
    proj_weight_rms_min: None,
    proj_weight_rms_max: None,
    name: "generic".into(),
});

pub fn rules_generic() -> Ruleset {
    GENERIC.clone()
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
    let pol: Policy = serde_yaml::from_str(&text)
        .map_err(|e| anyhow!("invalid policy yaml '{}': {}", path.display(), e))?;
    let rs = pol
        .rules
        .get(key)
        .ok_or_else(|| anyhow!("policy key '{}' not found in {}", key, path.display()))?;

    // Compile user-provided regex patterns with proper error handling
    let compiled_ln: Vec<Threshold> = rs
        .ln
        .iter()
        .map(|r| {
            Regex::new(&r.pattern)
                .map(|pattern| Threshold { pattern, min: r.min, max: r.max })
                .map_err(|e| {
                    anyhow!(
                        "invalid regex pattern in policy for key '{}': '{}' -> {}",
                        key,
                        r.pattern,
                        e
                    )
                })
        })
        .collect::<Result<_>>()?;

    Ok(Ruleset {
        ln: compiled_ln,
        proj_weight_rms_min: rs.proj_weight_rms_min,
        proj_weight_rms_max: rs.proj_weight_rms_max,
        name: rs.name.clone().unwrap_or_else(|| format!("policy:{}", key)),
    })
}
