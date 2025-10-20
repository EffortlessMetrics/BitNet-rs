//! Policy-driven model correction system
//!
//! This module provides a production-grade, policy-driven approach to handling known issues
//! in model files (e.g., quantized LayerNorm weights). Instead of ad-hoc environment flags,
//! corrections are defined in a YAML/JSON policy file and keyed by GGUF fingerprint.

use bitnet_common::{BitNetError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Version of the correction policy schema
pub const POLICY_VERSION: u32 = 1;

/// Correction policy containing all known model fixes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionPolicy {
    /// Schema version
    pub version: u32,
    /// Model-specific corrections keyed by fingerprint
    #[serde(default)]
    pub models: Vec<ModelCorrection>,
}

/// Corrections for a specific model (identified by fingerprint)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCorrection {
    /// SHA256 fingerprint of the GGUF file
    pub fingerprint: String,
    /// Human-readable notes about the model/issue
    #[serde(default)]
    pub notes: String,
    /// List of corrections to apply
    #[serde(default)]
    pub corrections: Vec<CorrectionAction>,
}

/// Specific correction action to apply
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CorrectionAction {
    /// Rescale LayerNorm gamma weights to target RMS
    #[serde(rename = "LN_GAMMA_RESCALE_RMS")]
    LnGammaRescaleRms {
        /// Target RMS value (typically 1.0)
        target_rms: f32,
        /// Clamp rescale factor to [min, max] for safety
        clamp: [f32; 2],
    },
    /// Override I2_S dequantization parameters for specific tensors
    #[serde(rename = "I2S_DEQUANT_OVERRIDE")]
    I2SDequantOverride {
        /// Tensor name patterns to match (e.g., ["q_proj.weight", "k_proj.weight"])
        tensors: Vec<String>,
        /// Invert scale (use 1/scale instead of scale)
        #[serde(default)]
        inv: bool,
        /// Scale multiplier (typically 1.0)
        #[serde(default = "default_k")]
        k: f32,
    },
}

fn default_k() -> f32 {
    1.0
}

/// Plan for applying corrections to a loaded model
#[derive(Debug, Clone)]
pub struct CorrectionPlan {
    /// Model fingerprint
    pub fingerprint: String,
    /// Notes about the model
    pub notes: String,
    /// Actions to apply
    pub actions: Vec<CorrectionAction>,
}

// CorrectionRecord is now defined in bitnet-common::types
// Re-export for convenience
pub use bitnet_common::CorrectionRecord;

impl CorrectionPolicy {
    /// Load policy from a YAML or JSON file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            BitNetError::Validation(format!(
                "Failed to read correction policy from {:?}: {}",
                path, e
            ))
        })?;

        // Try YAML first, then JSON
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            Self::from_json(&contents)
        } else {
            Self::from_yaml(&contents)
        }
    }

    /// Parse policy from YAML string
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml_ng::from_str(yaml).map_err(|e| {
            BitNetError::Validation(format!("Failed to parse correction policy YAML: {}", e))
        })
    }

    /// Parse policy from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            BitNetError::Validation(format!("Failed to parse correction policy JSON: {}", e))
        })
    }

    /// Find correction plan for a given fingerprint
    pub fn find_plan(&self, fingerprint: &str) -> Option<CorrectionPlan> {
        self.models.iter().find(|m| m.fingerprint == fingerprint).map(|m| CorrectionPlan {
            fingerprint: m.fingerprint.clone(),
            notes: m.notes.clone(),
            actions: m.corrections.clone(),
        })
    }

    /// Build a map from fingerprint to correction plan for fast lookups
    pub fn build_map(&self) -> HashMap<String, CorrectionPlan> {
        self.models
            .iter()
            .map(|m| {
                (
                    m.fingerprint.clone(),
                    CorrectionPlan {
                        fingerprint: m.fingerprint.clone(),
                        notes: m.notes.clone(),
                        actions: m.corrections.clone(),
                    },
                )
            })
            .collect()
    }

    /// Validate policy structure
    pub fn validate(&self) -> Result<()> {
        if self.version != POLICY_VERSION {
            return Err(BitNetError::Validation(format!(
                "Unsupported policy version: expected {}, got {}",
                POLICY_VERSION, self.version
            )));
        }

        // Check for duplicate fingerprints
        let mut seen = std::collections::HashSet::new();
        for model in &self.models {
            if !seen.insert(&model.fingerprint) {
                return Err(BitNetError::Validation(format!(
                    "Duplicate fingerprint in policy: {}",
                    model.fingerprint
                )));
            }

            // Validate fingerprint format (SHA256 hex string)
            if !model.fingerprint.starts_with("sha256-") {
                return Err(BitNetError::Validation(format!(
                    "Fingerprint must start with 'sha256-': {}",
                    model.fingerprint
                )));
            }

            let hash_part = &model.fingerprint[7..];
            if hash_part.len() != 64 || !hash_part.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(BitNetError::Validation(format!(
                    "Invalid SHA256 hash format: {}",
                    model.fingerprint
                )));
            }

            // Validate correction actions
            for action in &model.corrections {
                match action {
                    CorrectionAction::LnGammaRescaleRms { target_rms, clamp } => {
                        if !target_rms.is_finite() || *target_rms <= 0.0 {
                            return Err(BitNetError::Validation(format!(
                                "Invalid target_rms: {} (must be finite and positive)",
                                target_rms
                            )));
                        }
                        if clamp[0] <= 0.0 || clamp[1] <= clamp[0] || !clamp[1].is_finite() {
                            return Err(BitNetError::Validation(format!(
                                "Invalid clamp range: {:?} (must be [min, max] with 0 < min < max)",
                                clamp
                            )));
                        }
                    }
                    CorrectionAction::I2SDequantOverride { tensors, inv: _, k } => {
                        if tensors.is_empty() {
                            return Err(BitNetError::Validation(
                                "I2S_DEQUANT_OVERRIDE requires at least one tensor pattern"
                                    .to_string(),
                            ));
                        }
                        if !k.is_finite() || *k <= 0.0 {
                            return Err(BitNetError::Validation(format!(
                                "Invalid k value: {} (must be finite and positive)",
                                k
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for CorrectionPolicy {
    fn default() -> Self {
        Self { version: POLICY_VERSION, models: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_yaml_parsing() {
        let yaml = r#"
version: 1
models:
  - fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    notes: "Test model with quantized LN weights"
    corrections:
      - type: LN_GAMMA_RESCALE_RMS
        target_rms: 1.0
        clamp: [0.01, 100.0]
"#;

        let policy = CorrectionPolicy::from_yaml(yaml).unwrap();
        assert_eq!(policy.version, 1);
        assert_eq!(policy.models.len(), 1);
        assert_eq!(
            policy.models[0].fingerprint,
            "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        );
        assert_eq!(policy.models[0].notes, "Test model with quantized LN weights");
        assert_eq!(policy.models[0].corrections.len(), 1);

        match &policy.models[0].corrections[0] {
            CorrectionAction::LnGammaRescaleRms { target_rms, clamp } => {
                assert_eq!(*target_rms, 1.0);
                assert_eq!(clamp[0], 0.01);
                assert_eq!(clamp[1], 100.0);
            }
            _ => panic!("Expected LnGammaRescaleRms"),
        }
    }

    #[test]
    fn test_policy_json_parsing() {
        let json = r#"{
  "version": 1,
  "models": [
    {
      "fingerprint": "sha256-fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
      "notes": "Another test model",
      "corrections": [
        {
          "type": "LN_GAMMA_RESCALE_RMS",
          "target_rms": 1.0,
          "clamp": [0.01, 100.0]
        }
      ]
    }
  ]
}"#;

        let policy = CorrectionPolicy::from_json(json).unwrap();
        assert_eq!(policy.version, 1);
        assert_eq!(policy.models.len(), 1);
    }

    #[test]
    fn test_policy_validation() {
        let mut policy = CorrectionPolicy::default();
        policy.models.push(ModelCorrection {
            fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            notes: "Valid model".to_string(),
            corrections: vec![CorrectionAction::LnGammaRescaleRms {
                target_rms: 1.0,
                clamp: [0.01, 100.0],
            }],
        });

        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_policy_validation_bad_fingerprint() {
        let mut policy = CorrectionPolicy::default();
        policy.models.push(ModelCorrection {
            fingerprint: "invalid-fingerprint".to_string(),
            notes: "Bad model".to_string(),
            corrections: vec![],
        });

        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_policy_validation_duplicate_fingerprint() {
        let mut policy = CorrectionPolicy::default();
        let fp = "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        policy.models.push(ModelCorrection {
            fingerprint: fp.to_string(),
            notes: "Model 1".to_string(),
            corrections: vec![],
        });
        policy.models.push(ModelCorrection {
            fingerprint: fp.to_string(),
            notes: "Model 2".to_string(),
            corrections: vec![],
        });

        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_policy_validation_invalid_clamp() {
        let mut policy = CorrectionPolicy::default();
        policy.models.push(ModelCorrection {
            fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            notes: "Bad clamp".to_string(),
            corrections: vec![CorrectionAction::LnGammaRescaleRms {
                target_rms: 1.0,
                clamp: [100.0, 0.01], // inverted range
            }],
        });

        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_find_plan() {
        let mut policy = CorrectionPolicy::default();
        let fp = "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        policy.models.push(ModelCorrection {
            fingerprint: fp.to_string(),
            notes: "Test model".to_string(),
            corrections: vec![CorrectionAction::LnGammaRescaleRms {
                target_rms: 1.0,
                clamp: [0.01, 100.0],
            }],
        });

        let plan = policy.find_plan(fp);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.fingerprint, fp);
        assert_eq!(plan.notes, "Test model");
        assert_eq!(plan.actions.len(), 1);

        let no_plan = policy.find_plan("sha256-nonexistent");
        assert!(no_plan.is_none());
    }

    #[test]
    fn test_build_map() {
        let mut policy = CorrectionPolicy::default();
        let fp1 = "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let fp2 = "sha256-fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

        policy.models.push(ModelCorrection {
            fingerprint: fp1.to_string(),
            notes: "Model 1".to_string(),
            corrections: vec![],
        });
        policy.models.push(ModelCorrection {
            fingerprint: fp2.to_string(),
            notes: "Model 2".to_string(),
            corrections: vec![],
        });

        let map = policy.build_map();
        assert_eq!(map.len(), 2);
        assert!(map.contains_key(fp1));
        assert!(map.contains_key(fp2));
    }

    #[test]
    fn test_i2s_dequant_override_yaml_parsing() {
        let yaml = r#"
version: 1
models:
  - fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    notes: "Test model with inverted I2_S scales"
    corrections:
      - type: I2S_DEQUANT_OVERRIDE
        tensors:
          - "q_proj.weight"
          - "k_proj.weight"
          - "v_proj.weight"
        inv: true
        k: 1.0
"#;

        let policy = CorrectionPolicy::from_yaml(yaml).unwrap();
        assert_eq!(policy.version, 1);
        assert_eq!(policy.models.len(), 1);
        assert_eq!(policy.models[0].corrections.len(), 1);

        match &policy.models[0].corrections[0] {
            CorrectionAction::I2SDequantOverride { tensors, inv, k } => {
                assert_eq!(tensors.len(), 3);
                assert!(tensors.contains(&"q_proj.weight".to_string()));
                assert!(tensors.contains(&"k_proj.weight".to_string()));
                assert!(tensors.contains(&"v_proj.weight".to_string()));
                assert!(*inv);
                assert_eq!(*k, 1.0);
            }
            _ => panic!("Expected I2SDequantOverride"),
        }
    }

    #[test]
    fn test_i2s_dequant_override_validation() {
        let mut policy = CorrectionPolicy::default();
        policy.models.push(ModelCorrection {
            fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            notes: "Valid I2S override".to_string(),
            corrections: vec![CorrectionAction::I2SDequantOverride {
                tensors: vec!["q_proj.weight".to_string()],
                inv: true,
                k: 1.0,
            }],
        });

        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_i2s_dequant_override_validation_empty_tensors() {
        let mut policy = CorrectionPolicy::default();
        policy.models.push(ModelCorrection {
            fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            notes: "Invalid - empty tensors".to_string(),
            corrections: vec![CorrectionAction::I2SDequantOverride {
                tensors: vec![],
                inv: false,
                k: 1.0,
            }],
        });

        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_i2s_dequant_override_validation_invalid_k() {
        let mut policy = CorrectionPolicy::default();
        policy.models.push(ModelCorrection {
            fingerprint: "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
            notes: "Invalid - negative k".to_string(),
            corrections: vec![CorrectionAction::I2SDequantOverride {
                tensors: vec!["q_proj.weight".to_string()],
                inv: false,
                k: -1.0,
            }],
        });

        assert!(policy.validate().is_err());
    }
}
