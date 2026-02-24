// SPDX-License-Identifier: MIT OR Apache-2.0
//! Architecture-aware LayerNorm and projection weight validation rules.
//!
//! This crate provides shared validation logic used by:
//! - `bitnet-cli` (GGUF model inspection)
//! - `bitnet-st-tools` (SafeTensors export tools)
//!
//! # Quick start
//!
//! ```rust
//! use bitnet_validation::rules::{detect_rules, Ruleset};
//!
//! // Auto-detect ruleset from GGUF metadata
//! let ruleset = detect_rules("bitnet", 1); // arch="bitnet", file_type=1 (F16)
//! assert_eq!(ruleset.name, "bitnet-b1.58:f16");
//!
//! // Check if a LayerNorm weight is in the valid RMS envelope
//! let is_valid = ruleset.check_ln("blk.0.attn_norm.weight", 0.8);
//! assert!(is_valid);
//! ```

pub mod names;
pub mod rules;

pub use names::is_ln_gamma;
pub use rules::{Ruleset, Threshold, detect_rules, load_policy};
pub use rules::{rules_bitnet_b158_f16, rules_bitnet_b158_i2s, rules_generic};
