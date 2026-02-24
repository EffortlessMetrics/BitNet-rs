// SPDX-License-Identifier: MIT OR Apache-2.0
//! Architecture-aware LayerNorm and projection weight validation rules.
//!
//! This is a thin re-export shim; the implementation lives in `bitnet-validation`.

#[allow(unused_imports)]
pub use bitnet_validation::{
    Ruleset, Threshold, detect_rules, load_policy, rules_bitnet_b158_f16, rules_bitnet_b158_i2s,
    rules_generic,
};
