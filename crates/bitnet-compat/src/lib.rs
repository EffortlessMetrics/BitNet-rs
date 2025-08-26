//! BitNet GGUF compatibility layer
//!
//! This crate provides utilities to ensure GGUF model files are compatible with
//! both BitNet.rs and llama.cpp. It can diagnose and fix common compatibility
//! issues in GGUF files.
//!
//! # Main Features
//!
//! - **Diagnosis**: Detect missing or incorrect metadata in GGUF files
//! - **Auto-fixing**: Non-destructively patch GGUF files with missing metadata
//! - **Validation**: Ensure fixes are idempotent and preserve model integrity
//!
//! # Example
//!
//! ```no_run
//! use bitnet_compat::GgufCompatibilityFixer;
//!
//! // Diagnose issues in a GGUF file
//! let issues = GgufCompatibilityFixer::diagnose("model.gguf")?;
//! if !issues.is_empty() {
//!     println!("Found {} compatibility issues", issues.len());
//!     
//!     // Export a fixed version
//!     GgufCompatibilityFixer::export_fixed("model.gguf", "model_fixed.gguf")?;
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod gguf_fixer;

pub use gguf_fixer::GgufCompatibilityFixer;
