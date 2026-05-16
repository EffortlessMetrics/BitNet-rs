//! Model inspection commands for diagnostics and debugging.
//!
//! The command type and CLI argument surface live in this module; the
//! actual work is broken into single-responsibility submodules so that
//! each stage of the pipeline can be tested and reasoned about
//! independently:
//!
//! 1. [`model_io`] opens and hashes the model file once.
//! 2. [`ruleset_selector`] resolves the validation ruleset for the
//!    requested gate mode.
//! 3. [`tensor_scanner`] walks the GGUF tensor table, decoding
//!    LayerNorm/projection weights via [`tensor_decode`] and applying
//!    the ruleset.
//! 4. [`output`] renders the aggregated results as JSON or text.
//!
//! The orchestrator on `InspectCommand` only wires these stages together
//! and applies the strict-mode exit-code policy at the end.

mod model_io;
mod output;
mod ruleset_selector;
mod tensor_decode;
mod tensor_scanner;

use anyhow::Result;
use bitnet_models::formats::gguf::GgufReader;
use clap::Args;
use std::path::PathBuf;

use output::OutputContext;

/// Inspect command arguments
#[derive(Args)]
pub struct InspectCommand {
    /// Model file path
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Compute and display LayerNorm gamma statistics
    #[arg(long)]
    pub ln_stats: bool,

    /// Gate behavior: none|auto|policy
    #[arg(long, default_value = "auto")]
    pub gate: String,

    /// Policy file (YAML) for custom validation rules
    #[arg(long)]
    pub policy: Option<PathBuf>,

    /// Policy key (architecture ID) for rules lookup
    #[arg(long)]
    pub policy_key: Option<String>,

    /// Output format as JSON
    #[arg(long, default_value_t = false)]
    pub json: bool,
}

impl InspectCommand {
    pub async fn execute(&self) -> Result<()> {
        if self.ln_stats {
            self.check_ln_gamma_stats().await
        } else {
            anyhow::bail!(
                "No inspection mode specified. Use --ln-stats to check LayerNorm gamma statistics."
            );
        }
    }

    /// Check LayerNorm gamma statistics with architecture-aware validation.
    async fn check_ln_gamma_stats(&self) -> Result<()> {
        let loaded = model_io::open_and_hash(&self.model)?;
        let reader = GgufReader::new(&loaded.mmap)?;

        let strict_mode = read_strict_mode();

        let rules = ruleset_selector::select_ruleset(
            &reader,
            &self.gate,
            self.policy.as_deref(),
            self.policy_key.as_deref(),
        )?;

        let scan = tensor_scanner::scan(&reader, &rules)?;

        let ctx =
            OutputContext { model_sha256: &loaded.sha256, ruleset_name: &rules.name, strict_mode };

        if self.json {
            output::write_json(&ctx, &scan)?;
        } else {
            output::write_text(&ctx, &scan)?;
        }

        if scan.total_bad() > 0 && strict_mode {
            std::process::exit(crate::exit::EXIT_LN_SUSPICIOUS);
        }

        Ok(())
    }
}

/// Read the `BITNET_STRICT_MODE` environment variable as a boolean.
fn read_strict_mode() -> bool {
    std::env::var("BITNET_STRICT_MODE")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}
