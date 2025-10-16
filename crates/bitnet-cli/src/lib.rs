//! BitNet CLI library
//!
//! This library exposes internal modules for testing purposes.

#[cfg(feature = "full-cli")]
pub mod commands;
pub mod config;
pub mod exit;
pub mod ln_rules;

/// Build the CLI command for external use (e.g., in tests)
/// This duplicates the CLI structure from main.rs for library export
pub fn build_cli() -> clap::Command {
    use clap::CommandFactory;

    // Import the Cli struct from main to build command
    // This requires the main module structure
    // For now, we'll create a simple wrapper

    #[derive(clap::Parser)]
    #[command(name = "bitnet")]
    #[command(about = "BitNet.rs — 1-bit neural network inference with strict receipts")]
    #[command(version)]
    #[command(author = "BitNet Contributors")]
    #[command(
        after_help = "CLI Interface Version: 1.0.0\nDocs: https://docs.rs/bitnet\nIssues: https://github.com/EffortlessMetrics/BitNet-rs/issues"
    )]
    struct CliStub {}

    CliStub::command()
}
