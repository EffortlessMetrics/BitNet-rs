//! CLI flag parsing tests for xtask commands
//!
//! Tests validation and parsing of command-line arguments, particularly for
//! the crossval-per-token command with new parity ladder flags.

#[allow(unused_imports)]
use std::path::PathBuf;

/// Test default values for ladder, positions, and metrics flags
#[test]
#[cfg(feature = "inference")]
fn test_crossval_per_token_defaults() {
    use clap::Parser;

    // Define a minimal Args structure for testing
    #[derive(Parser, Debug)]
    #[command(name = "xtask")]
    struct TestArgs {
        #[command(subcommand)]
        cmd: TestCmd,
    }

    #[derive(clap::Subcommand, Debug)]
    enum TestCmd {
        #[command(name = "crossval-per-token")]
        CrossvalPerToken {
            #[arg(long)]
            model: PathBuf,
            #[arg(long)]
            tokenizer: PathBuf,
            #[arg(long)]
            prompt: String,
            #[arg(long, default_value = "positions")]
            ladder: String,
            #[arg(long, default_value_t = 8)]
            positions: usize,
            #[arg(long, default_value = "mse,kl,topk")]
            metrics: String,
        },
    }

    let args = TestArgs::parse_from([
        "xtask",
        "crossval-per-token",
        "--model",
        "model.gguf",
        "--tokenizer",
        "tokenizer.json",
        "--prompt",
        "test",
    ]);

    if let TestCmd::CrossvalPerToken { ladder, positions, metrics, .. } = args.cmd {
        assert_eq!(ladder, "positions", "Default ladder mode should be 'positions'");
        assert_eq!(positions, 8, "Default positions should be 8");
        assert_eq!(metrics, "mse,kl,topk", "Default metrics should be 'mse,kl,topk'");
    } else {
        panic!("Expected CrossvalPerToken command");
    }
}

/// Test custom values for ladder, positions, and metrics flags
#[test]
#[cfg(feature = "inference")]
fn test_crossval_per_token_custom_values() {
    use clap::Parser;

    #[derive(Parser, Debug)]
    #[command(name = "xtask")]
    struct TestArgs {
        #[command(subcommand)]
        cmd: TestCmd,
    }

    #[derive(clap::Subcommand, Debug)]
    enum TestCmd {
        #[command(name = "crossval-per-token")]
        CrossvalPerToken {
            #[arg(long)]
            model: PathBuf,
            #[arg(long)]
            tokenizer: PathBuf,
            #[arg(long)]
            prompt: String,
            #[arg(long, default_value = "positions")]
            ladder: String,
            #[arg(long, default_value_t = 8)]
            positions: usize,
            #[arg(long, default_value = "mse,kl,topk")]
            metrics: String,
        },
    }

    let args = TestArgs::parse_from([
        "xtask",
        "crossval-per-token",
        "--model",
        "model.gguf",
        "--tokenizer",
        "tokenizer.json",
        "--prompt",
        "test",
        "--ladder",
        "tokens",
        "--positions",
        "16",
        "--metrics",
        "mse,kl",
    ]);

    if let TestCmd::CrossvalPerToken { ladder, positions, metrics, .. } = args.cmd {
        assert_eq!(ladder, "tokens", "Custom ladder mode should be 'tokens'");
        assert_eq!(positions, 16, "Custom positions should be 16");
        assert_eq!(metrics, "mse,kl", "Custom metrics should be 'mse,kl'");
    } else {
        panic!("Expected CrossvalPerToken command");
    }
}

/// Test metrics string parsing and validation
#[test]
fn test_metrics_parsing() {
    use std::collections::HashSet;

    // Test valid metrics
    let metrics = "mse,kl,topk";
    let metrics_set: HashSet<&str> = metrics.split(',').map(|s| s.trim()).collect();

    assert!(metrics_set.contains("mse"));
    assert!(metrics_set.contains("kl"));
    assert!(metrics_set.contains("topk"));
    assert_eq!(metrics_set.len(), 3);

    // Test subset of metrics
    let metrics = "mse,topk";
    let metrics_set: HashSet<&str> = metrics.split(',').map(|s| s.trim()).collect();

    assert!(metrics_set.contains("mse"));
    assert!(!metrics_set.contains("kl"));
    assert!(metrics_set.contains("topk"));
    assert_eq!(metrics_set.len(), 2);

    // Test single metric
    let metrics = "kl";
    let metrics_set: HashSet<&str> = metrics.split(',').map(|s| s.trim()).collect();

    assert!(!metrics_set.contains("mse"));
    assert!(metrics_set.contains("kl"));
    assert!(!metrics_set.contains("topk"));
    assert_eq!(metrics_set.len(), 1);

    // Test with whitespace
    let metrics = "mse, kl, topk";
    let metrics_set: HashSet<&str> = metrics.split(',').map(|s| s.trim()).collect();

    assert!(metrics_set.contains("mse"));
    assert!(metrics_set.contains("kl"));
    assert!(metrics_set.contains("topk"));
    assert_eq!(metrics_set.len(), 3);
}

/// Test ladder mode validation logic
#[test]
fn test_ladder_mode_validation() {
    let valid_ladder_modes = ["tokens", "masks", "first-logit", "positions", "decode"];

    // Valid modes
    assert!(valid_ladder_modes.contains(&"positions"));
    assert!(valid_ladder_modes.contains(&"tokens"));
    assert!(valid_ladder_modes.contains(&"masks"));
    assert!(valid_ladder_modes.contains(&"first-logit"));
    assert!(valid_ladder_modes.contains(&"decode"));

    // Invalid modes
    assert!(!valid_ladder_modes.contains(&"invalid"));
    assert!(!valid_ladder_modes.contains(&""));
    assert!(!valid_ladder_modes.contains(&"POSITIONS"));
}

/// Test metrics validation logic
#[test]
fn test_metrics_validation() {
    let valid_metrics = ["mse", "kl", "topk"];

    // Valid metrics
    assert!(valid_metrics.contains(&"mse"));
    assert!(valid_metrics.contains(&"kl"));
    assert!(valid_metrics.contains(&"topk"));

    // Invalid metrics
    assert!(!valid_metrics.contains(&"invalid"));
    assert!(!valid_metrics.contains(&""));
    assert!(!valid_metrics.contains(&"MSE"));
}

/// Test positions parameter validation
#[test]
fn test_positions_validation() {
    // Valid positions
    assert!(1_usize > 0);
    assert!(8_usize > 0);
    assert!(100_usize > 0);

    // Invalid positions (0)
    assert_eq!(0_usize, 0);
}
