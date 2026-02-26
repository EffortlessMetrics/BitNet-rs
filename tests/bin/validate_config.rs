use bitnet_tests::config_validator::ConfigValidator;
use std::env;
use std::path::PathBuf;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    let validator = if args.len() > 1 {
        let config_path = PathBuf::from(&args[1]);
        match ConfigValidator::from_file(&config_path) {
            Ok(validator) => validator,
            Err(e) => {
                eprintln!("Error loading config file {:?}: {}", config_path, e);
                process::exit(1);
            }
        }
    } else {
        match ConfigValidator::new() {
            Ok(validator) => validator,
            Err(e) => {
                eprintln!("Error loading default config: {}", e);
                eprintln!("Try specifying a config file: {} <config-file>", args[0]);
                process::exit(1);
            }
        }
    };

    let result = validator.validate();

    println!("BitNet-rs Test Configuration Validation");
    println!("========================================");
    println!();

    if !result.errors.is_empty() {
        println!("❌ ERRORS:");
        for error in &result.errors {
            println!("  • {}: {}", error.field, error.message);
        }
        println!();
    }

    if !result.warnings.is_empty() {
        println!("⚠️  WARNINGS:");
        for warning in &result.warnings {
            println!("  • {}: {}", warning.field, warning.message);
        }
        println!();
    }

    if !result.info.is_empty() {
        println!("ℹ️  INFO:");
        for info in &result.info {
            println!("  • {}: {}", info.field, info.message);
        }
        println!();
    }

    println!("{}", result.summary());

    if result.is_valid() {
        println!("✅ Configuration is valid!");

        // Print some key configuration values
        let config = validator.config();
        println!();
        println!("Key Settings:");
        println!("  • Max parallel tests: {}", config.max_parallel_tests);
        println!("  • Test timeout: {:?}", config.test_timeout);
        println!("  • Cache directory: {:?}", config.cache_dir);
        println!("  • Log level: {}", config.log_level);
        println!("  • Coverage threshold: {:.1}%", config.coverage_threshold * 100.0);
        println!(
            "  • Cross-validation: {}",
            if config.crossval.enabled { "enabled" } else { "disabled" }
        );
        println!(
            "  • Auto-download fixtures: {}",
            if config.fixtures.auto_download { "enabled" } else { "disabled" }
        );

        process::exit(0);
    } else {
        println!("❌ Configuration has errors and cannot be used.");
        process::exit(1);
    }
}
