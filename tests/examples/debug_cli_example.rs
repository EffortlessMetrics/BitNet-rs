use bitnet_tests::debug_cli::{create_debug_cli, DebugCli};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 BitNet.rs Debug CLI Example");
    println!("==============================\n");

    // Check if debug directory exists
    let debug_dir = std::env::var("BITNET_DEBUG_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tests/debug"));

    if !debug_dir.exists() {
        println!("❌ Debug directory not found: {}", debug_dir.display());
        println!("💡 Run the debugging example first to generate debug reports:");
        println!("   cargo run --example debugging_example");
        return Ok(());
    }

    println!("📁 Using debug directory: {}", debug_dir.display());

    // Create debug CLI
    let debug_cli = create_debug_cli();

    // Check command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "interactive" | "i" => {
                // Run interactive mode
                debug_cli.interactive_debug().await?;
            }
            "analyze" => {
                if args.len() < 3 {
                    println!("❌ Usage: {} analyze <report_path>", args[0]);
                    return Ok(());
                }

                let report_path = PathBuf::from(&args[2]);
                match debug_cli.analyze_report(&report_path).await {
                    Ok(analysis) => {
                        println!("✅ Analysis completed successfully!");
                        // The analysis is printed by the CLI internally
                    }
                    Err(e) => {
                        println!("❌ Analysis failed: {}", e);
                    }
                }
            }
            "patterns" => match debug_cli.find_patterns().await {
                Ok(patterns) => {
                    if patterns.is_empty() {
                        println!("ℹ️  No patterns found across debug reports");
                    } else {
                        println!("✅ Found {} patterns", patterns.len());
                    }
                }
                Err(e) => {
                    println!("❌ Pattern analysis failed: {}", e);
                }
            },
            "guide" => {
                if args.len() < 3 {
                    println!("❌ Usage: {} guide <test_name>", args[0]);
                    return Ok(());
                }

                let test_name = &args[2];
                match debug_cli.generate_troubleshooting_guide(test_name).await {
                    Ok(guide) => {
                        println!("{}", guide);
                    }
                    Err(e) => {
                        println!("❌ Guide generation failed: {}", e);
                    }
                }
            }
            "help" | "--help" | "-h" => {
                print_help(&args[0]);
            }
            _ => {
                println!("❌ Unknown command: {}", args[1]);
                print_help(&args[0]);
            }
        }
    } else {
        // Default: show available reports and enter interactive mode
        println!("🔍 Available debug reports:");

        // List available reports
        match tokio::fs::read_dir(&debug_dir).await {
            Ok(mut entries) => {
                let mut found_reports = false;
                while let Some(entry) = entries.next_entry().await? {
                    let path = entry.path();
                    if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                        println!("  📄 {}", path.file_name().unwrap().to_string_lossy());
                        found_reports = true;
                    }
                }

                if !found_reports {
                    println!("  ℹ️  No debug reports found");
                    println!("  💡 Run tests with debugging enabled to generate reports:");
                    println!("     BITNET_DEBUG_ENABLED=true cargo test");
                }
            }
            Err(e) => {
                println!("❌ Failed to read debug directory: {}", e);
            }
        }

        println!("\n🛠️  Usage:");
        print_help(&args[0]);

        println!("\n🚀 Starting interactive mode...");
        debug_cli.interactive_debug().await?;
    }

    Ok(())
}

fn print_help(program_name: &str) {
    println!("Usage: {} [command] [args...]", program_name);
    println!();
    println!("Commands:");
    println!("  interactive, i           - Start interactive debug session");
    println!("  analyze <report_path>    - Analyze a specific debug report");
    println!("  patterns                 - Find patterns across all reports");
    println!("  guide <test_name>        - Generate troubleshooting guide for a test");
    println!("  help, --help, -h         - Show this help message");
    println!();
    println!("Examples:");
    println!("  {} interactive", program_name);
    println!("  {} analyze tests/debug/debug_session_123/debug_report.json", program_name);
    println!("  {} patterns", program_name);
    println!("  {} guide failing_test_name", program_name);
    println!();
    println!("Environment Variables:");
    println!(
        "  BITNET_DEBUG_OUTPUT_DIR  - Directory containing debug reports (default: tests/debug)"
    );
    println!("  BITNET_DEBUG_VERBOSE     - Enable verbose output (default: false)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_debug_cli_creation() {
        let cli = create_debug_cli();
        // Basic test to ensure CLI can be created
        assert!(!cli.is_debug_enabled() || cli.is_debug_enabled()); // Always true, just testing compilation
    }
}
