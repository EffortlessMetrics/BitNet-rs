#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CliConfigInput {
    raw_json: Vec<u8>,
    raw_toml: Vec<u8>,
    device: Vec<u8>,
    level: Vec<u8>,
    format: Vec<u8>,
    cpu_threads: Option<u32>,
    batch_size: u32,
    memory_opt: bool,
    timestamps: bool,
}

fuzz_target!(|input: CliConfigInput| {
    // Fuzz CliConfig deserialization from arbitrary JSON.
    if let Ok(s) = std::str::from_utf8(&input.raw_json) {
        if s.len() <= 4096 {
            let _ = serde_json::from_str::<bitnet_cli::config::CliConfig>(s);
        }
    }

    // Fuzz CliConfig deserialization from arbitrary TOML.
    if let Ok(s) = std::str::from_utf8(&input.raw_toml) {
        if s.len() <= 4096 {
            // TOML parsing (via serde) must never panic.
            let _ = toml::from_str::<bitnet_cli::config::CliConfig>(s);
        }
    }

    // Construct structured JSON from arbitrary fields.
    let device = std::str::from_utf8(&input.device).unwrap_or("auto");
    let level = std::str::from_utf8(&input.level).unwrap_or("info");
    let format = std::str::from_utf8(&input.format).unwrap_or("pretty");

    let obj = serde_json::json!({
        "default_device": device,
        "logging": {
            "level": level,
            "format": format,
            "timestamps": input.timestamps,
        },
        "performance": {
            "cpu_threads": input.cpu_threads,
            "batch_size": input.batch_size,
            "memory_optimization": input.memory_opt,
        }
    });

    let _ = serde_json::from_value::<bitnet_cli::config::CliConfig>(obj);

    // Edge: zero batch size, huge thread count.
    let edge = serde_json::json!({
        "default_device": "cpu",
        "logging": { "level": "error", "format": "json", "timestamps": false },
        "performance": {
            "cpu_threads": u64::MAX,
            "batch_size": 0,
            "memory_optimization": true,
        }
    });
    let _ = serde_json::from_value::<bitnet_cli::config::CliConfig>(edge);

    // Also test build_cli() doesn't panic on arbitrary args.
    let cli = bitnet_cli::build_cli();
    let args: Vec<String> = input
        .raw_json
        .chunks(8)
        .take(16)
        .filter_map(|chunk| std::str::from_utf8(chunk).ok())
        .map(|s| s.to_string())
        .collect();
    let mut full_args = vec!["bitnet".to_string()];
    full_args.extend(args);
    let _ = cli.try_get_matches_from(full_args);
});
