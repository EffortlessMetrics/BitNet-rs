//! Runtime fingerprint helpers shared by receipts and CLI tools.

use std::collections::HashMap;

/// Collect common runtime variables used in inference receipts.
///
/// Includes deterministic knobs when present and always reports Rust/BitNet/OS/CPU identity.
pub fn collect_runtime_fingerprint(bitnet_version: &str) -> HashMap<String, String> {
    let mut env_vars = HashMap::new();

    for key in ["BITNET_DETERMINISTIC", "BITNET_SEED", "RAYON_NUM_THREADS", "BITNET_GGUF"] {
        if let Ok(val) = std::env::var(key) {
            env_vars.insert(key.to_string(), val);
        }
    }

    env_vars.insert("RUST_VERSION".to_string(), rustc_version_runtime::version().to_string());
    env_vars.insert("BITNET_VERSION".to_string(), bitnet_version.to_string());
    env_vars
        .insert("OS".to_string(), format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH));
    env_vars.insert("CPU_BRAND".to_string(), detect_cpu_brand());

    env_vars
}

/// Detect CPU brand/model string (best-effort).
pub fn detect_cpu_brand() -> String {
    if cfg!(target_os = "linux") {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if let Some(brand) = line.strip_prefix("model name\t: ") {
                    return brand.trim().to_string();
                }
            }
        }
    }

    if cfg!(target_os = "macos") {
        if let Ok(output) =
            std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output()
            && output.status.success()
            && let Ok(brand) = String::from_utf8(output.stdout)
        {
            return brand.trim().to_string();
        }
    }

    std::env::consts::ARCH.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(bitnet_env)]
    fn includes_required_static_keys() {
        let vars = collect_runtime_fingerprint("9.9.9-test");
        assert!(vars.contains_key("RUST_VERSION"));
        assert_eq!(vars.get("BITNET_VERSION"), Some(&"9.9.9-test".to_string()));
        assert!(vars.contains_key("OS"));
        assert!(vars.contains_key("CPU_BRAND"));
    }

    #[test]
    #[serial(bitnet_env)]
    fn includes_optional_env_when_present() {
        temp_env::with_vars(
            [
                ("BITNET_DETERMINISTIC", Some("1")),
                ("BITNET_SEED", Some("42")),
                ("RAYON_NUM_THREADS", Some("8")),
                ("BITNET_GGUF", Some("/tmp/model.gguf")),
            ],
            || {
                let vars = collect_runtime_fingerprint("0.0.0");
                assert_eq!(vars.get("BITNET_DETERMINISTIC"), Some(&"1".to_string()));
                assert_eq!(vars.get("BITNET_SEED"), Some(&"42".to_string()));
                assert_eq!(vars.get("RAYON_NUM_THREADS"), Some(&"8".to_string()));
                assert_eq!(vars.get("BITNET_GGUF"), Some(&"/tmp/model.gguf".to_string()));
            },
        );
    }
}
