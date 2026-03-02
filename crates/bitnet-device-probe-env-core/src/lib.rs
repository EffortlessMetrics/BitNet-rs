use std::collections::HashSet;

/// Returns true when strict-mode is enabled and fake runtime overrides should be ignored.
#[must_use]
pub fn strict_mode_enabled() -> bool {
    strict_mode_enabled_from_value(std::env::var("BITNET_STRICT_MODE").ok().as_deref())
}

/// Parse strict-mode from an optional env value.
#[must_use]
pub fn strict_mode_enabled_from_value(value: Option<&str>) -> bool {
    value
        .map(|v| {
            let normalized = v.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true"
        })
        .unwrap_or(false)
}

/// Parse a fake GPU backend set from BITNET_GPU_FAKE.
///
/// Returns an empty set for `none` and supports delimiters: `, ; | <space>`.
#[must_use]
pub fn parse_fake_gpu_backends(value: &str) -> HashSet<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized == "none" {
        return HashSet::new();
    }

    normalized
        .split([',', ';', '|', ' '])
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Resolve fake GPU backend overrides from environment unless strict-mode is enabled.
#[must_use]
pub fn fake_gpu_backends() -> Option<HashSet<String>> {
    if strict_mode_enabled() {
        return None;
    }

    let fake = std::env::var("BITNET_GPU_FAKE").ok()?;
    Some(parse_fake_gpu_backends(&fake))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_mode_parsing_works() {
        assert!(strict_mode_enabled_from_value(Some("1")));
        assert!(strict_mode_enabled_from_value(Some("true")));
        assert!(strict_mode_enabled_from_value(Some("TRUE")));
        assert!(!strict_mode_enabled_from_value(Some("0")));
        assert!(!strict_mode_enabled_from_value(Some("false")));
        assert!(!strict_mode_enabled_from_value(None));
    }

    #[test]
    fn fake_backend_parsing_supports_multiple_delimiters() {
        let parsed = parse_fake_gpu_backends("cuda, rocm;oneapi|gpu");
        assert!(parsed.contains("cuda"));
        assert!(parsed.contains("rocm"));
        assert!(parsed.contains("oneapi"));
        assert!(parsed.contains("gpu"));
    }

    #[test]
    fn fake_backend_none_means_empty_override_set() {
        let parsed = parse_fake_gpu_backends("none");
        assert!(parsed.is_empty());
    }
}
