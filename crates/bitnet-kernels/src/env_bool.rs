use std::env;

const TRUTHY_VALUES: &[&str] = &["1", "true", "yes", "on"];
const FALSEY_VALUES: &[&str] = &["0", "false", "no", "off"];

fn normalized_env_bool(value: &str) -> Option<bool> {
    let normalized = value.trim().to_ascii_lowercase();

    if TRUTHY_VALUES.contains(&normalized.as_str()) {
        Some(true)
    } else if FALSEY_VALUES.contains(&normalized.as_str()) {
        Some(false)
    } else {
        None
    }
}

pub(crate) fn env_bool(name: &str) -> Option<bool> {
    env::var(name).ok().and_then(|value| normalized_env_bool(&value))
}

pub(crate) fn env_truthy(name: &str) -> bool {
    env_bool(name) == Some(true)
}

pub(crate) fn env_falsey(name: &str) -> bool {
    env_bool(name) == Some(false)
}

#[cfg(test)]
mod tests {
    use super::normalized_env_bool;

    #[test]
    fn normalized_env_bool_accepts_common_truthy_values() {
        for value in ["1", "true", "TRUE", " yes ", "On"] {
            assert_eq!(normalized_env_bool(value), Some(true), "{value}");
        }
    }

    #[test]
    fn normalized_env_bool_accepts_common_falsey_values() {
        for value in ["0", "false", "FALSE", " no ", "Off"] {
            assert_eq!(normalized_env_bool(value), Some(false), "{value}");
        }
    }

    #[test]
    fn normalized_env_bool_rejects_unknown_values() {
        for value in ["", "2", "enable", "disabled"] {
            assert_eq!(normalized_env_bool(value), None, "{value}");
        }
    }
}
