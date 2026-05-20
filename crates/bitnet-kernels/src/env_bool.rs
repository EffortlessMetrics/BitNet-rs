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
    use super::{env_bool, env_falsey, env_truthy, normalized_env_bool};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn env_guard() -> MutexGuard<'static, ()> {
        match env_lock().lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

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

    #[test]
    fn env_bool_reads_and_normalizes_variable_values() {
        let _guard = env_guard();
        let name = "BITNET_RS_ENV_BOOL_TEST";

        unsafe { std::env::set_var(name, "  TrUe  ") };
        assert_eq!(env_bool(name), Some(true));

        unsafe { std::env::set_var(name, "  oFf ") };
        assert_eq!(env_bool(name), Some(false));

        unsafe { std::env::remove_var(name) };
        assert_eq!(env_bool(name), None);
    }

    #[test]
    fn env_truthy_and_env_falsey_match_parsed_value() {
        let _guard = env_guard();
        let name = "BITNET_RS_ENV_BOOL_HELPERS_TEST";

        unsafe { std::env::set_var(name, "yes") };
        assert!(env_truthy(name));
        assert!(!env_falsey(name));

        unsafe { std::env::set_var(name, "0") };
        assert!(!env_truthy(name));
        assert!(env_falsey(name));

        unsafe { std::env::set_var(name, "maybe") };
        assert!(!env_truthy(name));
        assert!(!env_falsey(name));

        unsafe { std::env::remove_var(name) };
    }
}
