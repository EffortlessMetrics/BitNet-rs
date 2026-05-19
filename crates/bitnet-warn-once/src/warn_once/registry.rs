use std::collections::HashSet;
#[cfg(test)]
use std::sync::TryLockError;
use std::sync::{Mutex, OnceLock};

static WARN_REGISTRY: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

pub(crate) fn record_warning_occurrence(key: &str) -> bool {
    let registry = WARN_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    seen.insert(key.to_string())
}

#[cfg(test)]
pub(crate) fn clear_registry_for_test() {
    let registry = WARN_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    seen.clear();
}

#[cfg(test)]
pub(crate) fn contains_key_for_test(key: &str) -> bool {
    let registry = WARN_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()));
    let seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    seen.contains(key)
}

#[cfg(test)]
pub(crate) fn key_count_for_test() -> usize {
    let registry = WARN_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()));
    let seen = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    seen.len()
}

#[cfg(test)]
pub(crate) fn lock_available_for_test() -> bool {
    let registry = WARN_REGISTRY.get_or_init(|| Mutex::new(HashSet::new()));
    matches!(registry.try_lock(), Ok(_) | Err(TryLockError::Poisoned(_)))
}
