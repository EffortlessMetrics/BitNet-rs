use std::{collections::HashMap, env};

pub(crate) fn save_original_value(saved: &mut HashMap<String, Option<String>>, key: &str) {
    saved.entry(key.to_string()).or_insert_with(|| env::var(key).ok());
}

pub(crate) fn restore_saved_value(key: &str, original: &Option<String>) {
    // SAFETY: Caller guarantees ENV_LOCK is held while mutating process env.
    unsafe {
        match original {
            Some(value) => env::set_var(key, value),
            None => env::remove_var(key),
        }
    }
}
