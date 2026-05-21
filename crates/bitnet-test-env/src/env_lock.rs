use std::sync::{Mutex, OnceLock};

/// Global lock to serialize environment variable modifications across threads.
static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn get_env_lock() -> &'static Mutex<()> {
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}
