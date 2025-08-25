//! Runtime facade to allow either Tokio (default) or a wasm-friendly executor.
//! This abstraction allows the inference engine to work on both native and WASM targets.

#[cfg(any(feature = "rt-tokio", feature = "rt-wasm"))]
use std::time::Duration;

#[cfg(feature = "rt-tokio")]
use std::future::Future;

#[cfg(feature = "rt-tokio")]
pub mod time {
    use super::Duration;

    /// Sleep for a duration using the runtime's timer implementation
    pub async fn sleep(d: Duration) {
        tokio::time::sleep(d).await;
    }
}

#[cfg(feature = "rt-tokio")]
pub mod task {
    use super::Future;

    /// Spawn a future on the runtime
    pub fn spawn<F>(f: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        tokio::spawn(f);
    }
}

#[cfg(all(feature = "rt-wasm", not(feature = "rt-tokio")))]
pub mod time {
    use super::Duration;

    /// Sleep for a duration using the runtime's timer implementation
    pub async fn sleep(d: Duration) {
        gloo_timers::future::TimeoutFuture::new(d.as_millis() as u32).await;
    }
}

#[cfg(all(feature = "rt-wasm", not(feature = "rt-tokio")))]
pub mod task {
    use super::Future;

    /// Spawn a future on the runtime
    pub fn spawn<F>(f: F)
    where
        F: Future<Output = ()> + 'static,
    {
        wasm_bindgen_futures::spawn_local(f);
    }
}
