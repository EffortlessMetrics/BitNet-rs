## 2024-11-20 - BatchEngine Concurrency Flaw
**Learning:** The `BatchEngine` relies on `tokio::spawn` per request with `Semaphore::try_acquire`. This "greedy worker" pattern fails under high concurrency because blocked workers return immediately, leaving items in the queue if the active worker finishes its batch without checking for more work. A dedicated background loop or recursive check is needed for robust queue draining.
**Action:** When implementing batch processors in Rust/Tokio, prefer a long-running background task (loop) over spawn-per-request to ensure queue draining.
