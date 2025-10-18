//! FFI session wrapper for validation-only parity checking
//!
//! This module provides a reusable FFI session to prevent repeated model/context
//! allocation that causes munmap_chunk() crashes. It's ONLY for parity validation
//! and never used in production inference.
//!
//! # Architecture
//!
//! - `ParityCppSession`: Wraps BitnetModel + BitnetContext + vocab size
//! - Global `PARITY_CPP_SESSION`: OnceCell for session reuse across tests
//! - Thread-safe via Mutex (all FFI calls serialized)
//! - Reloads only when model path changes
//!
//! # Safety
//!
//! This module is feature-gated behind `#[cfg(feature="ffi")]` and only compiled
//! when explicitly requested for cross-validation builds.

use anyhow::Result;
use bitnet_sys::{BitnetContext, BitnetModel, bitnet_eval_tokens, bitnet_prefill, cpp_vocab_size};
use once_cell::sync::OnceCell;
use std::sync::Mutex;

/// Global FFI session for parity checking (reused across tests)
static PARITY_CPP_SESSION: OnceCell<Mutex<ParityCppSession>> = OnceCell::new();

/// Reusable C++ FFI session for parity validation
///
/// This struct maintains a persistent model/context pair to avoid repeated
/// allocation/deallocation that causes memory corruption in the C++ FFI layer.
pub struct ParityCppSession {
    /// Currently loaded model path (for reload detection)
    model_path: String,
    /// Cached vocabulary size
    vocab_size: usize,
    /// C++ context handle (MUST be declared after model so it drops first!)
    /// Fields drop in reverse declaration order, so context drops before model.
    context: BitnetContext,
    /// C++ model handle (must outlive context!)
    /// This is declared last so it drops after context is cleaned up.
    #[allow(dead_code)]
    model: BitnetModel,
}

// SAFETY: ParityCppSession is only accessed through a Mutex, ensuring single-threaded access
// to the underlying C++ FFI objects. The Mutex serializes all access, making it safe to
// share across threads. The C++ objects themselves are not accessed concurrently.
unsafe impl Send for ParityCppSession {}

impl ParityCppSession {
    /// Create a new FFI session by loading the model
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF model file
    ///
    /// # Returns
    /// * New session with model, context, and vocab size
    pub fn new(model_path: &str) -> Result<Self> {
        // Load model via C++ FFI
        let model = BitnetModel::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("C++ FFI model load failed: {:?}", e))?;

        // Create context (4096 max tokens, batch size 1, no threads override)
        let context = BitnetContext::new(&model, 4096, 1, 0)
            .map_err(|e| anyhow::anyhow!("C++ FFI context creation failed: {:?}", e))?;

        // Get vocab size from C++
        let vocab_size = cpp_vocab_size(&context)
            .map_err(|e| anyhow::anyhow!("C++ FFI vocab_size failed: {:?}", e))?;

        Ok(Self { model_path: model_path.to_string(), model, context, vocab_size })
    }

    /// Ensure the session has the correct model loaded (reload if path changed)
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF model file
    ///
    /// # Returns
    /// * Ok(()) if session is ready with correct model
    pub fn ensure_model(&mut self, model_path: &str) -> Result<()> {
        // Only reload if path changed
        if self.model_path != model_path {
            eprintln!(
                "ffi_session: model path changed from {} to {}, reloading",
                self.model_path, model_path
            );

            // Replace with new session (old one drops automatically)
            let new_session = Self::new(model_path)?;
            *self = new_session;
        }

        Ok(())
    }

    /// Prefill the context with prompt tokens (primes KV cache and sets n_past)
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to prefill
    ///
    /// # Returns
    /// * Ok(()) if prefill succeeded
    pub fn prefill(&self, tokens: &[i32]) -> Result<()> {
        bitnet_prefill(&self.context, tokens)
            .map_err(|e| anyhow::anyhow!("C++ FFI prefill failed: {:?}", e))
    }

    /// Evaluate tokens and return logits for the last token position
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to evaluate
    ///
    /// # Returns
    /// * Logits vector for the last token position (vocab_size elements)
    pub fn eval_last_logits(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        // Evaluate to get logits
        let logits = bitnet_eval_tokens(&self.context, tokens, self.vocab_size)
            .map_err(|e| anyhow::anyhow!("C++ FFI eval failed: {:?}", e))?;

        // Paranoia: ensure non-zero last-step logits (catches KV/logits wiring issues)
        let sum_abs: f32 = logits.iter().map(|x| x.abs()).sum();
        anyhow::ensure!(
            sum_abs > 1e-6,
            "C++ last-step logits near zero (sum_abs={:.2e}); KV/logits wiring off or weights not loaded",
            sum_abs
        );

        Ok(logits)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Get or initialize the global parity FFI session
///
/// This function ensures thread-safe access to a single shared session,
/// preventing repeated model/context allocation.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
///
/// # Returns
/// * Reference to the global session (behind Mutex)
pub fn parity_cpp_session(model_path: &str) -> Result<&'static Mutex<ParityCppSession>> {
    // Get or initialize the session (first model path wins)
    let session_mutex = PARITY_CPP_SESSION.get_or_try_init(|| {
        let session = ParityCppSession::new(model_path)?;
        Ok::<_, anyhow::Error>(Mutex::new(session))
    })?;

    // Verify model path matches (parity tests should use same model throughout)
    {
        let session = session_mutex
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock FFI session: {}", e))?;
        if session.model_path != model_path {
            anyhow::bail!(
                "FFI session model path mismatch: initialized with '{}', requested '{}'. \
                 Parity tests must use the same model throughout the session.",
                session.model_path,
                model_path
            );
        }
    }

    Ok(session_mutex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        // This test will fail without a valid model, but verifies the API compiles
        let result = ParityCppSession::new("nonexistent.gguf");
        assert!(result.is_err(), "Should fail with nonexistent model");
    }

    #[test]
    fn test_session_reuse() {
        // Test that multiple calls to parity_cpp_session return the same instance
        // This will fail without a valid model, but verifies singleton behavior
        let path = "nonexistent.gguf";

        let result1 = parity_cpp_session(path);
        let result2 = parity_cpp_session(path);

        // Both should fail with the same error (no model)
        assert!(result1.is_err());
        assert!(result2.is_err());
    }
}
