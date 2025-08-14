//! Streaming session management for the C API
//!
//! This module provides streaming session management for callback-based
//! token delivery and concurrent streaming operations.

use crate::{BitNetCError, StreamingSession};
use std::collections::HashMap;
use std::sync::Mutex;

/// Global streaming session manager
pub struct StreamingManager {
    sessions: Mutex<HashMap<u32, StreamingSession>>,
    next_id: Mutex<u32>,
}

impl StreamingManager {
    pub fn new() -> Self {
        Self { sessions: Mutex::new(HashMap::new()), next_id: Mutex::new(0) }
    }

    /// Store a streaming session and return its ID
    pub fn store_session(&self, session: StreamingSession) -> Result<u32, BitNetCError> {
        let session_id = {
            let mut next_id = self.next_id.lock().map_err(|_| {
                BitNetCError::ThreadSafety("Failed to acquire next ID lock".to_string())
            })?;
            let id = *next_id;
            *next_id += 1;
            id
        };

        let mut sessions = self.sessions.lock().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire sessions lock".to_string())
        })?;
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Remove a streaming session
    pub fn remove_session(&self, session_id: u32) -> Result<(), BitNetCError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire sessions lock".to_string())
        })?;

        match sessions.remove(&session_id) {
            Some(_) => Ok(()),
            None => {
                Err(BitNetCError::InvalidArgument(format!("Stream ID {} not found", session_id)))
            }
        }
    }

    /// Get the next token from a streaming session
    pub fn get_next_token(&self, session_id: u32) -> Result<Option<String>, BitNetCError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire sessions lock".to_string())
        })?;

        match sessions.get_mut(&session_id) {
            Some(session) => session.next_token(),
            None => {
                Err(BitNetCError::InvalidArgument(format!("Stream ID {} not found", session_id)))
            }
        }
    }

    /// Check if a streaming session exists
    pub fn has_session(&self, session_id: u32) -> Result<bool, BitNetCError> {
        let sessions = self.sessions.lock().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire sessions lock".to_string())
        })?;
        Ok(sessions.contains_key(&session_id))
    }

    /// Get the number of active streaming sessions
    pub fn active_session_count(&self) -> Result<usize, BitNetCError> {
        let sessions = self.sessions.lock().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire sessions lock".to_string())
        })?;
        Ok(sessions.len())
    }

    /// Clean up finished sessions
    pub fn cleanup_finished_sessions(&self) -> Result<usize, BitNetCError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire sessions lock".to_string())
        })?;

        let initial_count = sessions.len();
        sessions.retain(|_, session| !session.is_finished());
        let final_count = sessions.len();

        Ok(initial_count - final_count)
    }
}

// Global streaming manager instance
static STREAMING_MANAGER: std::sync::OnceLock<StreamingManager> = std::sync::OnceLock::new();

/// Get the global streaming manager instance
pub fn get_streaming_manager() -> &'static StreamingManager {
    STREAMING_MANAGER.get_or_init(|| StreamingManager::new())
}

/// Store a streaming session and return its ID
pub fn store_streaming_session(session: StreamingSession) -> u32 {
    match get_streaming_manager().store_session(session) {
        Ok(id) => id,
        Err(_) => u32::MAX, // Return invalid ID on error
    }
}

/// Remove a streaming session
pub fn remove_streaming_session(session_id: u32) -> Result<(), BitNetCError> {
    get_streaming_manager().remove_session(session_id)
}

/// Get the next token from a streaming session
pub fn get_next_token(session_id: u32) -> Result<Option<String>, BitNetCError> {
    get_streaming_manager().get_next_token(session_id)
}

/// Check if a streaming session exists
pub fn has_streaming_session(session_id: u32) -> bool {
    get_streaming_manager().has_session(session_id).unwrap_or(false)
}

/// Get the number of active streaming sessions
pub fn get_active_session_count() -> usize {
    get_streaming_manager().active_session_count().unwrap_or(0)
}

/// Clean up finished streaming sessions
pub fn cleanup_finished_sessions() -> usize {
    get_streaming_manager().cleanup_finished_sessions().unwrap_or(0)
}

/// Streaming callback wrapper for C callbacks
pub struct CallbackWrapper {
    callback: crate::config::BitNetCStreamCallback,
    user_data: *mut std::ffi::c_void,
}

impl CallbackWrapper {
    pub fn new(
        callback: crate::config::BitNetCStreamCallback,
        user_data: *mut std::ffi::c_void,
    ) -> Self {
        Self { callback, user_data }
    }

    /// Call the C callback with a token
    pub fn call(&self, token: &str) -> Result<bool, BitNetCError> {
        if let Some(callback_fn) = self.callback {
            let token_cstr = std::ffi::CString::new(token).map_err(|_| {
                BitNetCError::Internal("Failed to create C string for token".to_string())
            })?;

            let result = callback_fn(token_cstr.as_ptr(), self.user_data);
            Ok(result == 0) // 0 means continue, non-zero means stop
        } else {
            Err(BitNetCError::InvalidArgument("Callback function is null".to_string()))
        }
    }
}

unsafe impl Send for CallbackWrapper {}
unsafe impl Sync for CallbackWrapper {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_manager() {
        let manager = StreamingManager::new();

        // Test initial state
        assert_eq!(manager.active_session_count().unwrap(), 0);
        assert!(!manager.has_session(0).unwrap());

        // Test session storage would require a real StreamingSession
        // This is a placeholder test
        assert_eq!(manager.active_session_count().unwrap(), 0);
    }

    #[test]
    fn test_global_streaming_manager() {
        let manager = get_streaming_manager();
        assert_eq!(manager.active_session_count().unwrap_or(0), 0);
    }

    #[test]
    fn test_cleanup_finished_sessions() {
        let cleaned = cleanup_finished_sessions();
        // Should not panic and return a valid count
        assert!(cleaned == 0); // No sessions to clean initially
    }
}
