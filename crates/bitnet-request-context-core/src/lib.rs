//! Request context metadata primitives shared by server-side inference flows.
//!
//! Carries per-request metadata through an inference pipeline:
//! request ID, timing, model selection, and client info.

mod batch;
mod client;
mod context;
mod id;

pub use batch::RequestBatch;
pub use client::ClientInfo;
pub use context::RequestContext;
pub use id::RequestId;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_request_id() {
        let id = RequestId::new("test-123");
        assert_eq!(id.as_str(), "test-123");
        assert_eq!(format!("{id}"), "test-123");
    }

    #[test]
    fn test_sequential_id() {
        let id = RequestId::sequential(42);
        assert_eq!(id.as_str(), "req-42");
    }

    #[test]
    fn test_context_builder() {
        let ctx = RequestContext::new(RequestId::new("r1"))
            .with_model("phi-4")
            .with_max_tokens(100)
            .with_temperature(0.7)
            .with_stream(true);
        assert_eq!(ctx.model_id.as_deref(), Some("phi-4"));
        assert_eq!(ctx.max_tokens, 100);
        assert!(ctx.stream);
    }

    #[test]
    fn test_elapsed() {
        let ctx = RequestContext::new(RequestId::new("r"));
        std::thread::sleep(Duration::from_millis(10));
        assert!(ctx.elapsed() >= Duration::from_millis(5));
    }

    #[test]
    fn test_not_expired() {
        let ctx = RequestContext::new(RequestId::new("r")).with_deadline(Duration::from_mins(1));
        assert!(!ctx.is_expired());
    }

    #[test]
    fn test_expired() {
        let ctx = RequestContext::new(RequestId::new("r")).with_deadline(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        assert!(ctx.is_expired());
    }

    #[test]
    fn test_no_deadline() {
        let ctx = RequestContext::new(RequestId::new("r"));
        assert!(!ctx.is_expired());
        assert!(ctx.remaining().is_none());
    }

    #[test]
    fn test_remaining() {
        let ctx = RequestContext::new(RequestId::new("r")).with_deadline(Duration::from_mins(1));
        assert!(ctx.remaining().is_some());
    }

    #[test]
    fn test_batch() {
        let mut batch = RequestBatch::new();
        batch.add(RequestContext::new(RequestId::new("r1")).with_max_tokens(50));
        batch.add(RequestContext::new(RequestId::new("r2")).with_max_tokens(100).with_stream(true));
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.total_max_tokens(), 150);
        assert!(batch.has_streaming());
    }

    #[test]
    fn test_batch_empty() {
        let batch = RequestBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.total_max_tokens(), 0);
        assert!(!batch.has_streaming());
    }

    #[test]
    fn test_client_info() {
        let client = ClientInfo {
            ip: Some("127.0.0.1".into()),
            user_agent: Some("test/1.0".into()),
            api_key_id: None,
        };
        let ctx = RequestContext::new(RequestId::new("r")).with_client(client);
        assert_eq!(ctx.client.ip.as_deref(), Some("127.0.0.1"));
    }

    #[test]
    fn test_default_values() {
        let ctx = RequestContext::new(RequestId::new("r"));
        assert_eq!(ctx.max_tokens, 256);
        assert!((ctx.temperature - 1.0).abs() < f32::EPSILON);
        assert!(!ctx.stream);
    }
}
