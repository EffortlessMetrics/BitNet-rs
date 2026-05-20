use std::time::{Duration, Instant};

use crate::{ClientInfo, RequestId};

/// Request context carrying metadata through the pipeline.
#[derive(Debug)]
pub struct RequestContext {
    pub id: RequestId,
    pub created_at: Instant,
    pub model_id: Option<String>,
    pub client: ClientInfo,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stream: bool,
    deadlines: Option<Duration>,
}

impl RequestContext {
    pub fn new(id: RequestId) -> Self {
        Self {
            id,
            created_at: Instant::now(),
            model_id: None,
            client: ClientInfo::default(),
            max_tokens: 256,
            temperature: 1.0,
            stream: false,
            deadlines: None,
        }
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_id = Some(model.into());
        self
    }

    #[must_use]
    pub const fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    #[must_use]
    pub const fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    #[must_use]
    pub const fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    #[must_use]
    pub const fn with_deadline(mut self, timeout: Duration) -> Self {
        self.deadlines = Some(timeout);
        self
    }

    #[must_use]
    pub fn with_client(mut self, client: ClientInfo) -> Self {
        self.client = client;
        self
    }

    /// Time elapsed since request creation.
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check if the request has exceeded its deadline.
    pub fn is_expired(&self) -> bool {
        self.deadlines.is_some_and(|deadline| self.elapsed() > deadline)
    }

    /// Remaining time before deadline (None if no deadline or expired).
    pub fn remaining(&self) -> Option<Duration> {
        self.deadlines.and_then(|d| d.checked_sub(self.elapsed()))
    }
}
