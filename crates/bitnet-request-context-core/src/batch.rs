use crate::RequestContext;

/// Builder for batch request contexts.
#[derive(Debug)]
pub struct RequestBatch {
    requests: Vec<RequestContext>,
}

impl RequestBatch {
    pub const fn new() -> Self {
        Self { requests: Vec::new() }
    }

    pub fn add(&mut self, ctx: RequestContext) {
        self.requests.push(ctx);
    }

    pub const fn len(&self) -> usize {
        self.requests.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn total_max_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.max_tokens).sum()
    }

    pub fn has_streaming(&self) -> bool {
        self.requests.iter().any(|r| r.stream)
    }

    pub fn iter(&self) -> impl Iterator<Item = &RequestContext> {
        self.requests.iter()
    }

    pub fn expired_count(&self) -> usize {
        self.requests.iter().filter(|r| r.is_expired()).count()
    }
}

impl Default for RequestBatch {
    fn default() -> Self {
        Self::new()
    }
}
