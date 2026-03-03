//! Server-Sent Events (SSE) streaming response builder.
//!
//! Constructs SSE-formatted responses for token-by-token streaming
//! during inference, compatible with OpenAI API format.

/// A single SSE event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
}

impl SseEvent {
    pub fn data(data: impl Into<String>) -> Self {
        Self { event: None, data: data.into(), id: None }
    }

    pub fn named(event: impl Into<String>, data: impl Into<String>) -> Self {
        Self { event: Some(event.into()), data: data.into(), id: None }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Format as SSE wire format.
    pub fn to_sse(&self) -> String {
        let mut out = String::new();
        if let Some(ref id) = self.id {
            out.push_str(&format!("id: {id}\n"));
        }
        if let Some(ref event) = self.event {
            out.push_str(&format!("event: {event}\n"));
        }
        for line in self.data.lines() {
            out.push_str(&format!("data: {line}\n"));
        }
        if self.data.is_empty() {
            out.push_str("data: \n");
        }
        out.push('\n');
        out
    }
}

/// Builder for streaming generation responses.
#[derive(Debug)]
pub struct StreamingResponseBuilder {
    request_id: String,
    model: String,
    events: Vec<SseEvent>,
    token_count: usize,
}

impl StreamingResponseBuilder {
    pub fn new(request_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            model: model.into(),
            events: Vec::new(),
            token_count: 0,
        }
    }

    /// Add a token chunk event.
    pub fn add_token(&mut self, token_text: &str) -> &mut Self {
        self.token_count += 1;
        let data = format!(
            r#"{{"id":"{}","model":"{}","choices":[{{"index":0,"delta":{{"content":"{}"}}}}]}}"#,
            self.request_id,
            self.model,
            escape_json(token_text),
        );
        self.events.push(SseEvent::data(data));
        self
    }

    /// Add the final done event.
    pub fn finish(&mut self, finish_reason: &str) -> &mut Self {
        let data = format!(
            r#"{{"id":"{}","model":"{}","choices":[{{"index":0,"delta":{{}},"finish_reason":"{}"}}]}}"#,
            self.request_id, self.model, finish_reason,
        );
        self.events.push(SseEvent::data(data));
        self.events.push(SseEvent::data("[DONE]"));
        self
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Render all events as a complete SSE stream.
    pub fn render(&self) -> String {
        self.events.iter().map(|e| e.to_sse()).collect()
    }

    /// Get events as a slice.
    pub fn events(&self) -> &[SseEvent] {
        &self.events
    }
}

/// Create a heartbeat/keepalive event.
pub fn heartbeat_event() -> SseEvent {
    SseEvent::named("heartbeat", "")
}

/// Create an error event.
pub fn error_event(message: &str) -> SseEvent {
    SseEvent::named("error", format!(r#"{{"error":"{}"}}"#, escape_json(message)))
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_data_only() {
        let e = SseEvent::data("hello");
        let sse = e.to_sse();
        assert!(sse.contains("data: hello\n"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_sse_named() {
        let e = SseEvent::named("token", "world");
        let sse = e.to_sse();
        assert!(sse.contains("event: token\n"));
        assert!(sse.contains("data: world\n"));
    }

    #[test]
    fn test_sse_with_id() {
        let e = SseEvent::data("test").with_id("42");
        let sse = e.to_sse();
        assert!(sse.contains("id: 42\n"));
    }

    #[test]
    fn test_multiline_data() {
        let e = SseEvent::data("line1\nline2");
        let sse = e.to_sse();
        assert!(sse.contains("data: line1\n"));
        assert!(sse.contains("data: line2\n"));
    }

    #[test]
    fn test_streaming_builder() {
        let mut b = StreamingResponseBuilder::new("req-1", "phi-4");
        b.add_token("Hello");
        b.add_token(" world");
        b.finish("stop");
        assert_eq!(b.token_count(), 2);
        assert_eq!(b.event_count(), 4); // 2 tokens + finish + DONE
    }

    #[test]
    fn test_render() {
        let mut b = StreamingResponseBuilder::new("id", "m");
        b.add_token("hi");
        let rendered = b.render();
        assert!(rendered.contains("data: "));
        assert!(rendered.contains("\"content\":\"hi\""));
    }

    #[test]
    fn test_finish_event() {
        let mut b = StreamingResponseBuilder::new("id", "m");
        b.finish("max_tokens");
        let rendered = b.render();
        assert!(rendered.contains("max_tokens"));
        assert!(rendered.contains("[DONE]"));
    }

    #[test]
    fn test_heartbeat() {
        let e = heartbeat_event();
        let sse = e.to_sse();
        assert!(sse.contains("event: heartbeat"));
    }

    #[test]
    fn test_error_event() {
        let e = error_event("something broke");
        let sse = e.to_sse();
        assert!(sse.contains("event: error"));
        assert!(sse.contains("something broke"));
    }

    #[test]
    fn test_escape_json() {
        let mut b = StreamingResponseBuilder::new("id", "m");
        b.add_token("say \"hi\"");
        let rendered = b.render();
        assert!(rendered.contains("\\\"hi\\\""));
    }

    #[test]
    fn test_empty_builder() {
        let b = StreamingResponseBuilder::new("id", "m");
        assert_eq!(b.token_count(), 0);
        assert_eq!(b.event_count(), 0);
        assert!(b.render().is_empty());
    }

    #[test]
    fn test_empty_data_event() {
        let e = SseEvent::data("");
        let sse = e.to_sse();
        assert!(sse.contains("data: \n"));
    }
}
