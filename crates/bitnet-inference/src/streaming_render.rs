//! Streaming template rendering for real-time inference output.

/// A chunk of streamed output.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamChunk {
    /// Regular text content.
    Text(String),
    /// A special token was detected.
    SpecialToken(String),
    /// End of sequence marker.
    EndOfSequence,
}

/// State of the streaming renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderState {
    Idle,
    Streaming,
    Complete,
}

/// Streaming renderer that processes tokens and detects end-of-turn markers.
#[derive(Debug, Clone)]
pub struct StreamingRenderer {
    state: RenderState,
    buffer: String,
    tokens_rendered: usize,
    bytes_rendered: usize,
    eot_detector: EotDetector,
}

impl StreamingRenderer {
    pub fn new() -> Self {
        Self {
            state: RenderState::Idle,
            buffer: String::new(),
            tokens_rendered: 0,
            bytes_rendered: 0,
            eot_detector: EotDetector::new(),
        }
    }

    /// Feed a decoded token string, returning output chunks.
    pub fn feed_token(&mut self, token: &str) -> Vec<StreamChunk> {
        if self.state == RenderState::Complete {
            return vec![];
        }
        self.state = RenderState::Streaming;
        self.tokens_rendered += 1;
        self.bytes_rendered += token.len();

        // Check for end-of-turn markers
        if let Some(marker) = self.eot_detector.check(token) {
            self.state = RenderState::Complete;
            let mut chunks = vec![];
            // Flush any buffered text before the marker
            if !self.buffer.is_empty() {
                chunks.push(StreamChunk::Text(self.buffer.drain(..).collect()));
            }
            chunks.push(StreamChunk::SpecialToken(marker));
            chunks.push(StreamChunk::EndOfSequence);
            return chunks;
        }

        // Accumulate partial matches
        self.buffer.push_str(token);

        // Check if buffer might be a partial EoT match
        if self.eot_detector.is_partial_match(&self.buffer) {
            return vec![];
        }

        // No partial match — flush buffer as text
        let text: String = self.buffer.drain(..).collect();
        if text.is_empty() { vec![] } else { vec![StreamChunk::Text(text)] }
    }

    /// Signal end of generation.
    pub fn finish(&mut self) -> Vec<StreamChunk> {
        let mut chunks = vec![];
        if !self.buffer.is_empty() {
            chunks.push(StreamChunk::Text(self.buffer.drain(..).collect()));
        }
        if self.state != RenderState::Complete {
            chunks.push(StreamChunk::EndOfSequence);
            self.state = RenderState::Complete;
        }
        chunks
    }

    /// Reset for a new generation.
    pub fn reset(&mut self) {
        self.state = RenderState::Idle;
        self.buffer.clear();
        self.tokens_rendered = 0;
        self.bytes_rendered = 0;
        self.eot_detector.reset();
    }

    pub fn state(&self) -> RenderState {
        self.state
    }

    pub fn tokens_rendered(&self) -> usize {
        self.tokens_rendered
    }

    pub fn bytes_rendered(&self) -> usize {
        self.bytes_rendered
    }
}

impl Default for StreamingRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Detects end-of-turn markers in token streams.
#[derive(Debug, Clone)]
pub struct EotDetector {
    patterns: Vec<String>,
}

impl EotDetector {
    pub fn new() -> Self {
        Self { patterns: known_eot_patterns() }
    }

    /// Check if a token exactly matches an EoT pattern.
    pub fn check(&self, token: &str) -> Option<String> {
        for pattern in &self.patterns {
            if token == pattern {
                return Some(pattern.clone());
            }
        }
        None
    }

    /// Check if text could be a partial prefix of any EoT pattern.
    pub fn is_partial_match(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        self.patterns.iter().any(|p| p.starts_with(text) && p != text)
    }

    pub fn reset(&mut self) {
        // No state to reset currently
    }

    pub fn patterns(&self) -> &[String] {
        &self.patterns
    }
}

impl Default for EotDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Known end-of-turn/end-of-text patterns across model families.
pub fn known_eot_patterns() -> Vec<String> {
    vec![
        "<|im_end|>".to_string(),
        "<|endoftext|>".to_string(),
        "<|eot_id|>".to_string(),
        "</s>".to_string(),
        "<|end|>".to_string(),
        "<|end_of_turn|>".to_string(),
        "[/INST]".to_string(),
        "<|assistant|>".to_string(),
    ]
}

/// Buffer for collecting stream chunks.
#[derive(Debug, Clone, Default)]
pub struct StreamBuffer {
    chunks: Vec<StreamChunk>,
    total_text_bytes: usize,
}

impl StreamBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, chunk: StreamChunk) {
        if let StreamChunk::Text(ref t) = chunk {
            self.total_text_bytes += t.len();
        }
        self.chunks.push(chunk);
    }

    pub fn to_text(&self) -> String {
        self.chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn drain(&mut self) -> Vec<StreamChunk> {
        self.total_text_bytes = 0;
        std::mem::take(&mut self.chunks)
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn total_text_bytes(&self) -> usize {
        self.total_text_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_new() {
        let r = StreamingRenderer::new();
        assert_eq!(r.state(), RenderState::Idle);
        assert_eq!(r.tokens_rendered(), 0);
        assert_eq!(r.bytes_rendered(), 0);
    }

    #[test]
    fn test_feed_regular_token() {
        let mut r = StreamingRenderer::new();
        let chunks = r.feed_token("hello");
        assert_eq!(chunks, vec![StreamChunk::Text("hello".to_string())]);
        assert_eq!(r.state(), RenderState::Streaming);
        assert_eq!(r.tokens_rendered(), 1);
    }

    #[test]
    fn test_feed_eot_token() {
        let mut r = StreamingRenderer::new();
        let chunks = r.feed_token("<|im_end|>");
        assert!(chunks.contains(&StreamChunk::SpecialToken("<|im_end|>".to_string())));
        assert!(chunks.contains(&StreamChunk::EndOfSequence));
        assert_eq!(r.state(), RenderState::Complete);
    }

    #[test]
    fn test_feed_after_complete() {
        let mut r = StreamingRenderer::new();
        r.feed_token("<|im_end|>");
        let chunks = r.feed_token("more");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_finish() {
        let mut r = StreamingRenderer::new();
        r.feed_token("hello");
        let chunks = r.finish();
        assert!(chunks.contains(&StreamChunk::EndOfSequence));
        assert_eq!(r.state(), RenderState::Complete);
    }

    #[test]
    fn test_finish_with_buffer() {
        let mut renderer = StreamingRenderer::new();
        renderer.feed_token("world");
        renderer.buffer = "leftover".to_string();
        let chunks = renderer.finish();
        assert!(chunks.iter().any(|c| matches!(c, StreamChunk::Text(t) if t == "leftover")));
    }

    #[test]
    fn test_reset() {
        let mut r = StreamingRenderer::new();
        r.feed_token("hello");
        r.reset();
        assert_eq!(r.state(), RenderState::Idle);
        assert_eq!(r.tokens_rendered(), 0);
        assert_eq!(r.bytes_rendered(), 0);
    }

    #[test]
    fn test_bytes_counted() {
        let mut r = StreamingRenderer::new();
        r.feed_token("hi");
        r.feed_token("there");
        assert_eq!(r.bytes_rendered(), 7);
        assert_eq!(r.tokens_rendered(), 2);
    }

    #[test]
    fn test_multiple_tokens() {
        let mut r = StreamingRenderer::new();
        let c1 = r.feed_token("Hello");
        let c2 = r.feed_token(" world");
        assert_eq!(c1, vec![StreamChunk::Text("Hello".to_string())]);
        assert_eq!(c2, vec![StreamChunk::Text(" world".to_string())]);
    }

    #[test]
    fn test_eot_eos_token() {
        let mut r = StreamingRenderer::new();
        let chunks = r.feed_token("</s>");
        assert!(chunks.contains(&StreamChunk::SpecialToken("</s>".to_string())));
    }

    #[test]
    fn test_eot_endoftext() {
        let mut r = StreamingRenderer::new();
        let chunks = r.feed_token("<|endoftext|>");
        assert!(chunks.contains(&StreamChunk::SpecialToken("<|endoftext|>".to_string())));
    }

    #[test]
    fn test_eot_eot_id() {
        let mut r = StreamingRenderer::new();
        let chunks = r.feed_token("<|eot_id|>");
        assert!(chunks.contains(&StreamChunk::SpecialToken("<|eot_id|>".to_string())));
    }

    #[test]
    fn test_detector_new() {
        let d = EotDetector::new();
        assert!(!d.patterns().is_empty());
    }

    #[test]
    fn test_detector_check_hit() {
        let d = EotDetector::new();
        assert_eq!(d.check("<|im_end|>"), Some("<|im_end|>".to_string()));
    }

    #[test]
    fn test_detector_check_miss() {
        let d = EotDetector::new();
        assert_eq!(d.check("hello"), None);
    }

    #[test]
    fn test_detector_partial_match() {
        let d = EotDetector::new();
        assert!(d.is_partial_match("<|im_"));
        assert!(!d.is_partial_match("hello"));
    }

    #[test]
    fn test_known_patterns() {
        let patterns = known_eot_patterns();
        assert!(patterns.len() >= 6);
        assert!(patterns.contains(&"<|im_end|>".to_string()));
        assert!(patterns.contains(&"</s>".to_string()));
    }

    #[test]
    fn test_buffer_new() {
        let b = StreamBuffer::new();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn test_buffer_add_text() {
        let mut b = StreamBuffer::new();
        b.add(StreamChunk::Text("hello".to_string()));
        assert_eq!(b.len(), 1);
        assert_eq!(b.total_text_bytes(), 5);
        assert_eq!(b.to_text(), "hello");
    }

    #[test]
    fn test_buffer_drain() {
        let mut b = StreamBuffer::new();
        b.add(StreamChunk::Text("hi".to_string()));
        let chunks = b.drain();
        assert_eq!(chunks.len(), 1);
        assert!(b.is_empty());
    }

    #[test]
    fn test_buffer_multiple_chunks() {
        let mut b = StreamBuffer::new();
        b.add(StreamChunk::Text("hello ".to_string()));
        b.add(StreamChunk::Text("world".to_string()));
        b.add(StreamChunk::EndOfSequence);
        assert_eq!(b.to_text(), "hello world");
        assert_eq!(b.len(), 3);
        assert_eq!(b.total_text_bytes(), 11);
    }

    #[test]
    fn test_default_renderer() {
        let r = StreamingRenderer::default();
        assert_eq!(r.state(), RenderState::Idle);
    }

    #[test]
    fn test_default_detector() {
        let d = EotDetector::default();
        assert!(!d.patterns().is_empty());
    }

    #[test]
    fn test_default_buffer() {
        let b = StreamBuffer::default();
        assert!(b.is_empty());
    }

    #[test]
    fn test_full_generation_flow() {
        let mut r = StreamingRenderer::new();
        let mut all_text = String::new();
        for token in &["The", " answer", " is", " 42", "<|im_end|>"] {
            for chunk in r.feed_token(token) {
                if let StreamChunk::Text(t) = chunk {
                    all_text.push_str(&t);
                }
            }
        }
        assert_eq!(all_text, "The answer is 42");
        assert_eq!(r.state(), RenderState::Complete);
        assert_eq!(r.tokens_rendered(), 5);
    }
}
