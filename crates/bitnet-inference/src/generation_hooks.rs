//! Generation callback hooks.
//!
//! Define hooks that fire during token generation: on-token,
//! on-stop, on-error, and custom filters for controlling generation flow.

/// Action to take after a hook fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookAction {
    /// Continue generation normally.
    Continue,
    /// Stop generation immediately.
    Stop,
    /// Skip this token (don't add to output).
    Skip,
}

/// Event emitted during generation.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// A new token was generated.
    TokenGenerated { token_id: u32, token_text: String, position: usize },
    /// Generation completed normally.
    Completed { total_tokens: usize },
    /// Generation was stopped early.
    Stopped { reason: String, tokens_generated: usize },
    /// An error occurred.
    Error { message: String },
}

/// A hook that processes generation events.
pub trait GenerationHook: std::fmt::Debug {
    fn name(&self) -> &str;
    fn on_event(&mut self, event: &GenerationEvent) -> HookAction;
}

/// Token count limiter hook.
#[derive(Debug)]
pub struct MaxTokenHook {
    pub max_tokens: usize,
    pub current: usize,
}

impl MaxTokenHook {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens, current: 0 }
    }
}

impl GenerationHook for MaxTokenHook {
    fn name(&self) -> &str {
        "max_token_limit"
    }

    fn on_event(&mut self, event: &GenerationEvent) -> HookAction {
        if let GenerationEvent::TokenGenerated { .. } = event {
            self.current += 1;
            if self.current >= self.max_tokens {
                return HookAction::Stop;
            }
        }
        HookAction::Continue
    }
}

/// Stop word detector hook.
#[derive(Debug)]
pub struct StopWordHook {
    pub stop_words: Vec<String>,
    buffer: String,
}

impl StopWordHook {
    pub fn new(stop_words: Vec<String>) -> Self {
        Self { stop_words, buffer: String::new() }
    }
}

impl GenerationHook for StopWordHook {
    fn name(&self) -> &str {
        "stop_word_detector"
    }

    fn on_event(&mut self, event: &GenerationEvent) -> HookAction {
        if let GenerationEvent::TokenGenerated { token_text, .. } = event {
            self.buffer.push_str(token_text);
            for sw in &self.stop_words {
                if self.buffer.contains(sw.as_str()) {
                    return HookAction::Stop;
                }
            }
        }
        HookAction::Continue
    }
}

/// Token logger hook.
#[derive(Debug)]
pub struct LoggerHook {
    pub tokens: Vec<(u32, String)>,
}

impl LoggerHook {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }
    pub fn logged_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Default for LoggerHook {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationHook for LoggerHook {
    fn name(&self) -> &str {
        "token_logger"
    }

    fn on_event(&mut self, event: &GenerationEvent) -> HookAction {
        if let GenerationEvent::TokenGenerated { token_id, token_text, .. } = event {
            self.tokens.push((*token_id, token_text.clone()));
        }
        HookAction::Continue
    }
}

/// Hook pipeline: runs multiple hooks in sequence.
#[derive(Debug)]
pub struct HookPipeline {
    hooks: Vec<Box<dyn GenerationHook>>,
}

impl HookPipeline {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn add<H: GenerationHook + 'static>(&mut self, hook: H) {
        self.hooks.push(Box::new(hook));
    }

    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    /// Process an event through all hooks. Returns the most restrictive action.
    pub fn process(&mut self, event: &GenerationEvent) -> HookAction {
        let mut result = HookAction::Continue;
        for hook in &mut self.hooks {
            let action = hook.on_event(event);
            // Stop is most restrictive, then Skip, then Continue
            match action {
                HookAction::Stop => return HookAction::Stop,
                HookAction::Skip if result == HookAction::Continue => {
                    result = HookAction::Skip;
                }
                _ => {}
            }
        }
        result
    }

    pub fn hook_names(&self) -> Vec<&str> {
        self.hooks.iter().map(|h| h.name()).collect()
    }
}

impl Default for HookPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_event(id: u32, text: &str, pos: usize) -> GenerationEvent {
        GenerationEvent::TokenGenerated {
            token_id: id,
            token_text: text.to_string(),
            position: pos,
        }
    }

    #[test]
    fn test_max_token_hook() {
        let mut hook = MaxTokenHook::new(3);
        assert_eq!(hook.on_event(&token_event(1, "a", 0)), HookAction::Continue);
        assert_eq!(hook.on_event(&token_event(2, "b", 1)), HookAction::Continue);
        assert_eq!(hook.on_event(&token_event(3, "c", 2)), HookAction::Stop);
    }

    #[test]
    fn test_stop_word_hook() {
        let mut hook = StopWordHook::new(vec!["<|end|>".into()]);
        assert_eq!(hook.on_event(&token_event(1, "hello", 0)), HookAction::Continue);
        assert_eq!(hook.on_event(&token_event(2, "<|end|>", 1)), HookAction::Stop);
    }

    #[test]
    fn test_stop_word_across_tokens() {
        let mut hook = StopWordHook::new(vec!["stop".into()]);
        assert_eq!(hook.on_event(&token_event(1, "st", 0)), HookAction::Continue);
        assert_eq!(hook.on_event(&token_event(2, "op", 1)), HookAction::Stop);
    }

    #[test]
    fn test_logger_hook() {
        let mut hook = LoggerHook::new();
        hook.on_event(&token_event(1, "hello", 0));
        hook.on_event(&token_event(2, "world", 1));
        assert_eq!(hook.logged_count(), 2);
        assert_eq!(hook.tokens[0], (1, "hello".into()));
    }

    #[test]
    fn test_pipeline_continue() {
        let mut pipeline = HookPipeline::new();
        pipeline.add(LoggerHook::new());
        pipeline.add(MaxTokenHook::new(100));
        let action = pipeline.process(&token_event(1, "hi", 0));
        assert_eq!(action, HookAction::Continue);
    }

    #[test]
    fn test_pipeline_stop() {
        let mut pipeline = HookPipeline::new();
        pipeline.add(MaxTokenHook::new(1));
        pipeline.add(LoggerHook::new());
        let action = pipeline.process(&token_event(1, "a", 0));
        assert_eq!(action, HookAction::Stop);
    }

    #[test]
    fn test_pipeline_names() {
        let mut pipeline = HookPipeline::new();
        pipeline.add(LoggerHook::new());
        pipeline.add(MaxTokenHook::new(10));
        let names = pipeline.hook_names();
        assert!(names.contains(&"token_logger"));
        assert!(names.contains(&"max_token_limit"));
    }

    #[test]
    fn test_hook_count() {
        let mut pipeline = HookPipeline::new();
        assert_eq!(pipeline.hook_count(), 0);
        pipeline.add(LoggerHook::new());
        assert_eq!(pipeline.hook_count(), 1);
    }

    #[test]
    fn test_completed_event() {
        let mut hook = LoggerHook::new();
        let event = GenerationEvent::Completed { total_tokens: 10 };
        assert_eq!(hook.on_event(&event), HookAction::Continue);
    }

    #[test]
    fn test_error_event() {
        let mut hook = MaxTokenHook::new(5);
        let event = GenerationEvent::Error { message: "test".into() };
        assert_eq!(hook.on_event(&event), HookAction::Continue);
    }

    #[test]
    fn test_hook_action_eq() {
        assert_ne!(HookAction::Stop, HookAction::Continue);
        assert_ne!(HookAction::Skip, HookAction::Continue);
    }

    #[test]
    fn test_stop_word_no_match() {
        let mut hook = StopWordHook::new(vec!["xyz".into()]);
        assert_eq!(hook.on_event(&token_event(1, "hello", 0)), HookAction::Continue);
        assert_eq!(hook.on_event(&token_event(2, "world", 1)), HookAction::Continue);
    }
}
