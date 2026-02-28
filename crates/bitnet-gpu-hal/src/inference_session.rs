//! Inference session manager for multi-turn conversations.
//!
//! Provides [`InferenceSession`] for managing conversation state,
//! [`SessionManager`] for concurrent session handling, and
//! [`SessionPersistence`] for saving/loading sessions.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── Message roles ─────────────────────────────────────────────────────────

/// Role of a message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

// ── Message ───────────────────────────────────────────────────────────────

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub token_count: usize,
    pub timestamp: Instant,
}

impl Message {
    /// Create a new message with the given role and content.
    ///
    /// Token count is estimated as `content.split_whitespace().count()`.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        let content = content.into();
        let token_count = estimate_tokens(&content);
        Self { role, content, token_count, timestamp: Instant::now() }
    }

    /// Create a message with an explicit token count.
    pub fn with_token_count(
        role: Role,
        content: impl Into<String>,
        token_count: usize,
    ) -> Self {
        Self {
            role,
            content: content.into(),
            token_count,
            timestamp: Instant::now(),
        }
    }
}

/// Estimate token count from whitespace-split words.
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

// ── Session state ─────────────────────────────────────────────────────────

/// Lifecycle state of an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    Created,
    Active,
    Paused,
    Expired,
    Terminated,
}

// ── Truncation strategy ───────────────────────────────────────────────────

/// Strategy for truncating conversation history when the context
/// window is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Keep only the most recent messages.
    KeepRecent,
    /// Keep the system prompt plus the most recent messages.
    KeepSystemAndRecent,
    /// Placeholder for future summarisation support.
    Summarize,
}

// ── Session config ────────────────────────────────────────────────────────

/// Configuration for an [`InferenceSession`].
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub max_history_tokens: usize,
    pub context_window: usize,
    pub system_prompt: Option<String>,
    pub session_timeout: Duration,
    pub truncation_strategy: TruncationStrategy,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_history_tokens: 4096,
            context_window: 2048,
            system_prompt: None,
            session_timeout: Duration::from_secs(1800),
            truncation_strategy: TruncationStrategy::KeepSystemAndRecent,
        }
    }
}

// ── Context window ────────────────────────────────────────────────────────

/// Sliding window that tracks which tokens fit in the current context.
#[derive(Debug)]
pub struct ContextWindow {
    capacity: usize,
    /// Indices into the parent history that are currently in-context.
    active_indices: Vec<usize>,
    active_token_count: usize,
}

impl ContextWindow {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, active_indices: Vec::new(), active_token_count: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn active_token_count(&self) -> usize {
        self.active_token_count
    }

    pub fn active_indices(&self) -> &[usize] {
        &self.active_indices
    }

    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.active_token_count)
    }

    /// Rebuild the active window from a message list and truncation
    /// strategy.
    pub fn rebuild(
        &mut self,
        messages: &[Message],
        strategy: TruncationStrategy,
    ) {
        self.active_indices.clear();
        self.active_token_count = 0;

        match strategy {
            TruncationStrategy::KeepRecent => {
                self.fill_recent(messages, None);
            }
            TruncationStrategy::KeepSystemAndRecent => {
                // Reserve space for system message if present.
                let system_idx = messages
                    .iter()
                    .position(|m| m.role == Role::System);
                if let Some(idx) = system_idx {
                    let sys_tokens = messages[idx].token_count;
                    if sys_tokens <= self.capacity {
                        self.active_indices.push(idx);
                        self.active_token_count = sys_tokens;
                    }
                }
                self.fill_recent(messages, system_idx);
            }
            TruncationStrategy::Summarize => {
                // Fallback to KeepSystemAndRecent until summarisation
                // is implemented.
                self.rebuild(messages, TruncationStrategy::KeepSystemAndRecent);
            }
        }
    }

    /// Fill from most-recent messages backwards, skipping `skip_idx`.
    fn fill_recent(
        &mut self,
        messages: &[Message],
        skip_idx: Option<usize>,
    ) {
        let mut pending = Vec::new();
        for i in (0..messages.len()).rev() {
            if Some(i) == skip_idx {
                continue;
            }
            let tokens = messages[i].token_count;
            if self.active_token_count + tokens > self.capacity {
                break;
            }
            pending.push(i);
            self.active_token_count += tokens;
        }
        pending.reverse();
        self.active_indices.extend(pending);
        self.active_indices.sort_unstable();
    }
}

// ── Conversation history ──────────────────────────────────────────────────

/// Ordered list of messages forming a conversation.
#[derive(Debug)]
pub struct ConversationHistory {
    messages: Vec<Message>,
    total_tokens: usize,
}

impl ConversationHistory {
    pub fn new() -> Self {
        Self { messages: Vec::new(), total_tokens: 0 }
    }

    pub fn push(&mut self, msg: Message) {
        self.total_tokens += msg.token_count;
        self.messages.push(msg);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Remove all messages and reset the token count.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.total_tokens = 0;
    }
}

impl Default for ConversationHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ── Session metrics ───────────────────────────────────────────────────────

/// Aggregate metrics for an inference session.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub tokens_generated: usize,
    pub turn_count: usize,
    pub turn_latencies: Vec<Duration>,
    pub session_start: Instant,
}

impl SessionMetrics {
    pub fn new() -> Self {
        Self {
            tokens_generated: 0,
            turn_count: 0,
            turn_latencies: Vec::new(),
            session_start: Instant::now(),
        }
    }

    /// Total wall-clock time since session creation.
    pub fn total_time(&self) -> Duration {
        self.session_start.elapsed()
    }

    /// Average latency across all turns.
    pub fn average_turn_latency(&self) -> Option<Duration> {
        if self.turn_latencies.is_empty() {
            return None;
        }
        let sum: Duration = self.turn_latencies.iter().sum();
        Some(sum / self.turn_latencies.len() as u32)
    }

    pub fn record_turn(&mut self, tokens: usize, latency: Duration) {
        self.tokens_generated += tokens;
        self.turn_count += 1;
        self.turn_latencies.push(latency);
    }
}

impl Default for SessionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Inference session ─────────────────────────────────────────────────────

/// Error type for session operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// The session has expired due to idle timeout.
    Expired,
    /// The session was explicitly terminated.
    Terminated,
    /// The session is not in an active state.
    InvalidState { current: SessionState, expected: SessionState },
    /// Session ID was not found.
    NotFound(String),
    /// Persistence operation failed.
    PersistenceError(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expired => write!(f, "session has expired"),
            Self::Terminated => write!(f, "session was terminated"),
            Self::InvalidState { current, expected } => {
                write!(
                    f,
                    "invalid state: session is {current:?}, \
                     expected {expected:?}"
                )
            }
            Self::NotFound(id) => {
                write!(f, "session not found: {id}")
            }
            Self::PersistenceError(msg) => {
                write!(f, "persistence error: {msg}")
            }
        }
    }
}

impl std::error::Error for SessionError {}

/// Manages multi-turn conversation state for a single session.
pub struct InferenceSession {
    id: String,
    state: SessionState,
    config: SessionConfig,
    history: ConversationHistory,
    context_window: ContextWindow,
    metrics: SessionMetrics,
    last_activity: Instant,
}

impl InferenceSession {
    /// Create a new session with the given ID and configuration.
    pub fn new(id: impl Into<String>, config: SessionConfig) -> Self {
        let ctx_cap = config.context_window;
        let mut session = Self {
            id: id.into(),
            state: SessionState::Created,
            config: config.clone(),
            history: ConversationHistory::new(),
            context_window: ContextWindow::new(ctx_cap),
            metrics: SessionMetrics::new(),
            last_activity: Instant::now(),
        };
        if let Some(ref prompt) = config.system_prompt {
            session
                .history
                .push(Message::new(Role::System, prompt.clone()));
            session.rebuild_context();
        }
        session
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn history(&self) -> &ConversationHistory {
        &self.history
    }

    pub fn context_window(&self) -> &ContextWindow {
        &self.context_window
    }

    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }

    /// Activate the session, transitioning from Created or Paused.
    pub fn activate(&mut self) -> Result<(), SessionError> {
        self.check_timeout()?;
        match self.state {
            SessionState::Created | SessionState::Paused => {
                self.state = SessionState::Active;
                self.last_activity = Instant::now();
                Ok(())
            }
            SessionState::Expired => Err(SessionError::Expired),
            SessionState::Terminated => Err(SessionError::Terminated),
            SessionState::Active => Ok(()),
        }
    }

    /// Pause the session.
    pub fn pause(&mut self) -> Result<(), SessionError> {
        self.check_timeout()?;
        match self.state {
            SessionState::Active => {
                self.state = SessionState::Paused;
                Ok(())
            }
            SessionState::Paused => Ok(()),
            SessionState::Expired => Err(SessionError::Expired),
            SessionState::Terminated => Err(SessionError::Terminated),
            _ => Err(SessionError::InvalidState {
                current: self.state,
                expected: SessionState::Active,
            }),
        }
    }

    /// Terminate the session. This is irreversible.
    pub fn terminate(&mut self) {
        self.state = SessionState::Terminated;
    }

    /// Add a user message and return the updated context indices.
    pub fn add_user_message(
        &mut self,
        content: impl Into<String>,
    ) -> Result<&[usize], SessionError> {
        self.ensure_active()?;
        self.history.push(Message::new(Role::User, content));
        self.rebuild_context();
        self.last_activity = Instant::now();
        Ok(self.context_window.active_indices())
    }

    /// Add an assistant response, recording turn metrics.
    pub fn add_assistant_message(
        &mut self,
        content: impl Into<String>,
        latency: Duration,
    ) -> Result<(), SessionError> {
        self.ensure_active()?;
        let msg = Message::new(Role::Assistant, content);
        let tokens = msg.token_count;
        self.history.push(msg);
        self.metrics.record_turn(tokens, latency);
        self.rebuild_context();
        self.last_activity = Instant::now();
        Ok(())
    }

    /// Return messages currently within the context window.
    pub fn context_messages(&self) -> Vec<&Message> {
        self.context_window
            .active_indices()
            .iter()
            .filter_map(|&i| self.history.messages().get(i))
            .collect()
    }

    /// Check whether the session has exceeded its idle timeout.
    pub fn is_expired(&self) -> bool {
        self.last_activity.elapsed() > self.config.session_timeout
    }

    fn check_timeout(&mut self) -> Result<(), SessionError> {
        if self.is_expired() {
            self.state = SessionState::Expired;
            return Err(SessionError::Expired);
        }
        Ok(())
    }

    fn ensure_active(&mut self) -> Result<(), SessionError> {
        self.check_timeout()?;
        match self.state {
            SessionState::Active => Ok(()),
            SessionState::Expired => Err(SessionError::Expired),
            SessionState::Terminated => Err(SessionError::Terminated),
            other => Err(SessionError::InvalidState {
                current: other,
                expected: SessionState::Active,
            }),
        }
    }

    fn rebuild_context(&mut self) {
        self.context_window.rebuild(
            self.history.messages(),
            self.config.truncation_strategy,
        );
    }
}

// ── Session persistence ───────────────────────────────────────────────────

/// Serialisable snapshot of a session for persistence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionSnapshot {
    pub id: String,
    pub messages: Vec<(String, String)>, // (role, content)
    pub tokens_generated: usize,
    pub turn_count: usize,
}

/// Trait for persisting and restoring sessions.
pub trait SessionPersistence {
    fn save(&mut self, snapshot: &SessionSnapshot) -> Result<(), SessionError>;
    fn load(&self, id: &str) -> Result<SessionSnapshot, SessionError>;
    fn delete(&mut self, id: &str) -> Result<(), SessionError>;
    fn list(&self) -> Vec<String>;
}

/// In-memory persistence backend for testing.
#[derive(Debug, Default)]
pub struct InMemoryPersistence {
    store: HashMap<String, SessionSnapshot>,
}

impl InMemoryPersistence {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionPersistence for InMemoryPersistence {
    fn save(
        &mut self,
        snapshot: &SessionSnapshot,
    ) -> Result<(), SessionError> {
        self.store.insert(snapshot.id.clone(), snapshot.clone());
        Ok(())
    }

    fn load(&self, id: &str) -> Result<SessionSnapshot, SessionError> {
        self.store
            .get(id)
            .cloned()
            .ok_or_else(|| SessionError::NotFound(id.to_string()))
    }

    fn delete(&mut self, id: &str) -> Result<(), SessionError> {
        self.store
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| SessionError::NotFound(id.to_string()))
    }

    fn list(&self) -> Vec<String> {
        self.store.keys().cloned().collect()
    }
}

/// Create a [`SessionSnapshot`] from a live session.
pub fn snapshot_session(session: &InferenceSession) -> SessionSnapshot {
    let messages = session
        .history()
        .messages()
        .iter()
        .map(|m| (m.role.to_string(), m.content.clone()))
        .collect();
    SessionSnapshot {
        id: session.id().to_string(),
        messages,
        tokens_generated: session.metrics().tokens_generated,
        turn_count: session.metrics().turn_count,
    }
}

// ── Session manager ───────────────────────────────────────────────────────

/// Manages multiple concurrent inference sessions.
pub struct SessionManager {
    sessions: HashMap<String, InferenceSession>,
    default_config: SessionConfig,
}

impl SessionManager {
    pub fn new(default_config: SessionConfig) -> Self {
        Self { sessions: HashMap::new(), default_config }
    }

    /// Create a new session with the default config.
    pub fn create_session(
        &mut self,
        id: impl Into<String>,
    ) -> &mut InferenceSession {
        let id = id.into();
        let session =
            InferenceSession::new(id.clone(), self.default_config.clone());
        self.sessions.insert(id.clone(), session);
        self.sessions.get_mut(&id).expect("just inserted")
    }

    /// Create a session with a custom config.
    pub fn create_session_with_config(
        &mut self,
        id: impl Into<String>,
        config: SessionConfig,
    ) -> &mut InferenceSession {
        let id = id.into();
        let session = InferenceSession::new(id.clone(), config);
        self.sessions.insert(id.clone(), session);
        self.sessions.get_mut(&id).expect("just inserted")
    }

    pub fn get(&self, id: &str) -> Option<&InferenceSession> {
        self.sessions.get(id)
    }

    pub fn get_mut(
        &mut self,
        id: &str,
    ) -> Option<&mut InferenceSession> {
        self.sessions.get_mut(id)
    }

    /// Remove a session, terminating it first.
    pub fn remove(&mut self, id: &str) -> Option<InferenceSession> {
        if let Some(s) = self.sessions.get_mut(id) {
            s.terminate();
        }
        self.sessions.remove(id)
    }

    /// Number of tracked sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// IDs of all tracked sessions.
    pub fn session_ids(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    /// Remove all expired sessions and return their IDs.
    pub fn reap_expired(&mut self) -> Vec<String> {
        let expired: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.is_expired())
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            if let Some(s) = self.sessions.get_mut(id) {
                s.terminate();
            }
            self.sessions.remove(id);
        }
        expired
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────

    fn default_config() -> SessionConfig {
        SessionConfig {
            max_history_tokens: 100,
            context_window: 50,
            system_prompt: None,
            session_timeout: Duration::from_secs(300),
            truncation_strategy: TruncationStrategy::KeepRecent,
        }
    }

    fn config_with_system(prompt: &str) -> SessionConfig {
        SessionConfig {
            system_prompt: Some(prompt.to_string()),
            ..default_config()
        }
    }

    fn active_session(id: &str) -> InferenceSession {
        let mut s = InferenceSession::new(id, default_config());
        s.activate().unwrap();
        s
    }

    // ── Role display ─────────────────────────────────────────────────

    #[test]
    fn test_role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    // ── Message construction ─────────────────────────────────────────

    #[test]
    fn test_message_new_estimates_tokens() {
        let m = Message::new(Role::User, "hello world foo");
        assert_eq!(m.token_count, 3);
        assert_eq!(m.role, Role::User);
    }

    #[test]
    fn test_message_empty_content() {
        let m = Message::new(Role::User, "");
        assert_eq!(m.token_count, 0);
    }

    #[test]
    fn test_message_with_explicit_token_count() {
        let m = Message::with_token_count(Role::Assistant, "ok", 42);
        assert_eq!(m.token_count, 42);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("a b c"), 3);
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("single"), 1);
    }

    // ── Conversation history ─────────────────────────────────────────

    #[test]
    fn test_history_push_and_len() {
        let mut h = ConversationHistory::new();
        assert!(h.is_empty());
        h.push(Message::new(Role::User, "hi"));
        assert_eq!(h.len(), 1);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_history_total_tokens() {
        let mut h = ConversationHistory::new();
        h.push(Message::new(Role::User, "one two"));
        h.push(Message::new(Role::Assistant, "three"));
        assert_eq!(h.total_tokens(), 3);
    }

    #[test]
    fn test_history_clear() {
        let mut h = ConversationHistory::new();
        h.push(Message::new(Role::User, "hello world"));
        h.clear();
        assert!(h.is_empty());
        assert_eq!(h.total_tokens(), 0);
    }

    #[test]
    fn test_history_default() {
        let h = ConversationHistory::default();
        assert!(h.is_empty());
    }

    #[test]
    fn test_history_messages_slice() {
        let mut h = ConversationHistory::new();
        h.push(Message::new(Role::User, "a"));
        h.push(Message::new(Role::Assistant, "b"));
        assert_eq!(h.messages().len(), 2);
        assert_eq!(h.messages()[0].role, Role::User);
        assert_eq!(h.messages()[1].role, Role::Assistant);
    }

    // ── Session lifecycle ────────────────────────────────────────────

    #[test]
    fn test_session_initial_state() {
        let s = InferenceSession::new("s1", default_config());
        assert_eq!(s.state(), SessionState::Created);
        assert_eq!(s.id(), "s1");
    }

    #[test]
    fn test_session_activate_from_created() {
        let mut s = InferenceSession::new("s1", default_config());
        assert!(s.activate().is_ok());
        assert_eq!(s.state(), SessionState::Active);
    }

    #[test]
    fn test_session_activate_idempotent() {
        let mut s = active_session("s1");
        assert!(s.activate().is_ok());
        assert_eq!(s.state(), SessionState::Active);
    }

    #[test]
    fn test_session_pause_and_resume() {
        let mut s = active_session("s1");
        s.pause().unwrap();
        assert_eq!(s.state(), SessionState::Paused);
        s.activate().unwrap();
        assert_eq!(s.state(), SessionState::Active);
    }

    #[test]
    fn test_session_pause_idempotent() {
        let mut s = active_session("s1");
        s.pause().unwrap();
        assert!(s.pause().is_ok());
    }

    #[test]
    fn test_session_terminate() {
        let mut s = active_session("s1");
        s.terminate();
        assert_eq!(s.state(), SessionState::Terminated);
    }

    #[test]
    fn test_cannot_activate_terminated() {
        let mut s = active_session("s1");
        s.terminate();
        assert_eq!(
            s.activate().unwrap_err(),
            SessionError::Terminated
        );
    }

    #[test]
    fn test_cannot_pause_terminated() {
        let mut s = active_session("s1");
        s.terminate();
        assert_eq!(
            s.pause().unwrap_err(),
            SessionError::Terminated
        );
    }

    #[test]
    fn test_cannot_pause_created() {
        let mut s = InferenceSession::new("s1", default_config());
        let err = s.pause().unwrap_err();
        assert!(matches!(err, SessionError::InvalidState { .. }));
    }

    #[test]
    fn test_add_message_requires_active() {
        let mut s = InferenceSession::new("s1", default_config());
        let err = s.add_user_message("hi").unwrap_err();
        assert!(matches!(err, SessionError::InvalidState { .. }));
    }

    #[test]
    fn test_add_assistant_requires_active() {
        let mut s = InferenceSession::new("s1", default_config());
        let err = s
            .add_assistant_message("hi", Duration::from_millis(10))
            .unwrap_err();
        assert!(matches!(err, SessionError::InvalidState { .. }));
    }

    // ── Conversation flow ────────────────────────────────────────────

    #[test]
    fn test_single_turn_conversation() {
        let mut s = active_session("s1");
        s.add_user_message("hello").unwrap();
        s.add_assistant_message("hi", Duration::from_millis(5))
            .unwrap();
        assert_eq!(s.history().len(), 2);
    }

    #[test]
    fn test_multi_turn_conversation() {
        let mut s = active_session("s1");
        for i in 0..5 {
            s.add_user_message(format!("q{i}")).unwrap();
            s.add_assistant_message(
                format!("a{i}"),
                Duration::from_millis(1),
            )
            .unwrap();
        }
        assert_eq!(s.history().len(), 10);
        assert_eq!(s.metrics().turn_count, 5);
    }

    #[test]
    fn test_system_prompt_in_history() {
        let cfg = config_with_system("You are helpful.");
        let s = InferenceSession::new("s1", cfg);
        assert_eq!(s.history().len(), 1);
        assert_eq!(s.history().messages()[0].role, Role::System);
    }

    #[test]
    fn test_system_prompt_preserved_after_messages() {
        let cfg = config_with_system("Be concise.");
        let mut s = InferenceSession::new("s1", cfg);
        s.activate().unwrap();
        s.add_user_message("hello").unwrap();
        assert_eq!(s.history().messages()[0].role, Role::System);
        assert_eq!(s.history().messages()[0].content, "Be concise.");
    }

    // ── Context window ───────────────────────────────────────────────

    #[test]
    fn test_context_window_empty() {
        let cw = ContextWindow::new(100);
        assert_eq!(cw.capacity(), 100);
        assert_eq!(cw.active_token_count(), 0);
        assert!(cw.active_indices().is_empty());
        assert_eq!(cw.remaining(), 100);
    }

    #[test]
    fn test_context_window_rebuild_keep_recent() {
        let msgs = vec![
            Message::with_token_count(Role::User, "a", 10),
            Message::with_token_count(Role::Assistant, "b", 10),
            Message::with_token_count(Role::User, "c", 10),
        ];
        let mut cw = ContextWindow::new(25);
        cw.rebuild(&msgs, TruncationStrategy::KeepRecent);
        // Should keep last two messages (20 tokens ≤ 25).
        assert_eq!(cw.active_token_count(), 20);
        assert_eq!(cw.active_indices(), &[1, 2]);
    }

    #[test]
    fn test_context_window_all_fit() {
        let msgs = vec![
            Message::with_token_count(Role::User, "a", 5),
            Message::with_token_count(Role::Assistant, "b", 5),
        ];
        let mut cw = ContextWindow::new(100);
        cw.rebuild(&msgs, TruncationStrategy::KeepRecent);
        assert_eq!(cw.active_indices(), &[0, 1]);
        assert_eq!(cw.active_token_count(), 10);
    }

    #[test]
    fn test_context_window_keep_system_and_recent() {
        let msgs = vec![
            Message::with_token_count(Role::System, "sys", 5),
            Message::with_token_count(Role::User, "a", 10),
            Message::with_token_count(Role::Assistant, "b", 10),
            Message::with_token_count(Role::User, "c", 10),
        ];
        let mut cw = ContextWindow::new(25);
        cw.rebuild(&msgs, TruncationStrategy::KeepSystemAndRecent);
        // System (5) + last message (10) = 15, plus second-to-last
        // (10) = 25.
        assert!(cw.active_indices().contains(&0)); // system
        assert!(cw.active_indices().contains(&3)); // most recent
        assert_eq!(cw.active_token_count(), 25);
    }

    #[test]
    fn test_context_window_summarize_fallback() {
        let msgs = vec![
            Message::with_token_count(Role::System, "sys", 5),
            Message::with_token_count(Role::User, "a", 10),
        ];
        let mut cw = ContextWindow::new(100);
        cw.rebuild(&msgs, TruncationStrategy::Summarize);
        // Falls back to KeepSystemAndRecent.
        assert_eq!(cw.active_indices(), &[0, 1]);
    }

    #[test]
    fn test_context_window_remaining() {
        let msgs = vec![
            Message::with_token_count(Role::User, "a", 10),
        ];
        let mut cw = ContextWindow::new(50);
        cw.rebuild(&msgs, TruncationStrategy::KeepRecent);
        assert_eq!(cw.remaining(), 40);
    }

    #[test]
    fn test_context_messages_returns_correct_slice() {
        let cfg = SessionConfig {
            context_window: 20,
            ..default_config()
        };
        let mut s = InferenceSession::new("s1", cfg);
        s.activate().unwrap();
        s.add_user_message("first message here").unwrap();
        s.add_user_message("second message here").unwrap();
        let ctx = s.context_messages();
        assert!(!ctx.is_empty());
    }

    #[test]
    fn test_context_truncation_drops_old_messages() {
        let cfg = SessionConfig {
            context_window: 5,
            truncation_strategy: TruncationStrategy::KeepRecent,
            ..default_config()
        };
        let mut s = InferenceSession::new("s1", cfg);
        s.activate().unwrap();
        // Each "word" = 1 token. Adding many messages that exceed
        // context window of 5 tokens.
        s.add_user_message("alpha beta gamma").unwrap(); // 3 tokens
        s.add_user_message("delta epsilon").unwrap(); // 2 tokens
        s.add_user_message("zeta").unwrap(); // 1 token

        let ctx = s.context_messages();
        let total: usize = ctx.iter().map(|m| m.token_count).sum();
        assert!(total <= 5);
    }

    // ── Session timeout ──────────────────────────────────────────────

    #[test]
    fn test_session_not_expired_initially() {
        let s = InferenceSession::new("s1", default_config());
        assert!(!s.is_expired());
    }

    #[test]
    fn test_session_expires_with_zero_timeout() {
        let cfg = SessionConfig {
            session_timeout: Duration::ZERO,
            ..default_config()
        };
        let s = InferenceSession::new("s1", cfg);
        // With zero timeout, the session is immediately expired.
        assert!(s.is_expired());
    }

    #[test]
    fn test_expired_session_rejects_activate() {
        let cfg = SessionConfig {
            session_timeout: Duration::ZERO,
            ..default_config()
        };
        let mut s = InferenceSession::new("s1", cfg);
        std::thread::sleep(Duration::from_millis(1));
        let err = s.activate().unwrap_err();
        assert_eq!(err, SessionError::Expired);
        assert_eq!(s.state(), SessionState::Expired);
    }

    #[test]
    fn test_expired_session_rejects_messages() {
        let cfg = SessionConfig {
            session_timeout: Duration::ZERO,
            ..default_config()
        };
        let mut s = InferenceSession::new("s1", cfg);
        s.state = SessionState::Active; // force for test
        std::thread::sleep(Duration::from_millis(1));
        assert!(s.add_user_message("hi").is_err());
    }

    // ── Session metrics ──────────────────────────────────────────────

    #[test]
    fn test_metrics_initial() {
        let m = SessionMetrics::new();
        assert_eq!(m.tokens_generated, 0);
        assert_eq!(m.turn_count, 0);
        assert!(m.average_turn_latency().is_none());
    }

    #[test]
    fn test_metrics_default() {
        let m = SessionMetrics::default();
        assert_eq!(m.turn_count, 0);
    }

    #[test]
    fn test_metrics_record_turn() {
        let mut m = SessionMetrics::new();
        m.record_turn(10, Duration::from_millis(100));
        m.record_turn(20, Duration::from_millis(200));
        assert_eq!(m.tokens_generated, 30);
        assert_eq!(m.turn_count, 2);
        let avg = m.average_turn_latency().unwrap();
        assert_eq!(avg, Duration::from_millis(150));
    }

    #[test]
    fn test_metrics_total_time_increases() {
        let m = SessionMetrics::new();
        std::thread::sleep(Duration::from_millis(5));
        assert!(m.total_time() >= Duration::from_millis(1));
    }

    #[test]
    fn test_session_metrics_after_conversation() {
        let mut s = active_session("s1");
        s.add_user_message("hello").unwrap();
        s.add_assistant_message("hi there", Duration::from_millis(50))
            .unwrap();
        assert_eq!(s.metrics().turn_count, 1);
        assert_eq!(s.metrics().tokens_generated, 2); // "hi there"
    }

    #[test]
    fn test_metrics_multiple_turns() {
        let mut s = active_session("s1");
        for _ in 0..3 {
            s.add_user_message("q").unwrap();
            s.add_assistant_message("a", Duration::from_millis(10))
                .unwrap();
        }
        assert_eq!(s.metrics().turn_count, 3);
    }

    // ── Persistence ──────────────────────────────────────────────────

    #[test]
    fn test_in_memory_persistence_save_load() {
        let mut p = InMemoryPersistence::new();
        let snap = SessionSnapshot {
            id: "s1".to_string(),
            messages: vec![("user".into(), "hi".into())],
            tokens_generated: 5,
            turn_count: 1,
        };
        p.save(&snap).unwrap();
        let loaded = p.load("s1").unwrap();
        assert_eq!(loaded.id, "s1");
        assert_eq!(loaded.turn_count, 1);
    }

    #[test]
    fn test_persistence_load_not_found() {
        let p = InMemoryPersistence::new();
        let err = p.load("missing").unwrap_err();
        assert_eq!(err, SessionError::NotFound("missing".to_string()));
    }

    #[test]
    fn test_persistence_delete() {
        let mut p = InMemoryPersistence::new();
        let snap = SessionSnapshot {
            id: "s1".to_string(),
            messages: vec![],
            tokens_generated: 0,
            turn_count: 0,
        };
        p.save(&snap).unwrap();
        p.delete("s1").unwrap();
        assert!(p.load("s1").is_err());
    }

    #[test]
    fn test_persistence_delete_not_found() {
        let mut p = InMemoryPersistence::new();
        let err = p.delete("nope").unwrap_err();
        assert!(matches!(err, SessionError::NotFound(_)));
    }

    #[test]
    fn test_persistence_list() {
        let mut p = InMemoryPersistence::new();
        assert!(p.list().is_empty());
        let snap = SessionSnapshot {
            id: "s1".to_string(),
            messages: vec![],
            tokens_generated: 0,
            turn_count: 0,
        };
        p.save(&snap).unwrap();
        assert_eq!(p.list().len(), 1);
    }

    #[test]
    fn test_persistence_overwrite() {
        let mut p = InMemoryPersistence::new();
        let snap1 = SessionSnapshot {
            id: "s1".to_string(),
            messages: vec![],
            tokens_generated: 0,
            turn_count: 0,
        };
        let snap2 = SessionSnapshot {
            id: "s1".to_string(),
            messages: vec![("user".into(), "updated".into())],
            tokens_generated: 10,
            turn_count: 2,
        };
        p.save(&snap1).unwrap();
        p.save(&snap2).unwrap();
        let loaded = p.load("s1").unwrap();
        assert_eq!(loaded.turn_count, 2);
    }

    #[test]
    fn test_snapshot_session() {
        let mut s = active_session("s1");
        s.add_user_message("hello").unwrap();
        s.add_assistant_message("world", Duration::from_millis(1))
            .unwrap();
        let snap = snapshot_session(&s);
        assert_eq!(snap.id, "s1");
        assert_eq!(snap.messages.len(), 2);
        assert_eq!(snap.turn_count, 1);
    }

    #[test]
    fn test_snapshot_with_system_prompt() {
        let cfg = config_with_system("You are a bot.");
        let mut s = InferenceSession::new("s1", cfg);
        s.activate().unwrap();
        s.add_user_message("hi").unwrap();
        let snap = snapshot_session(&s);
        assert_eq!(snap.messages.len(), 2);
        assert_eq!(snap.messages[0].0, "system");
    }

    #[test]
    fn test_snapshot_roundtrip_through_persistence() {
        let mut s = active_session("s1");
        s.add_user_message("ping").unwrap();
        s.add_assistant_message("pong", Duration::from_millis(1))
            .unwrap();
        let snap = snapshot_session(&s);

        let mut p = InMemoryPersistence::new();
        p.save(&snap).unwrap();
        let loaded = p.load("s1").unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.tokens_generated, snap.tokens_generated);
    }

    // ── Session manager ──────────────────────────────────────────────

    #[test]
    fn test_manager_create_session() {
        let mut mgr = SessionManager::new(default_config());
        let s = mgr.create_session("s1");
        assert_eq!(s.id(), "s1");
        assert_eq!(s.state(), SessionState::Created);
    }

    #[test]
    fn test_manager_get_session() {
        let mut mgr = SessionManager::new(default_config());
        mgr.create_session("s1");
        assert!(mgr.get("s1").is_some());
        assert!(mgr.get("missing").is_none());
    }

    #[test]
    fn test_manager_get_mut_session() {
        let mut mgr = SessionManager::new(default_config());
        mgr.create_session("s1");
        let s = mgr.get_mut("s1").unwrap();
        s.activate().unwrap();
        assert_eq!(mgr.get("s1").unwrap().state(), SessionState::Active);
    }

    #[test]
    fn test_manager_remove_session() {
        let mut mgr = SessionManager::new(default_config());
        mgr.create_session("s1");
        let removed = mgr.remove("s1").unwrap();
        assert_eq!(removed.state(), SessionState::Terminated);
        assert!(mgr.get("s1").is_none());
    }

    #[test]
    fn test_manager_remove_nonexistent() {
        let mut mgr = SessionManager::new(default_config());
        assert!(mgr.remove("nope").is_none());
    }

    #[test]
    fn test_manager_session_count() {
        let mut mgr = SessionManager::new(default_config());
        assert_eq!(mgr.session_count(), 0);
        mgr.create_session("s1");
        mgr.create_session("s2");
        assert_eq!(mgr.session_count(), 2);
    }

    #[test]
    fn test_manager_session_ids() {
        let mut mgr = SessionManager::new(default_config());
        mgr.create_session("a");
        mgr.create_session("b");
        let mut ids = mgr.session_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn test_manager_create_with_custom_config() {
        let mut mgr = SessionManager::new(default_config());
        let custom = SessionConfig {
            context_window: 999,
            ..default_config()
        };
        let s = mgr.create_session_with_config("s1", custom);
        assert_eq!(s.config().context_window, 999);
    }

    #[test]
    fn test_manager_reap_expired() {
        let mut mgr = SessionManager::new(SessionConfig {
            session_timeout: Duration::ZERO,
            ..default_config()
        });
        mgr.create_session("s1");
        mgr.create_session("s2");
        std::thread::sleep(Duration::from_millis(1));
        let reaped = mgr.reap_expired();
        assert_eq!(reaped.len(), 2);
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_manager_reap_keeps_active() {
        let mut mgr = SessionManager::new(default_config());
        mgr.create_session("s1");
        let reaped = mgr.reap_expired();
        assert!(reaped.is_empty());
        assert_eq!(mgr.session_count(), 1);
    }

    // ── Error display ────────────────────────────────────────────────

    #[test]
    fn test_session_error_display() {
        assert_eq!(SessionError::Expired.to_string(), "session has expired");
        assert_eq!(
            SessionError::Terminated.to_string(),
            "session was terminated"
        );
        assert!(SessionError::NotFound("x".into())
            .to_string()
            .contains('x'));
        assert!(SessionError::PersistenceError("disk".into())
            .to_string()
            .contains("disk"));
    }

    #[test]
    fn test_invalid_state_error_display() {
        let e = SessionError::InvalidState {
            current: SessionState::Created,
            expected: SessionState::Active,
        };
        let s = e.to_string();
        assert!(s.contains("Created"));
        assert!(s.contains("Active"));
    }

    // ── Truncation strategy variants ─────────────────────────────────

    #[test]
    fn test_truncation_keep_recent_drops_oldest() {
        let msgs = vec![
            Message::with_token_count(Role::User, "old", 20),
            Message::with_token_count(Role::User, "mid", 20),
            Message::with_token_count(Role::User, "new", 20),
        ];
        let mut cw = ContextWindow::new(30);
        cw.rebuild(&msgs, TruncationStrategy::KeepRecent);
        // Only mid + new fit (40 > 30), so only new fits starting
        // from back. 20 <= 30, then 20+20 = 40 > 30.
        assert!(cw.active_indices().contains(&2));
        assert!(!cw.active_indices().contains(&0));
    }

    #[test]
    fn test_truncation_system_preserved_when_others_dropped() {
        let msgs = vec![
            Message::with_token_count(Role::System, "sys", 5),
            Message::with_token_count(Role::User, "old", 20),
            Message::with_token_count(Role::User, "new", 20),
        ];
        let mut cw = ContextWindow::new(30);
        cw.rebuild(&msgs, TruncationStrategy::KeepSystemAndRecent);
        assert!(cw.active_indices().contains(&0)); // system kept
        assert!(cw.active_indices().contains(&2)); // newest kept
    }

    #[test]
    fn test_truncation_large_system_prompt() {
        let msgs = vec![
            Message::with_token_count(Role::System, "huge", 100),
            Message::with_token_count(Role::User, "hi", 5),
        ];
        let mut cw = ContextWindow::new(50);
        cw.rebuild(&msgs, TruncationStrategy::KeepSystemAndRecent);
        // System prompt too large for window, only user fits.
        // System gets skipped because 100 > 50.
        assert!(!cw.active_indices().contains(&0));
    }

    #[test]
    fn test_context_window_no_messages() {
        let mut cw = ContextWindow::new(100);
        cw.rebuild(&[], TruncationStrategy::KeepRecent);
        assert!(cw.active_indices().is_empty());
        assert_eq!(cw.active_token_count(), 0);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_session_with_empty_system_prompt() {
        let cfg = config_with_system("");
        let s = InferenceSession::new("s1", cfg);
        assert_eq!(s.history().len(), 1);
        assert_eq!(s.history().messages()[0].token_count, 0);
    }

    #[test]
    fn test_session_config_default() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.max_history_tokens, 4096);
        assert_eq!(cfg.context_window, 2048);
        assert!(cfg.system_prompt.is_none());
        assert_eq!(cfg.session_timeout, Duration::from_secs(1800));
        assert_eq!(
            cfg.truncation_strategy,
            TruncationStrategy::KeepSystemAndRecent
        );
    }

    #[test]
    fn test_multiple_managers_independent() {
        let mut m1 = SessionManager::new(default_config());
        let mut m2 = SessionManager::new(default_config());
        m1.create_session("a");
        m2.create_session("b");
        assert!(m1.get("b").is_none());
        assert!(m2.get("a").is_none());
    }

    #[test]
    fn test_session_id_with_special_chars() {
        let mut s =
            InferenceSession::new("session/123-abc_XYZ", default_config());
        assert_eq!(s.id(), "session/123-abc_XYZ");
        assert!(s.activate().is_ok());
    }

    #[test]
    fn test_session_error_is_error_trait() {
        let e: Box<dyn std::error::Error> =
            Box::new(SessionError::Expired);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_large_conversation() {
        let mut s = active_session("big");
        for i in 0..100 {
            s.add_user_message(format!("question {i}")).unwrap();
            s.add_assistant_message(
                format!("answer {i}"),
                Duration::from_millis(1),
            )
            .unwrap();
        }
        assert_eq!(s.history().len(), 200);
        assert_eq!(s.metrics().turn_count, 100);
    }

    #[test]
    fn test_context_window_single_large_message() {
        let msgs = vec![
            Message::with_token_count(Role::User, "huge", 1000),
        ];
        let mut cw = ContextWindow::new(10);
        cw.rebuild(&msgs, TruncationStrategy::KeepRecent);
        // Message too large — nothing fits.
        assert!(cw.active_indices().is_empty());
    }

    #[test]
    fn test_context_window_exact_capacity() {
        let msgs = vec![
            Message::with_token_count(Role::User, "a", 25),
            Message::with_token_count(Role::User, "b", 25),
        ];
        let mut cw = ContextWindow::new(50);
        cw.rebuild(&msgs, TruncationStrategy::KeepRecent);
        assert_eq!(cw.active_token_count(), 50);
        assert_eq!(cw.remaining(), 0);
    }

    #[test]
    fn test_snapshot_empty_session() {
        let s = InferenceSession::new("empty", default_config());
        let snap = snapshot_session(&s);
        assert_eq!(snap.messages.len(), 0);
        assert_eq!(snap.tokens_generated, 0);
    }

    #[test]
    fn test_manager_replace_session() {
        let mut mgr = SessionManager::new(default_config());
        mgr.create_session("s1");
        mgr.create_session("s1"); // replace
        assert_eq!(mgr.session_count(), 1);
    }
}
