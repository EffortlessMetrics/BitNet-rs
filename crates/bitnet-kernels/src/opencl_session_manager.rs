//! Inference session lifecycle management for OpenCL GPU backends.
//!
//! Manages the full lifecycle of an inference request — creation,
//! prefilling, token generation, completion, cancellation, and error
//! recovery.  A [`SessionPool`] provides reusable sessions with LRU
//! eviction, and [`SessionCheckpoint`] enables pause/resume.
//!
//! When no OpenCL runtime is present this module provides CPU-reference
//! implementations that exercise the same state-machine logic.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// ── SessionId ───────────────────────────────────────────────────────

/// Unique, opaque identifier for an inference session.
///
/// Uses a monotonically increasing counter combined with a random
/// component to avoid collisions across process restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId {
    hi: u64,
    lo: u64,
}

static NEXT_SESSION_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl SessionId {
    /// Generate a new unique session id.
    pub fn new() -> Self {
        let counter = NEXT_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
        // Mix in a cheap hash of the counter for the `lo` half so that
        // consecutive ids are not trivially predictable.
        let lo = counter.wrapping_mul(0x517c_c1b7_2722_0a95);
        Self { hi: counter, lo }
    }

    /// Construct from raw halves (useful for deserialization).
    pub fn from_parts(hi: u64, lo: u64) -> Self {
        Self { hi, lo }
    }

    /// High 64 bits (monotonic counter).
    pub fn hi(&self) -> u64 {
        self.hi
    }

    /// Low 64 bits (hash-derived).
    pub fn lo(&self) -> u64 {
        self.lo
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sess-{:016x}-{:016x}", self.hi, self.lo)
    }
}

// ── SessionConfig ───────────────────────────────────────────────────

/// Configuration for a single inference session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (must be > 0.0).
    pub temperature: f32,
    /// Nucleus sampling threshold (0.0 .. 1.0].
    pub top_p: f32,
    /// Stop generation when any of these sequences is produced.
    pub stop_sequences: Vec<String>,
    /// Hard timeout for the entire generation request.
    pub timeout: Duration,
    /// Optional seed for reproducible sampling.
    pub seed: Option<u64>,
}

impl SessionConfig {
    /// Validate configuration, returning a descriptive error on failure.
    pub fn validate(&self) -> Result<(), SessionError> {
        if self.max_tokens == 0 {
            return Err(SessionError::InvalidConfig("max_tokens must be > 0".into()));
        }
        if self.temperature <= 0.0 {
            return Err(SessionError::InvalidConfig("temperature must be > 0.0".into()));
        }
        if !self.temperature.is_finite() {
            return Err(SessionError::InvalidConfig("temperature must be finite".into()));
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(SessionError::InvalidConfig("top_p must be in (0.0, 1.0]".into()));
        }
        if !self.top_p.is_finite() {
            return Err(SessionError::InvalidConfig("top_p must be finite".into()));
        }
        if self.timeout.is_zero() {
            return Err(SessionError::InvalidConfig("timeout must be > 0".into()));
        }
        Ok(())
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: Vec::new(),
            timeout: Duration::from_secs(60),
            seed: None,
        }
    }
}

// ── SessionState ────────────────────────────────────────────────────

/// States of the inference session state machine.
///
/// ```text
///  Idle ──► Prefilling ──► Generating ──► Complete
///   │           │              │
///   │           ▼              ▼
///   │        Error          Error
///   │           │              │
///   └───────────┴──► Cancelled ◄┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    /// Session created but generation has not started.
    Idle,
    /// Processing the prompt (prefill phase).
    Prefilling,
    /// Autoregressive token generation in progress.
    Generating,
    /// Generation finished normally.
    Complete,
    /// An unrecoverable error occurred.
    Error,
    /// The session was cancelled cooperatively.
    Cancelled,
}

impl SessionState {
    /// Returns `true` for terminal states (`Complete`, `Error`, `Cancelled`).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Error | Self::Cancelled)
    }

    /// Valid successor states from the current state.
    pub fn valid_transitions(self) -> &'static [SessionState] {
        match self {
            Self::Idle => &[Self::Prefilling, Self::Error, Self::Cancelled],
            Self::Prefilling => &[Self::Generating, Self::Error, Self::Cancelled],
            Self::Generating => &[Self::Complete, Self::Error, Self::Cancelled],
            Self::Complete | Self::Error | Self::Cancelled => &[],
        }
    }

    /// Whether transitioning to `next` is allowed.
    pub fn can_transition_to(self, next: Self) -> bool {
        self.valid_transitions().contains(&next)
    }
}

impl fmt::Display for SessionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Idle => "Idle",
            Self::Prefilling => "Prefilling",
            Self::Generating => "Generating",
            Self::Complete => "Complete",
            Self::Error => "Error",
            Self::Cancelled => "Cancelled",
        };
        f.write_str(label)
    }
}

// ── SessionError ────────────────────────────────────────────────────

/// Errors that can occur during session lifecycle management.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// The generation request exceeded the configured timeout.
    Timeout { elapsed: Duration, limit: Duration },
    /// The system ran out of memory for this session.
    OutOfMemory { requested_bytes: usize, available_bytes: usize },
    /// The session was cancelled via its [`CancellationToken`].
    Cancelled,
    /// The session configuration is invalid.
    InvalidConfig(String),
    /// An error occurred inside the model / kernel.
    ModelError(String),
    /// The requested state transition is not permitted.
    InvalidTransition { from: SessionState, to: SessionState },
    /// The session was not found in the pool.
    SessionNotFound(SessionId),
    /// The pool has reached its capacity limit.
    PoolExhausted { capacity: usize },
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout { elapsed, limit } => {
                write!(f, "session timed out after {elapsed:?} (limit {limit:?})")
            }
            Self::OutOfMemory { requested_bytes, available_bytes } => {
                write!(
                    f,
                    "out of memory: requested {requested_bytes} B, \
                     available {available_bytes} B"
                )
            }
            Self::Cancelled => f.write_str("session cancelled"),
            Self::InvalidConfig(msg) => {
                write!(f, "invalid config: {msg}")
            }
            Self::ModelError(msg) => write!(f, "model error: {msg}"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition: {from} → {to}")
            }
            Self::SessionNotFound(id) => {
                write!(f, "session not found: {id}")
            }
            Self::PoolExhausted { capacity } => {
                write!(f, "pool exhausted (capacity={capacity})")
            }
        }
    }
}

impl std::error::Error for SessionError {}

// ── CancellationToken ───────────────────────────────────────────────

/// Cooperative cancellation flag shared between the caller and the
/// session.  Both sides hold an `Arc` clone; the caller sets the flag
/// and the session checks it at safe yield points.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)) }
    }

    /// Signal cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Reset the token so it can be reused for a new generation.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// ── SessionMetrics ──────────────────────────────────────────────────

/// Performance counters collected during a session's lifetime.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    /// Tokens generated per second (decode phase).
    pub tokens_per_sec: f64,
    /// Time from start until the first generated token.
    pub time_to_first_token: Duration,
    /// Wall-clock time from start to completion.
    pub total_time: Duration,
    /// Peak memory usage in bytes.
    pub memory_used_bytes: usize,
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub generated_tokens: usize,
}

impl SessionMetrics {
    fn empty() -> Self {
        Self {
            tokens_per_sec: 0.0,
            time_to_first_token: Duration::ZERO,
            total_time: Duration::ZERO,
            memory_used_bytes: 0,
            prompt_tokens: 0,
            generated_tokens: 0,
        }
    }
}

impl Default for SessionMetrics {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for SessionMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prompt={} gen={} tok/s={:.2} ttft={:?} total={:?} mem={}B",
            self.prompt_tokens,
            self.generated_tokens,
            self.tokens_per_sec,
            self.time_to_first_token,
            self.total_time,
            self.memory_used_bytes,
        )
    }
}

// ── SessionCheckpoint ───────────────────────────────────────────────

/// Snapshot of a session's state sufficient to resume generation.
#[derive(Debug, Clone)]
pub struct SessionCheckpoint {
    /// The session id this checkpoint belongs to.
    pub session_id: SessionId,
    /// The state at checkpoint time.
    pub state: SessionState,
    /// Tokens generated so far.
    pub generated_tokens: Vec<u32>,
    /// The position in the KV-cache.
    pub kv_cache_position: usize,
    /// Metrics at checkpoint time.
    pub metrics: SessionMetrics,
    /// Opaque blob for backend-specific data (e.g. KV-cache).
    pub backend_state: Vec<u8>,
    /// Timestamp when checkpoint was taken.
    pub created_at: Instant,
}

impl SessionCheckpoint {
    /// Size estimate for memory accounting.
    pub fn estimated_size_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.generated_tokens.len() * std::mem::size_of::<u32>()
            + self.backend_state.len()
    }
}

// ── InferenceSession ────────────────────────────────────────────────

/// Manages one inference generation request through its lifecycle.
///
/// The session enforces the state-machine transitions and collects
/// metrics.  In this CPU-reference implementation, "generation" is
/// simulated with deterministic dummy tokens.
pub struct InferenceSession {
    id: SessionId,
    config: SessionConfig,
    state: SessionState,
    cancel_token: CancellationToken,
    metrics: SessionMetrics,
    generated_tokens: Vec<u32>,
    start_time: Option<Instant>,
    first_token_time: Option<Instant>,
    last_error: Option<String>,
}

impl InferenceSession {
    /// Create a new session. Validates `config` eagerly.
    pub fn new(config: SessionConfig) -> Result<Self, SessionError> {
        config.validate()?;
        Ok(Self {
            id: SessionId::new(),
            config,
            state: SessionState::Idle,
            cancel_token: CancellationToken::new(),
            metrics: SessionMetrics::empty(),
            generated_tokens: Vec::new(),
            start_time: None,
            first_token_time: None,
            last_error: None,
        })
    }

    /// Create with a specific id (used by pool / checkpoint restore).
    pub fn with_id(id: SessionId, config: SessionConfig) -> Result<Self, SessionError> {
        config.validate()?;
        Ok(Self {
            id,
            config,
            state: SessionState::Idle,
            cancel_token: CancellationToken::new(),
            metrics: SessionMetrics::empty(),
            generated_tokens: Vec::new(),
            start_time: None,
            first_token_time: None,
            last_error: None,
        })
    }

    // ── Accessors ───────────────────────────────────────────────────

    pub fn id(&self) -> SessionId {
        self.id
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }

    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated_tokens
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    // ── State transitions ───────────────────────────────────────────

    fn transition(&mut self, to: SessionState) -> Result<(), SessionError> {
        if !self.state.can_transition_to(to) {
            return Err(SessionError::InvalidTransition { from: self.state, to });
        }
        self.state = to;
        Ok(())
    }

    /// Begin the prefill phase.
    pub fn start_prefill(&mut self, prompt_tokens: usize) -> Result<(), SessionError> {
        self.transition(SessionState::Prefilling)?;
        self.start_time = Some(Instant::now());
        self.metrics.prompt_tokens = prompt_tokens;
        Ok(())
    }

    /// Transition from prefill to generation.
    pub fn start_generation(&mut self) -> Result<(), SessionError> {
        if self.cancel_token.is_cancelled() {
            self.state = SessionState::Cancelled;
            return Err(SessionError::Cancelled);
        }
        self.transition(SessionState::Generating)?;
        Ok(())
    }

    /// Append a generated token and update metrics.
    ///
    /// Returns `Err(SessionError::Cancelled)` if the cancellation token
    /// is set, `Err(SessionError::Timeout{..})` if the timeout has
    /// elapsed, or `Ok(true)` when the session should stop generating
    /// (max tokens reached or stop sequence found).
    pub fn push_token(&mut self, token: u32) -> Result<bool, SessionError> {
        if self.state != SessionState::Generating {
            return Err(SessionError::InvalidTransition {
                from: self.state,
                to: SessionState::Generating,
            });
        }

        // Cancellation check.
        if self.cancel_token.is_cancelled() {
            self.state = SessionState::Cancelled;
            return Err(SessionError::Cancelled);
        }

        // Timeout check.
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed();
            if elapsed >= self.config.timeout {
                self.state = SessionState::Error;
                self.last_error = Some(format!("timeout after {elapsed:?}"));
                return Err(SessionError::Timeout { elapsed, limit: self.config.timeout });
            }
        }

        if self.generated_tokens.is_empty() {
            self.first_token_time = Some(Instant::now());
        }
        self.generated_tokens.push(token);
        self.metrics.generated_tokens = self.generated_tokens.len();

        // Update throughput.
        if let Some(first_t) = self.first_token_time {
            let decode_secs = first_t.elapsed().as_secs_f64();
            if decode_secs > 0.0 {
                self.metrics.tokens_per_sec = self.generated_tokens.len() as f64 / decode_secs;
            }
        }

        // Max tokens reached?
        if self.generated_tokens.len() >= self.config.max_tokens {
            return Ok(true);
        }

        Ok(false)
    }

    /// Mark the session as complete and finalize metrics.
    pub fn complete(&mut self) -> Result<SessionMetrics, SessionError> {
        self.transition(SessionState::Complete)?;
        self.finalize_metrics();
        Ok(self.metrics.clone())
    }

    /// Mark the session as errored.
    pub fn fail(&mut self, reason: String) -> SessionError {
        // Allow transition to error from any non-terminal state.
        if !self.state.is_terminal() {
            self.state = SessionState::Error;
        }
        self.last_error = Some(reason.clone());
        self.finalize_metrics();
        SessionError::ModelError(reason)
    }

    /// Cancel the session cooperatively.
    pub fn cancel(&mut self) -> Result<(), SessionError> {
        if self.state.is_terminal() {
            return Err(SessionError::InvalidTransition {
                from: self.state,
                to: SessionState::Cancelled,
            });
        }
        self.cancel_token.cancel();
        self.state = SessionState::Cancelled;
        self.finalize_metrics();
        Ok(())
    }

    /// Reset the session so it can be reused by the pool.
    pub fn reset(&mut self, config: SessionConfig) -> Result<(), SessionError> {
        config.validate()?;
        self.config = config;
        self.state = SessionState::Idle;
        self.cancel_token.reset();
        self.metrics = SessionMetrics::empty();
        self.generated_tokens.clear();
        self.start_time = None;
        self.first_token_time = None;
        self.last_error = None;
        Ok(())
    }

    // ── Checkpointing ───────────────────────────────────────────────

    /// Save a checkpoint of the current session state.
    pub fn checkpoint(&self) -> SessionCheckpoint {
        SessionCheckpoint {
            session_id: self.id,
            state: self.state,
            generated_tokens: self.generated_tokens.clone(),
            kv_cache_position: self.generated_tokens.len(),
            metrics: self.metrics.clone(),
            backend_state: Vec::new(),
            created_at: Instant::now(),
        }
    }

    /// Restore from a checkpoint.  The session must be `Idle`.
    pub fn restore(&mut self, cp: &SessionCheckpoint) -> Result<(), SessionError> {
        if self.state != SessionState::Idle {
            return Err(SessionError::InvalidTransition { from: self.state, to: cp.state });
        }
        self.generated_tokens = cp.generated_tokens.clone();
        self.metrics = cp.metrics.clone();
        // Restored sessions resume in Generating if they were generating.
        if cp.state == SessionState::Generating || cp.state == SessionState::Prefilling {
            self.state = SessionState::Generating;
            self.start_time = Some(Instant::now());
            self.first_token_time = Some(Instant::now());
        } else {
            self.state = cp.state;
        }
        Ok(())
    }

    // ── CPU-reference "inference" ───────────────────────────────────

    /// Run a full inference pass (CPU reference).
    ///
    /// Simulates prefill + decode producing deterministic dummy tokens.
    pub fn run_cpu_reference(
        &mut self,
        prompt_tokens: usize,
    ) -> Result<SessionMetrics, SessionError> {
        self.start_prefill(prompt_tokens)?;
        self.start_generation()?;

        for i in 0..self.config.max_tokens {
            let token = ((i as u32).wrapping_mul(7) + 1) % 32000;
            match self.push_token(token) {
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => return Err(e),
            }
        }
        self.complete()
    }

    // ── Helpers ─────────────────────────────────────────────────────

    fn finalize_metrics(&mut self) {
        if let Some(start) = self.start_time {
            self.metrics.total_time = start.elapsed();
        }
        if let (Some(start), Some(first)) = (self.start_time, self.first_token_time) {
            self.metrics.time_to_first_token = first.duration_since(start);
        }
        // CPU-reference: estimate memory as 4 bytes per generated token
        // plus a flat overhead for the session object.
        self.metrics.memory_used_bytes = 1024 + self.generated_tokens.len() * 4;
    }
}

impl fmt::Debug for InferenceSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceSession")
            .field("id", &self.id)
            .field("state", &self.state)
            .field("generated", &self.generated_tokens.len())
            .finish()
    }
}

// ── SessionPool ─────────────────────────────────────────────────────

/// Configuration for the session pool.
#[derive(Debug, Clone)]
pub struct SessionPoolConfig {
    /// Maximum number of sessions in the pool.
    pub max_sessions: usize,
    /// Maximum idle time before a session is evicted.
    pub idle_timeout: Duration,
}

impl Default for SessionPoolConfig {
    fn default() -> Self {
        Self { max_sessions: 16, idle_timeout: Duration::from_secs(300) }
    }
}

/// Metadata tracked for each pooled session.
struct PoolEntry {
    session: InferenceSession,
    last_used: Instant,
}

/// Pool of reusable inference sessions with LRU eviction.
///
/// Sessions that have completed or errored can be returned to the pool,
/// reset, and reissued.  When the pool is full the least-recently-used
/// session is evicted.
pub struct SessionPool {
    config: SessionPoolConfig,
    /// Ordered from most-recently-used (back) to least-recently-used
    /// (front).
    sessions: VecDeque<PoolEntry>,
    /// Fast lookup by session id → index in `sessions`.
    index: HashMap<SessionId, usize>,
    /// Running counter of total sessions ever created.
    total_created: u64,
    /// Running counter of evictions.
    total_evicted: u64,
}

impl SessionPool {
    pub fn new(config: SessionPoolConfig) -> Self {
        Self {
            sessions: VecDeque::with_capacity(config.max_sessions),
            index: HashMap::with_capacity(config.max_sessions),
            config,
            total_created: 0,
            total_evicted: 0,
        }
    }

    /// Number of sessions currently in the pool.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Pool capacity.
    pub fn capacity(&self) -> usize {
        self.config.max_sessions
    }

    pub fn total_created(&self) -> u64 {
        self.total_created
    }

    pub fn total_evicted(&self) -> u64 {
        self.total_evicted
    }

    /// Acquire a new session from the pool.
    ///
    /// If the pool is at capacity the least-recently-used session is
    /// evicted and recycled.
    pub fn acquire(&mut self, config: SessionConfig) -> Result<SessionId, SessionError> {
        config.validate()?;

        if self.sessions.len() >= self.config.max_sessions {
            self.evict_lru()?;
        }

        let session = InferenceSession::new(config)?;
        let id = session.id();
        let entry = PoolEntry { session, last_used: Instant::now() };
        self.sessions.push_back(entry);
        self.rebuild_index();
        self.total_created += 1;
        Ok(id)
    }

    /// Return a session to the pool after use.
    pub fn release(
        &mut self,
        id: SessionId,
        new_config: SessionConfig,
    ) -> Result<(), SessionError> {
        let idx = self.index.get(&id).copied().ok_or(SessionError::SessionNotFound(id))?;

        self.sessions[idx].session.reset(new_config)?;
        self.sessions[idx].last_used = Instant::now();
        // Move to back (most recently used).
        let entry = self.sessions.remove(idx).unwrap();
        self.sessions.push_back(entry);
        self.rebuild_index();
        Ok(())
    }

    /// Look up a session mutably.
    pub fn get_mut(&mut self, id: SessionId) -> Result<&mut InferenceSession, SessionError> {
        let idx = self.index.get(&id).copied().ok_or(SessionError::SessionNotFound(id))?;
        self.sessions[idx].last_used = Instant::now();
        Ok(&mut self.sessions[idx].session)
    }

    /// Look up a session immutably.
    pub fn get(&self, id: SessionId) -> Result<&InferenceSession, SessionError> {
        let idx = self.index.get(&id).copied().ok_or(SessionError::SessionNotFound(id))?;
        Ok(&self.sessions[idx].session)
    }

    /// Remove a session from the pool entirely.
    pub fn remove(&mut self, id: SessionId) -> Result<InferenceSession, SessionError> {
        let idx = self.index.get(&id).copied().ok_or(SessionError::SessionNotFound(id))?;
        let entry = self.sessions.remove(idx).unwrap();
        self.rebuild_index();
        Ok(entry.session)
    }

    /// Evict all sessions idle longer than the configured timeout.
    pub fn evict_expired(&mut self) -> usize {
        let timeout = self.config.idle_timeout;
        let before = self.sessions.len();
        self.sessions.retain(|e| e.last_used.elapsed() < timeout);
        let evicted = before - self.sessions.len();
        if evicted > 0 {
            self.total_evicted += evicted as u64;
            self.rebuild_index();
        }
        evicted
    }

    /// Collect ids of all sessions currently in the pool.
    pub fn session_ids(&self) -> Vec<SessionId> {
        self.sessions.iter().map(|e| e.session.id()).collect()
    }

    // ── Internal ────────────────────────────────────────────────────

    fn evict_lru(&mut self) -> Result<(), SessionError> {
        if self.sessions.is_empty() {
            return Err(SessionError::PoolExhausted { capacity: self.config.max_sessions });
        }
        self.sessions.pop_front();
        self.total_evicted += 1;
        self.rebuild_index();
        Ok(())
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, entry) in self.sessions.iter().enumerate() {
            self.index.insert(entry.session.id(), i);
        }
    }
}

impl fmt::Debug for SessionPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionPool")
            .field("len", &self.sessions.len())
            .field("capacity", &self.config.max_sessions)
            .field("created", &self.total_created)
            .field("evicted", &self.total_evicted)
            .finish()
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a valid default config.
    fn default_cfg() -> SessionConfig {
        SessionConfig::default()
    }

    fn tiny_cfg(max_tokens: usize) -> SessionConfig {
        SessionConfig { max_tokens, ..default_cfg() }
    }

    // ── SessionId tests ─────────────────────────────────────────────

    #[test]
    fn session_id_unique() {
        let a = SessionId::new();
        let b = SessionId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn session_id_display_format() {
        let id = SessionId::from_parts(1, 2);
        let s = id.to_string();
        assert!(s.starts_with("sess-"));
        assert!(s.contains('-'));
    }

    #[test]
    fn session_id_ord() {
        let a = SessionId::from_parts(1, 0);
        let b = SessionId::from_parts(2, 0);
        assert!(a < b);
    }

    #[test]
    fn session_id_hash_eq() {
        let a = SessionId::from_parts(42, 99);
        let b = SessionId::from_parts(42, 99);
        assert_eq!(a, b);

        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn session_id_from_parts_roundtrip() {
        let id = SessionId::from_parts(0xDEAD, 0xBEEF);
        assert_eq!(id.hi(), 0xDEAD);
        assert_eq!(id.lo(), 0xBEEF);
    }

    #[test]
    fn session_id_default_is_new() {
        let a = SessionId::default();
        let b = SessionId::default();
        assert_ne!(a, b);
    }

    // ── SessionConfig tests ─────────────────────────────────────────

    #[test]
    fn default_config_valid() {
        assert!(default_cfg().validate().is_ok());
    }

    #[test]
    fn config_zero_max_tokens_invalid() {
        let cfg = SessionConfig { max_tokens: 0, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_negative_temperature_invalid() {
        let cfg = SessionConfig { temperature: -1.0, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_zero_temperature_invalid() {
        let cfg = SessionConfig { temperature: 0.0, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_nan_temperature_invalid() {
        let cfg = SessionConfig { temperature: f32::NAN, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_inf_temperature_invalid() {
        let cfg = SessionConfig { temperature: f32::INFINITY, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_zero_top_p_invalid() {
        let cfg = SessionConfig { top_p: 0.0, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_top_p_above_1_invalid() {
        let cfg = SessionConfig { top_p: 1.01, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_nan_top_p_invalid() {
        let cfg = SessionConfig { top_p: f32::NAN, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_top_p_exactly_1_valid() {
        let cfg = SessionConfig { top_p: 1.0, ..default_cfg() };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_zero_timeout_invalid() {
        let cfg = SessionConfig { timeout: Duration::ZERO, ..default_cfg() };
        assert!(matches!(cfg.validate(), Err(SessionError::InvalidConfig(_))));
    }

    #[test]
    fn config_with_stop_sequences_valid() {
        let cfg =
            SessionConfig { stop_sequences: vec!["</s>".into(), "\n".into()], ..default_cfg() };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_with_seed_valid() {
        let cfg = SessionConfig { seed: Some(42), ..default_cfg() };
        assert!(cfg.validate().is_ok());
    }

    // ── SessionState tests ──────────────────────────────────────────

    #[test]
    fn idle_is_not_terminal() {
        assert!(!SessionState::Idle.is_terminal());
    }

    #[test]
    fn prefilling_is_not_terminal() {
        assert!(!SessionState::Prefilling.is_terminal());
    }

    #[test]
    fn generating_is_not_terminal() {
        assert!(!SessionState::Generating.is_terminal());
    }

    #[test]
    fn complete_is_terminal() {
        assert!(SessionState::Complete.is_terminal());
    }

    #[test]
    fn error_is_terminal() {
        assert!(SessionState::Error.is_terminal());
    }

    #[test]
    fn cancelled_is_terminal() {
        assert!(SessionState::Cancelled.is_terminal());
    }

    #[test]
    fn idle_can_transition_to_prefilling() {
        assert!(SessionState::Idle.can_transition_to(SessionState::Prefilling));
    }

    #[test]
    fn idle_cannot_transition_to_generating() {
        assert!(!SessionState::Idle.can_transition_to(SessionState::Generating));
    }

    #[test]
    fn idle_cannot_transition_to_complete() {
        assert!(!SessionState::Idle.can_transition_to(SessionState::Complete));
    }

    #[test]
    fn prefilling_can_transition_to_generating() {
        assert!(SessionState::Prefilling.can_transition_to(SessionState::Generating));
    }

    #[test]
    fn generating_can_transition_to_complete() {
        assert!(SessionState::Generating.can_transition_to(SessionState::Complete));
    }

    #[test]
    fn terminal_states_have_no_transitions() {
        for state in [SessionState::Complete, SessionState::Error, SessionState::Cancelled] {
            assert!(state.valid_transitions().is_empty(), "{state} should have no transitions");
        }
    }

    #[test]
    fn any_non_terminal_can_transition_to_error() {
        for state in [SessionState::Idle, SessionState::Prefilling, SessionState::Generating] {
            assert!(
                state.can_transition_to(SessionState::Error),
                "{state} → Error should be valid"
            );
        }
    }

    #[test]
    fn any_non_terminal_can_transition_to_cancelled() {
        for state in [SessionState::Idle, SessionState::Prefilling, SessionState::Generating] {
            assert!(
                state.can_transition_to(SessionState::Cancelled),
                "{state} → Cancelled should be valid"
            );
        }
    }

    #[test]
    fn state_display() {
        assert_eq!(SessionState::Idle.to_string(), "Idle");
        assert_eq!(SessionState::Generating.to_string(), "Generating");
        assert_eq!(SessionState::Complete.to_string(), "Complete");
    }

    // Property: all 6 states are reachable.
    #[test]
    fn all_states_reachable() {
        let reachable = [
            SessionState::Idle,
            SessionState::Prefilling,
            SessionState::Generating,
            SessionState::Complete,
            SessionState::Error,
            SessionState::Cancelled,
        ];
        // Walk the state machine from Idle and collect every reachable state.
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![SessionState::Idle];
        while let Some(s) = stack.pop() {
            if visited.insert(s) {
                for &next in s.valid_transitions() {
                    stack.push(next);
                }
            }
        }
        for s in &reachable {
            assert!(visited.contains(s), "{s} is not reachable from Idle");
        }
    }

    // Property: terminal states cannot transition anywhere.
    #[test]
    fn terminal_states_are_absorbing() {
        let terminals = [SessionState::Complete, SessionState::Error, SessionState::Cancelled];
        let all_states = [
            SessionState::Idle,
            SessionState::Prefilling,
            SessionState::Generating,
            SessionState::Complete,
            SessionState::Error,
            SessionState::Cancelled,
        ];
        for &t in &terminals {
            for &s in &all_states {
                assert!(!t.can_transition_to(s), "{t} should not transition to {s}");
            }
        }
    }

    // ── SessionError tests ──────────────────────────────────────────

    #[test]
    fn error_display_timeout() {
        let e = SessionError::Timeout {
            elapsed: Duration::from_secs(10),
            limit: Duration::from_secs(5),
        };
        let msg = e.to_string();
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn error_display_oom() {
        let e = SessionError::OutOfMemory { requested_bytes: 1024, available_bytes: 512 };
        assert!(e.to_string().contains("out of memory"));
    }

    #[test]
    fn error_display_cancelled() {
        assert_eq!(SessionError::Cancelled.to_string(), "session cancelled");
    }

    #[test]
    fn error_display_invalid_config() {
        let e = SessionError::InvalidConfig("bad temp".into());
        assert!(e.to_string().contains("bad temp"));
    }

    #[test]
    fn error_display_model_error() {
        let e = SessionError::ModelError("kernel panic".into());
        assert!(e.to_string().contains("kernel panic"));
    }

    #[test]
    fn error_display_invalid_transition() {
        let e = SessionError::InvalidTransition {
            from: SessionState::Complete,
            to: SessionState::Generating,
        };
        assert!(e.to_string().contains("Complete"));
    }

    #[test]
    fn error_display_session_not_found() {
        let id = SessionId::from_parts(1, 2);
        let e = SessionError::SessionNotFound(id);
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn error_display_pool_exhausted() {
        let e = SessionError::PoolExhausted { capacity: 8 };
        assert!(e.to_string().contains("8"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(SessionError::Cancelled);
        assert!(!e.to_string().is_empty());
    }

    // ── CancellationToken tests ─────────────────────────────────────

    #[test]
    fn cancel_token_initially_false() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_token_cancel() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_token_reset() {
        let token = CancellationToken::new();
        token.cancel();
        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_token_clone_shares_state() {
        let token = CancellationToken::new();
        let clone = token.clone();
        token.cancel();
        assert!(clone.is_cancelled());
    }

    #[test]
    fn cancel_token_default() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    // ── SessionMetrics tests ────────────────────────────────────────

    #[test]
    fn metrics_default_zeroed() {
        let m = SessionMetrics::default();
        assert_eq!(m.tokens_per_sec, 0.0);
        assert_eq!(m.time_to_first_token, Duration::ZERO);
        assert_eq!(m.total_time, Duration::ZERO);
        assert_eq!(m.memory_used_bytes, 0);
        assert_eq!(m.prompt_tokens, 0);
        assert_eq!(m.generated_tokens, 0);
    }

    #[test]
    fn metrics_display() {
        let m = SessionMetrics::default();
        let s = m.to_string();
        assert!(s.contains("prompt="));
        assert!(s.contains("gen="));
    }

    // ── InferenceSession lifecycle tests ────────────────────────────

    #[test]
    fn session_create_valid() {
        let s = InferenceSession::new(default_cfg()).unwrap();
        assert_eq!(s.state(), SessionState::Idle);
    }

    #[test]
    fn session_create_invalid_config() {
        let cfg = SessionConfig { max_tokens: 0, ..default_cfg() };
        assert!(InferenceSession::new(cfg).is_err());
    }

    #[test]
    fn session_lifecycle_happy_path() {
        let mut s = InferenceSession::new(tiny_cfg(4)).unwrap();
        s.start_prefill(10).unwrap();
        assert_eq!(s.state(), SessionState::Prefilling);

        s.start_generation().unwrap();
        assert_eq!(s.state(), SessionState::Generating);

        for i in 0..4 {
            let done = s.push_token(i).unwrap();
            if i < 3 {
                assert!(!done);
            } else {
                assert!(done);
            }
        }

        let metrics = s.complete().unwrap();
        assert_eq!(s.state(), SessionState::Complete);
        assert_eq!(metrics.generated_tokens, 4);
    }

    #[test]
    fn session_cannot_prefill_twice() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        s.start_prefill(1).unwrap();
        assert!(s.start_prefill(1).is_err());
    }

    #[test]
    fn session_cannot_generate_from_idle() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        assert!(s.start_generation().is_err());
    }

    #[test]
    fn session_cannot_complete_from_idle() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        assert!(s.complete().is_err());
    }

    #[test]
    fn session_cannot_push_token_when_idle() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        assert!(s.push_token(0).is_err());
    }

    #[test]
    fn session_double_complete_fails() {
        let mut s = InferenceSession::new(tiny_cfg(1)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(42).unwrap();
        s.complete().unwrap();
        assert!(s.complete().is_err());
    }

    #[test]
    fn session_with_id() {
        let id = SessionId::from_parts(100, 200);
        let s = InferenceSession::with_id(id, default_cfg()).unwrap();
        assert_eq!(s.id(), id);
    }

    #[test]
    fn session_generated_tokens_collected() {
        let mut s = InferenceSession::new(tiny_cfg(3)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(10).unwrap();
        s.push_token(20).unwrap();
        assert_eq!(s.generated_tokens(), &[10, 20]);
    }

    #[test]
    fn session_metrics_populated_after_complete() {
        let mut s = InferenceSession::new(tiny_cfg(2)).unwrap();
        s.start_prefill(5).unwrap();
        s.start_generation().unwrap();
        s.push_token(1).unwrap();
        s.push_token(2).unwrap();
        let m = s.complete().unwrap();
        assert_eq!(m.prompt_tokens, 5);
        assert_eq!(m.generated_tokens, 2);
        assert!(m.total_time > Duration::ZERO);
    }

    // ── Cancellation tests ──────────────────────────────────────────

    #[test]
    fn cancel_idle_session() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        s.cancel().unwrap();
        assert_eq!(s.state(), SessionState::Cancelled);
    }

    #[test]
    fn cancel_during_generation() {
        let mut s = InferenceSession::new(tiny_cfg(100)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(1).unwrap();
        s.cancel().unwrap();
        assert_eq!(s.state(), SessionState::Cancelled);
    }

    #[test]
    fn cancel_token_stops_push_token() {
        let mut s = InferenceSession::new(tiny_cfg(100)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.cancel_token().cancel();
        let res = s.push_token(1);
        assert!(matches!(res, Err(SessionError::Cancelled)));
        assert_eq!(s.state(), SessionState::Cancelled);
    }

    #[test]
    fn cancel_token_stops_start_generation() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        s.start_prefill(1).unwrap();
        s.cancel_token().cancel();
        let res = s.start_generation();
        assert!(matches!(res, Err(SessionError::Cancelled)));
        assert_eq!(s.state(), SessionState::Cancelled);
    }

    #[test]
    fn cancel_terminal_session_fails() {
        let mut s = InferenceSession::new(tiny_cfg(1)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(1).unwrap();
        s.complete().unwrap();
        assert!(s.cancel().is_err());
    }

    // ── Timeout tests ───────────────────────────────────────────────

    #[test]
    fn timeout_very_short() {
        let cfg = SessionConfig {
            max_tokens: 1_000_000,
            timeout: Duration::from_nanos(1),
            ..default_cfg()
        };
        let mut s = InferenceSession::new(cfg).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();

        // The first push_token may or may not timeout depending on
        // scheduling, so just verify that eventually we get a timeout
        // or run out of tokens.
        let mut timed_out = false;
        for i in 0..1000 {
            match s.push_token(i) {
                Err(SessionError::Timeout { .. }) => {
                    timed_out = true;
                    break;
                }
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        assert!(timed_out, "expected timeout");
        assert_eq!(s.state(), SessionState::Error);
    }

    // ── Error state tests ───────────────────────────────────────────

    #[test]
    fn fail_sets_error_state() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        s.start_prefill(1).unwrap();
        let err = s.fail("boom".into());
        assert!(matches!(err, SessionError::ModelError(_)));
        assert_eq!(s.state(), SessionState::Error);
        assert_eq!(s.last_error(), Some("boom"));
    }

    #[test]
    fn fail_is_idempotent_in_terminal() {
        let mut s = InferenceSession::new(tiny_cfg(1)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(1).unwrap();
        s.complete().unwrap();
        // fail after complete doesn't change state to error
        let _ = s.fail("ignored".into());
        assert_eq!(s.state(), SessionState::Complete);
    }

    // ── Reset tests ─────────────────────────────────────────────────

    #[test]
    fn reset_returns_to_idle() {
        let mut s = InferenceSession::new(tiny_cfg(1)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(1).unwrap();
        s.complete().unwrap();
        s.reset(default_cfg()).unwrap();
        assert_eq!(s.state(), SessionState::Idle);
        assert!(s.generated_tokens().is_empty());
    }

    #[test]
    fn reset_with_invalid_config_fails() {
        let mut s = InferenceSession::new(default_cfg()).unwrap();
        let bad = SessionConfig { max_tokens: 0, ..default_cfg() };
        assert!(s.reset(bad).is_err());
    }

    // ── Checkpoint tests ────────────────────────────────────────────

    #[test]
    fn checkpoint_captures_tokens() {
        let mut s = InferenceSession::new(tiny_cfg(10)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(10).unwrap();
        s.push_token(20).unwrap();

        let cp = s.checkpoint();
        assert_eq!(cp.session_id, s.id());
        assert_eq!(cp.state, SessionState::Generating);
        assert_eq!(cp.generated_tokens, vec![10, 20]);
        assert_eq!(cp.kv_cache_position, 2);
    }

    #[test]
    fn checkpoint_restore_resumes_generating() {
        let mut s = InferenceSession::new(tiny_cfg(10)).unwrap();
        s.start_prefill(1).unwrap();
        s.start_generation().unwrap();
        s.push_token(10).unwrap();

        let cp = s.checkpoint();

        // Create a fresh session and restore.
        let mut s2 = InferenceSession::new(tiny_cfg(10)).unwrap();
        s2.restore(&cp).unwrap();
        assert_eq!(s2.state(), SessionState::Generating);
        assert_eq!(s2.generated_tokens(), &[10]);

        // Continue generating.
        s2.push_token(20).unwrap();
        assert_eq!(s2.generated_tokens(), &[10, 20]);
    }

    #[test]
    fn checkpoint_restore_fails_if_not_idle() {
        let mut s = InferenceSession::new(tiny_cfg(10)).unwrap();
        s.start_prefill(1).unwrap();

        let cp = s.checkpoint();

        let mut s2 = InferenceSession::new(tiny_cfg(10)).unwrap();
        s2.start_prefill(1).unwrap();
        assert!(s2.restore(&cp).is_err());
    }

    #[test]
    fn checkpoint_estimated_size() {
        let cp = SessionCheckpoint {
            session_id: SessionId::new(),
            state: SessionState::Generating,
            generated_tokens: vec![1, 2, 3],
            kv_cache_position: 3,
            metrics: SessionMetrics::default(),
            backend_state: vec![0; 100],
            created_at: Instant::now(),
        };
        assert!(cp.estimated_size_bytes() > 100);
    }

    // ── CPU-reference inference test ────────────────────────────────

    #[test]
    fn cpu_reference_happy_path() {
        let mut s = InferenceSession::new(tiny_cfg(8)).unwrap();
        let metrics = s.run_cpu_reference(10).unwrap();
        assert_eq!(s.state(), SessionState::Complete);
        assert_eq!(metrics.generated_tokens, 8);
        assert_eq!(metrics.prompt_tokens, 10);
    }

    #[test]
    fn cpu_reference_generates_deterministic_tokens() {
        let mut s1 = InferenceSession::new(tiny_cfg(4)).unwrap();
        s1.run_cpu_reference(5).unwrap();

        let mut s2 = InferenceSession::new(tiny_cfg(4)).unwrap();
        s2.run_cpu_reference(5).unwrap();

        assert_eq!(s1.generated_tokens(), s2.generated_tokens());
    }

    #[test]
    fn cpu_reference_cancelled_early() {
        let mut s = InferenceSession::new(tiny_cfg(100)).unwrap();
        // Cancel before running.
        s.cancel_token().cancel();
        let res = s.run_cpu_reference(1);
        // Should fail during start_generation.
        assert!(res.is_err());
    }

    // ── SessionPool tests ───────────────────────────────────────────

    #[test]
    fn pool_acquire_and_get() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let id = pool.acquire(default_cfg()).unwrap();
        let s = pool.get(id).unwrap();
        assert_eq!(s.state(), SessionState::Idle);
    }

    #[test]
    fn pool_acquire_invalid_config() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let bad = SessionConfig { max_tokens: 0, ..default_cfg() };
        assert!(pool.acquire(bad).is_err());
    }

    #[test]
    fn pool_len_and_empty() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        pool.acquire(default_cfg()).unwrap();
        assert!(!pool.is_empty());
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn pool_evicts_lru_on_overflow() {
        let cfg = SessionPoolConfig { max_sessions: 2, ..Default::default() };
        let mut pool = SessionPool::new(cfg);
        let id1 = pool.acquire(default_cfg()).unwrap();
        let _id2 = pool.acquire(default_cfg()).unwrap();

        // Pool is full; next acquire evicts LRU (id1).
        let _id3 = pool.acquire(default_cfg()).unwrap();
        assert_eq!(pool.len(), 2);
        assert!(pool.get(id1).is_err());
        assert_eq!(pool.total_evicted(), 1);
    }

    #[test]
    fn pool_release_resets_session() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let id = pool.acquire(tiny_cfg(2)).unwrap();

        {
            let s = pool.get_mut(id).unwrap();
            s.start_prefill(1).unwrap();
            s.start_generation().unwrap();
            s.push_token(1).unwrap();
            s.push_token(2).unwrap();
            s.complete().unwrap();
        }

        pool.release(id, default_cfg()).unwrap();
        let s = pool.get(id).unwrap();
        assert_eq!(s.state(), SessionState::Idle);
        assert!(s.generated_tokens().is_empty());
    }

    #[test]
    fn pool_release_nonexistent_fails() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let bogus = SessionId::from_parts(999, 999);
        assert!(pool.release(bogus, default_cfg()).is_err());
    }

    #[test]
    fn pool_remove() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let id = pool.acquire(default_cfg()).unwrap();
        let s = pool.remove(id).unwrap();
        assert_eq!(s.id(), id);
        assert!(pool.is_empty());
    }

    #[test]
    fn pool_remove_nonexistent_fails() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let bogus = SessionId::from_parts(888, 888);
        assert!(pool.remove(bogus).is_err());
    }

    #[test]
    fn pool_session_ids() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let id1 = pool.acquire(default_cfg()).unwrap();
        let id2 = pool.acquire(default_cfg()).unwrap();
        let ids = pool.session_ids();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn pool_capacity() {
        let cfg = SessionPoolConfig { max_sessions: 5, ..Default::default() };
        let pool = SessionPool::new(cfg);
        assert_eq!(pool.capacity(), 5);
    }

    #[test]
    fn pool_total_created() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        pool.acquire(default_cfg()).unwrap();
        pool.acquire(default_cfg()).unwrap();
        assert_eq!(pool.total_created(), 2);
    }

    #[test]
    fn pool_debug_format() {
        let pool = SessionPool::new(SessionPoolConfig::default());
        let dbg = format!("{pool:?}");
        assert!(dbg.contains("SessionPool"));
    }

    // ── Concurrent session tests (no state leakage) ─────────────────

    #[test]
    fn concurrent_sessions_independent() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let id1 = pool.acquire(tiny_cfg(3)).unwrap();
        let id2 = pool.acquire(tiny_cfg(5)).unwrap();

        // Advance session 1.
        {
            let s1 = pool.get_mut(id1).unwrap();
            s1.start_prefill(1).unwrap();
            s1.start_generation().unwrap();
            s1.push_token(100).unwrap();
        }

        // Session 2 should still be idle.
        {
            let s2 = pool.get(id2).unwrap();
            assert_eq!(s2.state(), SessionState::Idle);
            assert!(s2.generated_tokens().is_empty());
        }

        // Advance session 2 independently.
        {
            let s2 = pool.get_mut(id2).unwrap();
            s2.start_prefill(2).unwrap();
            s2.start_generation().unwrap();
            s2.push_token(200).unwrap();
            s2.push_token(201).unwrap();
        }

        // Verify no state leakage.
        let s1 = pool.get(id1).unwrap();
        assert_eq!(s1.generated_tokens(), &[100]);
        let s2 = pool.get(id2).unwrap();
        assert_eq!(s2.generated_tokens(), &[200, 201]);
    }

    #[test]
    fn concurrent_sessions_cancel_one() {
        let mut pool = SessionPool::new(SessionPoolConfig::default());
        let id1 = pool.acquire(tiny_cfg(10)).unwrap();
        let id2 = pool.acquire(tiny_cfg(10)).unwrap();

        {
            let s1 = pool.get_mut(id1).unwrap();
            s1.start_prefill(1).unwrap();
            s1.start_generation().unwrap();
            s1.push_token(1).unwrap();
            s1.cancel().unwrap();
        }

        // Session 2 unaffected.
        {
            let s2 = pool.get_mut(id2).unwrap();
            s2.start_prefill(1).unwrap();
            s2.start_generation().unwrap();
            s2.push_token(2).unwrap();
            assert_eq!(s2.state(), SessionState::Generating);
        }
    }

    #[test]
    fn pool_evict_expired() {
        let cfg = SessionPoolConfig { max_sessions: 10, idle_timeout: Duration::from_millis(1) };
        let mut pool = SessionPool::new(cfg);
        pool.acquire(default_cfg()).unwrap();
        pool.acquire(default_cfg()).unwrap();

        // Sleep just long enough for the timeout.
        std::thread::sleep(Duration::from_millis(5));

        let evicted = pool.evict_expired();
        assert_eq!(evicted, 2);
        assert!(pool.is_empty());
    }

    // ── Property-style tests ────────────────────────────────────────

    #[test]
    fn property_no_self_transitions() {
        let states = [
            SessionState::Idle,
            SessionState::Prefilling,
            SessionState::Generating,
            SessionState::Complete,
            SessionState::Error,
            SessionState::Cancelled,
        ];
        for &s in &states {
            assert!(!s.can_transition_to(s), "{s} should not self-transition");
        }
    }

    #[test]
    fn property_error_eq() {
        let a = SessionError::Cancelled;
        let b = SessionError::Cancelled;
        assert_eq!(a, b);

        let c = SessionError::InvalidConfig("x".into());
        let d = SessionError::InvalidConfig("x".into());
        assert_eq!(c, d);
    }

    #[test]
    fn property_session_debug() {
        let s = InferenceSession::new(default_cfg()).unwrap();
        let dbg = format!("{s:?}");
        assert!(dbg.contains("InferenceSession"));
        assert!(dbg.contains("Idle"));
    }
}
