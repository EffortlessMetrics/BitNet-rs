//! Composable middleware chain for request processing.
//!
//! Priority-ordered middleware entries with enable/disable,
//! timing tracking, preset chains, and action types.

use std::collections::HashMap;
use std::time::Duration;

/// Middleware priority (lower = earlier in chain).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u32);

impl Priority {
    pub const FIRST: Self = Self(0);
    pub const AUTH: Self = Self(100);
    pub const RATE_LIMIT: Self = Self(200);
    pub const LOGGING: Self = Self(300);
    pub const CORS: Self = Self(400);
    pub const TRANSFORM: Self = Self(500);
    pub const LAST: Self = Self(u32::MAX);
}

/// Result of middleware processing.
#[derive(Debug, Clone, PartialEq)]
pub enum MiddlewareAction {
    Continue,
    Reject { status: u16, message: String },
    SetHeader { key: String, value: String },
}

/// A middleware entry in the chain.
#[derive(Debug, Clone)]
pub struct MiddlewareEntry {
    pub name: String,
    pub priority: Priority,
    pub enabled: bool,
}

impl MiddlewareEntry {
    pub fn new(name: impl Into<String>, priority: Priority) -> Self {
        Self { name: name.into(), priority, enabled: true }
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// Ordered chain of middleware entries.
#[derive(Debug, Default)]
pub struct MiddlewareChain {
    entries: Vec<MiddlewareEntry>,
    sorted: bool,
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, entry: MiddlewareEntry) -> &mut Self {
        self.entries.push(entry);
        self.sorted = false;
        self
    }

    pub fn sort(&mut self) {
        self.entries.sort_by_key(|e| e.priority);
        self.sorted = true;
    }

    pub fn enabled_entries(&mut self) -> Vec<&MiddlewareEntry> {
        if !self.sorted {
            self.sort();
        }
        self.entries.iter().filter(|e| e.enabled).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn enabled_count(&self) -> usize {
        self.entries.iter().filter(|e| e.enabled).count()
    }

    pub fn remove_by_name(&mut self, name: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.name != name);
        self.entries.len() < before
    }

    pub fn names(&mut self) -> Vec<String> {
        self.enabled_entries().iter().map(|e| e.name.clone()).collect()
    }
}

/// Execution timing tracker for middleware.
#[derive(Debug, Default)]
pub struct MiddlewareTimings {
    timings: HashMap<String, Duration>,
}

impl MiddlewareTimings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, name: &str, duration: Duration) {
        self.timings.insert(name.to_string(), duration);
    }

    pub fn get(&self, name: &str) -> Option<Duration> {
        self.timings.get(name).copied()
    }

    pub fn total(&self) -> Duration {
        self.timings.values().sum()
    }

    pub fn slowest(&self) -> Option<(&str, Duration)> {
        self.timings.iter().max_by_key(|(_, d)| *d).map(|(n, d)| (n.as_str(), *d))
    }

    pub fn count(&self) -> usize {
        self.timings.len()
    }
}

/// Production middleware chain preset.
pub fn production_chain() -> MiddlewareChain {
    let mut chain = MiddlewareChain::new();
    chain.add(MiddlewareEntry::new("request_id", Priority::FIRST));
    chain.add(MiddlewareEntry::new("auth", Priority::AUTH));
    chain.add(MiddlewareEntry::new("rate_limit", Priority::RATE_LIMIT));
    chain.add(MiddlewareEntry::new("logging", Priority::LOGGING));
    chain.add(MiddlewareEntry::new("cors", Priority::CORS));
    chain.add(MiddlewareEntry::new("transform", Priority::TRANSFORM));
    chain
}

/// Minimal chain with only logging.
pub fn minimal_chain() -> MiddlewareChain {
    let mut chain = MiddlewareChain::new();
    chain.add(MiddlewareEntry::new("logging", Priority::LOGGING));
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::FIRST < Priority::AUTH);
        assert!(Priority::AUTH < Priority::RATE_LIMIT);
        assert!(Priority::CORS < Priority::LAST);
    }

    #[test]
    fn test_chain_add_and_sort() {
        let mut chain = MiddlewareChain::new();
        chain.add(MiddlewareEntry::new("cors", Priority::CORS));
        chain.add(MiddlewareEntry::new("auth", Priority::AUTH));
        let entries = chain.enabled_entries();
        assert_eq!(entries[0].name, "auth");
        assert_eq!(entries[1].name, "cors");
    }

    #[test]
    fn test_disabled_entry() {
        let mut chain = MiddlewareChain::new();
        chain.add(MiddlewareEntry::new("auth", Priority::AUTH).disabled());
        chain.add(MiddlewareEntry::new("logging", Priority::LOGGING));
        assert_eq!(chain.enabled_count(), 1);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_remove_by_name() {
        let mut chain = MiddlewareChain::new();
        chain.add(MiddlewareEntry::new("auth", Priority::AUTH));
        chain.add(MiddlewareEntry::new("cors", Priority::CORS));
        assert!(chain.remove_by_name("auth"));
        assert_eq!(chain.len(), 1);
        assert!(!chain.remove_by_name("nonexistent"));
    }

    #[test]
    fn test_middleware_action_reject() {
        let action = MiddlewareAction::Reject { status: 429, message: "rate limited".into() };
        if let MiddlewareAction::Reject { status, .. } = &action {
            assert_eq!(*status, 429);
        }
    }

    #[test]
    fn test_timings_record_and_total() {
        let mut timings = MiddlewareTimings::new();
        timings.record("auth", Duration::from_millis(5));
        timings.record("cors", Duration::from_millis(10));
        assert_eq!(timings.count(), 2);
        assert_eq!(timings.total(), Duration::from_millis(15));
    }

    #[test]
    fn test_timings_slowest() {
        let mut timings = MiddlewareTimings::new();
        timings.record("fast", Duration::from_millis(1));
        timings.record("slow", Duration::from_millis(100));
        let (name, dur) = timings.slowest().unwrap();
        assert_eq!(name, "slow");
        assert_eq!(dur, Duration::from_millis(100));
    }

    #[test]
    fn test_production_chain() {
        let mut chain = production_chain();
        let entries = chain.enabled_entries();
        assert_eq!(entries.len(), 6);
        assert_eq!(entries[0].name, "request_id");
    }

    #[test]
    fn test_minimal_chain() {
        let chain = minimal_chain();
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_names_sorted() {
        let mut chain = MiddlewareChain::new();
        chain.add(MiddlewareEntry::new("b", Priority(2)));
        chain.add(MiddlewareEntry::new("a", Priority(1)));
        let names = chain.names();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_empty_chain() {
        let chain = MiddlewareChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_set_header_action() {
        let action =
            MiddlewareAction::SetHeader { key: "X-Request-Id".into(), value: "abc123".into() };
        if let MiddlewareAction::SetHeader { key, value } = &action {
            assert_eq!(key, "X-Request-Id");
            assert_eq!(value, "abc123");
        }
    }
}
