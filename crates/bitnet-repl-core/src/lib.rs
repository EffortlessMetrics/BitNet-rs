//! Core, reusable building blocks for chat-style REPL flows.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Parsed result of one line entered into the REPL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplInput {
    Exit,
    Help,
    Clear,
    Metrics,
    Message(String),
}

/// Parse a raw line from stdin into a REPL action.
///
/// Empty or whitespace-only lines are ignored and return `None`.
pub fn parse_repl_input(line: &str) -> Option<ReplInput> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    Some(match line {
        "/exit" | "/quit" => ReplInput::Exit,
        "/help" => ReplInput::Help,
        "/clear" => ReplInput::Clear,
        "/metrics" => ReplInput::Metrics,
        _ => ReplInput::Message(line.to_owned()),
    })
}

/// Performance metrics for a chat session.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ChatMetrics {
    pub total_tokens_generated: usize,
    pub total_time_ms: u64,
    pub num_exchanges: usize,
}

impl ChatMetrics {
    pub fn add_exchange(&mut self, tokens: usize, elapsed_ms: u64) {
        self.total_tokens_generated += tokens;
        self.total_time_ms += elapsed_ms;
        self.num_exchanges += 1;
    }

    pub fn average_tps(&self) -> f64 {
        if self.total_time_ms > 0 {
            self.total_tokens_generated as f64 / (self.total_time_ms as f64 / 1000.0)
        } else {
            0.0
        }
    }
}

/// FIFO history buffer with an optional max size.
#[derive(Debug, Clone)]
pub struct BoundedHistory<T> {
    items: Vec<T>,
    limit: Option<usize>,
}

impl<T> BoundedHistory<T> {
    pub fn new(limit: Option<usize>) -> Self {
        Self { items: Vec::new(), limit }
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item);
        self.enforce_limit();
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.items.clone()
    }

    fn enforce_limit(&mut self) {
        if let Some(limit) = self.limit
            && self.items.len() > limit
        {
            let excess = self.items.len() - limit;
            self.items.drain(0..excess);
        }
    }
}

/// Copy receipt from effective receipt path to timestamped file in the target directory.
pub fn copy_receipt_if_present(src: &Path, dir: &Path) -> Result<Option<PathBuf>> {
    if !src.exists() {
        return Ok(None);
    }

    fs::create_dir_all(dir)?;
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let dst = dir.join(format!("chat-{}.json", ts));
    fs::copy(src, &dst)?;
    Ok(Some(dst))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_commands_and_messages() {
        assert_eq!(parse_repl_input("   "), None);
        assert_eq!(parse_repl_input("/help"), Some(ReplInput::Help));
        assert_eq!(parse_repl_input("/quit"), Some(ReplInput::Exit));
        assert_eq!(parse_repl_input("hello"), Some(ReplInput::Message("hello".into())));
    }

    #[test]
    fn bounded_history_trims_fifo() {
        let mut history = BoundedHistory::new(Some(2));
        history.push(1);
        history.push(2);
        history.push(3);

        let items: Vec<_> = history.iter().copied().collect();
        assert_eq!(items, vec![2, 3]);
    }

    #[test]
    fn metrics_average_is_zero_without_time() {
        let mut metrics = ChatMetrics::default();
        metrics.add_exchange(5, 0);
        assert_eq!(metrics.average_tps(), 0.0);
    }

    #[test]
    fn copies_receipt_if_present() {
        let src_dir = tempdir().expect("temp dir should be created");
        let dst_dir = tempdir().expect("temp dir should be created");
        let src = src_dir.path().join("receipt.json");
        fs::write(&src, "{}\n").expect("receipt fixture should be written");

        let copied = copy_receipt_if_present(&src, dst_dir.path())
            .expect("copy should succeed")
            .expect("receipt should be copied");

        assert!(copied.exists());
    }
}
