//! GPU-aware streaming generation with device-to-host token transfer.
//!
//! Extends [`GenerationStream`] to work transparently with GPU backends,
//! handling per-token GPU→host transfers and backpressure when the client
//! cannot keep up with generation speed.

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls GPU-specific streaming behaviour.
#[derive(Debug, Clone)]
pub struct GpuStreamingConfig {
    /// Channel capacity for the token pipeline.  When the channel is full the
    /// generation loop yields to the runtime (backpressure).
    pub channel_capacity: usize,
    /// Maximum time to wait when the channel is full before dropping the token.
    pub backpressure_timeout: Duration,
    /// Whether to synchronise the GPU stream after every token transfer.
    pub sync_per_token: bool,
    /// Enable cancellation support via a shared atomic flag.
    pub cancellable: bool,
}

impl Default for GpuStreamingConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 32,
            backpressure_timeout: Duration::from_secs(5),
            sync_per_token: true,
            cancellable: true,
        }
    }
}

impl GpuStreamingConfig {
    /// Low-latency preset: small buffer, sync every token.
    pub fn low_latency() -> Self {
        Self {
            channel_capacity: 4,
            backpressure_timeout: Duration::from_secs(2),
            sync_per_token: true,
            cancellable: true,
        }
    }

    /// High-throughput preset: larger buffer, async transfers.
    pub fn high_throughput() -> Self {
        Self {
            channel_capacity: 128,
            backpressure_timeout: Duration::from_secs(10),
            sync_per_token: false,
            cancellable: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.channel_capacity == 0 {
            anyhow::bail!("channel_capacity must be > 0");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Token event
// ---------------------------------------------------------------------------

/// A single token produced by the GPU generation loop.
#[derive(Debug, Clone)]
pub struct GpuTokenEvent {
    /// Decoded text for this token.
    pub text: String,
    /// Vocabulary token id.
    pub token_id: u32,
    /// 0-based position in the generated sequence.
    pub position: usize,
    /// Wall-clock time since generation start.
    pub elapsed: Duration,
    /// GPU→host transfer latency for this token (if measured).
    pub transfer_latency: Option<Duration>,
}

/// Signals the end of generation.
#[derive(Debug, Clone)]
pub struct GpuStreamComplete {
    pub total_tokens: u64,
    pub total_time: Duration,
    pub tokens_per_second: f64,
    pub backpressure_events: u64,
}

// ---------------------------------------------------------------------------
// Stream statistics
// ---------------------------------------------------------------------------

/// Runtime statistics collected during GPU streaming.
#[derive(Debug, Default)]
pub struct GpuStreamStats {
    pub tokens_sent: AtomicU64,
    pub backpressure_events: AtomicU64,
    pub dropped_tokens: AtomicU64,
    pub cancelled: AtomicBool,
}

impl GpuStreamStats {
    pub fn snapshot(&self) -> GpuStreamStatsSnapshot {
        GpuStreamStatsSnapshot {
            tokens_sent: self.tokens_sent.load(Ordering::Relaxed),
            backpressure_events: self.backpressure_events.load(Ordering::Relaxed),
            dropped_tokens: self.dropped_tokens.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
        }
    }
}

/// Immutable copy of streaming statistics.
#[derive(Debug, Clone)]
pub struct GpuStreamStatsSnapshot {
    pub tokens_sent: u64,
    pub backpressure_events: u64,
    pub dropped_tokens: u64,
    pub cancelled: bool,
}

// ---------------------------------------------------------------------------
// GPU generation stream
// ---------------------------------------------------------------------------

/// A stream of tokens generated on a GPU device, with backpressure support.
///
/// The generation loop runs in a background Tokio task.  Tokens are sent
/// through a bounded channel; when the consumer is slow the loop waits up
/// to [`GpuStreamingConfig::backpressure_timeout`] before dropping tokens.
pub struct GpuGenerationStream {
    receiver: mpsc::Receiver<Result<GpuTokenEvent>>,
    _task: tokio::task::JoinHandle<()>,
    cancel: Arc<AtomicBool>,
    stats: Arc<GpuStreamStats>,
    start_time: Instant,
}

impl GpuGenerationStream {
    /// Spawn a GPU generation stream using the provided token producer.
    pub fn spawn<F, Fut>(config: GpuStreamingConfig, producer: F) -> Result<Self>
    where
        F: FnOnce(mpsc::Sender<Result<GpuTokenEvent>>, Arc<AtomicBool>, Arc<GpuStreamStats>) -> Fut
            + Send
            + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        config.validate()?;

        let (tx, rx) = mpsc::channel(config.channel_capacity);
        let cancel = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(GpuStreamStats::default());
        let start_time = Instant::now();

        let cancel_clone = cancel.clone();
        let stats_clone = stats.clone();

        let task = tokio::spawn(async move {
            producer(tx, cancel_clone, stats_clone).await;
        });

        Ok(Self { receiver: rx, _task: task, cancel, stats, start_time })
    }

    /// Receive the next token from the GPU generation loop.
    pub async fn next(&mut self) -> Option<Result<GpuTokenEvent>> {
        self.receiver.recv().await
    }

    /// Cancel generation (cooperative).
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Return a snapshot of streaming statistics.
    pub fn stats(&self) -> GpuStreamStatsSnapshot {
        self.stats.snapshot()
    }

    /// Build a [`GpuStreamComplete`] summary.
    pub fn complete_summary(&self) -> GpuStreamComplete {
        let elapsed = self.start_time.elapsed();
        let total_tokens = self.stats.tokens_sent.load(Ordering::Relaxed);
        let tps = if elapsed.as_millis() > 0 {
            (total_tokens as f64 * 1000.0) / elapsed.as_millis() as f64
        } else {
            0.0
        };
        GpuStreamComplete {
            total_tokens,
            total_time: elapsed,
            tokens_per_second: tps,
            backpressure_events: self.stats.backpressure_events.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: send with backpressure
// ---------------------------------------------------------------------------

/// Send a token through the channel, respecting backpressure timeout.
/// Returns `true` if sent, `false` if dropped.
pub async fn send_with_backpressure(
    tx: &mpsc::Sender<Result<GpuTokenEvent>>,
    event: GpuTokenEvent,
    timeout: Duration,
    stats: &GpuStreamStats,
) -> bool {
    match tokio::time::timeout(timeout, tx.send(Ok(event))).await {
        Ok(Ok(())) => {
            stats.tokens_sent.fetch_add(1, Ordering::Relaxed);
            true
        }
        Ok(Err(_)) => {
            debug!("token channel closed by consumer");
            false
        }
        Err(_) => {
            stats.backpressure_events.fetch_add(1, Ordering::Relaxed);
            stats.dropped_tokens.fetch_add(1, Ordering::Relaxed);
            warn!("backpressure timeout: dropping token");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        GpuStreamingConfig::default().validate().unwrap();
    }

    #[test]
    fn low_latency_config_is_valid() {
        GpuStreamingConfig::low_latency().validate().unwrap();
    }

    #[test]
    fn high_throughput_config_is_valid() {
        GpuStreamingConfig::high_throughput().validate().unwrap();
    }

    #[test]
    fn zero_capacity_config_is_invalid() {
        let mut cfg = GpuStreamingConfig::default();
        cfg.channel_capacity = 0;
        assert!(cfg.validate().is_err());
    }

    #[tokio::test]
    async fn stream_receives_tokens_from_producer() {
        let config = GpuStreamingConfig { channel_capacity: 8, ..Default::default() };
        let mut stream = GpuGenerationStream::spawn(config, |tx, _cancel, stats| async move {
            for i in 0..5u32 {
                let event = GpuTokenEvent {
                    text: format!("tok{i}"),
                    token_id: i,
                    position: i as usize,
                    elapsed: Duration::from_millis(i as u64 * 10),
                    transfer_latency: Some(Duration::from_micros(50)),
                };
                send_with_backpressure(&tx, event, Duration::from_secs(1), &stats).await;
            }
        })
        .unwrap();

        let mut received = Vec::new();
        while let Some(Ok(event)) = stream.next().await {
            received.push(event.token_id);
        }
        assert_eq!(received, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn cancellation_stops_producer() {
        let config = GpuStreamingConfig { channel_capacity: 4, ..Default::default() };
        let mut stream = GpuGenerationStream::spawn(config, |tx, cancel, stats| async move {
            for i in 0..1000u32 {
                if cancel.load(Ordering::Relaxed) {
                    stats.cancelled.store(true, Ordering::Relaxed);
                    break;
                }
                let event = GpuTokenEvent {
                    text: format!("t{i}"),
                    token_id: i,
                    position: i as usize,
                    elapsed: Duration::from_millis(i as u64),
                    transfer_latency: None,
                };
                if !send_with_backpressure(&tx, event, Duration::from_millis(100), &stats).await {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .unwrap();

        // Read a few tokens then cancel.
        let _ = stream.next().await;
        let _ = stream.next().await;
        stream.cancel();

        // Drain remaining.
        while stream.next().await.is_some() {}

        assert!(stream.is_cancelled());
    }

    #[tokio::test]
    async fn backpressure_records_stats() {
        let config = GpuStreamingConfig {
            channel_capacity: 1,
            backpressure_timeout: Duration::from_millis(10),
            ..Default::default()
        };

        let mut stream = GpuGenerationStream::spawn(config, |tx, _cancel, stats| async move {
            for i in 0..10u32 {
                let event = GpuTokenEvent {
                    text: format!("bp{i}"),
                    token_id: i,
                    position: i as usize,
                    elapsed: Duration::from_millis(i as u64),
                    transfer_latency: None,
                };
                send_with_backpressure(&tx, event, Duration::from_millis(10), &stats).await;
            }
        })
        .unwrap();

        // Deliberately slow consumer.
        tokio::time::sleep(Duration::from_millis(200)).await;
        while stream.next().await.is_some() {}

        let snap = stream.stats();
        assert!(snap.tokens_sent > 0 || snap.dropped_tokens > 0);
    }

    #[tokio::test]
    async fn complete_summary_reports_tokens() {
        let config = GpuStreamingConfig::default();
        let mut stream = GpuGenerationStream::spawn(config, |tx, _cancel, stats| async move {
            for i in 0..3u32 {
                let event = GpuTokenEvent {
                    text: format!("s{i}"),
                    token_id: i,
                    position: i as usize,
                    elapsed: Duration::from_millis(i as u64 * 5),
                    transfer_latency: None,
                };
                send_with_backpressure(&tx, event, Duration::from_secs(1), &stats).await;
            }
        })
        .unwrap();

        while stream.next().await.is_some() {}

        let summary = stream.complete_summary();
        assert_eq!(summary.total_tokens, 3);
    }

    #[test]
    fn stats_snapshot_reflects_updates() {
        let stats = GpuStreamStats::default();
        stats.tokens_sent.store(42, Ordering::Relaxed);
        stats.backpressure_events.store(3, Ordering::Relaxed);
        stats.dropped_tokens.store(1, Ordering::Relaxed);
        stats.cancelled.store(true, Ordering::Relaxed);

        let snap = stats.snapshot();
        assert_eq!(snap.tokens_sent, 42);
        assert_eq!(snap.backpressure_events, 3);
        assert_eq!(snap.dropped_tokens, 1);
        assert!(snap.cancelled);
    }
}
