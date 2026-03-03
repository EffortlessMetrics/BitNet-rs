//! Multi-stream dispatch for overlapping compute and memory transfers.

use crate::config::LaunchConfig;

/// Direction of a memory transfer relative to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host → Device.
    HostToDevice,
    /// Device → Host.
    DeviceToHost,
    /// Device → Device (peer or same-GPU).
    DeviceToDevice,
}

/// A single slot in a stream timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamSlot {
    /// A kernel launch.
    Kernel {
        /// Logical name for profiling / debugging.
        tag: String,
        /// Launch configuration.
        config: LaunchConfig,
    },
    /// An asynchronous memory transfer.
    Transfer {
        /// Logical name for profiling / debugging.
        tag: String,
        /// Transfer direction.
        direction: TransferDirection,
        /// Number of bytes to transfer.
        bytes: u64,
    },
    /// A stream-level barrier (wait for an event from another stream).
    WaitEvent {
        /// Stream ordinal that records the event.
        source_stream: u32,
        /// Ordinal of the slot in the source stream after which the event
        /// is recorded.
        after_slot: usize,
    },
}

impl StreamSlot {
    /// Create a kernel slot.
    #[must_use]
    pub fn kernel(tag: impl Into<String>, config: LaunchConfig) -> Self {
        Self::Kernel { tag: tag.into(), config }
    }

    /// Create a transfer slot.
    #[must_use]
    pub fn transfer(tag: impl Into<String>, direction: TransferDirection, bytes: u64) -> Self {
        Self::Transfer { tag: tag.into(), direction, bytes }
    }
}

/// A plan describing work across multiple CUDA streams, enabling
/// compute / transfer overlap.
#[derive(Debug, Clone)]
pub struct StreamPlan {
    streams: Vec<Vec<StreamSlot>>,
}

impl StreamPlan {
    /// Create a plan with `n` streams.
    #[must_use]
    pub fn new(num_streams: usize) -> Self {
        Self { streams: vec![Vec::new(); num_streams.max(1)] }
    }

    /// Number of streams.
    #[must_use]
    pub const fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Push a slot onto a given stream. Returns the slot index within
    /// that stream.
    ///
    /// # Panics
    ///
    /// Panics if `stream_id >= num_streams()`.
    pub fn push(&mut self, stream_id: usize, slot: StreamSlot) -> usize {
        let stream = &mut self.streams[stream_id];
        let idx = stream.len();
        stream.push(slot);
        idx
    }

    /// Convenience: push a kernel onto a stream.
    pub fn push_kernel(
        &mut self,
        stream_id: usize,
        tag: impl Into<String>,
        config: LaunchConfig,
    ) -> usize {
        self.push(stream_id, StreamSlot::kernel(tag, config))
    }

    /// Convenience: push a transfer onto a stream.
    pub fn push_transfer(
        &mut self,
        stream_id: usize,
        tag: impl Into<String>,
        direction: TransferDirection,
        bytes: u64,
    ) -> usize {
        self.push(stream_id, StreamSlot::transfer(tag, direction, bytes))
    }

    /// Insert a barrier: `waiter_stream` will wait for `source_stream`
    /// to complete slot `after_slot`.
    pub fn insert_barrier(
        &mut self,
        waiter_stream: usize,
        source_stream: u32,
        after_slot: usize,
    ) -> usize {
        self.push(waiter_stream, StreamSlot::WaitEvent { source_stream, after_slot })
    }

    /// Read-only view of a single stream's slots.
    #[must_use]
    pub fn stream_slots(&self, stream_id: usize) -> &[StreamSlot] {
        &self.streams[stream_id]
    }

    /// Total kernel launches across all streams.
    #[must_use]
    pub fn total_kernels(&self) -> usize {
        self.streams
            .iter()
            .flat_map(|s| s.iter())
            .filter(|slot| matches!(slot, StreamSlot::Kernel { .. }))
            .count()
    }

    /// Total transfer bytes across all streams.
    #[must_use]
    pub fn total_transfer_bytes(&self) -> u64 {
        self.streams
            .iter()
            .flat_map(|s| s.iter())
            .filter_map(|slot| match slot {
                StreamSlot::Transfer { bytes, .. } => Some(*bytes),
                _ => None,
            })
            .sum()
    }

    /// Validate the plan: barrier sources must reference valid streams
    /// and slots.
    pub fn validate(&self) -> Result<(), StreamValidationError> {
        for (sid, stream) in self.streams.iter().enumerate() {
            for (slot_idx, slot) in stream.iter().enumerate() {
                if let StreamSlot::WaitEvent { source_stream, after_slot } = slot {
                    let src = *source_stream as usize;
                    if src >= self.streams.len() {
                        return Err(StreamValidationError::InvalidSourceStream {
                            stream: sid,
                            slot: slot_idx,
                            source: src,
                        });
                    }
                    if *after_slot >= self.streams[src].len() {
                        return Err(StreamValidationError::InvalidSourceSlot {
                            stream: sid,
                            slot: slot_idx,
                            source_stream: src,
                            source_slot: *after_slot,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Errors from stream plan validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamValidationError {
    /// A `WaitEvent` references a non-existent stream.
    InvalidSourceStream {
        /// Stream containing the bad barrier.
        stream: usize,
        /// Slot index of the bad barrier.
        slot: usize,
        /// The invalid source stream index.
        source: usize,
    },
    /// A `WaitEvent` references a non-existent slot in the source stream.
    InvalidSourceSlot {
        /// Stream containing the bad barrier.
        stream: usize,
        /// Slot index of the bad barrier.
        slot: usize,
        /// The source stream index.
        source_stream: usize,
        /// The invalid slot index in the source stream.
        source_slot: usize,
    },
}

impl std::fmt::Display for StreamValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSourceStream { stream, slot, source } => write!(
                f,
                "stream {stream} slot {slot} references \
                 non-existent source stream {source}"
            ),
            Self::InvalidSourceSlot { stream, slot, source_stream, source_slot } => write!(
                f,
                "stream {stream} slot {slot} references \
                 non-existent slot {source_slot} in stream {source_stream}"
            ),
        }
    }
}

impl std::error::Error for StreamValidationError {}

// ── Helper: build a compute+transfer overlap plan ──────────────────────

/// Build a simple two-stream plan: stream 0 for compute, stream 1 for
/// host↔device transfers, with barriers so transfers complete before
/// compute reads the data and compute completes before results are read
/// back.
///
/// This is the canonical double-buffered dispatch pattern.
#[must_use]
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn double_buffered_plan(
    upload_bytes: u64,
    kernel_tag: impl Into<String>,
    kernel_config: LaunchConfig,
    download_bytes: u64,
) -> StreamPlan {
    let tag = kernel_tag.into();
    let mut plan = StreamPlan::new(2);

    // Stream 1: upload H→D
    let upload_slot = plan.push_transfer(
        1,
        format!("{tag}_upload"),
        TransferDirection::HostToDevice,
        upload_bytes,
    );

    // Stream 0: wait for upload, then run kernel
    plan.insert_barrier(0, 1, upload_slot);
    let kern_slot = plan.push_kernel(0, &tag, kernel_config);

    // Stream 1: wait for kernel, then download D→H
    plan.insert_barrier(1, 0, kern_slot);
    plan.push_transfer(
        1,
        format!("{tag}_download"),
        TransferDirection::DeviceToHost,
        download_bytes,
    );

    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LaunchConfig {
        LaunchConfig::for_elements(256)
    }

    // ── Construction ───────────────────────────────────────────────────

    #[test]
    fn new_creates_streams() {
        let p = StreamPlan::new(3);
        assert_eq!(p.num_streams(), 3);
    }

    #[test]
    fn new_at_least_one() {
        let p = StreamPlan::new(0);
        assert_eq!(p.num_streams(), 1);
    }

    // ── Push operations ────────────────────────────────────────────────

    #[test]
    fn push_kernel_slot() {
        let mut p = StreamPlan::new(1);
        let idx = p.push_kernel(0, "k1", cfg());
        assert_eq!(idx, 0);
        assert_eq!(p.total_kernels(), 1);
    }

    #[test]
    fn push_transfer_slot() {
        let mut p = StreamPlan::new(1);
        p.push_transfer(0, "xfer", TransferDirection::HostToDevice, 4096);
        assert_eq!(p.total_transfer_bytes(), 4096);
    }

    #[test]
    fn push_multiple_streams() {
        let mut p = StreamPlan::new(2);
        p.push_kernel(0, "k0", cfg());
        p.push_kernel(1, "k1", cfg());
        assert_eq!(p.total_kernels(), 2);
    }

    // ── Barriers ───────────────────────────────────────────────────────

    #[test]
    fn barrier_valid() {
        let mut p = StreamPlan::new(2);
        let slot = p.push_kernel(0, "k0", cfg());
        p.insert_barrier(1, 0, slot);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn barrier_invalid_stream() {
        let mut p = StreamPlan::new(2);
        p.push(0, StreamSlot::WaitEvent { source_stream: 5, after_slot: 0 });
        assert!(matches!(
            p.validate().unwrap_err(),
            StreamValidationError::InvalidSourceStream { .. }
        ));
    }

    #[test]
    fn barrier_invalid_slot() {
        let mut p = StreamPlan::new(2);
        p.push(1, StreamSlot::WaitEvent { source_stream: 0, after_slot: 0 });
        assert!(matches!(
            p.validate().unwrap_err(),
            StreamValidationError::InvalidSourceSlot { .. }
        ));
    }

    // ── Aggregate helpers ──────────────────────────────────────────────

    #[test]
    fn total_transfer_bytes_multiple() {
        let mut p = StreamPlan::new(2);
        p.push_transfer(0, "a", TransferDirection::HostToDevice, 1000);
        p.push_transfer(1, "b", TransferDirection::DeviceToHost, 2000);
        assert_eq!(p.total_transfer_bytes(), 3000);
    }

    #[test]
    fn total_kernels_mixed() {
        let mut p = StreamPlan::new(1);
        p.push_kernel(0, "k", cfg());
        p.push_transfer(0, "t", TransferDirection::HostToDevice, 100);
        p.push_kernel(0, "k2", cfg());
        assert_eq!(p.total_kernels(), 2);
    }

    // ── Double-buffered plan ───────────────────────────────────────────

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn double_buffered_structure() {
        let p = double_buffered_plan(1024, "matmul", cfg(), 512);
        assert_eq!(p.num_streams(), 2);
        assert_eq!(p.total_kernels(), 1);
        assert_eq!(p.total_transfer_bytes(), 1024 + 512);
        assert!(p.validate().is_ok());
    }

    // ── Stream slots accessor ──────────────────────────────────────────

    #[test]
    fn stream_slots_accessor() {
        let mut p = StreamPlan::new(2);
        p.push_kernel(0, "a", cfg());
        p.push_kernel(0, "b", cfg());
        assert_eq!(p.stream_slots(0).len(), 2);
        assert_eq!(p.stream_slots(1).len(), 0);
    }

    // ── Validation errors display ──────────────────────────────────────

    #[test]
    fn validation_error_display() {
        let e = StreamValidationError::InvalidSourceStream { stream: 1, slot: 0, source: 5 };
        assert!(e.to_string().contains("non-existent source stream 5"));
    }

    // ── StreamSlot constructors ────────────────────────────────────────

    #[test]
    fn slot_kernel_constructor() {
        let slot = StreamSlot::kernel("test", cfg());
        assert!(matches!(slot, StreamSlot::Kernel { .. }));
    }

    #[test]
    fn slot_transfer_constructor() {
        let slot = StreamSlot::transfer("xfer", TransferDirection::DeviceToDevice, 999);
        if let StreamSlot::Transfer { bytes, direction, .. } = slot {
            assert_eq!(bytes, 999);
            assert_eq!(direction, TransferDirection::DeviceToDevice);
        } else {
            panic!("expected Transfer");
        }
    }

    // ── TransferDirection equality ─────────────────────────────────────

    #[test]
    fn transfer_direction_eq() {
        assert_eq!(TransferDirection::HostToDevice, TransferDirection::HostToDevice);
        assert_ne!(TransferDirection::HostToDevice, TransferDirection::DeviceToHost);
    }
}
