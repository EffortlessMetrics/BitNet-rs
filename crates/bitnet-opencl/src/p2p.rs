//! GPU-to-GPU peer-to-peer (P2P) memory transfer.
//!
//! Detects P2P capability between devices (OpenCL SVM, CUDA P2P),
//! performs direct GPU-GPU buffer copies when possible, and falls back to
//! host-staged copies when P2P is unavailable. Includes bandwidth
//! measurement and reporting.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Device identifier
// ---------------------------------------------------------------------------

/// Lightweight identifier for a GPU device participating in P2P transfers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId {
    /// Platform-specific index (e.g. CUDA ordinal or OpenCL device index).
    pub index: usize,
    /// Backend type.
    pub backend: BackendKind,
}

/// Supported P2P backend discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cuda,
    OpenCl,
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}:{}", self.backend, self.index)
    }
}

// ---------------------------------------------------------------------------
// P2P capability
// ---------------------------------------------------------------------------

/// Result of probing P2P capability between two devices.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct P2PCapability {
    pub src: DeviceId,
    pub dst: DeviceId,
    pub direct_supported: bool,
    /// Reason when direct P2P is not available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Trait abstracting the platform-specific P2P capability probe.
pub trait P2PProbe: Send + Sync {
    /// Check if direct P2P is possible between `src` and `dst`.
    fn probe(&self, src: &DeviceId, dst: &DeviceId) -> P2PCapability;
}

/// Default probe that always reports P2P as unavailable (safe fallback).
#[derive(Debug, Default)]
pub struct FallbackProbe;

impl P2PProbe for FallbackProbe {
    fn probe(&self, src: &DeviceId, dst: &DeviceId) -> P2PCapability {
        P2PCapability {
            src: src.clone(),
            dst: dst.clone(),
            direct_supported: false,
            reason: Some("No runtime P2P driver available".to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer path
// ---------------------------------------------------------------------------

/// The path chosen for a particular copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferPath {
    /// Direct GPU-GPU DMA (peer-to-peer).
    Direct,
    /// GPU → Host → GPU (fallback).
    HostStaged,
}

// ---------------------------------------------------------------------------
// Transfer result & bandwidth
// ---------------------------------------------------------------------------

/// Bandwidth measurement from a transfer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BandwidthReport {
    pub bytes_transferred: u64,
    pub duration: Duration,
    pub bandwidth_gbps: f64,
    pub path: TransferPath,
    pub src: DeviceId,
    pub dst: DeviceId,
}

impl BandwidthReport {
    pub fn new(
        bytes: u64,
        duration: Duration,
        path: TransferPath,
        src: DeviceId,
        dst: DeviceId,
    ) -> Self {
        let secs = duration.as_secs_f64();
        let gbps = if secs > 0.0 { (bytes as f64 * 8.0) / (secs * 1e9) } else { 0.0 };
        Self { bytes_transferred: bytes, duration, bandwidth_gbps: gbps, path, src, dst }
    }
}

// ---------------------------------------------------------------------------
// Transfer engine
// ---------------------------------------------------------------------------

/// Errors from P2P transfer operations.
#[derive(Debug, thiserror::Error)]
pub enum P2PError {
    #[error("Transfer failed: {0}")]
    TransferFailed(String),
    #[error("Device not found: {0}")]
    DeviceNotFound(String),
}

/// Engine that orchestrates GPU-GPU copies, choosing direct P2P or
/// host-staged fallback automatically.
pub struct TransferEngine<P: P2PProbe = FallbackProbe> {
    probe: P,
}

impl TransferEngine<FallbackProbe> {
    /// Create an engine with the default (fallback) probe.
    pub fn new() -> Self {
        Self { probe: FallbackProbe }
    }
}

impl Default for TransferEngine<FallbackProbe> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: P2PProbe> TransferEngine<P> {
    /// Create an engine with a custom probe.
    pub fn with_probe(probe: P) -> Self {
        Self { probe }
    }

    /// Detect P2P capability between two devices.
    pub fn detect(&self, src: &DeviceId, dst: &DeviceId) -> P2PCapability {
        self.probe.probe(src, dst)
    }

    /// Transfer a buffer, choosing the best path automatically.
    ///
    /// `data` is the raw bytes to move. In a real implementation the
    /// runtime would handle GPU pointers; here we simulate the copy to
    /// measure overhead and exercise the path-selection logic.
    pub fn transfer(
        &self,
        src: &DeviceId,
        dst: &DeviceId,
        data: &[u8],
    ) -> Result<BandwidthReport, P2PError> {
        let cap = self.probe.probe(src, dst);
        let path =
            if cap.direct_supported { TransferPath::Direct } else { TransferPath::HostStaged };

        let start = Instant::now();
        match path {
            TransferPath::Direct => {
                self.direct_copy(src, dst, data)?;
            }
            TransferPath::HostStaged => {
                self.host_staged_copy(src, dst, data)?;
            }
        }
        let elapsed = start.elapsed();
        Ok(BandwidthReport::new(data.len() as u64, elapsed, path, src.clone(), dst.clone()))
    }

    /// Simulate a direct GPU-GPU DMA copy.
    fn direct_copy(&self, _src: &DeviceId, _dst: &DeviceId, _data: &[u8]) -> Result<(), P2PError> {
        // In production this would issue a P2P DMA via CUDA or OpenCL SVM.
        Ok(())
    }

    /// Simulate a host-staged copy (GPU→host→GPU).
    fn host_staged_copy(
        &self,
        _src: &DeviceId,
        _dst: &DeviceId,
        _data: &[u8],
    ) -> Result<(), P2PError> {
        // Allocate a temporary host buffer and copy through it.
        // Simulated here — real implementation would use pinned memory.
        let _staging = Vec::from(_data);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_dev(idx: usize) -> DeviceId {
        DeviceId { index: idx, backend: BackendKind::Cuda }
    }

    fn ocl_dev(idx: usize) -> DeviceId {
        DeviceId { index: idx, backend: BackendKind::OpenCl }
    }

    // Probe that says P2P is supported for same-backend, different device.
    struct TestProbe;
    impl P2PProbe for TestProbe {
        fn probe(&self, src: &DeviceId, dst: &DeviceId) -> P2PCapability {
            let supported = src.backend == dst.backend && src.index != dst.index;
            P2PCapability {
                src: src.clone(),
                dst: dst.clone(),
                direct_supported: supported,
                reason: if supported {
                    None
                } else {
                    Some("cross-backend or same device".to_string())
                },
            }
        }
    }

    #[test]
    fn test_fallback_probe_always_false() {
        let probe = FallbackProbe;
        let cap = probe.probe(&cuda_dev(0), &cuda_dev(1));
        assert!(!cap.direct_supported);
        assert!(cap.reason.is_some());
    }

    #[test]
    fn test_detect_p2p_same_backend() {
        let engine = TransferEngine::with_probe(TestProbe);
        let cap = engine.detect(&cuda_dev(0), &cuda_dev(1));
        assert!(cap.direct_supported);
    }

    #[test]
    fn test_detect_p2p_cross_backend() {
        let engine = TransferEngine::with_probe(TestProbe);
        let cap = engine.detect(&cuda_dev(0), &ocl_dev(1));
        assert!(!cap.direct_supported);
    }

    #[test]
    fn test_transfer_uses_direct_path() {
        let engine = TransferEngine::with_probe(TestProbe);
        let data = vec![0u8; 1024];
        let report = engine.transfer(&cuda_dev(0), &cuda_dev(1), &data).expect("transfer ok");
        assert_eq!(report.path, TransferPath::Direct);
        assert_eq!(report.bytes_transferred, 1024);
    }

    #[test]
    fn test_transfer_falls_back_to_host_staged() {
        let engine = TransferEngine::new(); // FallbackProbe → always host-staged
        let data = vec![0u8; 2048];
        let report = engine.transfer(&cuda_dev(0), &cuda_dev(1), &data).expect("transfer ok");
        assert_eq!(report.path, TransferPath::HostStaged);
        assert_eq!(report.bytes_transferred, 2048);
    }

    #[test]
    fn test_bandwidth_calculation() {
        let report = BandwidthReport::new(
            1_000_000_000, // 1 GB
            Duration::from_secs(1),
            TransferPath::Direct,
            cuda_dev(0),
            cuda_dev(1),
        );
        // 1 GB in 1 s = 8 Gbps
        assert!((report.bandwidth_gbps - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_bandwidth_zero_duration() {
        let report = BandwidthReport::new(
            1024,
            Duration::ZERO,
            TransferPath::HostStaged,
            ocl_dev(0),
            ocl_dev(1),
        );
        assert_eq!(report.bandwidth_gbps, 0.0);
    }

    #[test]
    fn test_device_id_display() {
        let dev = cuda_dev(3);
        assert_eq!(format!("{}", dev), "Cuda:3");
        let dev = ocl_dev(0);
        assert_eq!(format!("{}", dev), "OpenCl:0");
    }
}
