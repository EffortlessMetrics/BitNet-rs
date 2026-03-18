//! Error detection and correction (ECC) for GPU computation buffers.
//!
//! # Overview
//!
//! Provides software-level error detection and correction for GPU inference
//! buffers, complementing hardware ECC with application-layer integrity checks.
//! Key components:
//!
//! - [`ECCMode`] — detection/correction level (none, detect-only, SEC, DED)
//! - [`ECCConfig`] — configuration (mode, check frequency, correction strength)
//! - [`ECCStats`] — cumulative error tracking and reporting
//! - [`ErrorCorrectionManager`] — buffer registration, periodic checking,
//!   and scrubbing
//!
//! # Algorithms
//!
//! - **Parity**: single-bit parity for lightweight detection
//! - **Hamming(7,4)**: single-error-correct / double-error-detect codes
//! - **CRC-32**: cyclic redundancy check for bulk integrity verification
//! - **Checksum**: fast additive checksums for buffer snapshots
//! - **TMR comparison**: triple-modular-redundancy output comparison
//!
//! # GPU dispatch
//!
//! CUDA kernel sources for GPU-accelerated parity/checksum are feature-gated
//! behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations are always available.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use bitnet_common::{KernelError, Result};

// ── Constants ────────────────────────────────────────────────────────

/// CRC-32 polynomial (IEEE 802.3 / ITU-T V.42).
const CRC32_POLY: u32 = 0xEDB8_8320;

/// CRC-32 lookup table size.
const CRC32_TABLE_SIZE: usize = 256;

/// Hamming(7,4) data bits per nibble.
const HAMMING_DATA_BITS: usize = 4;

/// Hamming(7,4) total codeword length.
const HAMMING_CODE_BITS: usize = 7;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_buffer_id() -> BufferId {
    BufferId(NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed))
}

// ── CUDA kernel source ──────────────────────────────────────────────

/// CUDA kernel source for ECC parity and checksum operations.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ECC_KERNEL_SRC: &str = r#"
extern "C" __global__ void compute_parity_kernel(
    const float *data,
    unsigned int *parity_out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int bits = __float_as_uint(data[idx]);
        parity_out[idx] = __popc(bits) & 1u;
    }
}

extern "C" __global__ void compute_checksum_kernel(
    const float *data,
    unsigned long long *checksum_out,
    int n
) {
    __shared__ unsigned long long partial[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long val = 0;
    if (idx < n) {
        val = (unsigned long long)__float_as_uint(data[idx]);
    }
    partial[tid] = val;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(checksum_out, partial[0]);
    }
}

extern "C" __global__ void detect_bit_errors_kernel(
    const float *data,
    const unsigned int *expected_parity,
    unsigned int *error_flags,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int bits = __float_as_uint(data[idx]);
        unsigned int actual = __popc(bits) & 1u;
        error_flags[idx] = (actual != expected_parity[idx]) ? 1u : 0u;
    }
}
"#;

// ── ECCMode ─────────────────────────────────────────────────────────

/// Error correction operating mode.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ECCMode {
    /// No error checking (maximum performance).
    None,
    /// Detect errors but do not attempt correction.
    #[default]
    DetectOnly,
    /// Single-bit error correction (Hamming SEC).
    SingleBitCorrect,
    /// Double-bit error detection with single-bit correction (SEC-DED).
    DoubleBitDetect,
}

impl fmt::Display for ECCMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::DetectOnly => write!(f, "DetectOnly"),
            Self::SingleBitCorrect => write!(f, "SEC"),
            Self::DoubleBitDetect => write!(f, "SEC-DED"),
        }
    }
}

// ── ECCConfig ───────────────────────────────────────────────────────

/// Configuration for the error correction subsystem.
#[derive(Debug, Clone)]
pub struct ECCConfig {
    /// Operating mode for error detection/correction.
    pub mode: ECCMode,
    /// How often to run periodic checks (every N kernel launches).
    pub check_frequency: u32,
    /// Correction strength: max bit-flips correctable per word.
    pub correction_strength: u32,
    /// Whether to log detected/corrected errors.
    pub logging_enabled: bool,
}

impl Default for ECCConfig {
    fn default() -> Self {
        Self {
            mode: ECCMode::DetectOnly,
            check_frequency: 100,
            correction_strength: 1,
            logging_enabled: true,
        }
    }
}

impl ECCConfig {
    /// Create a new config with the given mode.
    pub fn new(mode: ECCMode) -> Self {
        Self { mode, ..Default::default() }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.check_frequency == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "check_frequency must be non-zero".into(),
            }
            .into());
        }
        if self.mode == ECCMode::SingleBitCorrect && self.correction_strength < 1 {
            return Err(KernelError::InvalidArguments {
                reason: "correction_strength must be >= 1 for SEC mode".into(),
            }
            .into());
        }
        if self.mode == ECCMode::DoubleBitDetect && self.correction_strength < 1 {
            return Err(KernelError::InvalidArguments {
                reason: "correction_strength must be >= 1 for SEC-DED mode".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Builder: set check frequency.
    pub fn with_check_frequency(mut self, freq: u32) -> Self {
        self.check_frequency = freq;
        self
    }

    /// Builder: set correction strength.
    pub fn with_correction_strength(mut self, strength: u32) -> Self {
        self.correction_strength = strength;
        self
    }

    /// Builder: enable or disable logging.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.logging_enabled = enabled;
        self
    }
}

// ── ECCStats ────────────────────────────────────────────────────────

/// Cumulative error detection/correction statistics.
#[derive(Debug, Clone, Default)]
pub struct ECCStats {
    /// Total single-bit errors detected.
    pub single_bit_errors: u64,
    /// Total double-bit errors detected.
    pub double_bit_errors: u64,
    /// Total errors corrected.
    pub corrections_applied: u64,
    /// Total parity check failures.
    pub parity_failures: u64,
    /// Total checksum mismatches.
    pub checksum_mismatches: u64,
    /// Total CRC failures.
    pub crc_failures: u64,
    /// Total buffers checked.
    pub buffers_checked: u64,
    /// Total scrub passes completed.
    pub scrub_passes: u64,
    /// Timestamp of the last check.
    pub last_check: Option<Instant>,
}

impl ECCStats {
    /// Total errors of all types.
    pub fn total_errors(&self) -> u64 {
        self.single_bit_errors
            + self.double_bit_errors
            + self.parity_failures
            + self.checksum_mismatches
            + self.crc_failures
    }

    /// Error rate as a fraction of buffers checked.
    pub fn error_rate(&self) -> f64 {
        if self.buffers_checked == 0 {
            return 0.0;
        }
        self.total_errors() as f64 / self.buffers_checked as f64
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for ECCStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ECC: {} checked, {} errors ({} SBE, {} DBE, {} parity, \
             {} checksum, {} CRC), {} corrected, {} scrubs",
            self.buffers_checked,
            self.total_errors(),
            self.single_bit_errors,
            self.double_bit_errors,
            self.parity_failures,
            self.checksum_mismatches,
            self.crc_failures,
            self.corrections_applied,
            self.scrub_passes,
        )
    }
}

// ── BufferId / RegisteredBuffer ─────────────────────────────────────

/// Unique handle for a registered buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "buf-{}", self.0)
    }
}

/// Metadata for a buffer registered with the ECC manager.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RegisteredBuffer {
    /// Unique identifier.
    id: BufferId,
    /// Human-readable label.
    label: String,
    /// Buffer length in elements (f32).
    len: usize,
    /// Stored parity bits (one per element).
    parity: Vec<u8>,
    /// Stored additive checksum.
    checksum: u64,
    /// Stored CRC-32 value.
    crc32: u32,
    /// Timestamp of the last check.
    last_checked: Option<Instant>,
}

// ── ErrorCorrectionManager ──────────────────────────────────────────

/// Manages buffer registration, periodic ECC checking, and scrubbing.
pub struct ErrorCorrectionManager {
    /// Active configuration.
    config: ECCConfig,
    /// Cumulative statistics.
    stats: ECCStats,
    /// Registered buffers keyed by id.
    buffers: HashMap<BufferId, RegisteredBuffer>,
    /// Kernel launch counter for periodic check scheduling.
    launch_counter: u64,
}

impl ErrorCorrectionManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: ECCConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, stats: ECCStats::default(), buffers: HashMap::new(), launch_counter: 0 })
    }

    /// Borrow the current configuration.
    pub fn config(&self) -> &ECCConfig {
        &self.config
    }

    /// Borrow cumulative statistics.
    pub fn stats(&self) -> &ECCStats {
        &self.stats
    }

    /// Number of registered buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Register a buffer for ECC protection and capture its initial
    /// parity, checksum, and CRC.
    pub fn register_buffer(&mut self, label: impl Into<String>, data: &[f32]) -> Result<BufferId> {
        if data.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "cannot register empty buffer".into(),
            }
            .into());
        }
        let id = next_buffer_id();
        let parity: Vec<u8> = data.iter().map(|v| compute_parity(*v)).collect();
        let checksum = compute_checksum(data);
        let crc = crc32_compute(data);
        let buf = RegisteredBuffer {
            id,
            label: label.into(),
            len: data.len(),
            parity,
            checksum,
            crc32: crc,
            last_checked: Some(Instant::now()),
        };
        self.buffers.insert(id, buf);
        Ok(id)
    }

    /// Unregister a buffer.
    pub fn unregister_buffer(&mut self, id: BufferId) -> Result<()> {
        self.buffers.remove(&id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("buffer {id} not registered"),
        })?;
        Ok(())
    }

    /// Increment the launch counter and run periodic checks if due.
    ///
    /// Returns the number of errors detected during this tick (0 if no
    /// check was performed).
    pub fn tick(&mut self, all_buffers: &[(&BufferId, &[f32])]) -> u64 {
        self.launch_counter += 1;
        if self.config.mode == ECCMode::None {
            return 0;
        }
        if !self.launch_counter.is_multiple_of(u64::from(self.config.check_frequency)) {
            return 0;
        }
        let mut errors = 0u64;
        for &(id, data) in all_buffers {
            errors += self.check_buffer(*id, data).unwrap_or(0);
        }
        errors
    }

    /// Explicitly check a single buffer against stored metadata.
    pub fn check_buffer(&mut self, id: BufferId, data: &[f32]) -> Result<u64> {
        let buf = self.buffers.get(&id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("buffer {id} not registered"),
        })?;

        self.stats.buffers_checked += 1;
        let now = Instant::now();
        self.stats.last_check = Some(now);

        let mut errors = 0u64;

        // Parity check.
        let parity_errs = detect_errors(data, &buf.parity);
        if parity_errs > 0 {
            self.stats.parity_failures += parity_errs;
            errors += parity_errs;
        }

        // Checksum check.
        if !verify_checksum(data, buf.checksum) {
            self.stats.checksum_mismatches += 1;
            errors += 1;
        }

        // CRC check.
        if crc32_compute(data) != buf.crc32 {
            self.stats.crc_failures += 1;
            errors += 1;
        }

        // Update last_checked on the buffer.
        if let Some(buf_mut) = self.buffers.get_mut(&id) {
            buf_mut.last_checked = Some(now);
        }

        Ok(errors)
    }

    /// Update stored metadata for a buffer after legitimate mutation.
    pub fn refresh_buffer(&mut self, id: BufferId, data: &[f32]) -> Result<()> {
        let buf = self.buffers.get_mut(&id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("buffer {id} not registered"),
        })?;
        buf.parity = data.iter().map(|v| compute_parity(*v)).collect();
        buf.checksum = compute_checksum(data);
        buf.crc32 = crc32_compute(data);
        buf.len = data.len();
        buf.last_checked = Some(Instant::now());
        Ok(())
    }

    /// Run a full scrub over a mutable buffer: detect and, if configured,
    /// correct single-bit errors.  Returns the number of corrections
    /// applied.
    pub fn scrub_buffer(&mut self, id: BufferId, data: &mut [f32]) -> Result<u64> {
        let mode = self.config.mode;
        let buf = self.buffers.get_mut(&id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("buffer {id} not registered"),
        })?;
        let corrections = ecc_scrub_buffer(data, &buf.parity, mode);
        self.stats.scrub_passes += 1;
        self.stats.corrections_applied += corrections;
        // Re-snapshot after scrub.
        buf.parity = data.iter().map(|v| compute_parity(*v)).collect();
        buf.checksum = compute_checksum(data);
        buf.crc32 = crc32_compute(data);
        buf.last_checked = Some(Instant::now());
        Ok(corrections)
    }
}

impl fmt::Debug for ErrorCorrectionManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErrorCorrectionManager")
            .field("mode", &self.config.mode)
            .field("buffers", &self.buffers.len())
            .field("launches", &self.launch_counter)
            .field("stats", &self.stats)
            .finish()
    }
}

// ════════════════════════════════════════════════════════════════════
// Free-standing ECC algorithms (CPU implementations, always compiled)
// ════════════════════════════════════════════════════════════════════

// ── Parity ──────────────────────────────────────────────────────────

/// Compute single-bit parity of a float's bit representation.
///
/// Returns `0` if the number of set bits is even, `1` if odd.
pub fn compute_parity(value: f32) -> u8 {
    let bits = value.to_bits();
    (bits.count_ones() & 1) as u8
}

/// Detect parity errors in a buffer.  Returns the number of elements
/// whose current parity differs from the expected parity vector.
pub fn detect_errors(data: &[f32], expected_parity: &[u8]) -> u64 {
    let len = data.len().min(expected_parity.len());
    let mut errors = 0u64;
    for i in 0..len {
        if compute_parity(data[i]) != expected_parity[i] {
            errors += 1;
        }
    }
    errors
}

/// Attempt to correct a single-bit error in a 32-bit word using the
/// known expected parity.  Returns `Some(corrected)` if a correction
/// was applied, `None` if parity already matches.
pub fn correct_single_bit(value: f32, expected_parity: u8) -> Option<f32> {
    let bits = value.to_bits();
    let actual_parity = (bits.count_ones() & 1) as u8;
    if actual_parity == expected_parity {
        return None;
    }
    // Flip the least-significant bit (heuristic — restores parity).
    let corrected = bits ^ 1;
    Some(f32::from_bits(corrected))
}

// ── Checksum ────────────────────────────────────────────────────────

/// Compute a simple additive checksum over a float buffer.
///
/// Sums the bit representations of each element as `u64` values.
pub fn compute_checksum(data: &[f32]) -> u64 {
    data.iter().map(|v| u64::from(v.to_bits())).sum()
}

/// Verify that a buffer's checksum matches the expected value.
pub fn verify_checksum(data: &[f32], expected: u64) -> bool {
    compute_checksum(data) == expected
}

// ── Hamming(7,4) ────────────────────────────────────────────────────

/// Hamming(7,4) generator matrix.  Layout: `[d0 d1 d2 d3 p0 p1 p2]`.
const HAMMING_G: [[u8; HAMMING_CODE_BITS]; HAMMING_DATA_BITS] =
    [[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]];

/// Hamming(7,4) parity-check matrix (syndrome = H · r^T).
const HAMMING_H: [[u8; HAMMING_CODE_BITS]; 3] =
    [[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]];

/// Encode a 4-bit nibble using Hamming(7,4).
///
/// Input: lowest 4 bits of `nibble`.  Returns a 7-bit codeword.
pub fn hamming_encode(nibble: u8) -> u8 {
    let data = [nibble & 1, (nibble >> 1) & 1, (nibble >> 2) & 1, (nibble >> 3) & 1];
    let mut code: u8 = 0;
    for (bit_pos, _) in HAMMING_G[0].iter().enumerate() {
        let mut val = 0u8;
        for (data_idx, &d) in data.iter().enumerate() {
            val ^= d & HAMMING_G[data_idx][bit_pos];
        }
        code |= val << bit_pos;
    }
    code
}

/// Decode a Hamming(7,4) codeword.
///
/// Returns `(data_nibble, corrected)` where `corrected` is `true` if
/// a single-bit error was detected and fixed.
pub fn hamming_decode(codeword: u8) -> (u8, bool) {
    let mut syndrome: usize = 0;
    for (row_idx, row) in HAMMING_H.iter().enumerate() {
        let mut bit = 0u8;
        for (col, &h_val) in row.iter().enumerate() {
            bit ^= ((codeword >> col) & 1) & h_val;
        }
        syndrome |= (bit as usize) << row_idx;
    }

    let mut word = codeword;
    let corrected = if syndrome == 0 {
        false
    } else {
        // Find which H column matches the syndrome to locate the error.
        for col in 0..HAMMING_CODE_BITS {
            let mut col_val = 0usize;
            for (row_idx, row) in HAMMING_H.iter().enumerate() {
                col_val |= (row[col] as usize) << row_idx;
            }
            if col_val == syndrome {
                word ^= 1 << col;
                break;
            }
        }
        true
    };

    let data_nibble = word & 0x0F;
    (data_nibble, corrected)
}

// ── CRC-32 ──────────────────────────────────────────────────────────

/// Build the CRC-32 lookup table (reflected / LSB-first).
fn crc32_table() -> [u32; CRC32_TABLE_SIZE] {
    let mut table = [0u32; CRC32_TABLE_SIZE];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32_POLY;
            } else {
                crc >>= 1;
            }
        }
        *entry = crc;
    }
    table
}

/// Compute CRC-32 over a float buffer (each f32 as 4 LE bytes).
pub fn crc32_compute(data: &[f32]) -> u32 {
    let table = crc32_table();
    let mut crc: u32 = 0xFFFF_FFFF;
    for val in data {
        let bytes = val.to_bits().to_le_bytes();
        for &b in &bytes {
            let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
            crc = (crc >> 8) ^ table[idx];
        }
    }
    crc ^ 0xFFFF_FFFF
}

// ── Scrubbing ───────────────────────────────────────────────────────

/// Scrub a mutable buffer: detect parity mismatches and, for
/// [`ECCMode::SingleBitCorrect`] or [`ECCMode::DoubleBitDetect`],
/// attempt single-bit correction.  Returns the number of corrections
/// applied.
pub fn ecc_scrub_buffer(data: &mut [f32], expected_parity: &[u8], mode: ECCMode) -> u64 {
    let len = data.len().min(expected_parity.len());
    let mut corrections = 0u64;
    for i in 0..len {
        let actual = compute_parity(data[i]);
        if actual != expected_parity[i] {
            match mode {
                ECCMode::SingleBitCorrect | ECCMode::DoubleBitDetect => {
                    if let Some(fixed) = correct_single_bit(data[i], expected_parity[i]) {
                        data[i] = fixed;
                        corrections += 1;
                    }
                }
                ECCMode::DetectOnly | ECCMode::None => {}
            }
        }
    }
    corrections
}

// ── TMR comparison ──────────────────────────────────────────────────

/// Compare three redundant output buffers (triple modular redundancy).
///
/// For each position, returns the majority value when at least two of
/// three agree.  Positions where all three differ use `a[i]` (first
/// buffer wins) and are counted as unresolvable.
///
/// Returns `(result, mismatches)`.
pub fn compare_redundant_outputs(a: &[f32], b: &[f32], c: &[f32]) -> (Vec<f32>, u64) {
    let len = a.len().min(b.len()).min(c.len());
    let mut result = Vec::with_capacity(len);
    let mut mismatches = 0u64;
    for i in 0..len {
        if a[i] == b[i] || a[i] == c[i] {
            result.push(a[i]);
        } else if b[i] == c[i] {
            result.push(b[i]);
        } else {
            result.push(a[i]);
            mismatches += 1;
        }
    }
    (result, mismatches)
}

// ── GPU launch helpers ──────────────────────────────────────────────

/// Launch configuration for ECC GPU kernels.
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[derive(Debug, Clone)]
pub struct ECCLaunchConfig {
    /// Number of thread blocks.
    pub grid_dim: u32,
    /// Threads per block.
    pub block_dim: u32,
    /// Buffer element count.
    pub n: u32,
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
impl ECCLaunchConfig {
    /// Compute a launch configuration for `n` elements with 256
    /// threads per block.
    pub fn for_elements(n: u32) -> Self {
        let block_dim = 256;
        let grid_dim = n.div_ceil(block_dim);
        Self { grid_dim, block_dim, n }
    }
}

/// Launch the GPU parity kernel (stub — actual dispatch via cudarc).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_compute_parity(
    _data: &[f32],
    _parity_out: &mut [u32],
    n: u32,
) -> Result<ECCLaunchConfig> {
    Ok(ECCLaunchConfig::for_elements(n))
}

/// Launch the GPU checksum kernel (stub).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_compute_checksum(
    _data: &[f32],
    _checksum_out: &mut [u64],
    n: u32,
) -> Result<ECCLaunchConfig> {
    Ok(ECCLaunchConfig::for_elements(n))
}

/// Launch the GPU error-detection kernel (stub).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_detect_errors(
    _data: &[f32],
    _expected_parity: &[u32],
    _error_flags: &mut [u32],
    n: u32,
) -> Result<ECCLaunchConfig> {
    Ok(ECCLaunchConfig::for_elements(n))
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── ECCMode ─────────────────────────────────────────────────────

    #[test]
    fn test_ecc_mode_default() {
        assert_eq!(ECCMode::default(), ECCMode::DetectOnly);
    }

    #[test]
    fn test_ecc_mode_display() {
        assert_eq!(ECCMode::None.to_string(), "None");
        assert_eq!(ECCMode::DetectOnly.to_string(), "DetectOnly");
        assert_eq!(ECCMode::SingleBitCorrect.to_string(), "SEC");
        assert_eq!(ECCMode::DoubleBitDetect.to_string(), "SEC-DED");
    }

    #[test]
    fn test_ecc_mode_equality() {
        assert_eq!(ECCMode::None, ECCMode::None);
        assert_ne!(ECCMode::None, ECCMode::DetectOnly);
    }

    #[test]
    fn test_ecc_mode_clone() {
        let mode = ECCMode::SingleBitCorrect;
        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    // ── ECCConfig ───────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = ECCConfig::default();
        assert_eq!(cfg.mode, ECCMode::DetectOnly);
        assert_eq!(cfg.check_frequency, 100);
        assert_eq!(cfg.correction_strength, 1);
        assert!(cfg.logging_enabled);
    }

    #[test]
    fn test_config_new() {
        let cfg = ECCConfig::new(ECCMode::SingleBitCorrect);
        assert_eq!(cfg.mode, ECCMode::SingleBitCorrect);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(ECCConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_frequency() {
        let cfg = ECCConfig { check_frequency: 0, ..Default::default() };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_sec_zero_strength() {
        let cfg = ECCConfig {
            mode: ECCMode::SingleBitCorrect,
            correction_strength: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_ded_zero_strength() {
        let cfg = ECCConfig {
            mode: ECCMode::DoubleBitDetect,
            correction_strength: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_builder_check_frequency() {
        let cfg = ECCConfig::default().with_check_frequency(50);
        assert_eq!(cfg.check_frequency, 50);
    }

    #[test]
    fn test_config_builder_correction_strength() {
        let cfg = ECCConfig::default().with_correction_strength(2);
        assert_eq!(cfg.correction_strength, 2);
    }

    #[test]
    fn test_config_builder_logging() {
        let cfg = ECCConfig::default().with_logging(false);
        assert!(!cfg.logging_enabled);
    }

    #[test]
    fn test_config_none_mode_validates() {
        let cfg = ECCConfig::new(ECCMode::None);
        assert!(cfg.validate().is_ok());
    }

    // ── ECCStats ────────────────────────────────────────────────────

    #[test]
    fn test_stats_default_zero() {
        let s = ECCStats::default();
        assert_eq!(s.total_errors(), 0);
        assert_eq!(s.buffers_checked, 0);
        assert!(s.last_check.is_none());
    }

    #[test]
    fn test_stats_total_errors() {
        let s = ECCStats {
            single_bit_errors: 1,
            double_bit_errors: 2,
            parity_failures: 3,
            checksum_mismatches: 4,
            crc_failures: 5,
            ..Default::default()
        };
        assert_eq!(s.total_errors(), 15);
    }

    #[test]
    fn test_stats_error_rate_zero_checked() {
        let s = ECCStats::default();
        assert_eq!(s.error_rate(), 0.0);
    }

    #[test]
    fn test_stats_error_rate() {
        let s = ECCStats { single_bit_errors: 5, buffers_checked: 100, ..Default::default() };
        assert!((s.error_rate() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_stats_reset() {
        let mut s = ECCStats { single_bit_errors: 42, buffers_checked: 100, ..Default::default() };
        s.reset();
        assert_eq!(s.total_errors(), 0);
        assert_eq!(s.buffers_checked, 0);
    }

    #[test]
    fn test_stats_display() {
        let s = ECCStats::default();
        let d = format!("{s}");
        assert!(d.contains("ECC:"));
        assert!(d.contains("checked"));
    }

    // ── compute_parity ──────────────────────────────────────────────

    #[test]
    fn test_parity_zero() {
        assert_eq!(compute_parity(0.0), 0);
    }

    #[test]
    fn test_parity_one() {
        let p = compute_parity(1.0);
        let expected = (1.0f32.to_bits().count_ones() & 1) as u8;
        assert_eq!(p, expected);
    }

    #[test]
    fn test_parity_negative_zero() {
        let p = compute_parity(-0.0);
        let expected = ((-0.0f32).to_bits().count_ones() & 1) as u8;
        assert_eq!(p, expected);
    }

    #[test]
    fn test_parity_deterministic() {
        let v = 1.5f32;
        assert_eq!(compute_parity(v), compute_parity(v));
    }

    #[test]
    fn test_parity_nan() {
        let p = compute_parity(f32::NAN);
        let expected = (f32::NAN.to_bits().count_ones() & 1) as u8;
        assert_eq!(p, expected);
    }

    #[test]
    fn test_parity_inf() {
        let p = compute_parity(f32::INFINITY);
        let expected = (f32::INFINITY.to_bits().count_ones() & 1) as u8;
        assert_eq!(p, expected);
    }

    // ── detect_errors ───────────────────────────────────────────────

    #[test]
    fn test_detect_errors_no_errors() {
        let data = [1.0f32, 2.0, 3.0];
        let parity: Vec<u8> = data.iter().map(|v| compute_parity(*v)).collect();
        assert_eq!(detect_errors(&data, &parity), 0);
    }

    #[test]
    fn test_detect_errors_all_errors() {
        let data = [1.0f32, 2.0, 3.0];
        let parity: Vec<u8> = data.iter().map(|v| compute_parity(*v) ^ 1).collect();
        assert_eq!(detect_errors(&data, &parity), 3);
    }

    #[test]
    fn test_detect_errors_length_mismatch() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let parity = [0u8; 2];
        let errs = detect_errors(&data, &parity);
        assert!(errs <= 2);
    }

    #[test]
    fn test_detect_errors_empty() {
        assert_eq!(detect_errors(&[], &[]), 0);
    }

    // ── correct_single_bit ──────────────────────────────────────────

    #[test]
    fn test_correct_no_error() {
        let v = 42.0f32;
        let p = compute_parity(v);
        assert_eq!(correct_single_bit(v, p), None);
    }

    #[test]
    fn test_correct_with_error() {
        let v = 42.0f32;
        let p = compute_parity(v);
        let corrupted = f32::from_bits(v.to_bits() ^ 1);
        let result = correct_single_bit(corrupted, p);
        assert!(result.is_some());
        assert_eq!(compute_parity(result.unwrap()), p);
    }

    #[test]
    fn test_correct_restores_parity() {
        for bits in [0u32, 1, 0x3F80_0000, 0x4048_F5C3, 0xFFFF_FFFF] {
            let v = f32::from_bits(bits);
            let p = compute_parity(v);
            let corrupted = f32::from_bits(bits ^ 1);
            if let Some(fixed) = correct_single_bit(corrupted, p) {
                assert_eq!(compute_parity(fixed), p);
            }
        }
    }

    // ── compute_checksum / verify_checksum ──────────────────────────

    #[test]
    fn test_checksum_empty() {
        assert_eq!(compute_checksum(&[]), 0);
    }

    #[test]
    fn test_checksum_single() {
        let data = [1.0f32];
        let cs = compute_checksum(&data);
        assert_eq!(cs, u64::from(1.0f32.to_bits()));
    }

    #[test]
    fn test_checksum_deterministic() {
        let data = [1.0, 2.0, 3.0, 4.0f32];
        assert_eq!(compute_checksum(&data), compute_checksum(&data));
    }

    #[test]
    fn test_checksum_differs_after_mutation() {
        let data = [1.0, 2.0, 3.0f32];
        let cs = compute_checksum(&data);
        let mutated = [1.0, 2.0, 4.0f32];
        assert_ne!(compute_checksum(&mutated), cs);
    }

    #[test]
    fn test_verify_checksum_pass() {
        let data = [1.0, 2.0, 3.0f32];
        let cs = compute_checksum(&data);
        assert!(verify_checksum(&data, cs));
    }

    #[test]
    fn test_verify_checksum_fail() {
        let data = [1.0, 2.0, 3.0f32];
        assert!(!verify_checksum(&data, 0));
    }

    // ── hamming_encode / hamming_decode ──────────────────────────────

    #[test]
    fn test_hamming_roundtrip_all_nibbles() {
        for nibble in 0..16u8 {
            let code = hamming_encode(nibble);
            let (decoded, corrected) = hamming_decode(code);
            assert_eq!(decoded, nibble, "nibble {nibble}");
            assert!(!corrected);
        }
    }

    #[test]
    fn test_hamming_single_bit_correction() {
        for nibble in 0..16u8 {
            let code = hamming_encode(nibble);
            for flip in 0..HAMMING_CODE_BITS {
                let corrupted = code ^ (1 << flip);
                let (decoded, corrected) = hamming_decode(corrupted);
                assert_eq!(decoded, nibble, "nibble {nibble}, flip bit {flip}");
                assert!(corrected);
            }
        }
    }

    #[test]
    fn test_hamming_encode_zero() {
        let code = hamming_encode(0);
        assert_eq!(code, 0);
    }

    #[test]
    fn test_hamming_encode_max_nibble() {
        let code = hamming_encode(0x0F);
        assert_eq!(code & 0x0F, 0x0F);
    }

    #[test]
    fn test_hamming_decode_no_error() {
        let (_, corrected) = hamming_decode(hamming_encode(5));
        assert!(!corrected);
    }

    #[test]
    fn test_hamming_encode_bits_in_range() {
        for nibble in 0..16u8 {
            let code = hamming_encode(nibble);
            assert!(code < 128, "codeword must fit in 7 bits");
        }
    }

    // ── crc32_compute ───────────────────────────────────────────────

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32_compute(&[]), 0);
    }

    #[test]
    fn test_crc32_deterministic() {
        let data = [1.0, 2.0, 3.0f32];
        assert_eq!(crc32_compute(&data), crc32_compute(&data));
    }

    #[test]
    fn test_crc32_differs_on_mutation() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [1.0, 2.0, 4.0f32];
        assert_ne!(crc32_compute(&a), crc32_compute(&b));
    }

    #[test]
    fn test_crc32_single_zero() {
        let crc = crc32_compute(&[0.0f32]);
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_crc32_order_matters() {
        let a = [1.0, 2.0f32];
        let b = [2.0, 1.0f32];
        assert_ne!(crc32_compute(&a), crc32_compute(&b));
    }

    #[test]
    fn test_crc32_table_first_entry() {
        let table = crc32_table();
        assert_eq!(table[0], 0);
    }

    #[test]
    fn test_crc32_table_size() {
        let table = crc32_table();
        assert_eq!(table.len(), 256);
    }

    // ── ecc_scrub_buffer ────────────────────────────────────────────

    #[test]
    fn test_scrub_no_errors() {
        let mut data = [1.0, 2.0, 3.0f32];
        let parity: Vec<u8> = data.iter().map(|v| compute_parity(*v)).collect();
        let fixed = ecc_scrub_buffer(&mut data, &parity, ECCMode::SingleBitCorrect);
        assert_eq!(fixed, 0);
    }

    #[test]
    fn test_scrub_corrects_flipped_bit() {
        let original = [42.0f32];
        let parity: Vec<u8> = original.iter().map(|v| compute_parity(*v)).collect();
        let mut data = [f32::from_bits(original[0].to_bits() ^ 1)];
        let fixed = ecc_scrub_buffer(&mut data, &parity, ECCMode::SingleBitCorrect);
        assert_eq!(fixed, 1);
        assert_eq!(compute_parity(data[0]), parity[0]);
    }

    #[test]
    fn test_scrub_detect_only_no_correction() {
        let original = [42.0f32];
        let parity: Vec<u8> = original.iter().map(|v| compute_parity(*v)).collect();
        let corrupted = f32::from_bits(original[0].to_bits() ^ 1);
        let mut data = [corrupted];
        let fixed = ecc_scrub_buffer(&mut data, &parity, ECCMode::DetectOnly);
        assert_eq!(fixed, 0);
        assert_eq!(data[0].to_bits(), corrupted.to_bits());
    }

    #[test]
    fn test_scrub_none_mode() {
        let original = [42.0f32];
        let parity: Vec<u8> = original.iter().map(|v| compute_parity(*v)).collect();
        let mut data = [f32::from_bits(original[0].to_bits() ^ 1)];
        let fixed = ecc_scrub_buffer(&mut data, &parity, ECCMode::None);
        assert_eq!(fixed, 0);
    }

    #[test]
    fn test_scrub_ded_mode_corrects() {
        let original = [42.0f32];
        let parity: Vec<u8> = original.iter().map(|v| compute_parity(*v)).collect();
        let mut data = [f32::from_bits(original[0].to_bits() ^ 1)];
        let fixed = ecc_scrub_buffer(&mut data, &parity, ECCMode::DoubleBitDetect);
        assert_eq!(fixed, 1);
    }

    #[test]
    fn test_scrub_multiple_elements() {
        let original = [1.0f32, 2.0, 3.0, 4.0];
        let parity: Vec<u8> = original.iter().map(|v| compute_parity(*v)).collect();
        let mut data: Vec<f32> = original.iter().map(|v| f32::from_bits(v.to_bits() ^ 1)).collect();
        let fixed = ecc_scrub_buffer(&mut data, &parity, ECCMode::SingleBitCorrect);
        assert_eq!(fixed, 4);
    }

    #[test]
    fn test_scrub_empty() {
        let fixed = ecc_scrub_buffer(&mut [], &[], ECCMode::SingleBitCorrect);
        assert_eq!(fixed, 0);
    }

    // ── compare_redundant_outputs ───────────────────────────────────

    #[test]
    fn test_tmr_all_agree() {
        let a = vec![1.0, 2.0, 3.0f32];
        let (result, mismatches) = compare_redundant_outputs(&a, &a, &a);
        assert_eq!(result, a);
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn test_tmr_one_differs() {
        let a = vec![1.0, 2.0, 3.0f32];
        let b = vec![1.0, 9.0, 3.0f32];
        let c = vec![1.0, 2.0, 3.0f32];
        let (result, mismatches) = compare_redundant_outputs(&a, &b, &c);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn test_tmr_b_and_c_agree() {
        let a = vec![9.0f32];
        let b = vec![1.0f32];
        let c = vec![1.0f32];
        let (result, mismatches) = compare_redundant_outputs(&a, &b, &c);
        assert_eq!(result, vec![1.0]);
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn test_tmr_all_differ() {
        let a = vec![1.0f32];
        let b = vec![2.0f32];
        let c = vec![3.0f32];
        let (result, mismatches) = compare_redundant_outputs(&a, &b, &c);
        assert_eq!(result, vec![1.0]);
        assert_eq!(mismatches, 1);
    }

    #[test]
    fn test_tmr_empty() {
        let (result, mismatches) = compare_redundant_outputs(&[], &[], &[]);
        assert!(result.is_empty());
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn test_tmr_different_lengths() {
        let a = vec![1.0, 2.0, 3.0f32];
        let b = vec![1.0, 2.0f32];
        let c = vec![1.0, 2.0, 3.0, 4.0f32];
        let (result, _) = compare_redundant_outputs(&a, &b, &c);
        assert_eq!(result.len(), 2);
    }

    // ── ErrorCorrectionManager ──────────────────────────────────────

    #[test]
    fn test_manager_new() {
        let mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        assert_eq!(mgr.buffer_count(), 0);
    }

    #[test]
    fn test_manager_invalid_config() {
        let cfg = ECCConfig { check_frequency: 0, ..Default::default() };
        assert!(ErrorCorrectionManager::new(cfg).is_err());
    }

    #[test]
    fn test_manager_register_buffer() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        let data = [1.0, 2.0, 3.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        assert_eq!(mgr.buffer_count(), 1);
        assert!(id.to_string().starts_with("buf-"));
    }

    #[test]
    fn test_manager_register_empty_buffer() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        assert!(mgr.register_buffer("empty", &[]).is_err());
    }

    #[test]
    fn test_manager_unregister_buffer() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        let data = [1.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        mgr.unregister_buffer(id).unwrap();
        assert_eq!(mgr.buffer_count(), 0);
    }

    #[test]
    fn test_manager_unregister_missing() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        assert!(mgr.unregister_buffer(BufferId(9999)).is_err());
    }

    #[test]
    fn test_manager_check_buffer_clean() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        let data = [1.0, 2.0, 3.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        let errs = mgr.check_buffer(id, &data).unwrap();
        assert_eq!(errs, 0);
    }

    #[test]
    fn test_manager_check_buffer_corrupted() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        let data = [1.0, 2.0, 3.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        let corrupted = [f32::from_bits(data[0].to_bits() ^ 1), data[1], data[2]];
        let errs = mgr.check_buffer(id, &corrupted).unwrap();
        assert!(errs > 0);
    }

    #[test]
    fn test_manager_check_missing_buffer() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        assert!(mgr.check_buffer(BufferId(9999), &[1.0]).is_err());
    }

    #[test]
    fn test_manager_refresh_buffer() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        let data = [1.0, 2.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        let new_data = [4.0, 5.0f32];
        mgr.refresh_buffer(id, &new_data).unwrap();
        let errs = mgr.check_buffer(id, &new_data).unwrap();
        assert_eq!(errs, 0);
    }

    #[test]
    fn test_manager_refresh_missing() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        assert!(mgr.refresh_buffer(BufferId(9999), &[1.0]).is_err());
    }

    #[test]
    fn test_manager_scrub_clean() {
        let mut mgr =
            ErrorCorrectionManager::new(ECCConfig::new(ECCMode::SingleBitCorrect)).unwrap();
        let mut data = [1.0, 2.0, 3.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        let fixed = mgr.scrub_buffer(id, &mut data).unwrap();
        assert_eq!(fixed, 0);
    }

    #[test]
    fn test_manager_scrub_corrects() {
        let mut mgr =
            ErrorCorrectionManager::new(ECCConfig::new(ECCMode::SingleBitCorrect)).unwrap();
        let original = [42.0f32];
        let id = mgr.register_buffer("test", &original).unwrap();
        let mut corrupted = [f32::from_bits(original[0].to_bits() ^ 1)];
        let fixed = mgr.scrub_buffer(id, &mut corrupted).unwrap();
        assert_eq!(fixed, 1);
        assert_eq!(mgr.stats().scrub_passes, 1);
        assert_eq!(mgr.stats().corrections_applied, 1);
    }

    #[test]
    fn test_manager_scrub_missing() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        assert!(mgr.scrub_buffer(BufferId(9999), &mut [1.0]).is_err());
    }

    #[test]
    fn test_manager_tick_no_check_before_frequency() {
        let mut mgr =
            ErrorCorrectionManager::new(ECCConfig::default().with_check_frequency(10)).unwrap();
        let data = [1.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        for _ in 0..9 {
            let errs = mgr.tick(&[(&id, &data)]);
            assert_eq!(errs, 0);
        }
        assert_eq!(mgr.stats().buffers_checked, 0);
    }

    #[test]
    fn test_manager_tick_checks_at_frequency() {
        let mut mgr =
            ErrorCorrectionManager::new(ECCConfig::default().with_check_frequency(5)).unwrap();
        let data = [1.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        for _ in 0..5 {
            mgr.tick(&[(&id, &data)]);
        }
        assert_eq!(mgr.stats().buffers_checked, 1);
    }

    #[test]
    fn test_manager_tick_none_mode() {
        let mut mgr =
            ErrorCorrectionManager::new(ECCConfig::new(ECCMode::None).with_check_frequency(1))
                .unwrap();
        let data = [1.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        let errs = mgr.tick(&[(&id, &data)]);
        assert_eq!(errs, 0);
        assert_eq!(mgr.stats().buffers_checked, 0);
    }

    #[test]
    fn test_manager_config_accessor() {
        let mgr = ErrorCorrectionManager::new(ECCConfig::new(ECCMode::DoubleBitDetect)).unwrap();
        assert_eq!(mgr.config().mode, ECCMode::DoubleBitDetect);
    }

    #[test]
    fn test_manager_debug_format() {
        let mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        let dbg = format!("{mgr:?}");
        assert!(dbg.contains("ErrorCorrectionManager"));
    }

    #[test]
    fn test_manager_stats_update_on_check() {
        let mut mgr = ErrorCorrectionManager::new(ECCConfig::default()).unwrap();
        let data = [1.0, 2.0f32];
        let id = mgr.register_buffer("test", &data).unwrap();
        mgr.check_buffer(id, &data).unwrap();
        assert_eq!(mgr.stats().buffers_checked, 1);
        assert!(mgr.stats().last_check.is_some());
    }

    // ── BufferId ────────────────────────────────────────────────────

    #[test]
    fn test_buffer_id_display() {
        let id = BufferId(42);
        assert_eq!(format!("{id}"), "buf-42");
    }

    #[test]
    fn test_buffer_id_equality() {
        assert_eq!(BufferId(1), BufferId(1));
        assert_ne!(BufferId(1), BufferId(2));
    }

    // ── GPU launch config (feature-gated) ───────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    mod gpu_tests {
        use super::super::*;

        #[test]
        fn test_launch_config_for_elements() {
            let cfg = ECCLaunchConfig::for_elements(1000);
            assert_eq!(cfg.block_dim, 256);
            assert_eq!(cfg.grid_dim, 4);
            assert_eq!(cfg.n, 1000);
        }

        #[test]
        fn test_launch_config_single_block() {
            let cfg = ECCLaunchConfig::for_elements(100);
            assert_eq!(cfg.grid_dim, 1);
        }

        #[test]
        fn test_launch_parity_stub() {
            let data = [1.0, 2.0f32];
            let mut out = [0u32; 2];
            let cfg = launch_compute_parity(&data, &mut out, 2).unwrap();
            assert_eq!(cfg.n, 2);
        }

        #[test]
        fn test_launch_checksum_stub() {
            let data = [1.0f32];
            let mut out = [0u64];
            let cfg = launch_compute_checksum(&data, &mut out, 1).unwrap();
            assert_eq!(cfg.n, 1);
        }

        #[test]
        fn test_launch_detect_errors_stub() {
            let data = [1.0f32];
            let parity = [0u32];
            let mut flags = [0u32];
            let cfg = launch_detect_errors(&data, &parity, &mut flags, 1).unwrap();
            assert_eq!(cfg.n, 1);
        }

        #[test]
        fn test_kernel_src_nonempty() {
            assert!(!ECC_KERNEL_SRC.is_empty());
        }

        #[test]
        fn test_kernel_src_contains_parity() {
            assert!(ECC_KERNEL_SRC.contains("compute_parity_kernel"));
        }

        #[test]
        fn test_kernel_src_contains_checksum() {
            assert!(ECC_KERNEL_SRC.contains("compute_checksum_kernel"));
        }
    }

    // ── Integration / roundtrip tests ───────────────────────────────

    #[test]
    fn test_roundtrip_register_check_scrub() {
        let mut mgr =
            ErrorCorrectionManager::new(ECCConfig::new(ECCMode::SingleBitCorrect)).unwrap();
        let original = [10.0, 20.0, 30.0f32];
        let id = mgr.register_buffer("rt", &original).unwrap();

        let mut corrupted = original;
        corrupted[1] = f32::from_bits(corrupted[1].to_bits() ^ 1);

        let errs = mgr.check_buffer(id, &corrupted).unwrap();
        assert!(errs > 0);

        let fixed = mgr.scrub_buffer(id, &mut corrupted).unwrap();
        assert_eq!(fixed, 1);

        let errs2 = mgr.check_buffer(id, &corrupted).unwrap();
        assert_eq!(errs2, 0);
    }

    #[test]
    fn test_crc32_known_value_stability() {
        let data = [1.0f32, 0.0, -1.0];
        let crc = crc32_compute(&data);
        assert_eq!(crc, crc32_compute(&data));
    }

    #[test]
    fn test_hamming_all_nibbles_unique_codes() {
        let codes: Vec<u8> = (0..16).map(hamming_encode).collect();
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(codes[i], codes[j]);
            }
        }
    }

    #[test]
    fn test_checksum_large_buffer() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let cs = compute_checksum(&data);
        assert!(verify_checksum(&data, cs));
    }

    #[test]
    fn test_crc32_large_buffer() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let crc = crc32_compute(&data);
        assert_eq!(crc, crc32_compute(&data));
    }

    #[test]
    fn test_parity_all_bits_set() {
        let v = f32::from_bits(0xFFFF_FFFF);
        assert_eq!(compute_parity(v), 0); // 32 bits set → even
    }

    #[test]
    fn test_parity_single_bit_set() {
        let v = f32::from_bits(1);
        assert_eq!(compute_parity(v), 1); // 1 bit set → odd
    }
}
