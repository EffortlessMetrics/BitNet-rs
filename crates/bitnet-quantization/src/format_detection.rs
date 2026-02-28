//! Improved quantization format auto-detection heuristics.
//!
//! Determines the most likely quantization format from tensor metadata
//! (element count, byte size, block layout) so callers don't need to
//! specify the format manually. Detection follows the priority order:
//!   QK256 > I2S-BitNet32-F16 > TL1/TL2.

use bitnet_common::QuantizationType;

/// Block-size constants for each format.
const QK256_BLOCK: usize = 256;
const QK256_PACKED_BYTES_PER_BLOCK: usize = 64;
const BITNET32_BLOCK: usize = 32;
const BITNET32_BYTES_PER_BLOCK: usize = 10; // 8 packed + 2 F16 scale

/// Confidence level for a format detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DetectionConfidence {
    /// No match at all.
    None,
    /// Possible match (size compatible but ambiguous).
    Low,
    /// Likely match (size and structure align).
    Medium,
    /// Near-certain (unique structural fingerprint).
    High,
}

/// Result of a single format probe.
#[derive(Debug, Clone)]
pub struct FormatCandidate {
    pub format: QuantizationType,
    pub confidence: DetectionConfidence,
    pub reason: &'static str,
}

/// Detect the quantization format of packed tensor data.
///
/// # Arguments
///
/// * `num_elements` – logical element count of the tensor (e.g. rows × cols).
/// * `data_bytes`   – number of bytes in the packed payload.
///
/// Returns candidates sorted by confidence (highest first). An empty vec
/// means no known format matches.
pub fn detect_format(num_elements: usize, data_bytes: usize) -> Vec<FormatCandidate> {
    let mut candidates = Vec::new();

    // --- QK256 probe ---
    if num_elements > 0 {
        let blocks = num_elements.div_ceil(QK256_BLOCK);
        let expected = blocks * QK256_PACKED_BYTES_PER_BLOCK;
        let diff = data_bytes.abs_diff(expected);
        if diff == 0 {
            candidates.push(FormatCandidate {
                format: QuantizationType::I2S,
                confidence: DetectionConfidence::High,
                reason: "Exact QK256 byte match (256-elem blocks, 64B/block)",
            });
        } else if diff <= 128 {
            candidates.push(FormatCandidate {
                format: QuantizationType::I2S,
                confidence: DetectionConfidence::Medium,
                reason: "QK256 byte match within alignment tolerance (≤128B)",
            });
        }
    }

    // --- I2S-BitNet32-F16 probe ---
    if num_elements > 0 {
        let blocks = num_elements.div_ceil(BITNET32_BLOCK);
        let expected = blocks * BITNET32_BYTES_PER_BLOCK;
        let diff = data_bytes.abs_diff(expected);
        if diff == 0 {
            candidates.push(FormatCandidate {
                format: QuantizationType::I2S,
                confidence: DetectionConfidence::Medium,
                reason: "Exact BitNet32-F16 byte match (32-elem blocks, 10B/block)",
            });
        } else if diff <= 64 {
            candidates.push(FormatCandidate {
                format: QuantizationType::I2S,
                confidence: DetectionConfidence::Low,
                reason: "BitNet32-F16 byte match within tolerance (≤64B)",
            });
        }
    }

    // --- TL1 probe (ARM-style 64-elem blocks) ---
    if num_elements > 0 {
        let blocks_64 = num_elements.div_ceil(64);
        // TL1: 2 bits/elem → 16 bytes packed per 64-elem block + 4-byte scale
        let expected = blocks_64 * 20;
        if data_bytes.abs_diff(expected) <= 32 {
            candidates.push(FormatCandidate {
                format: QuantizationType::TL1,
                confidence: DetectionConfidence::Low,
                reason: "Compatible with TL1 64-elem block layout",
            });
        }
    }

    // --- TL2 probe (x86-style 128-elem blocks) ---
    if num_elements > 0 {
        let blocks_128 = num_elements.div_ceil(128);
        // TL2: 2 bits/elem → 32 bytes packed per 128-elem block + 4-byte scale
        let expected = blocks_128 * 36;
        if data_bytes.abs_diff(expected) <= 32 {
            candidates.push(FormatCandidate {
                format: QuantizationType::TL2,
                confidence: DetectionConfidence::Low,
                reason: "Compatible with TL2 128-elem block layout",
            });
        }
    }

    // Sort highest confidence first, then by format discriminant for stability.
    candidates.sort_by(|a, b| b.confidence.cmp(&a.confidence));
    candidates
}

/// Convenience wrapper: returns the best-matching format or `None`.
pub fn detect_best_format(num_elements: usize, data_bytes: usize) -> Option<FormatCandidate> {
    detect_format(num_elements, data_bytes).into_iter().next()
}

/// Provide a human-readable explanation for why detection chose a format.
pub fn explain_detection(num_elements: usize, data_bytes: usize) -> String {
    let candidates = detect_format(num_elements, data_bytes);
    if candidates.is_empty() {
        return format!(
            "No known quantization format matches {num_elements} elements / {data_bytes} bytes. \
             Expected QK256: {} bytes, BitNet32-F16: {} bytes.",
            num_elements.div_ceil(QK256_BLOCK) * QK256_PACKED_BYTES_PER_BLOCK,
            num_elements.div_ceil(BITNET32_BLOCK) * BITNET32_BYTES_PER_BLOCK,
        );
    }
    let mut out = String::new();
    for (i, c) in candidates.iter().enumerate() {
        out.push_str(&format!("{}. {:?} ({:?}): {}\n", i + 1, c.format, c.confidence, c.reason));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_qk256_is_high_confidence() {
        let elems = 1024;
        let bytes = (elems / QK256_BLOCK) * QK256_PACKED_BYTES_PER_BLOCK; // 4 * 64 = 256
        let cands = detect_format(elems, bytes);
        assert!(!cands.is_empty());
        assert_eq!(cands[0].confidence, DetectionConfidence::High);
        assert_eq!(cands[0].format, QuantizationType::I2S);
    }

    #[test]
    fn qk256_with_padding_is_medium() {
        let elems = 1024;
        let exact = (elems / QK256_BLOCK) * QK256_PACKED_BYTES_PER_BLOCK;
        let cands = detect_format(elems, exact + 64);
        assert!(!cands.is_empty());
        assert_eq!(cands[0].confidence, DetectionConfidence::Medium);
    }

    #[test]
    fn bitnet32_exact_match() {
        let elems = 1024;
        let bytes = (elems / BITNET32_BLOCK) * BITNET32_BYTES_PER_BLOCK; // 32 * 10 = 320
        let cands = detect_format(elems, bytes);
        assert!(cands.iter().any(|c| c.reason.contains("BitNet32")));
    }

    #[test]
    fn no_match_returns_empty() {
        let cands = detect_format(1024, 7); // nonsense byte count
        assert!(cands.is_empty());
    }

    #[test]
    fn detect_best_format_returns_highest() {
        let elems = 512;
        let bytes = (elems / QK256_BLOCK) * QK256_PACKED_BYTES_PER_BLOCK;
        let best = detect_best_format(elems, bytes);
        assert!(best.is_some());
        assert_eq!(best.unwrap().confidence, DetectionConfidence::High);
    }

    #[test]
    fn explain_detection_no_match_includes_expected() {
        let expl = explain_detection(1024, 7);
        assert!(expl.contains("No known quantization format"));
        assert!(expl.contains("QK256"));
    }

    #[test]
    fn explain_detection_match_includes_candidates() {
        let elems = 1024;
        let bytes = (elems / QK256_BLOCK) * QK256_PACKED_BYTES_PER_BLOCK;
        let expl = explain_detection(elems, bytes);
        assert!(expl.contains("High"));
    }

    #[test]
    fn zero_elements_returns_empty() {
        let cands = detect_format(0, 0);
        assert!(cands.is_empty());
    }

    #[test]
    fn candidates_sorted_by_confidence() {
        // Choose a size that matches multiple formats at different confidence levels
        let elems = 256;
        let bytes = QK256_PACKED_BYTES_PER_BLOCK; // exact QK256
        let cands = detect_format(elems, bytes);
        for w in cands.windows(2) {
            assert!(w[0].confidence >= w[1].confidence);
        }
    }
}
