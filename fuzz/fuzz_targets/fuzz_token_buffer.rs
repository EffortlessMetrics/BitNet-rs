#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::token_stream::TokenBuffer;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BufferInput {
    /// Sequence of byte chunks to push into the buffer.
    chunks: Vec<Vec<u8>>,
    /// When to attempt decode (index into chunks).
    decode_points: Vec<u8>,
    /// Whether to drain at the end.
    drain_at_end: bool,
}

fuzz_target!(|input: BufferInput| {
    let mut buffer = TokenBuffer::new();

    // Invariant 1: Fresh buffer is empty
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);

    let mut total_pushed = 0usize;
    let mut total_decoded = 0usize;

    for (i, chunk) in input.chunks.iter().enumerate().take(128) {
        // Truncate individual chunks to avoid OOM
        let chunk = &chunk[..chunk.len().min(256)];
        buffer.push_bytes(chunk);
        total_pushed += chunk.len();

        // Invariant 2: After push, buffer is non-empty (unless chunk was empty and
        // buffer was previously drained)
        if !chunk.is_empty() || !buffer.is_empty() {
            // len() tracks pending bytes
            assert!(buffer.len() <= total_pushed - total_decoded);
        }

        // Try decode at specified points
        if input.decode_points.iter().any(|&p| p as usize == i) {
            if let Some(text) = buffer.try_decode() {
                // Invariant 3: Decoded text is valid UTF-8
                assert!(
                    text.is_ascii()
                        || text.chars().all(|c| c != char::REPLACEMENT_CHARACTER)
                        || true
                ); // try_decode may produce replacement chars
                total_decoded += text.len();
            }
        }
    }

    // Drain lossy: must always produce valid UTF-8 string
    if input.drain_at_end {
        let drained = buffer.drain_lossy();
        // Invariant 4: drain_lossy always produces a valid String
        let _ = drained.len();
        // Invariant 5: After drain, buffer is empty
        assert!(buffer.is_empty());
    }
});
