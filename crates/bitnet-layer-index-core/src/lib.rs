//! Reusable helpers for extracting layer indices from tensor names.
//!
//! This crate centralizes the low-level parsing used by multiple crates
//! (`bitnet-models`, `bitnet-gpu-hal`) so layer-name handling follows one
//! implementation.

/// Parse an unsigned integer from the start of `bytes` until the first non-digit.
///
/// Returns `None` if no leading digits are found or on overflow.
pub fn parse_usize_prefix(bytes: &[u8]) -> Option<usize> {
    let mut value: usize = 0;
    let mut found_any = false;

    for &b in bytes {
        if b.is_ascii_digit() {
            value = value.checked_mul(10)?.checked_add((b - b'0') as usize)?;
            found_any = true;
        } else {
            break;
        }
    }

    found_any.then_some(value)
}

/// Extract any numeric segment from a dot-separated tensor name.
///
/// Example: `"model.layers.12.attn.q_proj.weight" -> Some(12)`.
pub fn extract_any_layer_index(name: &str) -> Option<usize> {
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
        }
    }
    None
}

/// Extract a layer index from known GGUF/transformer patterns.
///
/// Supports `blk.<i>.*` and `layers.<i>.*`.
pub fn extract_structured_layer_index(name: &str) -> Option<usize> {
    parse_prefix_layer_index(name, "blk.").or_else(|| parse_prefix_layer_index(name, "layers."))
}

/// Extract a `blk.<i>.*` block index.
///
/// Example: `"blk.3.attn_q.weight" -> Some(3)`.
pub fn parse_block_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 && parts[0] == "blk" { parts[1].parse().ok() } else { None }
}

fn parse_prefix_layer_index(name: &str, prefix: &str) -> Option<usize> {
    name.find(prefix)
        .and_then(|pos| name.as_bytes().get(pos + prefix.len()..))
        .and_then(parse_usize_prefix)
}

#[cfg(test)]
mod tests {
    use super::{
        extract_any_layer_index, extract_structured_layer_index, parse_block_index,
        parse_usize_prefix,
    };

    #[test]
    fn parse_prefix_digits() {
        assert_eq!(parse_usize_prefix(b"123.attn"), Some(123));
        assert_eq!(parse_usize_prefix(b"001"), Some(1));
        assert_eq!(parse_usize_prefix(b"x1"), None);
    }

    #[test]
    fn extract_any_index_from_dotted_name() {
        assert_eq!(extract_any_layer_index("model.layers.39.mlp"), Some(39));
        assert_eq!(extract_any_layer_index("layers.0.wq"), Some(0));
        assert_eq!(extract_any_layer_index("embed_tokens.weight"), None);
    }

    #[test]
    fn extract_structured_index() {
        assert_eq!(extract_structured_layer_index("blk.123.attn_q.weight"), Some(123));
        assert_eq!(extract_structured_layer_index("model.layers.7.mlp.gate"), Some(7));
        assert_eq!(extract_structured_layer_index("token_embd.weight"), None);
    }

    #[test]
    fn parse_block_only() {
        assert_eq!(parse_block_index("blk.3.attn_q.weight"), Some(3));
        assert_eq!(parse_block_index("blk.x.attn_q.weight"), None);
        assert_eq!(parse_block_index("layers.3.attn_q.weight"), None);
    }
}
