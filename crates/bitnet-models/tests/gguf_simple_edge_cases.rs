//! Edge-case tests for `gguf_simple` module — the largest module (~1800 lines)
//! in the bitnet-models crate.
//!
//! Covers:
//! - `GGUFLoaderConfig` defaults and custom values
//! - `load_gguf_full` error paths: nonexistent file, empty file, truncated file,
//!   invalid magic, file too small to be GGUF
//! - `load_gguf` (deprecated shim) backward-compatibility
//! - Successful loading of a minimal valid GGUF via `bitnet-st2gguf` GgufWriter

#![allow(deprecated)] // load_gguf is deprecated; we test it for backward compat

#[cfg(any(feature = "cpu", feature = "gpu", feature = "crossval"))]
mod tests {
    use bitnet_common::Device;
    use bitnet_models::gguf_simple::{GGUFLoaderConfig, load_gguf, load_gguf_full};
    use bitnet_st2gguf::writer::{GgufWriter, MetadataValue, TensorDType, TensorEntry};
    use std::path::Path;
    use tempfile::TempDir;

    // ── GGUFLoaderConfig defaults ────────────────────────────────────────

    #[test]
    fn config_default_is_permissive() {
        let cfg = GGUFLoaderConfig::default();
        assert!(!cfg.strict_mode, "default should be permissive");
        assert_eq!(cfg.tolerance_bytes, 128);
    }

    #[test]
    fn config_strict_mode_overrides() {
        let cfg = GGUFLoaderConfig { strict_mode: true, ..Default::default() };
        assert!(cfg.strict_mode);
        // tolerance_bytes is still the default; it's simply ignored in strict mode
        assert_eq!(cfg.tolerance_bytes, 128);
    }

    #[test]
    fn config_custom_tolerance() {
        let cfg = GGUFLoaderConfig { strict_mode: false, tolerance_bytes: 512 };
        assert!(!cfg.strict_mode);
        assert_eq!(cfg.tolerance_bytes, 512);
    }

    #[test]
    fn config_zero_tolerance() {
        let cfg = GGUFLoaderConfig { strict_mode: false, tolerance_bytes: 0 };
        assert_eq!(cfg.tolerance_bytes, 0);
    }

    #[test]
    fn config_clone_and_debug() {
        let cfg = GGUFLoaderConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.strict_mode, cfg.strict_mode);
        assert_eq!(cloned.tolerance_bytes, cfg.tolerance_bytes);
        // Debug impl exists
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("strict_mode"));
        assert!(dbg.contains("tolerance_bytes"));
    }

    // ── load_gguf_full: nonexistent file ─────────────────────────────────

    #[test]
    fn load_nonexistent_file_returns_error() {
        let path = Path::new("totally_nonexistent_model_file.gguf");
        let result = load_gguf_full(path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_err(), "nonexistent file must fail");
        let err_msg = format!("{}", result.err().expect("should be Err"));
        assert!(
            err_msg.contains("Failed to open") || err_msg.contains("nonexistent"),
            "error should mention open failure: {err_msg}"
        );
    }

    // ── load_gguf_full: empty file ───────────────────────────────────────

    #[test]
    fn load_empty_file_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.gguf");
        std::fs::write(&path, b"").unwrap();
        let result = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_err(), "empty file must fail");
    }

    // ── load_gguf_full: file too small to be GGUF ────────────────────────

    #[test]
    fn load_file_too_small_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tiny.gguf");
        // 3 bytes — not even enough for the 4-byte magic
        std::fs::write(&path, b"GGU").unwrap();
        let result = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_err(), "3-byte file must fail");
    }

    // ── load_gguf_full: invalid magic bytes ──────────────────────────────

    #[test]
    fn load_invalid_magic_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad_magic.gguf");
        // Valid-length header with wrong magic
        let mut data = Vec::new();
        data.extend_from_slice(b"XXXX"); // wrong magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        data.extend_from_slice(&0u64.to_le_bytes()); // n_kv
        data.resize(256, 0); // pad to something bigger
        std::fs::write(&path, &data).unwrap();

        let result = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_err(), "invalid magic must fail");
    }

    // ── load_gguf_full: truncated header ─────────────────────────────────

    #[test]
    fn load_truncated_header_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("truncated.gguf");
        // Valid magic but header is cut short
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        // Missing n_tensors and n_kv
        std::fs::write(&path, &data).unwrap();

        let result = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_err(), "truncated header must fail");
    }

    // ── load_gguf_full: random garbage bytes ─────────────────────────────

    #[test]
    fn load_random_garbage_returns_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("garbage.gguf");
        // Fill with deterministic non-GGUF bytes
        let garbage: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        std::fs::write(&path, &garbage).unwrap();
        let result = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_err(), "random garbage must fail");
    }

    // ── load_gguf_full: strict vs permissive config ──────────────────────

    #[test]
    fn load_nonexistent_strict_mode_still_errors() {
        let path = Path::new("no_such_file_strict.gguf");
        let cfg = GGUFLoaderConfig { strict_mode: true, ..Default::default() };
        let result = load_gguf_full(path, Device::Cpu, cfg);
        assert!(result.is_err());
    }

    // ── load_gguf (deprecated shim) ──────────────────────────────────────

    #[test]
    fn deprecated_load_gguf_nonexistent_errors() {
        let path = Path::new("no_such_deprecated.gguf");
        let result = load_gguf(path, Device::Cpu);
        assert!(result.is_err(), "deprecated shim must also fail on missing file");
    }

    // ── load_gguf_full with a minimal valid GGUF ─────────────────────────

    /// Build a minimal but real GGUF file via GgufWriter with enough metadata
    /// and tensors for the loader to parse successfully.
    fn build_minimal_gguf(dir: &TempDir) -> std::path::PathBuf {
        let path = dir.path().join("minimal.gguf");
        let hidden = 32usize;
        let vocab = 64usize;
        let n_layers = 1usize;
        let intermediate = 64usize;
        let n_heads = 4u32;

        let mut w = GgufWriter::new();
        // Required metadata expected by extract_config_from_gguf
        w.add_metadata("llama.embedding_length", MetadataValue::U32(hidden as u32));
        w.add_metadata("llama.block_count", MetadataValue::U32(n_layers as u32));
        w.add_metadata("llama.feed_forward_length", MetadataValue::U32(intermediate as u32));
        w.add_metadata("llama.attention.head_count", MetadataValue::U32(n_heads));
        w.add_metadata("llama.attention.head_count_kv", MetadataValue::U32(n_heads));

        // Helper: create an F32 tensor entry
        let f32_tensor = |name: &str, rows: usize, cols: usize| -> TensorEntry {
            let numel = rows * cols;
            let data: Vec<u8> = vec![0u8; numel * 4]; // zeros
            TensorEntry::new(
                name.to_string(),
                vec![rows as u64, cols as u64],
                TensorDType::F32,
                data,
            )
        };

        let f32_vec = |name: &str, len: usize| -> TensorEntry {
            let data = vec![0x3f, 0x80, 0x00, 0x00].repeat(len); // f32 1.0 repeated
            TensorEntry::new(name.to_string(), vec![len as u64], TensorDType::F32, data)
        };

        // Embedding + output
        w.add_tensor(f32_tensor("token_embd.weight", vocab, hidden));
        w.add_tensor(f32_tensor("output.weight", hidden, vocab));

        // One transformer layer
        let pfx = "blk.0";
        for suffix in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            w.add_tensor(f32_tensor(&format!("{pfx}.{suffix}.weight"), hidden, hidden));
        }
        for suffix in ["ffn_gate", "ffn_up"] {
            w.add_tensor(f32_tensor(&format!("{pfx}.{suffix}.weight"), intermediate, hidden));
        }
        w.add_tensor(f32_tensor(&format!("{pfx}.ffn_down.weight"), hidden, intermediate));
        for suffix in ["attn_norm", "ffn_norm"] {
            w.add_tensor(f32_vec(&format!("{pfx}.{suffix}.weight"), hidden));
        }
        w.add_tensor(f32_vec("output_norm.weight", hidden));

        w.write_to_file(&path).expect("GgufWriter should succeed");
        path
    }

    #[test]
    fn load_minimal_valid_gguf_succeeds() {
        let dir = TempDir::new().unwrap();
        let path = build_minimal_gguf(&dir);
        let result = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_ok(), "minimal valid GGUF should load: {:?}", result.err());
    }

    #[test]
    fn load_minimal_gguf_has_expected_tensors() {
        let dir = TempDir::new().unwrap();
        let path = build_minimal_gguf(&dir);
        let res = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default()).unwrap();

        // Should have token_embd and output at minimum
        assert!(res.tensors.contains_key("token_embd.weight"), "missing token_embd.weight");
        // output.weight might be stored as-is or normalized; check either key
        let has_output =
            res.tensors.contains_key("output.weight") || res.tensors.contains_key("lm_head.weight");
        assert!(has_output, "missing output projection tensor");
    }

    #[test]
    fn load_minimal_gguf_config_matches_metadata() {
        let dir = TempDir::new().unwrap();
        let path = build_minimal_gguf(&dir);
        let res = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default()).unwrap();

        assert_eq!(res.config.model.hidden_size, 32);
        assert_eq!(res.config.model.num_layers, 1);
        assert_eq!(res.config.model.num_heads, 4);
    }

    #[test]
    fn load_minimal_gguf_strict_mode_succeeds() {
        let dir = TempDir::new().unwrap();
        let path = build_minimal_gguf(&dir);
        let cfg = GGUFLoaderConfig { strict_mode: true, ..Default::default() };
        let result = load_gguf_full(&path, Device::Cpu, cfg);
        assert!(result.is_ok(), "strict mode should still load valid F32 GGUF: {:?}", result.err());
    }

    #[test]
    fn deprecated_load_gguf_on_valid_file_returns_tuple() {
        let dir = TempDir::new().unwrap();
        let path = build_minimal_gguf(&dir);
        let result = load_gguf(&path, Device::Cpu);
        assert!(result.is_ok(), "deprecated shim should load valid file: {:?}", result.err());
        let (config, tensors) = result.unwrap();
        assert_eq!(config.model.hidden_size, 32);
        assert!(!tensors.is_empty());
    }

    // ── GgufLoadResult fields ────────────────────────────────────────────

    #[test]
    fn load_result_i2s_qk256_empty_for_f32_model() {
        let dir = TempDir::new().unwrap();
        let path = build_minimal_gguf(&dir);
        let res = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default()).unwrap();
        assert!(res.i2s_qk256.is_empty(), "F32-only model should have no QK256 tensors");
    }
}
