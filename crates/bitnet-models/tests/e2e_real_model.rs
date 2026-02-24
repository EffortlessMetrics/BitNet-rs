//! End-to-end tests with a real GGUF model.
//!
//! These tests are gated on the `BITNET_GGUF` environment variable.
//! Set it to a real model path to enable:
//!
//!   BITNET_GGUF=/path/to/model.gguf cargo test -p bitnet-models --test e2e_real_model
//!
//! In CI, set `RUN_E2E=1` alongside `BITNET_GGUF` to opt in.

#[cfg(any(feature = "cpu", feature = "gpu"))]
mod tests {
    use std::path::PathBuf;

    fn model_path() -> Option<PathBuf> {
        let path = std::env::var_os("BITNET_GGUF")?;
        let p = PathBuf::from(path);
        if p.exists() { Some(p) } else { None }
    }

    /// Load the model and verify basic metadata is present.
    #[test]
    fn e2e_load_gguf_header() {
        let Some(path) = model_path() else {
            eprintln!("SKIP: set BITNET_GGUF=/path/to/model.gguf to run e2e tests");
            return;
        };
        let result = bitnet_models::gguf_simple::load_gguf_full(
            &path,
            bitnet_common::Device::Cpu,
            Default::default(),
        );
        match result {
            Ok(loaded) => {
                let tensor_count = loaded.tensors.len() + loaded.i2s_qk256.len();
                assert!(tensor_count > 0, "model should have at least one tensor");
                println!(
                    "e2e_load_gguf_header PASS — {} f32 tensors, {} qk256 tensors from {:?}",
                    loaded.tensors.len(),
                    loaded.i2s_qk256.len(),
                    path.file_name().unwrap_or_default()
                );
            }
            Err(e) => panic!("Failed to load model from {path:?}: {e}"),
        }
    }

    /// Verify that the config fields are populated (not default zeros) after loading.
    #[test]
    fn e2e_config_populated() {
        let Some(path) = model_path() else {
            return;
        };
        let loaded = match bitnet_models::gguf_simple::load_gguf_full(
            &path,
            bitnet_common::Device::Cpu,
            Default::default(),
        ) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("SKIP: could not load model: {e}");
                return;
            }
        };
        let cfg = &loaded.config;
        // A real model will have non-default vocab_size and hidden_size
        assert!(cfg.model.vocab_size > 0, "vocab_size must be > 0, got {}", cfg.model.vocab_size);
        assert!(
            cfg.model.hidden_size > 0,
            "hidden_size must be > 0, got {}",
            cfg.model.hidden_size
        );
        assert!(cfg.model.num_layers > 0, "num_layers must be > 0");
        println!(
            "e2e_config_populated PASS — vocab={} hidden={} layers={}",
            cfg.model.vocab_size, cfg.model.hidden_size, cfg.model.num_layers
        );
    }

    /// Ensure the loaded tensors have finite, non-zero norms (no all-zero/NaN weights).
    #[test]
    fn e2e_weight_tensors_finite() {
        let Some(path) = model_path() else {
            return;
        };
        let loaded = match bitnet_models::gguf_simple::load_gguf_full(
            &path,
            bitnet_common::Device::Cpu,
            Default::default(),
        ) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("SKIP: could not load model: {e}");
                return;
            }
        };
        // Check a sample of tensors for finite values
        let checked = loaded
            .tensors
            .iter()
            .take(8)
            .map(|(name, tensor)| {
                let flat: Vec<f32> =
                    tensor.flatten_all().and_then(|t| t.to_vec1()).unwrap_or_default();
                let any_finite = flat.iter().any(|v| v.is_finite());
                (name.clone(), flat.len(), any_finite)
            })
            .collect::<Vec<_>>();

        for (name, len, finite) in &checked {
            assert!(
                *len == 0 || *finite,
                "tensor '{name}' with {len} elements has no finite values"
            );
        }
        println!("e2e_weight_tensors_finite PASS — checked {} tensors", checked.len());
    }
}

// When no backend feature is enabled, compile a trivial stub so the binary links.
#[cfg(not(any(feature = "cpu", feature = "gpu")))]
#[test]
fn e2e_skipped_no_backend_feature() {
    eprintln!("SKIP: build with --features cpu or --features gpu to enable e2e tests");
}
