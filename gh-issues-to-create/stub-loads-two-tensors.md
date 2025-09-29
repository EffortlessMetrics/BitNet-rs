# Stub code: `loads_two_tensors` test in `gguf_min.rs` is ignored

The `loads_two_tensors` test in `crates/bitnet-models/src/gguf_min.rs` is ignored with a `#[ignore]` attribute. This suggests that the test is not fully implemented or is not intended to be run. This is a form of stubbing.

**File:** `crates/bitnet-models/src/gguf_min.rs`

**Function:** `loads_two_tensors`

**Code:**
```rust
    #[test]
    #[ignore] // set BITNET_GGUF to a real path to run
    fn loads_two_tensors() {
        let p = std::env::var_os("BITNET_GGUF").expect("set BITNET_GGUF");
        let two = load_two(p).unwrap();
        assert!(two.vocab > 0 && two.dim > 0);
        assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
        assert_eq!(two.lm_head.len(), two.dim * two.vocab);
    }
```

## Proposed Fix

The `loads_two_tensors` test should be implemented to actually load two tensors from a GGUF file and assert that the `vocab` and `dim` are correctly extracted. This would involve:

1.  **Creating a dummy GGUF file:** Create a dummy GGUF file with two tensors.
2.  **Loading the GGUF file:** Call `load_two` with the dummy GGUF file.
3.  **Asserting the result:** Assert that the `vocab` and `dim` are correctly extracted.

### Example Implementation

```rust
    #[test]
    fn loads_two_tensors() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_path_buf();

        // Write a minimal GGUF header with two tensors
        let gguf_header = b"GGUF\x03\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        temp_file.write_all(gguf_header).expect("Failed to write GGUF header");

        // Write tensor info for tok_embeddings
        write_gguf_tensor_info(&mut temp_file, "tok_embeddings.weight", &[100, 50], 0, 0);

        // Write tensor info for lm_head
        write_gguf_tensor_info(&mut temp_file, "lm_head.weight", &[50, 100], 0, 0);

        let two = load_two(&path).unwrap();
        assert_eq!(two.vocab, 100);
        assert_eq!(two.dim, 50);
        assert_eq!(two.tok_embeddings.len(), two.vocab * two.dim);
        assert_eq!(two.lm_head.len(), two.dim * two.vocab);
    }

    // Helper function to write GGUF tensor info (simplified)
    fn write_gguf_tensor_info<W: Write>(writer: &mut W, name: &str, shape: &[u64], ty: u32, offset: u64) {
        // ... implementation to write GGUF tensor info ...
    }
```
