# Stub code: `read_metadata_value` in `gguf_parity.rs` has a placeholder for `ARRAY` type

The `read_metadata_value` function in `crates/bitnet-models/src/gguf_parity.rs` has a comment "For now, just return a placeholder" for the `ARRAY` type. It just skips the array and returns `"[array]"`. This is a form of stubbing.

**File:** `crates/bitnet-models/src/gguf_parity.rs`

**Function:** `read_metadata_value`

**Code:**
```rust
fn read_metadata_value(reader: &mut BufReader<File>) -> Result<String> {
    let mut type_bytes = [0u8; 4];
    reader.read_exact(&mut type_bytes)?;
    let value_type = u32::from_le_bytes(type_bytes);

    match value_type {
        // ...
        9 => {
            // ARRAY
            // For now, just return a placeholder
            skip_array(reader)?;
            Ok("[array]".to_string())
        }
        // ...
    }
}
```

## Proposed Fix

The `read_metadata_value` function should be implemented to actually read the array values. This would involve reading the element type and count, and then reading each element based on its type.

### Example Implementation

```rust
fn read_metadata_value(reader: &mut BufReader<File>) -> Result<String> {
    let mut type_bytes = [0u8; 4];
    reader.read_exact(&mut type_bytes)?;
    let value_type = u32::from_le_bytes(type_bytes);

    match value_type {
        // ...
        9 => {
            // ARRAY
            let elem_type = read_u32(reader)?;
            let count = read_u64(reader)?;
            let mut values = Vec::new();
            for _ in 0..count {
                values.push(read_gguf_value(reader, elem_type)?);
            }
            Ok(format!("{:?}", values))
        }
        // ...
    }
}

fn read_gguf_value(reader: &mut BufReader<File>, ty: u32) -> Result<String> {
    // ... implementation to read a single GGUF value ...
    Ok("value".to_string()) // Placeholder
}
```
