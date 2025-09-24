# GGUF Parser Edge Cases - Fuzzing Corpus

## Validated Edge Cases (No Crashes Found)

### Invalid Magic Numbers
- Corrupted GGUF magic bytes
- Partial header data (< 16 bytes)
- Invalid version numbers

### Metadata Corruption
- UTF-8 string corruption handling
- Key-value pair boundary overflow
- Missing tensor metadata

### Tensor Alignment Issues
- V3 header alignment validation
- Data offset corruption
- Tensor count overflow protection

## Property-Based Test Coverage
- I2S block alignment correctness
- Block underflow prevention
- Transpose roundtrip validation

## Status: All edge cases handled gracefully with proper error returns