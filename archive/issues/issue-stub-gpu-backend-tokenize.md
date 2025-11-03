# [IMPLEMENTATION] Replace placeholder tokenization in GpuBackend with proper tokenizer integration

## Problem Description
The `GpuBackend::tokenize` function in `crates/bitnet-inference/src/gpu.rs` uses character-to-u32 conversion instead of proper tokenization, preventing real GPU inference workflows.

## Environment
- **File**: `crates/bitnet-inference/src/gpu.rs`
- **Function**: `GpuBackend::tokenize`
- **Current State**: Placeholder implementation

## Root Cause Analysis
```rust
fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
    // Placeholder implementation - in practice would use a proper tokenizer
    Ok(text.chars().map(|c| c as u32).collect())
}
```

**Issues:**
1. No real tokenization - just character ASCII conversion
2. Incompatible with real model vocabularies
3. Prevents actual GPU inference testing
4. Inconsistent with CPU tokenization pipeline

## Proposed Solution
```rust
impl GpuBackend {
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Use the same tokenizer as the inference engine
        self.tokenizer.encode(text, true, true)
            .map_err(|e| BitNetError::Tokenization(e))
    }
}

// Update GpuBackend construction to include tokenizer
pub struct GpuBackend {
    device: CudaDevice,
    tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer>,
    // ... other fields
}
```

## Implementation Plan
### Phase 1: Tokenizer Integration (1 day)
- [ ] Add tokenizer field to GpuBackend struct
- [ ] Update constructor to accept tokenizer parameter
- [ ] Implement proper tokenization using bitnet_tokenizers

### Phase 2: Error Handling & Testing (1 day)
- [ ] Add comprehensive error handling
- [ ] Create tests with real tokenizer
- [ ] Validate consistency with CPU tokenization

## Acceptance Criteria
- [ ] Real tokenization using bitnet_tokenizers crate
- [ ] Consistent behavior with CPU backend
- [ ] Proper error handling for invalid inputs
- [ ] Test coverage with real models

**Labels**: `implementation`, `gpu`, `tokenization`, `P2-medium`
**Effort**: 2 days
