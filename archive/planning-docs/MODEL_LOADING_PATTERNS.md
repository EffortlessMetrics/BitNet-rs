# BitNet-rs Model Loading and GGUF Integration Patterns

## Executive Summary

BitNet-rs implements a sophisticated, layered model loading system with automatic format detection, comprehensive validation, and memory-efficient handling. The system integrates GGUF parsing, tokenization, and cross-validation into a cohesive pipeline with strong error handling and security boundaries.

---

## 1. Model Loading Flow Architecture

### 1.1 Entry Points

```
ModelLoader (generic dispatcher)
├── GgufLoader (GGUF format)
├── SafeTensorsLoader (SafeTensors format)
└── HuggingFaceLoader (HuggingFace directory layout)
```

**Key File**: `/crates/bitnet-models/src/loader.rs`

The `ModelLoader` implements a multi-strategy format detection:

1. **Extension-based detection** (fastest): `.gguf`, `.safetensors`
2. **Magic byte detection**: File header analysis
3. **Directory structure detection**: HuggingFace layout analysis

### 1.2 Primary Loading Pipeline

```
File Path → Format Detection → Metadata Extraction → Tensor Loading → 
Model Validation → BitNetModel Construction → Ready for Inference
```

**Code Path**:
```rust
// File: bitnet-models/src/loader.rs
pub fn load_with_config<P: AsRef<Path>>(
    &self,
    path: P,
    config: &LoadConfig,
) -> Result<Box<dyn Model>>
```

**Flow**:
1. Detect format loader (GGUF/SafeTensors/HuggingFace)
2. Extract model metadata (name, arch, dimensions)
3. Validate model compatibility
4. Load model weights via format-specific loader
5. Return boxed trait object

### 1.3 Production vs. Standard Loading

**Standard Loader** (`ModelLoader`):
- File: `bitnet-models/src/loader.rs`
- Used: Inference, CLI, general use
- Config: `LoadConfig` (mmap, checksums, progress callbacks)

**Production Loader** (`ProductionModelLoader`):
- File: `bitnet-models/src/production_loader.rs`
- Features: Enhanced validation, tensor alignment checks, memory profiling
- Config: `ProductionLoadConfig` (strict mode, max model size, memory analysis)
- Status: Infrastructure ready but not fully activated (feature-gated)

---

## 2. GGUF Parsing and Metadata Extraction

### 2.1 GGUF Reader Architecture

**File**: `/crates/bitnet-models/src/formats/gguf/reader.rs`

Supports **GGUF v2 and v3** with comprehensive security hardening:

```rust
pub struct GgufReader<'a> {
    data: &'a [u8],
    pub header: GgufHeader,
    metadata: Vec<GgufMetadata>,
    tensor_infos: Vec<TensorInfo>,
    data_start: usize,
}
```

**Header Format**:
- **v2**: magic (4) + version (4) + n_tensors (8) + n_kv (8)
- **v3**: v2 + alignment (4) + data_offset (8)

**Alignment Strategy**:
- v2: Fixed 32-byte alignment
- v3: Reads alignment from header; clamped to power-of-two; defaults to 32 if invalid
- Data offset: Validated against file bounds and alignment before use

### 2.2 Metadata Extraction Pattern

**File**: `/crates/bitnet-models/src/formats/gguf/loader.rs`

```rust
fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata> {
    // Memory-maps or reads entire file
    let reader = GgufReader::new(data)?;
    
    // Extract basic info
    name: reader.get_string_metadata("general.name")
    architecture: reader.get_string_metadata("general.architecture")
    
    // Extract model config
    vocab_size: get_u32_any(reader, &["llama.vocab_size", ...])
    hidden_size: get_u32_any(reader, &["llama.embedding_length", ...])
    num_heads: get_u32_any(reader, &["llama.attention.head_count", ...])
    num_layers: get_u32_any(reader, &["llama.block_count", ...])
    
    // Infer missing config from tensor shapes
    if hidden_size == 0:
        infer_hidden_size_from_tensors(reader)
    if num_layers == 0:
        infer_num_layers_from_tensors(reader)
    if num_key_value_heads == 0:
        infer_kv_heads_from_tensors(reader)
}
```

### 2.3 Metadata Lookup Patterns

BitNet-rs uses **multi-key fallback patterns** for metadata extraction:

```
Priority chains (example):
1. BitNet-specific: "bitnet-b1.58.embedding_length"
2. LLaMA standard: "llama.embedding_length"
3. Generic: "n_embd", "hidden_size"
4. Inferred from tensors if all else fails
```

**Key Metadata Keys Extracted**:

| Category | Keys | Fallback |
|----------|------|----------|
| Vocab Size | `llama.vocab_size`, `bitnet-b1.58.vocab_size`, `tokenizer.ggml.tokens` | Token count from array |
| Hidden | `llama.embedding_length`, `bitnet-b1.58.embedding_length`, `n_embd` | Embedding tensor shape |
| Layers | `llama.block_count`, `bitnet-b1.58.block_count` | Max layer number in tensors |
| Attention Heads | `llama.attention.head_count`, `bitnet-b1.58.attention.head_count`, `n_head` | KV tensor shapes |
| KV Heads | `llama.attention.head_count_kv`, `bitnet-b1.58.attention.head_count_kv` | k_proj tensor dimensions |
| RoPE Freq Base | `bitnet-b1.58.rope.freq_base`, `llama.rope.freq_base`, `rope.freq_base` | 10000.0 (default) |
| Block Size | `bitnet.block_size` | Architecture-dependent |
| Precision | `bitnet.precision` | Auto-detect |

### 2.4 Chat Template Metadata Extraction

**Not explicitly shown in loader**, but metadata keys available:
- `general.chat_template`: GGUF chat template metadata
- Tokenizer-specific special token IDs:
  - `tokenizer.ggml.bos_token_id`
  - `tokenizer.ggml.eos_token_id`
  - `tokenizer.ggml.eot_token_id` (LLaMA-3)

**Note**: Chat template detection is handled by tokenizer/CLI layer, not model loader.

---

## 3. Tensor Loading Strategies

### 3.1 Tensor Loading Entry Point

**File**: `/crates/bitnet-models/src/formats/gguf/loader.rs`

```rust
fn load(
    &self,
    path: &Path,
    device: &Device,
    config: &LoadConfig
) -> Result<Box<dyn Model>>
```

**Strategy**:
1. **Memory mapping** (if `config.use_mmap`): Direct OS mmap for efficient large file access
2. **Full buffer load** (fallback): Read entire file into memory if mmap unavailable
3. Parse GGUF structure with security limits
4. Load tensors with device-aware type selection

### 3.2 Memory Mapping Patterns

**File**: `/crates/bitnet-models/src/loader.rs`

```rust
struct MmapFile {
    _file: File,
    mmap: Mmap,
}

impl MmapFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { _file: file, mmap })
    }
    
    pub fn as_slice(&self) -> &[u8] { &self.mmap }
}
```

**Benefits**:
- Zero-copy access to large models (>10GB)
- OS page cache management (automatic)
- Efficient memory usage for multi-process inference

### 3.3 Tensor Type Handling

BitNet-rs supports multiple quantization formats with **automatic flavor detection**:

**I2_S Flavor Detection**:
- **BitNet32-F16**: 32-element blocks with inline F16 scales (production)
- **QK256**: 256-element blocks with separate scales (GGML-compatible, MVP scalar)

**Detection Priority**:
```
1. Check tensor size against expected QK256 shape
2. Match QK256 (256-elem blocks) before BitNet32-F16
3. Automatic routing to scalar or AVX2 kernels
4. FFI bridge to C++ if BITNET_CPP_DIR set
```

### 3.4 Embedding Transposition Logic

Microsoft models ship with transposed embeddings:

```rust
fn embedding_is_transposed(dims: &[usize]) -> bool {
    dims.len() == 2 && dims[0] < dims[1] && dims[1] >= 32768
}
```

**Heuristic**: Detects [hidden, vocab] layout; transposes to [vocab, hidden].

### 3.5 Projection Tensor Normalization

All attention/FFN projections normalized to `[out_dim, in_dim]`:

```rust
fn maybe_transpose_to_out_in(shape: &[usize], name: &str) -> bool {
    is_projection_weight(name) && shape.len() == 2
}
```

**Projection patterns detected**:
- Attention: `attn_{q,k,v}.weight`, `{q,k,v,o}_proj.weight`
- FFN: `ffn_{gate,up,down}.weight`, `{gate,up,down}_proj.weight`

---

## 4. Model Validation Patterns

### 4.1 Validation Stages

**File**: `/crates/bitnet-models/src/formats/gguf/loader.rs`

**Stage 1: GGUF Structure Validation**
```rust
reader.validate()?
```
- Header magic bytes check
- Version support (v2, v3)
- Tensor count bounds (max 100K)
- Metadata count bounds (max 10K)
- Offset bounds verification

**Stage 2: LayerNorm Gamma Statistics**
```rust
fn check_ln_gamma_stats(name: &str, w: &Tensor) -> Result<()>
```
- RMS (root mean square) check: acceptable range [0.5, 2.0]
- Failure mode: Indicates quantized LayerNorm (should be F16/F32)
- Strict mode: `BITNET_STRICT_MODE=1` exits with code 8 on failure
- Non-strict: Logs warning and continues

**Stage 3: Tensor Presence Validation**
```rust
// In BitNetModel::from_gguf
let has_embeddings = check_required_tensors();
let has_output = check_output_tensors();
// Fail if critical tensors missing
```

### 4.2 Architecture Detection

**Inferred from Tensor Analysis**:

```rust
fn infer_num_layers_from_tensors(reader: &GgufReader) -> Option<usize>
fn infer_hidden_size_from_tensors(reader: &GgufReader) -> Option<usize>
fn infer_intermediate_size_from_tensors(...) -> Option<usize>
fn infer_kv_heads_from_tensors(...) -> Result<usize>
```

**Detection Patterns**:
- Layer count: Find max layer index in `blk.N.` or `layers.N.` patterns
- Hidden size: Use embedding tensor shape (vocab vs hidden)
- Intermediate: FFN gate/up projection shapes
- KV heads: k_proj output dimension / head_dim

### 4.3 Correction Policies

**File**: `/crates/bitnet-models/src/correction_policy.rs`

For known-bad models, apply targeted corrections:

```rust
struct CorrectionPlan {
    fingerprint: String,
    actions: Vec<CorrectionAction>,
}

enum CorrectionAction {
    LnGammaRescaleRms { target_rms: f32, clamp: [f32; 2] },
    // ... other corrections
}
```

**Activation**:
- Requires both: `BITNET_CORRECTION_POLICY=/path/to/policy.yml`
- AND: `BITNET_ALLOW_RUNTIME_CORRECTIONS=1`
- CI blocks these flags (production safety)

---

## 5. Model Integration with Inference Engine

### 5.1 Model-to-Inference Pipeline

**File**: `/crates/bitnet-models/src/bitnet.rs`

```rust
pub fn from_gguf(
    config: BitNetConfig,
    tensors: HashMap<String, CandleTensor>,
    raw_tensors: HashMap<String, CandleTensor>,  // QK256 raw tensors
    device: Device,
) -> Result<Self>
```

**Flow**:
1. Validate required tensors present (embeddings, output)
2. Build transformer model from weight maps
3. Remap GGUF tensor names to transformer structure
4. Normalize embeddings/lm_head dimensions
5. Create TransformerModel with kernel dispatch

### 5.2 Weight Remapping

**File**: `/crates/bitnet-models/src/weight_mapper.rs`

Canonicalizes tensor names across vendor formats:

```
BitNet format:        blk.0.attn_q.weight
↓
Canonical format:     layers.0.attention.q_proj.weight
↓
TransformerModel API: Used directly by transformer implementation
```

**Regex patterns for vendor detection**:
- Microsoft BitNet: `blk.N.*` → `layers.N.attention.*`
- LLaMA style: `layers.N.self_attn.wq.*` → `layers.N.attention.q_proj.*`
- Generic: Normalizes feed-forward, attention norms, layer norms

### 5.3 Config Inference and Validation

**Dynamic Config Updates**:
```rust
// Detect from tensor shapes if metadata missing/incorrect
detected_vocab = normalize_model_tensors(&mut mapped)
detected_hidden = normalize_model_tensors(&mut mapped)

if updated_config.vocab_size != detected_vocab:
    tracing::info!("Updating vocab_size to {}", detected_vocab)
```

**Config Validation** (in CLI):
```rust
fn validate_model_config(&self, config: &BitNetConfig) -> Result<()>
```
- d_model divisible by n_heads
- n_heads divisible by n_kv_heads (for GQA)
- Vocab size > 0
- Hidden size > 0

---

## 6. Integration with Tokenization

### 6.1 Tokenizer Auto-Discovery

**File**: `/crates/bitnet-tokenizers/src/gguf_loader.rs`

Tokenizers can be loaded directly from GGUF metadata:

```rust
pub struct RustTokenizer {
    kind: GgufTokKind,  // Spm or Bpe
    spm: Option<SentencePieceProcessor>,
    bpe: Option<tokenizers::Tokenizer>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    eot_id: Option<u32>,
}

impl RustTokenizer {
    pub fn from_gguf(reader: &GgufReader) -> Result<Self>
}
```

**Metadata Keys**:
- `tokenizer.ggml.model`: Type ("llama" → SPM, "gpt2"/"bpe" → BPE)
- `tokenizer.model`: Serialized protobuf (SPM)
- `tokenizer.ggml.tokens`: Vocabulary array
- `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`: Special tokens
- `tokenizer.ggml.eot_token_id`: LLaMA-3 end-of-turn token

### 6.2 Chat Template Detection

**File**: `/crates/bitnet-cli/src/commands/inference.rs` (template auto-detection logic)

Priority order:
1. Explicit CLI flag: `--prompt-template`
2. GGUF metadata: `general.chat_template` (detects LLaMA-3 special tokens)
3. Model path heuristics: Contains "llama3", "instruct", "chat"
4. Tokenizer path heuristics: Similar patterns
5. Fallback: "instruct" template (more reliable than raw)

---

## 7. Integration with Cross-Validation

### 7.1 Model Passing to crossval

**File**: `/crossval/src/token_parity.rs`

Cross-validation framework validates token sequence parity:

```rust
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
) -> Result<()>
```

**Purpose**: Fail-fast before expensive logits comparisons if tokenization diverges.

**Exit Codes**:
- Success (0): Tokens match
- Error (2): Token mismatch detected
- Other: Infrastructure errors

### 7.2 Model Loading in xtask

**Feature-gated**: `--features crossval-all` or `--features inference`

Loads model and tokenizer for:
- Per-token logits comparison (`crossval-per-token`)
- Receipt verification (`verify-receipt`)
- Benchmark runs (`benchmark`)

---

## 8. Error Handling and Validation

### 8.1 Security Boundaries

**File**: `/crates/bitnet-models/src/formats/gguf/types.rs`

Security limits enforced:

```rust
pub struct SecurityLimits {
    max_metadata_size: usize,      // Default: 100MB
    max_string_length: usize,      // Default: 1MB
    max_tensor_count: u64,         // Default: 100K
    max_metadata_count: u64,       // Default: 10K
    max_dimensions: u32,           // Default: 4
}
```

**Validation Gates**:
1. File size: ≤ 100 × max_metadata_size (∼10GB default)
2. Tensor count: ≤ 100K
3. Metadata count: ≤ 10K
4. String lengths: ≤ 1MB each
5. Tensor dimensions: ≤ 4 per tensor
6. Offset bounds: Absolute addresses within file
7. No integer overflow on offset calculations

### 8.2 Error Propagation Patterns

```rust
// Fail-fast on required tensors
if !has_embeddings {
    return Err(BitNetError::Validation("Missing embeddings"))
}

// Warn on suspicious stats (unless strict mode)
if !ok && BITNET_STRICT_MODE {
    return Err(BitNetError::Security(...))
} else {
    tracing::info!("Warning: {} (continuing)", msg)
}

// Gracefully degrade (infer from tensors if metadata missing)
if hidden_size == 0 && let Some(h) = infer_hidden_size {
    config.hidden_size = h
}
```

---

## 9. Performance Patterns

### 9.1 Memory Mapping Strategy

**Advantages**:
- Large models loaded without full memory copy
- OS handles page swapping automatically
- Multiple processes can share pages

**Default Behavior**:
```rust
let config = LoadConfig {
    use_mmap: true,        // Default: yes
    validate_checksums: true,
    progress_callback: None,
}
```

### 9.2 Lazy Tensor Loading

Tensors created via candle with:
- Device-aware allocation (CPU/CUDA/Metal)
- Lazy evaluation for shape validation
- Contiguous layout for kernel efficiency

### 9.3 QK256 Fast Path (v0.2 Foundation)

Runtime dispatch to AVX2-optimized dequantization:

```
QK256 tensor detected
├─ Runtime CPU feature check
├─ If AVX2 available: AVX2 kernel (∼1.2× uplift)
└─ Fallback: Scalar kernel (correct, MVP speed)
```

---

## 10. Key Integration Points

### 10.1 Model-to-CLI Integration

```
CLI Inference Request
├─ Parse flags (--model, --tokenizer, --prompt-template)
├─ Load model: ModelLoader::new() → load()
├─ Auto-detect tokenizer from GGUF or file
├─ Auto-detect prompt template
├─ Initialize generation engine
└─ Stream output with receipt tracking
```

### 10.2 Model-to-Validation Integration

```
Validation Pipeline
├─ Load GGUF: GgufReader::new()
├─ Extract metadata: extract_config()
├─ Validate LayerNorm stats: check_ln_gamma_stats()
├─ Load tensors with correction policies
├─ Build transformer: build_transformer()
└─ Inspect tool: bin/inspect --ln-stats
```

### 10.3 Model-to-Crossval Integration

```
Cross-Validation
├─ Load Rust model: ModelLoader
├─ Load tokenizer from GGUF: RustTokenizer::from_gguf()
├─ Tokenize prompt
├─ Validate token parity with C++
├─ Compare logits position-by-position
├─ Report cosine similarity & exact match rate
└─ Write receipt (ci/inference.json)
```

---

## 11. Known Limitations and Future Improvements

### 11.1 Current Limitations

1. **QK256 Performance**: MVP scalar kernels (~0.1 tok/s for 2B models)
   - v0.2 planned: Nibble-LUT + FMA tiling for ≥3× uplift

2. **LayerNorm Validation**: Heuristic RMS checking
   - Cannot catch all quantization issues
   - Relies on external policy for known-bad models

3. **Shape Inference**: Limited to standard tensor naming patterns
   - Fails on unusual or custom architectures

4. **Production Loader**: Infrastructure ready but not fully activated
   - `ProductionModelLoader` class exists but marked `#[allow(dead_code)]`

### 11.2 Future Directions

1. **Advanced Validation**: ML-based LayerNorm detection
2. **Streaming Loading**: Load tensors on-demand for huge models
3. **Quantization Auto-Selection**: Detect optimal format per model
4. **Hardware Profiling**: Auto-tune batch size and device placement

---

## Summary: Model Loading Data Flow

```
┌─────────────────────────────────────────────────┐
│ User specifies model path & optional tokenizer   │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │ Format Detection │ (ext, magic, structure)
        └────────┬────────┘
                 │
      ┌──────────▼──────────┐
      │ GgufReader::new()    │ Parse GGUF structure
      └──────────┬──────────┘
                 │
      ┌──────────▼──────────────────┐
      │ extract_config()             │ Metadata + inference
      │ (with shape-based fallback)  │
      └──────────┬──────────────────┘
                 │
      ┌──────────▼──────────┐
      │ load_tensors()       │ Device-aware loading
      │ (mmap or buffer)     │
      └──────────┬──────────┘
                 │
      ┌──────────▼──────────┐
      │ Validation Checks    │
      │ • LayerNorm stats    │
      │ • Offset bounds      │
      │ • Tensor presence    │
      └──────────┬──────────┘
                 │
      ┌──────────▼──────────┐
      │ build_transformer()  │ Weight remapping
      │ (remap_gguf_weights)│ Shape normalization
      └──────────┬──────────┘
                 │
      ┌──────────▼──────────┐
      │ BitNetModel::from_  │ Ready for inference
      │ gguf() returns Model│
      └──────────┬──────────┘
                 │
      ┌──────────▼──────────────┐
      │ Optional integrations:  │
      │ • Tokenizer loading     │
      │ • Template detection    │
      │ • Cross-validation      │
      └─────────────────────────┘
```

