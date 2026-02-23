# Quick Start: Per-Position Logits Parity & Tracing

**TL;DR**: Find the exact token and layer where Rust vs C++ diverge.

---

## 🚀 Quick Commands

### Find First Diverging Token

```bash
cargo run -p xtask --features inference -- crossval-per-token \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 4
```

**Output**:
```
✓ t=0 cosine=0.99998
✓ t=1 cosine=0.99997
✗ t=3 cosine=0.99231 ← First divergence
```

---

### Find First Diverging Layer (at token t=3)

```bash
# Step 1: Generate Rust traces
BITNET_TRACE_DIR=/tmp/rust cargo run -p bitnet-cli \
  --features cpu,trace -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 1

# Step 2: Generate C++ traces (implement equivalent in C++)

# Step 3: Compare
python3 scripts/trace_diff.py /tmp/rust /tmp/cpp
```

**Output**:
```
✗ First divergence at seq=3, layer=6, stage=attn_out
  Rust blake3: 407a12f3...
  C++ blake3:  19b4ce8d...
```

---

## 📊 Output Formats

### Text (Human-Readable)
```bash
--format text
```
Shows per-token metrics with ✓/✗ indicators

### JSON (CI/Automation)
```bash
--format json
```
Returns structured data:
```json
{
  "first_divergence_token": 3,
  "per_token_cosine_sim": [0.999, 0.998, 0.992],
  "status": "diverged"
}
```

---

## 🔧 Configuration

### Cosine Similarity Tolerance
```bash
--cos-tol 0.999  # Default: 0.999 (99.9% similarity)
```
Lower = more strict (0.0 = orthogonal, 1.0 = identical)

### Maximum Tokens
```bash
--max-tokens 4  # Default: 4
```
Limit generation to reduce test time

---

## 📁 Trace Files

**Location**: `$BITNET_TRACE_DIR/*.trace`

**Format**: JSONL with one record per tensor

**Example**:
```json
{
  "name": "embeddings",
  "shape": [1, 2560],
  "dtype": "F32",
  "blake3": "407a12f3abc98d12...",
  "rms": 0.998,
  "seq": 0,
  "layer": -1,
  "stage": "embeddings"
}
```

**Tracepoints**:
- `seq=0, layer=-1, stage="embeddings"` - Token embeddings
- `seq=t, layer=-2, stage="all_layers_out"` - Output after all layers
- `seq=t, layer=-1, stage="logits"` - Logits per position

---

## 🛠️ Compilation

### Enable Tracing
```bash
cargo build --features cpu,trace
```

### Enable Inference Command
```bash
cargo build -p xtask --features inference
```

### Run Tests
```bash
cargo test -p bitnet-trace --features trace
cargo test -p bitnet-models --features cpu,trace
```

---

## 🐛 Troubleshooting

### "C++ FFI not available"
```bash
# Compile with crossval feature or set BITNET_CPP_DIR
cargo build --features crossval
export BITNET_CPP_DIR=/path/to/bitnet.cpp
```

### Trace files not generated
```bash
# Ensure BITNET_TRACE_DIR is set and trace feature enabled
export BITNET_TRACE_DIR=/tmp/traces
mkdir -p /tmp/traces
cargo run --features cpu,trace -- ...
```

### "Shape mismatch" in trace_diff.py
```bash
# Check both runs used same tokenizer and prompt
# Verify both Rust and C++ traced the same inputs
```

---

## 📚 Documentation

- **Full Implementation**: `IMPLEMENTATION_SUMMARY.md`
- **Exploration Reports**: `docs/` directory
- **Trace API**: `crates/bitnet-trace/README.md`
- **Crossval Framework**: `crossval/README.md`

---

## ⚡ Performance Notes

- **Tracing overhead**: ~10-50ms per trace (I/O-bound)
- **Feature-gated**: Zero cost when `--features trace` not used
- **Selective tracing**: Use `BITNET_TRACE_SELECT` (Sprint 3) to reduce volume

---

## 🎯 Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All positions/traces match |
| `1` | Divergence found |
| `2` | Error (missing files, invalid args) |

Use in CI pipelines:
```bash
if cargo run -p xtask --features inference -- crossval-per-token ...; then
  echo "✅ Parity check passed"
else
  echo "❌ Divergence detected"
  exit 1
fi
```

---

## 🚀 Next Steps (Sprint 3 - Coming Soon)

- **BITNET_TRACE_SELECT**: Filter traces by seq/layer
- **Makefile targets**: One-command workflows
- **--dump-raw**: Export .npy files for numerical debugging
- **Tolerance control**: Configurable numeric thresholds

---

**Questions?** See `IMPLEMENTATION_SUMMARY.md` for detailed architecture and design decisions.
