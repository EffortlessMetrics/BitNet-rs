# Validation Gates Technical Reference

**Audience:** Developers implementing or extending the validation system, and advanced users needing technical details.

**Purpose:** Technical specification of the architecture-aware validation gate system for LayerNorm and projection weight validation.

---

## Overview

The BitNet.rs validation gate system provides architecture-aware statistical validation of GGUF models to detect:

- Quantized LayerNorm weights (should be F16/F32)
- Corrupted projection weight scales
- Inverted I2_S dequantization parameters
- Export format mismatches

The system uses pattern-based threshold validation with architecture-specific rulesets derived from empirical analysis of clean models.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Validation Gate System                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │ Gate Mode    │─────▶│ Ruleset      │─────▶│ Tensor   │  │
│  │ Selection    │      │ Selection    │      │ Validator│  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
│        │                      │                     │        │
│        │                      │                     │        │
│   ┌────▼────┐           ┌────▼────┐          ┌────▼────┐   │
│   │ none    │           │Built-in │          │  RMS    │   │
│   │ auto    │           │ Rules   │          │  Check  │   │
│   │ policy  │           │  YAML   │          │ Pattern │   │
│   └─────────┘           └─────────┘          │  Match  │   │
│                                               └─────────┘   │
│                                                              │
│  Exit Codes:                                                │
│    0 = EXIT_SUCCESS (all checks passed)                     │
│    8 = EXIT_LN_SUSPICIOUS (validation failed, strict mode)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Gate Mode Selection**: Determine validation strategy (`none`, `auto`, `policy`)
2. **Ruleset Loading**: Load architecture-specific thresholds
3. **Tensor Iteration**: Scan all tensors in GGUF file
4. **Pattern Matching**: Match tensor names against ruleset patterns
5. **RMS Validation**: Compare computed RMS against threshold envelope
6. **Exit Code Determination**: Return appropriate exit code based on results and strict mode

---

## Gate Modes

### Mode: `none`

**Behavior:** Skip validation entirely. Uses generic fallback ruleset with permissive envelopes.

**Ruleset:** `generic`
- LayerNorm: `[0.80, 1.20]` for all `.*norm\.weight$` patterns
- Projection: No validation

**Use Cases:**
- Debugging validation system implementation
- Testing with experimental models
- Performance benchmarking without validation overhead

**Exit Codes:**
- Always returns `0` (no validation performed)

**Example:**

```bash
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate none model.gguf
```

---

### Mode: `auto` (Default)

**Behavior:** Auto-detect architecture from GGUF metadata and select appropriate built-in ruleset.

**Detection Logic:**

```rust
pub fn detect_rules(arch: &str, file_type: u32) -> Ruleset {
    let arch_l = arch.to_ascii_lowercase();
    if arch_l.contains("bitnet") || arch_l.contains("b1.58") {
        match file_type {
            1 => rules_bitnet_b158_f16(),  // F16 clean export
            _ => rules_bitnet_b158_i2s(),  // Quantized (I2_S, etc.)
        }
    } else {
        rules_generic()  // LLaMA-style fallback
    }
}
```

**Metadata Keys:**
- `general.architecture` (string): Model architecture identifier
- `general.file_type` (u32): File type indicator
  - `1` = F16 (all weights in half precision)
  - Other values = Quantized (I2_S, Q4_0, etc.)

**Ruleset Selection Table:**

| Architecture | File Type | Ruleset | Description |
|--------------|-----------|---------|-------------|
| Contains `"bitnet"` or `"b1.58"` | `1` (F16) | `bitnet-b1.58:f16` | Clean F16 BitNet export |
| Contains `"bitnet"` or `"b1.58"` | Other | `bitnet-b1.58:i2_s` | Quantized BitNet (I2_S, etc.) |
| Other | Any | `generic` | LLaMA/Mistral/standard RMSNorm |

**Exit Codes:**
- `0`: All validations passed
- `8` (`EXIT_LN_SUSPICIOUS`): Validation failed in strict mode

**Example:**

```bash
# Auto-detect from GGUF metadata
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto model.gguf

# Or via environment variable
export BITNET_VALIDATION_GATE=auto
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
```

---

### Mode: `policy`

**Behavior:** Load custom ruleset from YAML policy file using explicit key.

**Required Arguments:**
- `--policy PATH`: Path to YAML policy file
- `--policy-key KEY`: Key in policy file (format: `architecture:variant`)

**Policy File Structure:**

```yaml
version: 1

rules:
  # Policy key format: architecture:variant
  my-model:f16:
    name: "Human-readable ruleset name"

    # LayerNorm validation rules (pattern-based)
    ln:
      - pattern: "regex_pattern_1"
        min: 0.85
        max: 1.15
        description: "Optional description"

      - pattern: "regex_pattern_2"
        min: 0.40
        max: 1.50

    # Projection weight RMS envelope (optional)
    proj_weight_rms_min: 0.015
    proj_weight_rms_max: 0.35

    notes: |
      Optional notes about this ruleset
```

**Exit Codes:**
- `0`: All validations passed
- `8` (`EXIT_LN_SUSPICIOUS`): Validation failed in strict mode
- `1`: Policy file not found or key not found

**Example:**

```bash
# Explicit policy mode
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats \
  --gate policy \
  --policy examples/policies/custom-model.yml \
  --policy-key my-model:f16 \
  model.gguf

# Or via environment variables
export BITNET_VALIDATION_GATE=policy
export BITNET_VALIDATION_POLICY=examples/policies/custom-model.yml
export BITNET_VALIDATION_POLICY_KEY=my-model:f16
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
```

---

## Built-in Rulesets

### Ruleset: `bitnet-b1.58:f16`

**Purpose:** Validation for BitNet b1.58 models exported in F16 precision (clean, unquantized).

**Characteristics:**
- All weights in F16 format
- LayerNorm gamma weights have natural RMS distribution
- FFN LayerNorm often has legitimately low RMS (~0.05-0.10)

**LayerNorm Rules:**

| Pattern | Min | Max | Description |
|---------|-----|-----|-------------|
| `ffn_layernorm\.weight$` | `0.05` | `2.0` | FFN LayerNorm (architectural low gamma) |
| `post_attention_layernorm\.weight$` | `0.25` | `2.0` | Post-attention LayerNorm |
| `input_layernorm\.weight$` | `0.35` | `2.0` | Input LayerNorm |
| `final_(layer)?norm\.weight$` | `0.50` | `2.0` | Final output norm |
| `(attn\|ffn\|rms).*norm\.weight$` | `0.50` | `2.0` | Generic attention/FFN/RMS norms |
| `.*norm\.weight$` | `0.50` | `2.0` | Fallback for any norm |

**Projection Weight Envelope:**
- **Min:** `0.01`
- **Max:** `0.40`
- **Rationale:** F16 projection weights (Q/K/V/O, FFN) typically have RMS ~0.01-0.25 after F16 export

**Implementation:**

```rust
pub fn rules_bitnet_b158_f16() -> Ruleset {
    Ruleset {
        ln: vec![
            Threshold {
                pattern: re(r"ffn_layernorm\.weight$"),
                min: 0.05,
                max: 2.0,
            },
            Threshold {
                pattern: re(r"post_attention_layernorm\.weight$"),
                min: 0.25,
                max: 2.0,
            },
            Threshold {
                pattern: re(r"input_layernorm\.weight$"),
                min: 0.35,
                max: 2.0,
            },
            Threshold {
                pattern: re(r"final_(layer)?norm\.weight$"),
                min: 0.50,
                max: 2.0,
            },
            Threshold {
                pattern: re(r"(attn|ffn|rms).*norm\.weight$"),
                min: 0.50,
                max: 2.0,
            },
            Threshold {
                pattern: re(r".*norm\.weight$"),
                min: 0.50,
                max: 2.0,
            },
        ],
        proj_weight_rms_min: Some(0.01),
        proj_weight_rms_max: Some(0.40),
        name: "bitnet-b1.58:f16".into(),
    }
}
```

**Source:** Empirical analysis of clean F16 exports from st2gguf converter.

---

### Ruleset: `bitnet-b1.58:i2_s`

**Purpose:** Validation for BitNet b1.58 models quantized to I2_S (2-bit signed).

**Characteristics:**
- Projection weights quantized to I2_S
- LayerNorm weights should remain in F16/F32 (not quantized)
- Attention norm RMS legitimately drops to ~0.01-0.02 after quantization side effects
- FFN norm should remain close to 1.0

**LayerNorm Rules:**

| Pattern | Min | Max | Description |
|---------|-----|-----|-------------|
| `attn_norm\.weight$` | `0.01` | `2.0` | Attention norm (low RMS is legitimate) |
| `ffn_norm\.weight$` | `0.50` | `2.0` | FFN norm (should stay near 1.0) |
| `final_(layer)?norm\.weight$` | `0.50` | `2.0` | Final output norm |
| `.*norm\.weight$` | `0.25` | `2.0` | Fallback for any norm |

**Projection Weight Envelope:**
- **Min:** `0.002`
- **Max:** `0.20`
- **Rationale:** I2_S dequantization produces smaller RMS values (~0.002-0.10 typical)

**Implementation:**

```rust
pub fn rules_bitnet_b158_i2s() -> Ruleset {
    Ruleset {
        ln: vec![
            Threshold {
                pattern: re(r"attn_norm\.weight$"),
                min: 0.01,
                max: 2.0,
            },
            Threshold {
                pattern: re(r"ffn_norm\.weight$"),
                min: 0.50,
                max: 2.0,
            },
            Threshold {
                pattern: re(r"final_(layer)?norm\.weight$"),
                min: 0.50,
                max: 2.0,
            },
            Threshold {
                pattern: re(r".*norm\.weight$"),
                min: 0.25,
                max: 2.0,
            },
        ],
        proj_weight_rms_min: Some(0.002),
        proj_weight_rms_max: Some(0.20),
        name: "bitnet-b1.58:i2_s".into(),
    }
}
```

**Source:** Empirical analysis of Microsoft BitNet I2_S GGUF models.

**Important Note:** The low attn_norm RMS (~0.01-0.02) in I2_S models is **expected and legitimate**. This is not corruption. If you see this pattern, verify your model is actually I2_S quantized before flagging as error.

---

### Ruleset: `generic`

**Purpose:** Fallback validation for standard RMSNorm transformers (LLaMA, Mistral, etc.).

**Characteristics:**
- Standard RMSNorm with gamma weights near 1.0
- No architectural quirks (ffn_norm follows same pattern as attn_norm)
- Conservative envelope suitable for most standard architectures

**LayerNorm Rules:**

| Pattern | Min | Max | Description |
|---------|-----|-----|-------------|
| `.*norm\.weight$` | `0.80` | `1.20` | All LayerNorm weights (standard RMSNorm) |

**Projection Weight Envelope:**
- **Min:** None (no validation)
- **Max:** None (no validation)
- **Rationale:** Projection RMS varies widely across architectures; no universal threshold

**Implementation:**

```rust
pub fn rules_generic() -> Ruleset {
    Ruleset {
        ln: vec![Threshold {
            pattern: re(r".*norm\.weight$"),
            min: 0.80,
            max: 1.20,
        }],
        proj_weight_rms_min: None,
        proj_weight_rms_max: None,
        name: "generic".into(),
    }
}
```

**Source:** Standard RMSNorm behavior observed in LLaMA family models.

---

## Validation Algorithm

### LayerNorm Validation

**Step 1: Tensor Identification**

```rust
use bitnet_models::names::is_layernorm_weight;

for tensor in gguf_reader.tensors() {
    if is_layernorm_weight(&tensor.name) {
        // This is a LayerNorm gamma tensor
        validate_layernorm(&tensor, &ruleset);
    }
}
```

**Step 2: RMS Computation**

```rust
fn compute_rms(tensor: &Tensor) -> Result<f32> {
    // Convert to F32 for reliable statistics
    let t32 = tensor.to_dtype(DType::F32)?;

    // Compute mean of squares
    let mean_sq = t32.sqr()?.mean_all()?.to_scalar::<f32>()?;

    // Return square root (RMS)
    Ok(mean_sq.sqrt())
}
```

**Mathematical Definition:**

$$
\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}
$$

For LayerNorm gamma weights initialized near 1.0, RMS ≈ 1.0 is expected.

**Step 3: Pattern Matching**

```rust
fn check_ln(&self, name: &str, rms: f32) -> bool {
    for threshold in &self.ln {
        if threshold.pattern.is_match(name) {
            return rms >= threshold.min && rms <= threshold.max;
        }
    }
    // No match => best-effort generic envelope
    rms >= 0.50 && rms <= 2.0
}
```

**Pattern Priority:**
1. Check patterns in ruleset order (first match wins)
2. If no pattern matches, use fallback envelope `[0.50, 2.0]`

**Step 4: Result Aggregation**

```rust
let mut ln_bad_count = 0;
let mut ln_total_count = 0;

for ln_tensor in ln_tensors {
    ln_total_count += 1;
    let is_ok = ruleset.check_ln(&ln_tensor.name, ln_tensor.rms);
    if !is_ok {
        ln_bad_count += 1;
    }
}
```

---

### Projection Weight Validation

**Step 1: Tensor Identification**

```rust
use bitnet_models::names::is_projection_weight;

for tensor in gguf_reader.tensors() {
    if is_projection_weight(&tensor.name) {
        // This is a projection weight (Q/K/V/O, FFN gate/up/down)
        validate_projection(&tensor, &ruleset);
    }
}
```

**Step 2: Type Filtering**

```rust
// Only validate RMS for float tensors
if !matches!(tensor.tensor_type, GgufTensorType::F32 | GgufTensorType::F16) {
    // Skip quantized tensors (I2_S, Q4, etc.)
    continue;
}
```

**Rationale:** Quantized projection weights are expected (e.g., I2_S models). RMS validation only applies to float weights where corruption would manifest as unusual RMS values.

**Step 3: RMS Validation**

```rust
fn check_proj_rms(&self, rms: f32) -> bool {
    match (self.proj_weight_rms_min, self.proj_weight_rms_max) {
        (Some(min), Some(max)) => rms >= min && rms <= max,
        _ => true, // No validation (no opinion)
    }
}
```

**Step 4: Result Aggregation**

```rust
let mut proj_bad_count = 0;
let mut proj_total_count = 0;

for proj_tensor in proj_tensors {
    proj_total_count += 1;
    let is_ok = ruleset.check_proj_rms(proj_tensor.rms);
    if !is_ok {
        proj_bad_count += 1;
    }
}
```

---

## Exit Code Handling

### Exit Code: `0` (Success)

**Condition:** All validation checks passed, or strict mode is disabled.

**Behavior:**
- All LayerNorm RMS values within envelope
- All projection RMS values within envelope (if ruleset defines envelope)
- Process exits with code `0`

**Example:**

```bash
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
echo $?  # Output: 0
```

---

### Exit Code: `8` (Suspicious LayerNorm)

**Name:** `EXIT_LN_SUSPICIOUS`

**Condition:** One or more LayerNorm or projection weights failed validation **and** strict mode is enabled.

**Strict Mode Activation:**

```bash
# Via environment variable
BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
echo $?  # Output: 8 (if validation fails)

# Check in Rust code
let strict_mode = std::env::var("BITNET_STRICT_MODE")
    .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
    .unwrap_or(false);

if total_bad > 0 && strict_mode {
    std::process::exit(EXIT_LN_SUSPICIOUS);
}
```

**Use Cases:**
- CI/CD pipelines requiring zero-tolerance validation
- Production qualification gates
- Release validation workflows

**Example CI Check:**

```bash
BITNET_STRICT_MODE=1 ./scripts/validate_gguf.sh model.gguf tokenizer.json
if [ $? -eq 8 ]; then
  echo "ERROR: Model has suspicious LayerNorm weights"
  echo "Regenerate GGUF with float LayerNorm weights"
  exit 1
fi
```

---

## Pattern Syntax

### Regex Patterns

Validation rules use Rust regex syntax for pattern matching:

```yaml
ln:
  - pattern: "attn_norm\\.weight$"  # Literal dot, end of string
  - pattern: "blk\\.[0-9]+\\..*"    # Layer prefix with number
  - pattern: "final_(layer)?norm"   # Optional "layer" group
  - pattern: "(attn|ffn)_norm"      # Alternation
```

**Common Patterns:**

| Pattern | Description | Example Matches |
|---------|-------------|-----------------|
| `attn_norm\.weight$` | Attention norm weights (exact suffix) | `blk.0.attn_norm.weight` |
| `ffn.*norm\.weight$` | FFN norm weights (any middle part) | `blk.0.ffn_layernorm.weight` |
| `final_norm\.weight$` | Final norm (no layer) | `output_norm.weight`, `final_norm.weight` |
| `blk\.[0-9]+\.` | Any layer tensor | `blk.0.attn_q.weight`, `blk.15.ffn_gate.weight` |
| `.*norm\.weight$` | Any norm weight (fallback) | `blk.0.attn_norm.weight`, `custom_norm.weight` |

**Pattern Priority:**

Patterns are evaluated in order. First match determines the threshold:

```yaml
ln:
  # Specific pattern (checked first)
  - pattern: "ffn_layernorm\\.weight$"
    min: 0.05
    max: 2.0

  # Generic pattern (checked last)
  - pattern: ".*norm\\.weight$"
    min: 0.50
    max: 2.0
```

If `blk.0.ffn_layernorm.weight` is checked:
1. Matches first pattern → use `[0.05, 2.0]`
2. Second pattern is not evaluated

---

## Threshold Derivation

### Empirical Analysis Methodology

**Step 1: Collect Clean Models**

```bash
# Export multiple clean F16 GGUFs from same architecture
for checkpoint in checkpoint_*.safetensors; do
  cargo run --release -p bitnet-st2gguf -- \
    --input "$checkpoint" \
    --output "clean_$(basename $checkpoint .safetensors).gguf"
done
```

**Step 2: Extract RMS Statistics**

```bash
# Inspect each model
for model in clean_*.gguf; do
  cargo run -p bitnet-cli -- inspect --ln-stats --json "$model" \
    > "stats_$(basename $model .gguf).json"
done
```

**Step 3: Aggregate Statistics**

```python
import json
import numpy as np
from collections import defaultdict

stats_by_pattern = defaultdict(list)

for stats_file in stats_files:
    with open(stats_file) as f:
        data = json.load(f)
        for tensor in data['tensors']:
            if tensor['kind'] == 'layernorm':
                # Group by suffix pattern
                name = tensor['name']
                if 'attn_norm' in name:
                    pattern = 'attn_norm'
                elif 'ffn' in name:
                    pattern = 'ffn_norm'
                else:
                    pattern = 'other_norm'

                stats_by_pattern[pattern].append(float(tensor['rms']))

# Compute min/max with safety margin
for pattern, rms_values in stats_by_pattern.items():
    observed_min = np.min(rms_values)
    observed_max = np.max(rms_values)

    # Add 10% safety margin
    policy_min = observed_min * 0.90
    policy_max = observed_max * 1.10

    print(f"{pattern}:")
    print(f"  Observed: [{observed_min:.3f}, {observed_max:.3f}]")
    print(f"  Policy:   [{policy_min:.3f}, {policy_max:.3f}]")
```

**Step 4: Define Policy**

```yaml
version: 1

rules:
  architecture:variant:
    name: "Architecture Variant"
    ln:
      # Use policy min/max from step 3
      - pattern: "attn_norm\\.weight$"
        min: 0.85  # policy_min
        max: 1.15  # policy_max
        description: "Derived from empirical analysis (observed [0.92, 1.05])"
```

---

### Safety Margin Guidelines

**5-10% Margin:**

Most architectures should use 5-10% margin beyond observed min/max:

```
policy_min = observed_min * 0.95  # 5% looser
policy_max = observed_max * 1.05  # 5% looser
```

**Stricter for Critical Layers:**

Final output norms should have tighter envelopes (2-3% margin):

```
# Final norm is critical for stability
- pattern: "final_norm\\.weight$"
  min: 0.98  # observed_min * 0.98
  max: 1.02  # observed_max * 1.02
```

**Looser for Variable Layers:**

FFN LayerNorm with architectural low gamma may need wider envelope:

```
# FFN LayerNorm legitimately has low gamma
- pattern: "ffn.*norm\\.weight$"
  min: 0.05  # observed_min * 0.50 (50% looser)
  max: 2.00  # observed_max * 2.00 (100% looser)
```

---

## Environment Variables

### `BITNET_VALIDATION_GATE`

**Values:** `none`, `auto`, `policy`

**Default:** `auto`

**Description:** Validation gate mode. Overrides `--gate` CLI argument.

**Example:**

```bash
export BITNET_VALIDATION_GATE=auto
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
```

---

### `BITNET_VALIDATION_POLICY`

**Values:** Path to YAML policy file

**Default:** None

**Description:** Policy file path for `gate=policy` mode.

**Example:**

```bash
export BITNET_VALIDATION_POLICY=examples/policies/custom.yml
export BITNET_VALIDATION_POLICY_KEY=my-model:f16
export BITNET_VALIDATION_GATE=policy
cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
```

---

### `BITNET_VALIDATION_POLICY_KEY`

**Values:** String (format: `architecture:variant`)

**Default:** Uses `general.architecture` from GGUF metadata

**Description:** Policy key for rules lookup in YAML file.

**Example:**

```bash
export BITNET_VALIDATION_POLICY_KEY=bitnet-b1.58:f16
```

---

### `BITNET_STRICT_MODE`

**Values:** `0`, `1`, `true`, `false`, `yes`, `no`, `on`, `off`

**Default:** `0` (disabled)

**Description:** Enable strict validation. When enabled, validation failures cause non-zero exit code (`EXIT_LN_SUSPICIOUS=8`).

**Example:**

```bash
BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --ln-stats model.gguf
if [ $? -ne 0 ]; then
  echo "Validation failed in strict mode"
fi
```

---

## Implementation Details

### File Locations

| Component | Path |
|-----------|------|
| Main validation logic | `crates/bitnet-cli/src/commands/inspect.rs` |
| Ruleset definitions | `crates/bitnet-cli/src/ln_rules.rs` |
| Exit code constants | `crates/bitnet-cli/src/exit.rs` |
| Tensor name utilities | `crates/bitnet-models/src/names.rs` |
| GGUF reader | `crates/bitnet-models/src/formats/gguf/reader.rs` |

---

### Key Data Structures

**`Threshold`:**

```rust
pub struct Threshold {
    pub pattern: Regex,  // Regex for tensor name matching
    pub min: f32,        // Minimum acceptable RMS
    pub max: f32,        // Maximum acceptable RMS
}
```

**`Ruleset`:**

```rust
pub struct Ruleset {
    pub ln: Vec<Threshold>,              // LayerNorm validation rules
    pub proj_weight_rms_min: Option<f32>, // Projection RMS min (None = skip)
    pub proj_weight_rms_max: Option<f32>, // Projection RMS max (None = skip)
    pub name: String,                     // Human-readable ruleset name
}
```

**`TensorStat`:**

```rust
struct TensorStat {
    name: String,      // Tensor name (e.g., "blk.0.attn_norm.weight")
    rms: f32,          // Computed RMS value
    is_ok: bool,       // Within envelope?
    kind: TensorKind,  // LayerNorm or Projection
}
```

---

### RMS Computation Implementation

```rust
fn compute_rms(tensor: &Tensor) -> Result<f32> {
    // Convert to F32 for reliable statistics
    let t32 = tensor
        .to_dtype(DType::F32)
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    // Compute mean of squared values
    let mean_sq = t32
        .sqr()
        .map_err(|e| BitNetError::Validation(e.to_string()))?
        .mean_all()
        .map_err(|e| BitNetError::Validation(e.to_string()))?
        .to_scalar::<f32>()
        .map_err(|e| BitNetError::Validation(e.to_string()))?;

    // Return square root (RMS)
    Ok(mean_sq.sqrt())
}
```

**Numerical Considerations:**

- **Precision:** Always compute in F32, even if tensor is F16
- **Stability:** Use `sqr()` → `mean()` → `sqrt()` (numerically stable)
- **Edge Cases:** Handle empty tensors (return error) and NaN values (propagate error)

---

## Testing and Validation

### Unit Tests

**Test coverage:**

1. **Ruleset selection:** `detect_rules()` returns correct ruleset for each architecture
2. **Pattern matching:** Regex patterns match expected tensor names
3. **RMS validation:** `check_ln()` and `check_proj_rms()` enforce thresholds correctly
4. **Exit codes:** Strict mode returns correct exit codes

**Example test:**

```rust
#[test]
fn test_bitnet_f16_ruleset() {
    let rules = rules_bitnet_b158_f16();

    // FFN LayerNorm: low RMS is OK
    assert!(rules.check_ln("blk.0.ffn_layernorm.weight", 0.08));

    // Attention norm: should be near 1.0
    assert!(rules.check_ln("blk.0.attn_norm.weight", 0.95));
    assert!(!rules.check_ln("blk.0.attn_norm.weight", 0.02));  // Too low
}
```

---

### Integration Tests

**Test clean models:**

```bash
# Test against known-good F16 model
cargo run -p bitnet-cli -- inspect --ln-stats --gate auto \
  tests/fixtures/clean-bitnet-f16.gguf

# Should output:
# ✅ LN RMS gate passed (bitnet-b1.58:f16)
```

**Test known-bad models:**

```bash
# Test against model with quantized LayerNorm
BITNET_STRICT_MODE=1 \
  cargo run -p bitnet-cli -- inspect --ln-stats --gate auto \
  tests/fixtures/bad-bitnet-quantized-ln.gguf

# Should output:
# ❌ LN RMS gate failed: 24/24 out of envelope
# Exit code: 8
```

---

### CI Integration

**Example GitHub Actions workflow:**

```yaml
- name: Validate GGUF Models
  run: |
    for model in tests/fixtures/*.gguf; do
      echo "Validating $model"
      BITNET_STRICT_MODE=1 \
        cargo run -p bitnet-cli -- inspect --ln-stats --gate auto "$model"

      if [ $? -ne 0 ]; then
        echo "ERROR: Validation failed for $model"
        exit 1
      fi
    done
```

---

## Performance Considerations

### Computational Cost

**RMS computation per tensor:**

```
O(n) where n = tensor element count

Typical LayerNorm tensor: 2560 elements (hidden_dim)
Typical projection tensor: 2560 × 2560 = 6.5M elements

RMS computation: ~0.01ms (LayerNorm), ~10ms (projection)
```

**Total validation time:**

```
BitNet b1.58 2B model:
- 24 layers × 2 LN tensors/layer = 48 LayerNorm tensors
- 24 layers × 7 proj tensors/layer = 168 projection tensors (F16 only)

Total RMS computations: ~50 LayerNorm + ~20 F16 projections
Validation time: ~0.5ms + ~200ms = ~200ms total
```

**Optimization opportunities:**

1. **Skip quantized tensors:** Only compute RMS for F16/F32 weights
2. **Parallel computation:** Use Rayon for tensor iteration (future work)
3. **Cached results:** Memoize RMS for repeated validation (future work)

---

### Memory Usage

**Peak memory:**

```
Single tensor RMS computation:
- F32 conversion: tensor_size × 4 bytes
- Intermediate squared tensor: tensor_size × 4 bytes
- Total: tensor_size × 8 bytes

Largest tensor (projection): 2560 × 2560 × 8 = ~52 MB
```

**Memory optimization:**

- Tensors are validated sequentially (not loaded into memory simultaneously)
- F32 conversions are temporary (freed after RMS computation)
- Total memory overhead: ~100 MB peak (negligible)

---

## Future Extensions

### Planned Features

1. **Dynamic threshold learning:**
   - Auto-generate policies from clean model corpus
   - Machine learning-based anomaly detection

2. **Cross-layer consistency checks:**
   - Verify RMS is consistent across layers
   - Detect layer-specific corruption

3. **Tensor content validation:**
   - Check for NaN/Inf values
   - Validate weight magnitude distribution

4. **Performance profiling:**
   - Report validation time per tensor
   - Identify slow validation steps

5. **Policy versioning:**
   - Support multiple policy versions in single file
   - Backward compatibility with older policy formats

---

## Receipt Honesty Validation (Issue #453)

BitNet.rs extends the validation gate system to include receipt honesty validation, ensuring inference receipts accurately reflect the actual computation paths used. This prevents false performance claims and enables trustworthy baselines.

### Receipt Validation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Receipt Honesty Validation                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │ Schema       │─────▶│ Kernel ID    │─────▶│ Compute  │  │
│  │ Validation   │      │ Matching     │      │ Path     │  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
│        │                      │                     │        │
│        │                      │                     │        │
│   ┌────▼────┐           ┌────▼────┐          ┌────▼────┐   │
│   │ v1.0.0  │           │Quantized│          │ "real"  │   │
│   │ Fields  │           │ Kernels │          │ Claims  │   │
│   └─────────┘           └─────────┘          └─────────┘   │
│                                                              │
│  Exit Codes:                                                │
│    0 = Receipt validation passed                            │
│    1 = Receipt validation failed (false claims detected)    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Receipt Schema v1.0.0

Receipts generated by BitNet.rs include the following fields for validation:

```json
{
  "schema_version": "1.0.0",
  "backend": "cpu" | "cuda",
  "compute_path": "real" | "fallback" | "mock",
  "kernels": ["kernel_id_1", "kernel_id_2", ...],
  "tokens_per_second": 18.5,
  "tokens_generated": 128,
  "environment": {
    "BITNET_STRICT_MODE": "1",
    "BITNET_DETERMINISTIC": "1"
  },
  "timestamp": "2025-10-14T12:34:56.789Z"
}
```

### Kernel ID Naming Conventions

**Quantized Kernel IDs (Native 1/2-bit Arithmetic):**

| Device | Quantization | Pattern | Examples |
|--------|--------------|---------|----------|
| GPU | I2S | `gemm_*`, `i2s_gpu_*`, `wmma_*` | `gemm_fp16`, `i2s_gpu_quantize`, `wmma_matmul` |
| CPU | I2S | `i2s_gemv`, `i2s_matmul_*`, `quantized_matmul_i2s` | `i2s_gemv`, `quantized_matmul_i2s` |
| CPU (ARM) | TL1 | `tl1_neon_*`, `tl1_lookup_*` | `tl1_neon_matmul`, `tl1_lookup` |
| CPU (x86) | TL2 | `tl2_avx_*`, `tl2_avx512_*` | `tl2_avx_matmul`, `tl2_avx512_pack` |

**Fallback Kernel IDs (FP32 Dequantization):**

| Pattern | Meaning | Examples |
|---------|---------|----------|
| `dequant_*` | Dequantization to FP32 | `dequant_fp32`, `dequant_i2s_to_fp32` |
| `fp32_*` | FP32 computation | `fp32_matmul`, `fp32_gemm` |
| `fallback_*` | Generic fallback path | `fallback_compute`, `fallback_matmul` |
| `scalar_*` | Scalar (non-SIMD) fallback | `scalar_matmul`, `scalar_quantization` |
| `mock_*` | Mock/test stub | `mock_kernel`, `mock_inference` |

### Receipt Validation Rules

**Rule 1: Schema Validation**

- `schema_version` must be `"1.0.0"` or compatible
- All required fields must be present: `backend`, `compute_path`, `kernels`, `tokens_per_second`
- `kernels` array must be non-empty

**Rule 2: Compute Path Correlation**

- `compute_path="real"` requires ≥1 quantized kernel ID
- `compute_path="fallback"` may have fallback kernel IDs
- `compute_path="mock"` is rejected in strict mode

**Rule 3: Backend Correlation**

- `backend="cuda"` receipts must have GPU kernel IDs (not CPU kernels)
- `backend="cpu"` receipts must have CPU kernel IDs

**Rule 4: Performance Realism**

- `tokens_per_second` must be within realistic range for device and quantization type
- Values >150 tok/s flagged as suspicious (potential mock computation)

**Rule 5: Kernel ID Hygiene**

- Kernel IDs must be non-empty strings
- Kernel ID length ≤128 characters
- Total kernel count ≤10,000 (prevents abuse)

### Validation Commands

**Basic Receipt Validation:**

```bash
# Validate receipt schema and basic honesty
cargo run -p xtask -- verify-receipt ci/inference.json

# Expected output:
# ✓ Schema version: 1.0.0 (valid)
# ✓ Required fields present
# ✓ Compute path: real (valid)
# ✓ Backend: cpu (valid)
# ✓ Kernel count: 2 kernels
# ✓ Receipt validation: PASS
```

**Quantized Kernel Validation:**

```bash
# Require quantized kernels for "real" claims
cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json

# Expected output:
# ✓ Schema validation: PASS
# ✓ Kernel validation: 2 quantized kernels detected
#   - i2s_gemv (CPU quantized matmul)
#   - quantized_matmul_i2s (CPU quantized matmul)
# ✓ Compute path validation: "real" correlates with quantized kernels
# ✓ Fallback detection: No fallback indicators found
# ✓ Receipt validation: PASS
```

**GPU Kernel Validation:**

```bash
# Require GPU kernels for GPU backend claims
cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/inference.json

# Expected output (success):
# ✓ GPU kernel validation: 3 GPU kernels detected
#   - gemm_fp16 (GPU mixed precision matmul)
#   - i2s_gpu_quantize (GPU quantization)
#   - wmma_matmul (Tensor Core acceleration)
# ✓ Backend correlation: "cuda" matches GPU kernels
# ✓ Receipt validation: PASS

# Expected output (failure - silent CPU fallback):
# ✗ GPU kernel validation: FAIL
# Error: Receipt claims backend="cuda" but no GPU kernels detected
# Found CPU kernels: ["i2s_gemv", "quantized_matmul_i2s"]
# This indicates silent fallback from GPU to CPU occurred.
```

**Performance Metrics Validation:**

```bash
# Validate performance metrics for realism
cargo run -p xtask -- verify-receipt --validate-performance ci/inference.json

# Expected output (success):
# ✓ Performance validation: PASS
# tokens_per_second: 18.5 (within realistic range for CPU I2S: 10-20 tok/s)

# Expected output (failure - suspicious performance):
# ✗ Performance validation: FAIL
# Error: Suspicious performance detected: 250.0 tok/s (threshold: 150.0)
# CPU inference claiming 250 tok/s is unrealistic. This suggests mock inference.
```

### Receipt Validation Integration

**CI/CD Pipeline:**

```yaml
# .github/workflows/receipt-validation.yml
- name: Run benchmark with strict mode
  env:
    BITNET_STRICT_MODE: "1"
    BITNET_DETERMINISTIC: "1"
    BITNET_SEED: "42"
  run: cargo run -p xtask -- benchmark --model model.gguf --tokens 128

- name: Verify receipt schema
  run: cargo run -p xtask -- verify-receipt ci/inference.json

- name: Verify quantized kernels
  run: cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json

- name: Verify performance metrics
  run: cargo run -p xtask -- verify-receipt --validate-performance ci/inference.json

- name: Check for fallback indicators
  run: |
    if jq -e '.kernels[] | select(contains("dequant") or contains("fp32_") or contains("fallback_"))' ci/inference.json; then
      echo "ERROR: Fallback kernels detected"
      exit 1
    fi
```

**Programmatic Usage:**

```rust
use bitnet_common::strict_mode::StrictModeEnforcer;

// Validate receipt honesty programmatically
let receipt = load_receipt("ci/inference.json")?;
let enforcer = StrictModeEnforcer::new();

// Validate performance metrics
enforcer.validate_performance_metrics(&receipt.performance)?;

// Validate kernel IDs match compute_path claim
verify_quantization_claims(&receipt)?;

// Validate GPU claims have GPU kernel IDs
if receipt.backend == "cuda" {
    verify_gpu_kernels(&receipt.kernels)?;
}
```

### Exit Codes for Receipt Validation

| Code | Name | Condition | Use Case |
|------|------|-----------|----------|
| `0` | `EXIT_SUCCESS` | Receipt validation passed | Normal success |
| `1` | `EXIT_GENERIC_FAIL` | Receipt validation failed | False claims detected |
| `8` | `EXIT_LN_SUSPICIOUS` | Model validation failed | Model has suspicious weights |

### Common Receipt Validation Failures

**Failure 1: False Quantization Claims**

```json
{
  "compute_path": "real",  // ← Claims quantized computation
  "kernels": ["dequant_fp32", "fp32_matmul"]  // ← But uses fallback!
}
```

**Detection:**
```bash
cargo run -p xtask -- verify-receipt --require-quantized-kernels ci/inference.json

# Error: Receipt claims compute_path="real" but kernels contain only fallback indicators
```

**Failure 2: Silent CPU Fallback on GPU**

```json
{
  "backend": "cuda",  // ← Claims GPU
  "kernels": ["i2s_gemv", "quantized_matmul_i2s"]  // ← But uses CPU kernels!
}
```

**Detection:**
```bash
cargo run -p xtask -- verify-receipt --require-gpu-kernels ci/inference.json

# Error: Receipt claims backend="cuda" but no GPU kernels detected
```

**Failure 3: Suspicious Performance**

```json
{
  "backend": "cpu",
  "kernels": ["i2s_gemv"],
  "tokens_per_second": 250.0  // ← Unrealistic for CPU!
}
```

**Detection:**
```bash
cargo run -p xtask -- verify-receipt --validate-performance ci/inference.json

# Error: Suspicious performance detected: 250.0 tok/s (threshold: 150.0)
```

### Implementation Files

| Component | Path |
|-----------|------|
| Receipt verification logic | `xtask/src/main.rs` (verify_receipt_cmd) |
| Kernel ID pattern matching | `xtask/src/main.rs` (is_quantized_kernel, is_fallback_kernel) |
| Strict mode enforcer | `crates/bitnet-common/src/strict_mode.rs` |
| Receipt schema types | `crates/bitnet-inference/src/receipts.rs` |
| Test fixtures | `crates/bitnet-inference/tests/strict_quantization_test.rs` |

## Parity Validation (Dual I2_S Flavor Support)

BitNet.rs validates correctness through systematic comparison with C++ reference implementation.

### Parity Validation Architecture

**Parity Harness Components:**

```
┌────────────────────────────────────────────────────────────┐
│              Parity Validation System                      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐     ┌──────────┐  │
│  │ Load Model   │─────▶│ Detect I2S   │────▶│ Route to │  │
│  │ (Rust)       │      │ Flavor       │     │ Kernel   │  │
│  └──────────────┘      └──────────────┘     └──────────┘  │
│        │                      │                     │       │
│   Rust Tokenizer        BitNet32F16         Rust/FFI      │
│   Auto-discovery        or QK256NoScale     Selection      │
│                                                   │         │
│                                            ┌────▼─────┐   │
│                                            │ Compute   │   │
│                                            │ Logits    │   │
│                                            └────┬─────┘   │
│                                                 │          │
│                                          ┌──────▼─────┐   │
│                                          │ Calculate  │   │
│                                          │ Parity     │   │
│                                          │ Metrics    │   │
│                                          └────┬─────┘   │
│                                               │         │
│                                        ┌──────▼─────┐   │
│                                        │ Receipt    │   │
│                                        │ Generation │   │
│                                        └────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────────┘
```

### Parity Receipt Schema v1.1.0

```json
{
  "schema_version": "1.1.0",
  "validation": {
    "backend": "rust | cpp_ffi",
    "crossval_source": "rust | cpp_ffi",
    "i2s_flavor_detected": "BitNet32F16 | GgmlQk256NoScale | mixed",
    "scale_tensor_present": true,
    "tokenizer": "rust",
    "compute": "rust | cpp_ffi"
  },
  "parity": {
    "cpp_available": true,
    "cosine_similarity": 0.9923,
    "exact_match_rate": 1.0,
    "max_logit_diff": 0.0001234,
    "status": "ok"
  },
  "compute_path": "real",
  "kernels": [
    "i2s_qk256_scalar",
    "quantized_matmul_i2s",
    "attention_kv_cache_update"
  ],
  "tensors": [
    {
      "name": "layers.0.attention.q_proj.weight",
      "qtype": "I2_S",
      "flavor": "GgmlQk256NoScale",
      "blocks": 256,
      "block_size": 256,
      "has_scales": true,
      "kernel_id": "i2s_qk256_scalar"
    }
  ],
  "timestamp": "2025-10-17T12:00:00Z"
}
```

### Parity Metrics Validation

| Metric | Target | Meaning | Command |
|--------|--------|---------|---------|
| **Cosine Similarity** | ≥ 0.99 | Logit vector alignment | `cargo run -p xtask -- crossval --metric cosine` |
| **Exact Match Rate** | = 1.0 | Greedy decode token match (N=4) | `cargo run -p xtask -- crossval --metric exact-match` |
| **Max Logit Diff** | < 1e-4 | Largest per-token divergence | `cargo run -p xtask -- crossval --metric max-diff` |
| **Runtime Latency** | < 110% of C++ | Relative performance | `cargo run -p xtask -- crossval --metric latency` |

### Parity Validation Commands

**One-Command Smoke Test:**

```bash
# Validates both BitNet32F16 and QK256 formats
scripts/parity_smoke.sh models/model.gguf

# Expected output:
# ✓ Rust tokenizer parity: PASS
# ✓ BitNet32F16 logits: PASS (cosine=0.9999)
# ✓ QK256 logits: PASS (cosine=0.9923)
# ✓ Greedy decode match: 100% (4/4 tokens)
```

**Full Cross-Validation with Receipts:**

```bash
# Set C++ reference path for FFI validation
export BITNET_CPP_DIR=/path/to/BitNet.cpp

# Run cross-validation with deterministic mode
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Cross-validate with receipt generation
cargo run -p xtask -- crossval --model models/model.gguf --tokens 128

# Verify receipt metrics
cargo run -p xtask -- verify-receipt ci/inference.json
```

**Per-Flavor Validation:**

```bash
# Test BitNet32F16 format
cargo test -p bitnet-models --no-default-features --features "cpu,crossval" \
  test_i2s_bitnet32_parity -- --nocapture

# Test QK256 format
cargo test -p bitnet-models --no-default-features --features "cpu,crossval" \
  test_i2s_qk256_parity -- --nocapture
```

### Flavor Detection Impact on Parity

**BitNet32F16 (Existing Format):**
- Block size: 32 elements
- Scales: Inline F16 (2 bytes per block)
- Parity: Direct comparison with C++ BitNet implementation
- Status: Mature (100% parity, <5% latency variance)

**QK256 (GGML Format - MVP):**
- Block size: 256 elements (QK_K)
- Scales: Separate F32 tensor
- Parity: FFI session routes to C++ for Phase 1 validation
- Status: MVP (scalar kernels), parity ≥ 0.99 cosine similarity
- Kernel IDs: `i2s_qk256_scalar` (Phase 1), `i2s_qk256_avx2`/`i2s_qk256_neon` (Phase 2)

**Mixed Flavor Models:**
- Receipts track detected flavors (`"i2s_flavor_detected": "mixed"`)
- Each tensor mapped to appropriate kernel
- Parity calculated per-flavor then aggregated

### Production vs Validation Paths

**Production Code** (default builds):
- Fail-closed on unsupported flavors
- No FFI routing (100% Rust)
- Strict mode prevents FP32 fallback

**Parity Validation** (with `BITNET_CPP_DIR` set):
- Routes ggml I2_S to C++ FFI when Rust kernel unavailable
- Tokenizer always Rust (for determinism)
- Enables incremental validation before Phase 2 completion

### Exit Codes

| Code | Condition | Meaning |
|------|-----------|---------|
| 0 | Parity metrics pass | All flavors validated successfully |
| 1 | Cosine < 0.99 | Logit divergence exceeds threshold |
| 2 | Exact match < 100% | Greedy decode tokens diverged |
| 4 | Latency > 110% of C++ | Performance regression detected |
| 8 | Flavor detection failed | I2_S format not recognized |

## Related Documentation

- **Tutorial:** [Getting Started with Strict Mode](../tutorials/strict-mode-quantization-validation.md) - Learning-oriented introduction
- **How-To:** [Running Strict Mode Validation Workflows](../how-to/strict-mode-validation-workflows.md) - Problem-oriented workflows
- **How-To:** [Verifying Receipt Honesty](../how-to/receipt-verification.md) - Detailed receipt validation guide
- **How-To:** [Use QK256 Models](../howto/use-qk256-models.md) - QK256 GGML format usage guide
- **Reference:** [Quantization Support](./quantization-support.md#strict-quantization-guards) - Strict mode technical details
- **Reference:** [Quantization Support - I2S QK256](./quantization-support.md#i2s-qk256ggml---pure-rust-production-ready) - QK256 format specification
- **Reference:** [Environment Variables](../environment-variables.md#strict-mode-variables) - Complete variable documentation
- **Explanation:** [Strict Quantization Guards Specification](../explanation/strict-quantization-guards.md) - Complete feature specification
- **Explanation:** [I2_S Dual-Flavor Architecture](../explanation/i2s-dual-flavor.md) - Detailed dual-flavor design
- **[Validation Workflow Guide](../howto/validate-models.md)**: User-facing validation documentation
- **[Export Clean GGUF Guide](../howto/export-clean-gguf.md)**: How to create clean models
- **[Correction Policy Documentation](../explanation/correction-policy.md)**: Runtime correction system
- **[Policy Examples](../../examples/policies/README.md)**: Example policies and creation guide
- **[LayerNorm Rules Implementation](../../crates/bitnet-cli/src/ln_rules.rs)**: Source code

---

## References

### Academic References

- **RMSNorm:** Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
- **BitNet:** Wang et al. (2023), "BitNet: Scaling 1-bit Transformers for Large Language Models"
- **GGUF Format:** [ggml-org/gguf](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

### Implementation References

- **Rust regex crate:** [regex](https://docs.rs/regex)
- **Candle tensor library:** [candle-core](https://docs.rs/candle-core)
- **GGUF reader:** `crates/bitnet-models/src/formats/gguf/reader.rs`
- **Receipt verification:** `xtask/src/main.rs` (verify_receipt_cmd)
- **Strict mode enforcer:** `crates/bitnet-common/src/strict_mode.rs`

---

## Appendix: Exit Code Summary

| Code | Name | Condition | Use Case |
|------|------|-----------|----------|
| `0` | `EXIT_SUCCESS` | All validations passed | Normal success |
| `1` | `EXIT_GENERIC_FAIL` | Generic failure (file not found, receipt validation failed) | Error handling |
| `8` | `EXIT_LN_SUSPICIOUS` | LayerNorm/projection validation failed in strict mode | CI/CD gates |

**See also:** `crates/bitnet-cli/src/exit.rs` for complete exit code definitions.

---

## Appendix: Pattern Examples

### Common BitNet Tensor Names

```
token_embd.weight                    (not validated)
blk.0.attn_norm.weight              (LayerNorm)
blk.0.attn_q.weight                 (Projection)
blk.0.attn_k.weight                 (Projection)
blk.0.attn_v.weight                 (Projection)
blk.0.attn_o.weight                 (Projection)
blk.0.ffn_norm.weight               (LayerNorm)
blk.0.ffn_gate.weight               (Projection)
blk.0.ffn_up.weight                 (Projection)
blk.0.ffn_down.weight               (Projection)
...
blk.23.attn_norm.weight
blk.23.ffn_norm.weight
output_norm.weight                  (LayerNorm)
output.weight                       (not validated)
```

### Pattern Matching Examples

| Tensor Name | Matched Pattern | Min | Max | Ruleset |
|-------------|----------------|-----|-----|---------|
| `blk.0.ffn_layernorm.weight` | `ffn_layernorm\.weight$` | 0.05 | 2.0 | `bitnet-b1.58:f16` |
| `blk.0.attn_norm.weight` | `attn_norm\.weight$` | 0.01 | 2.0 | `bitnet-b1.58:i2_s` |
| `output_norm.weight` | `final_(layer)?norm\.weight$` | 0.50 | 2.0 | `bitnet-b1.58:f16` |
| `custom_model.ln.weight` | `.*norm\.weight$` (fallback) | 0.80 | 1.20 | `generic` |

---

For questions or issues, see:
- **GitHub Issues**: [BitNet-rs/issues](https://github.com/microsoft/BitNet/issues)
- **Documentation Index**: `docs/` directory
- **Source Code**: `crates/bitnet-cli/src/ln_rules.rs`
