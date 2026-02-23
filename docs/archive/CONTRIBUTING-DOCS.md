# Documentation Contributing Guide

**Audience:** Contributors writing or updating documentation for BitNet.rs

**Goal:** Maintain consistent, high-quality documentation that serves different audiences and use cases

---

## Table of Contents

- [Documentation Standards](#documentation-standards)
- [Structural Patterns](#structural-patterns)
- [Code Examples](#code-examples)
- [Markdown Style](#markdown-style)
- [File Organization](#file-organization)
- [Review Process](#review-process)
- [Common Patterns Reference](#common-patterns-reference)

---

## Documentation Standards

### Headings Hierarchy

Use proper heading hierarchy for scanability and navigation:

```markdown
# Page Title (H1 - one per document)

## Major Section (H2)

### Subsection (H3)

#### Details (H4)
```

**Rules:**

- **H1**: Document title only (one per file)
- **H2**: Major sections (build commands, testing, troubleshooting)
- **H3**: Subsections within major sections
- **H4**: Details, examples, or edge cases
- Never skip levels (e.g., H1 → H3)
- Use sentence case for headings: "Build commands reference" not "Build Commands Reference"
- Keep headings concise (≤ 60 characters when possible)

### Tone and Voice

BitNet.rs documentation uses a **direct, technical, and precise** tone:

**Do:**

- Use active voice: "Run the validation script" not "The validation script should be run"
- Be explicit about requirements: "You must specify features" not "It's recommended to specify features"
- State facts clearly: "Default features are empty" not "By default, no features are enabled"
- Use present tense: "The tool validates weights" not "The tool will validate weights"

**Don't:**

- Use marketing language or hype
- Use unnecessary qualifiers: "simply", "just", "easily"
- Anthropomorphize code: "The validator is happy" (use "The validator passes")
- Use vague language: "some configurations" (specify which ones)

### Audience Targeting

Every documentation file should declare its audience and goal in the front matter:

```markdown
# Document Title

**Audience:** Developers building BitNet.rs from source

**Goal:** Successfully compile and test the project on Linux/macOS/Windows

---
```

**Common audiences:**

- **Developers**: Contributors writing code
- **Researchers**: Users experimenting with models and quantization
- **Operators**: Production deployment and monitoring
- **Users**: End users running inference
- **Maintainers**: CI/CD and release management

### Content Types (Diátaxis Framework)

BitNet.rs documentation follows the [Diátaxis framework](https://diataxis.fr/) with four content types:

| Type | Purpose | Location | Example |
|------|---------|----------|---------|
| **Tutorial** | Learning-oriented | `docs/tutorials/` | Step-by-step first inference |
| **How-to** | Problem-oriented | `docs/howto/` | Export clean GGUF model |
| **Reference** | Information-oriented | `docs/reference/` | Quantization format specs |
| **Explanation** | Understanding-oriented | `docs/explanation/` | Why strict mode exists |

**Choose the right type:**

- **Tutorial**: "I want to learn how to use X"
- **How-to**: "I need to accomplish specific task Y"
- **Reference**: "I need to look up technical detail Z"
- **Explanation**: "I want to understand why W works this way"

---

## Structural Patterns

### Front Matter (Required)

Every documentation file starts with front matter:

```markdown
# Document Title

**Audience:** [Target audience]

**Goal:** [What the reader will accomplish]

---
```

**Example:**

```markdown
# Model Validation Workflow

**Audience:** Developers and researchers who need to validate GGUF models

**Goal:** Ensure models pass strict validation without runtime corrections

---
```

### Breadcrumbs and Navigation

For documents in subdirectories, include navigation breadcrumbs:

```markdown
# Document Title

**Audience:** [Target audience]

**Goal:** [What the reader will accomplish]

**Navigation:** [Home](../README.md) > [Development](README.md) > Build Commands

---
```

**Rules:**

- Use relative links to maintain portability
- Include parent directory and immediate siblings
- Place after front matter, before first H2

### Table of Contents (For Long Documents)

Documents > 200 lines should include a TOC after front matter:

```markdown
# Document Title

**Audience:** [Target audience]

**Goal:** [What the reader will accomplish]

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---
```

**TOC Guidelines:**

- Use markdown anchor links (auto-generated from headings)
- Include only H2 and select H3 headings
- Keep TOC ≤ 15 entries (split document if larger)
- Place horizontal rule (`---`) after TOC

### Cross-References and Related Docs

End documents with related documentation:

```markdown
---

## Related Documentation

- **Tutorial:** [Getting Started](../tutorials/getting-started.md) - First inference workflow
- **How-To:** [Export Clean GGUF](../howto/export-clean-gguf.md) - Model export guide
- **Reference:** [Validation Gates](../reference/validation-gates.md) - Technical specification
- **Explanation:** [Why Strict Mode](../explanation/strict-mode-rationale.md) - Design rationale

---
```

**Format:**

- Use bold for link type prefix
- Include brief description after dash
- Group by content type (Tutorial, How-To, Reference, Explanation)
- Use relative paths

### Metadata Footer (Optional)

For living documents, include metadata footer:

```markdown
---

**Document Status:** Living document (updated 2025-10-23)

**Maintainers:** @username1, @username2

**Related Issues:** #254, #260, #469

**Last Review:** 2025-10-23
```

---

## Code Examples

### Always Specify Feature Flags

BitNet.rs has **empty default features**. All code examples must explicitly specify features:

**Correct:**

```bash
# Build with CPU support
cargo build --no-default-features --features cpu

# Run tests with GPU support
cargo test --no-default-features --features gpu --workspace
```

**Incorrect (missing features):**

```bash
# Wrong - uses empty default features
cargo build

# Wrong - missing --no-default-features flag
cargo test --features cpu
```

### Include Descriptive Comments

Add comments explaining what each command does:

```bash
# Build validation tools in release mode
cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli

# Run strict mode validation (fails on suspicious weights)
BITNET_STRICT_MODE=1 \
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  inspect --ln-stats --gate auto models/model.gguf

# Validate with deterministic settings
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
cargo run -p bitnet-cli --no-default-features --features cpu -- run --model model.gguf
```

### Multi-Line Commands

Use backslash continuation for readability:

```bash
# Correct: Multi-line with proper indentation
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/model.gguf \
  --tokenizer models/tokenizer.json \
  --prompt "What is 2+2?" \
  --max-tokens 32 \
  --temperature 0.7
```

**Rules:**

- Use `\` at line end for continuation
- Indent continuation lines by 2 spaces
- Align related flags vertically when possible
- Break at logical boundaries (before major flags)

### Expected Output

Show expected output for non-trivial commands:

```bash
# Validate GGUF model
./scripts/validate_gguf.sh models/model.gguf models/tokenizer.json
```

**Expected Output:**

```
===================================================
1/3: LayerNorm Statistics Check (Strict Mode)
===================================================
INFO: ✅ Found 24 healthy LayerNorm layers (RMS in [0.5, 2.0])

===================================================
2/3: Projection Weight RMS Check
===================================================
INFO: ✅ Projection weights loaded (see RMS values above)

===================================================
3/3: Greedy Inference Probe (Linguistic Sanity)
===================================================
Generated output:
----------------------------------------
The capital of France is Paris.
----------------------------------------
INFO: ✅ Output contains recognizable words

===================================================
Validation Report
===================================================
INFO: ✅✅✅ ALL VALIDATION CHECKS PASSED ✅✅✅
```

### Code Examples in Rust

When showing Rust code, include:

1. **Feature gates** if relevant
2. **Error handling** patterns
3. **Comments** explaining non-obvious logic

```rust
use bitnet_common::strict_mode::StrictModeEnforcer;
use bitnet_common::{Device, QuantizationType, Result};

// Production inference with strict mode enabled
std::env::set_var("BITNET_STRICT_MODE", "1");
let enforcer = StrictModeEnforcer::new_detailed();

// Validate inference path (fails on mock usage)
enforcer.validate_inference_path(&inference_path)?;

// Validate quantization kernel availability
enforcer.validate_kernel_availability(&kernel_scenario)?;
```

**Include feature gates:**

```rust
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_function() -> Result<()> {
    // GPU-specific implementation
    Ok(())
}

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
pub fn gpu_function() -> Result<()> {
    Err(BitNetError::FeatureNotEnabled("gpu or cuda"))
}
```

---

## Markdown Style

### Fenced Code Blocks

Always use fenced code blocks with language tags:

````markdown
```bash
# Shell commands
cargo build --no-default-features --features cpu
```

```rust
// Rust code
fn main() {
    println!("Hello, BitNet!");
}
```

```json
{
  "schema_version": "1.0.0",
  "backend": "cpu"
}
```

```yaml
# YAML configuration
name: ci
on: push
```
````

**Supported language tags:**

- `bash` - Shell commands (preferred over `sh` or `shell`)
- `rust` - Rust code
- `toml` - Cargo.toml, config files
- `json` - JSON data
- `yaml` - YAML configuration
- `text` - Plain text output
- `console` - Terminal session with prompts

### Line Length

**Code:** No hard limit (readability over strict limits)

**Prose:** Aim for 100-120 characters per line

- Improves readability in side-by-side diffs
- Allows comfortable viewing without horizontal scrolling
- Soft limit, not enforced by tools

**Exception:** Long URLs, code examples, tables may exceed limit

### Lists

Use consistent list formatting:

**Unordered lists:**

```markdown
- First item
- Second item with details:
  - Nested sub-item
  - Another sub-item
- Third item
```

**Ordered lists:**

```markdown
1. First step
2. Second step with details:
   - Sub-detail A
   - Sub-detail B
3. Third step
```

**Rules:**

- Use `-` for unordered lists (not `*` or `+`)
- Indent nested lists by 2 spaces
- Add blank line before/after lists
- Use ordered lists for sequential steps
- Use unordered lists for non-sequential items

### Tables

Use tables for structured data:

```markdown
| Feature Flag | Description | Example |
|--------------|-------------|---------|
| `cpu` | CPU inference | `--features cpu` |
| `gpu` | GPU acceleration | `--features gpu` |
| `ffi` | C++ FFI bridge | `--features cpu,ffi` |
```

**Table guidelines:**

- Always include header row
- Use `|` alignment for readability
- Keep tables ≤ 5 columns (split if wider)
- Use blank line before/after table
- Align columns for source readability (not required but preferred)

### Emphasis

Use emphasis consistently:

- **Bold** (`**text**`): Important concepts, UI elements, file names
- *Italic* (`*text*`): Emphasis, technical terms on first use, variable names
- `Code` (`` `text` ``): Commands, file paths, code identifiers, values

**Examples:**

- "The **LayerNorm weights** must be in *float format* (F16 or F32)."
- "Set `BITNET_STRICT_MODE=1` to enable **strict validation**."
- "Edit the `Cargo.toml` file in the *bitnet-cli* crate."

### Horizontal Rules

Use `---` for horizontal rules (not `***` or `___`):

```markdown
## Section 1

Content here.

---

## Section 2

Content here.
```

**When to use:**

- Before/after front matter
- Before/after TOC
- Between major sections (H2)
- Before metadata footer

### Links

Use descriptive link text (not "click here"):

**Correct:**

```markdown
See the [validation framework documentation](../development/validation-framework.md) for details.

For GPU setup, consult the [CUDA configuration guide](../GPU_SETUP.md).
```

**Incorrect:**

```markdown
Click [here](../development/validation-framework.md) for details.

For more information, see [this guide](../GPU_SETUP.md).
```

**Link formats:**

- **Internal links**: Use relative paths (`../howto/export-clean-gguf.md`)
- **External links**: Use full URLs (`https://github.com/microsoft/BitNet`)
- **Anchor links**: Use auto-generated anchors (`#troubleshooting`)

### Admonitions (Callouts)

Use blockquotes with emoji prefixes for callouts:

```markdown
⚠️ **Warning:** LayerNorm weights must be in float format (F16/F32), not quantized.

✅ **Tip:** Use `--no-default-features --features cpu` to avoid linker errors.

💡 **Note:** Default features are empty in BitNet.rs. Always specify features explicitly.

🔒 **Security:** Never enable runtime corrections in CI. Use only for known-bad models in local development.
```

**Callout types:**

- ⚠️ **Warning:** Potential errors or issues
- ✅ **Tip:** Best practices or helpful suggestions
- 💡 **Note:** Additional context or clarification
- 🔒 **Security:** Security-relevant information
- 🚧 **Status:** Development status or limitations

---

## File Organization

### Directory Structure

BitNet.rs documentation follows the Diátaxis framework:

```
docs/
├── README.md                  # Documentation index
├── CLAUDE.md                  # Project-wide guidance for AI assistants
├── CONTRIBUTING-DOCS.md       # This file
├── quickstart.md              # 5-minute getting started
├── getting-started.md         # Comprehensive introduction
│
├── tutorials/                 # Learning-oriented guides
│   ├── first-inference.md
│   └── tokenizer-discovery-tutorial.md
│
├── howto/                     # Problem-oriented guides
│   ├── export-clean-gguf.md
│   ├── validate-models.md
│   └── deterministic-inference-setup.md
│
├── reference/                 # Information-oriented specs
│   ├── quantization-support.md
│   ├── validation-gates.md
│   └── tokenizer-discovery-api.md
│
├── explanation/               # Understanding-oriented docs
│   ├── FEATURES.md
│   ├── ROADMAP.md
│   ├── i2s-dual-flavor.md
│   └── correction-policy.md
│
├── development/               # Developer guides
│   ├── build-commands.md
│   ├── test-suite.md
│   ├── validation-framework.md
│   ├── validation-ci.md
│   ├── gpu-development.md
│   └── xtask.md
│
├── baselines/                 # Performance baselines
│   └── README.md
│
└── adr/                       # Architecture Decision Records
    ├── README.md
    └── 0001-configuration-layering.md
```

### When to Create New Files

**Create a new file when:**

- Document exceeds 500 lines
- Topic is distinct and self-contained
- Content serves different audience than parent doc
- Multiple cross-references point to same content

**Don't create new files for:**

- Short content (< 100 lines) that fits in parent doc
- One-off examples or edge cases
- Temporary or experimental content

### Naming Conventions

**File names:**

- Use kebab-case: `export-clean-gguf.md`
- Be descriptive: `validation-framework.md` not `validation.md`
- Use `.md` extension (lowercase)
- Avoid abbreviations unless widely known (OK: `gpu`, `ci`, `ffi`)

**Directory names:**

- Use lowercase, plural when appropriate: `tutorials/`, `howto/`
- Keep shallow hierarchy (≤ 3 levels deep)

---

## Review Process

### Pre-Submission Checklist

Before submitting documentation changes:

- [ ] Run markdownlint (see below)
- [ ] Verify all links (internal and external)
- [ ] Test all code examples
- [ ] Check for typos and grammar
- [ ] Ensure front matter is present
- [ ] Verify feature flags in examples
- [ ] Add related docs section if applicable

### Markdownlint

BitNet.rs uses markdownlint for style consistency:

```bash
# Install markdownlint-cli
npm install -g markdownlint-cli

# Lint all documentation
markdownlint 'docs/**/*.md'

# Lint specific file
markdownlint docs/CONTRIBUTING-DOCS.md

# Auto-fix common issues
markdownlint --fix 'docs/**/*.md'
```

**Common issues:**

- MD001: Heading levels should increment by one
- MD013: Line length (disabled for code blocks)
- MD022: Headings should be surrounded by blank lines
- MD025: Multiple H1 headings
- MD033: Inline HTML (generally discouraged)

**Configuration:** See `.markdownlint.json` for project rules

### Link Checking

Verify all links are valid:

```bash
# Install markdown-link-check
npm install -g markdown-link-check

# Check all links in document
markdown-link-check docs/CONTRIBUTING-DOCS.md

# Check all docs
find docs -name '*.md' -exec markdown-link-check {} \;
```

**Link issues:**

- Broken internal links (file moved/renamed)
- Broken external links (404, domain expired)
- Incorrect anchor links (heading changed)

### Code Example Testing

Test all code examples before committing:

```bash
# Extract bash commands from markdown
grep -A 10 '```bash' docs/howto/export-clean-gguf.md

# Run commands in clean environment
docker run --rm -it rust:1.90 bash

# Verify expected output matches
```

**Testing strategy:**

- Test in clean environment (not your dev setup)
- Verify feature flags are correct
- Check exit codes match documentation
- Ensure environment variables are set properly

### Technical Review

Documentation changes require technical review:

**Review criteria:**

- **Accuracy**: Technical details are correct
- **Completeness**: No critical steps missing
- **Clarity**: Examples are clear and tested
- **Consistency**: Follows style guide
- **Audience**: Appropriate for target audience

**Reviewers should:**

- Run code examples
- Verify links
- Check for outdated content
- Suggest improvements

---

## Common Patterns Reference

### Bash Command Template

```bash
# [Brief description of what this command does]
[ENV_VAR=value] cargo [subcommand] [flags] [package] [--features flags] [-- args]

# Example: Build CLI with validation features
cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli
```

### Inference Command Template

```bash
# [Description of inference scenario]
RUST_LOG=warn cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- run \
  --model models/path/to/model.gguf \
  --tokenizer models/path/to/tokenizer.json \
  --prompt-template [auto|raw|instruct|llama3-chat] \
  --prompt "Your prompt here" \
  --max-tokens [number] \
  --temperature [0.0-1.0]
```

### Test Command Template

```bash
# [Description of test scope]
cargo test -p [package] --no-default-features --features [cpu|gpu] [--test test_name]

# Example: Run validation workflow tests
cargo test -p bitnet-cli --test validation_workflow \
  --no-default-features --features cpu,full-cli
```

### Troubleshooting Section Template

```markdown
## Troubleshooting

### [Problem Description]

**Symptom:**

```text
[Exact error message or behavior]
```

**Cause:** [Explanation of root cause]

**Solution:**

1. [First step]
2. [Second step]
3. [Third step]

**Verification:**

```bash
# [Command to verify fix]
```

**See also:** [Link to related documentation]
```

### Step-by-Step Guide Template

```markdown
## Step-by-Step Guide

### Step 1: [Action]

[Brief explanation of step]

```bash
# [Commands for this step]
```

**Expected Output:**

```
[Show expected output]
```

### Step 2: [Next Action]

[Explanation]

```bash
# [Commands]
```

[Continue pattern...]
```

### Configuration Table Template

```markdown
## Configuration Options

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BITNET_STRICT_MODE` | Boolean | `0` | Enable strict validation |
| `BITNET_DETERMINISTIC` | Boolean | `0` | Enable reproducible inference |
| `BITNET_SEED` | Integer | Random | Random seed for inference |

**Usage:**

```bash
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
```
```

---

## Summary

Key principles for BitNet.rs documentation:

1. **Always specify feature flags** - Never assume default features
2. **Use descriptive examples** - Show realistic, tested commands
3. **Follow Diátaxis** - Choose right content type (tutorial/howto/reference/explanation)
4. **Target your audience** - Declare audience and goal upfront
5. **Link related docs** - Help readers find what they need next
6. **Test everything** - Run all code examples before committing

**Quality checklist:**

- ✅ Front matter present (audience, goal)
- ✅ Feature flags explicit in all examples
- ✅ Code examples tested
- ✅ Links verified
- ✅ Markdownlint passes
- ✅ Related docs section included
- ✅ Technical review complete

---

**Document Status:** Living document (updated 2025-10-23)

**Maintainers:** Core documentation team

**Related Documentation:**

- **Reference:** [CLAUDE.md](../CLAUDE.md) - Project-wide guidance
- **How-To:** [Getting Started](getting-started.md) - First-time contributor guide
- **Explanation:** [Documentation Structure](README.md) - Documentation index

---
