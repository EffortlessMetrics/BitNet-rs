# BitNet.rs CLI Argument Parsing Patterns Report

## Executive Summary

This report documents the CLI argument parsing patterns used in BitNet.rs, focusing on:
1. The **clap** derive macro patterns (both `#[derive(Parser)]` and `#[derive(Args)]`)
2. How to add new flags and enums
3. Validation patterns (both built-in and custom)
4. Environment variable integration
5. Help text and documentation patterns
6. Examples for implementing new features

**Key Files:**
- Main CLI: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/main.rs` (1800+ lines)
- Xtask CLI: `/home/steven/code/Rust/BitNet-rs/xtask/src/main.rs` (6000+ lines)
- Inference Command: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-cli/src/commands/inference.rs`
- Inference Library: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/prompt_template.rs`

---

## Part 1: Core Clap Structures

### Structure 1: Top-Level CLI with Subcommands (bitnet-cli)

```rust
#[derive(Parser)]
#[command(name = "bitnet")]
#[command(about = "BitNet.rs — 1-bit neural network inference with strict receipts")]
#[command(long_about = r#"Long detailed help text..."#)]
#[command(version = bitnet_version())]  // Dynamic version function
#[command(author = "BitNet Contributors")]
#[command(after_help = format!("CLI Interface Version: {}\n...", INTERFACE_VERSION))]
struct Cli {
    /// Global flags (apply to all subcommands)
    #[arg(short, long, value_name = "PATH", global = true)]
    config: Option<std::path::PathBuf>,

    #[arg(short, long, value_name = "DEVICE", global = true)]
    device: Option<String>,

    #[arg(long, value_name = "LEVEL", global = true)]
    log_level: Option<String>,

    #[arg(long, value_name = "N", global = true)]
    threads: Option<usize>,

    #[arg(long, value_name = "SIZE", global = true)]
    batch_size: Option<usize>,

    /// Shell completions generation
    #[arg(long, value_name = "SHELL")]
    completions: Option<Shell>,

    /// Configuration file output
    #[arg(long, value_name = "PATH")]
    save_config: Option<std::path::PathBuf>,

    /// Version information
    #[arg(long)]
    interface_version: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Run { /* args */ },
    Chat { /* args */ },
    // ... other commands
}
```

**Key Patterns:**
- Use `#[command(name = "...")]` for command name
- Use `#[command(long_about = r#"..."#)]` for multi-line help with examples
- Use `global = true` for flags that apply to all subcommands
- Use `#[command(subcommand)]` for nested command structure
- Dynamic version via function call: `version = bitnet_version()`

---

### Structure 2: Args struct for Nested Commands (inference.rs pattern)

```rust
#[derive(Args, Debug, Default)]
pub struct InferenceCommand {
    /// Path to the model file
    #[arg(short, long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Maximum number of tokens (with aliases)
    #[arg(
        long = "max-tokens",
        visible_aliases = ["max-new-tokens", "n-predict"],
        default_value = "512",
        value_name = "N"
    )]
    pub max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value = "0.7", value_name = "TEMP")]
    pub temperature: f32,

    /// Top-k sampling parameter (optional)
    #[arg(long, value_name = "K")]
    pub top_k: Option<usize>,

    /// System prompt for chat models
    #[arg(long, value_name = "TEXT")]
    pub system_prompt: Option<String>,

    /// Prompt template selection
    #[arg(long, value_name = "TEMPLATE", default_value = "auto")]
    pub prompt_template: String,

    /// Stop sequences (repeatable)
    #[arg(
        long = "stop",
        visible_alias = "stop-sequence",
        visible_alias = "stop_sequences",
        value_name = "SEQ"
    )]
    pub stop: Vec<String>,

    /// Stop token IDs (numeric, repeatable)
    #[arg(long = "stop-id", value_name = "ID")]
    pub stop_id: Vec<u32>,

    /// Q&A mode bundling defaults
    #[arg(long)]
    pub qa: bool,
}
```

**Key Patterns:**
- Use `#[derive(Args)]` for command-specific argument structs
- Default values: `default_value = "512"` for strings, `default_value_t = 0.7` for typed values
- For bools without default: omit `default_value_t` (defaults to false)
- Aliases: `visible_aliases = ["alias1", "alias2"]` (shows in help)
- Repeatable args: `Vec<T>` type automatically enables `--flag val1 --flag val2`
- Optional args: `Option<T>` type
- Environment fallback: `env = "VAR_NAME"` (e.g., `env = "BITNET_ALLOW_MOCK"`)

---

## Part 2: Enum-Based Flags (The Right Way)

### Example 1: TemplateType Enum with FromStr

**File:** `crates/bitnet-inference/src/prompt_template.rs`

```rust
pub enum TemplateType {
    /// Raw text (no formatting)
    Raw,
    /// Simple Q&A instruct format
    Instruct,
    /// LLaMA-3 chat format with special tokens
    Llama3Chat,
}

impl std::str::FromStr for TemplateType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "raw" => Ok(Self::Raw),
            "instruct" => Ok(Self::Instruct),
            "llama3-chat" | "llama3_chat" => Ok(Self::Llama3Chat),  // Alternative spellings
            _ => bail!("Unknown template type: {}. Supported: raw, instruct, llama3-chat", s),
        }
    }
}

impl std::fmt::Display for TemplateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Instruct => write!(f, "instruct"),
            Self::Llama3Chat => write!(f, "llama3-chat"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_templates() {
        assert_eq!("raw".parse::<TemplateType>().unwrap(), TemplateType::Raw);
        assert_eq!("instruct".parse::<TemplateType>().unwrap(), TemplateType::Instruct);
        assert_eq!("llama3-chat".parse::<TemplateType>().unwrap(), TemplateType::Llama3Chat);
        assert_eq!("llama3_chat".parse::<TemplateType>().unwrap(), TemplateType::Llama3Chat);
        assert!("invalid".parse::<TemplateType>().is_err());
    }
}
```

**Usage in CLI:**

```rust
#[derive(Args)]
pub struct MyCommand {
    /// Prompt template selection
    #[arg(long, value_name = "TEMPLATE", default_value = "auto")]
    pub prompt_template: String,  // Stays as String, parse later with .parse::<TemplateType>()?
}

// In handler function:
let template_type: TemplateType = if prompt_template == "auto" {
    TemplateType::Instruct  // Default fallback
} else {
    prompt_template.parse().with_context(|| {
        format!(
            "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat",
            prompt_template
        )
    })?
};
```

**Why this pattern?**
- Clap's `value_parser` for custom types is newer; keeping as `String` and parsing manually works everywhere
- Allows custom error messages with context
- `FromStr` + `Display` impls are standard Rust patterns
- Supports alternative spellings (`llama3-chat` and `llama3_chat`)
- Tests are straightforward

---

## Part 3: Adding New Flags - Complete Examples

### Example 1: Add `--cpp-backend` Enum Flag

Following the TemplateType pattern:

**Step 1: Define the enum in a library crate** (e.g., `bitnet-inference/src/lib.rs`):

```rust
pub enum CppBackend {
    BitNet,
    Llama,
}

impl std::str::FromStr for CppBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "bitnet" => Ok(Self::BitNet),
            "llama" => Ok(Self::Llama),
            _ => bail!("Unknown C++ backend '{}'. Supported: bitnet, llama", s),
        }
    }
}

impl std::fmt::Display for CppBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BitNet => write!(f, "bitnet"),
            Self::Llama => write!(f, "llama"),
        }
    }
}
```

**Step 2: Add flag to CLI** (in clap struct):

```rust
#[derive(Args)]
pub struct CrossvalCommand {
    /// C++ backend to use for cross-validation
    #[arg(long, value_name = "BACKEND", default_value = "bitnet")]
    pub cpp_backend: String,
    
    // ... other args
}
```

**Step 3: Parse in handler**:

```rust
async fn handle_crossval(cmd: CrossvalCommand) -> Result<()> {
    let backend: CppBackend = cmd.cpp_backend.parse()
        .with_context(|| format!("Invalid C++ backend: {}", cmd.cpp_backend))?;
    
    match backend {
        CppBackend::BitNet => { /* ... */ },
        CppBackend::Llama => { /* ... */ },
    }
    
    Ok(())
}
```

**Step 4: Test**:

```rust
#[test]
fn test_cpp_backend_parsing() {
    assert!("bitnet".parse::<CppBackend>().is_ok());
    assert!("llama".parse::<CppBackend>().is_ok());
    assert!("invalid".parse::<CppBackend>().is_err());
}
```

---

### Example 2: Add `--prompt-template` Flag

**In bitnet-cli/src/main.rs (Run command):**

```rust
#[derive(Subcommand)]
enum Commands {
    Run {
        // ... existing fields ...
        
        /// Prompt template: auto (detect), raw (no formatting), instruct (Q&A), llama3-chat
        #[arg(long, value_name = "TEMPLATE", default_value = "auto")]
        prompt_template: String,

        #[command(subcommand)]
        command: Option<Commands>,
    },
}
```

**Implementation pattern** (from `crates/bitnet-cli/src/main.rs` lines ~906-920):

```rust
use bitnet_inference::TemplateType;

let template_type: TemplateType = if prompt_template == "auto" {
    // Auto-detect will be done after loading tokenizer
    TemplateType::Instruct // Default fallback
} else {
    prompt_template.parse().with_context(|| {
        format!(
            "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat",
            prompt_template
        )
    })?
};

// Auto-detect if "auto" was requested (lines ~1064-1075)
let template_type = if prompt_template == "auto" {
    if tokenizer.token_to_id("<|eot_id|>").is_some() {
        debug!("Auto-detected llama3-chat template (tokenizer has <|eot_id|>)");
        TemplateType::Llama3Chat
    } else {
        debug!("Auto-detected instruct template (fallback)");
        TemplateType::Instruct
    }
} else {
    template_type
};
```

---

### Example 3: Add `--system-prompt` Flag

**In clap struct:**

```rust
#[derive(Args)]
pub struct InferenceCommand {
    // ... existing args ...
    
    /// System prompt for chat models
    #[arg(long, value_name = "TEXT")]
    pub system_prompt: Option<String>,
}
```

**Implementation** (from `crates/bitnet-cli/src/main.rs` lines ~1077-1078):

```rust
// Format prompt using the template with optional system prompt
let formatted_prompt = template_type.apply(&prompt, system_prompt.as_deref());
```

**Key characteristics:**
- Optional (uses `Option<String>`)
- No default value needed for optional fields
- `value_name = "TEXT"` shows as help text
- Use `.as_deref()` to pass `Option<&str>` instead of `Option<String>`

---

### Example 4: Add `--dump-ids` Boolean Flag

**In clap struct:**

```rust
#[derive(Args)]
pub struct RunCommand {
    // ... existing args ...
    
    /// Dump token IDs to stdout
    #[arg(long, default_value_t = false)]
    pub dump_ids: bool,
    
    /// Dump logit steps during generation (max steps)
    #[arg(long)]
    pub dump_logit_steps: Option<usize>,

    /// Top-k tokens to include in logit dump
    #[arg(long, default_value = "10", value_name = "K")]
    pub logits_topk: usize,
}
```

**Implementation** (from `crates/bitnet-cli/src/main.rs` lines ~1439-1441):

```rust
if dump_ids {
    println!("Token IDs: {:?}", generated_tokens);
}
```

**Key characteristics:**
- Use `default_value_t = false` for bools (not `default_value`)
- No short flag (`-d`) to avoid confusion with other flags
- Optional `Option<usize>` for related args that need values

---

### Example 5: Add `--dump-cpp-ids` Flag (Xtask Pattern)

**Pattern from xtask** (showing various flag types):

```rust
#[derive(Subcommand)]
enum Cmd {
    /// Benchmark model performance
    Benchmark {
        /// Path to GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// Path to tokenizer file (required unless --allow-mock)
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Number of tokens to generate for benchmark
        #[arg(long, default_value_t = 128)]
        tokens: usize,

        /// Benchmark prompt (affects prefill time)
        #[arg(long, default_value = "The capital of France is")]
        prompt: String,

        /// Use GPU if available
        #[arg(long, default_value_t = false)]
        gpu: bool,

        /// Allow mock tokenizer for testing
        #[arg(long, default_value_t = false)]
        allow_mock: bool,

        /// Suppress generation output (default: true)
        #[arg(long, default_value_t = true)]
        no_output: bool,

        /// Write detailed results to JSON file
        #[arg(long)]
        json: Option<PathBuf>,

        /// Number of warmup tokens to generate and discard
        #[arg(long, default_value_t = 10)]
        warmup_tokens: usize,
    },
}
```

**For `--dump-cpp-ids`:**

```rust
#[derive(Subcommand)]
enum Cmd {
    CrossvalPerToken {
        // ... existing fields ...

        /// Dump C++ token IDs for comparison
        #[arg(long, default_value_t = false)]
        pub dump_cpp_ids: bool,
    },
}
```

---

## Part 4: Validation Patterns

### Pattern 1: Environment Variable Fallback

```rust
#[arg(long, env = "BITNET_ALLOW_MOCK", default_value_t = false)]
pub allow_mock: bool,
```

**Usage in code:**

```rust
// Env var is automatically used; no explicit code needed
if allow_mock {
    // User provided --allow-mock OR env BITNET_ALLOW_MOCK=1
    println!("Mock mode enabled");
}
```

### Pattern 2: Conflicting Arguments

```rust
#[arg(long, conflicts_with = "file")]
pub text: Option<String>,

#[arg(long, conflicts_with = "text")]
pub file: Option<PathBuf>,
```

Clap automatically prevents both being set at once.

### Pattern 3: Manual Validation with Context

```rust
let template_type: TemplateType = prompt_template.parse()
    .with_context(|| {
        format!(
            "Invalid prompt template '{}'. Supported: raw, instruct, llama3-chat",
            prompt_template
        )
    })?;
```

### Pattern 4: Alias Support

```rust
#[arg(
    long = "max-tokens",
    visible_aliases = ["max-new-tokens", "n-predict"],
    default_value = "512"
)]
pub max_tokens: usize,
```

**Shows in help:**
```
Options:
  --max-tokens <N>           [default: 512] [aliases: max-new-tokens, n-predict]
```

### Pattern 5: Multiple Aliases for Subcommands

```rust
#[derive(Subcommand)]
enum Commands {
    #[command(alias = "generate")]
    Run { /* ... */ },
    
    #[command(alias = "infer")]
    Inference(Box<InferenceCommand>),
    
    #[command(name = "setup-cpp-auto")]
    SetupCppAuto { /* ... */ },
}
```

---

## Part 5: Help Text Patterns

### Pattern 1: Single-line help (docstring)

```rust
#[arg(short, long)]
pub verbose: bool,  // No docstring = no help text
```

### Pattern 2: Doc comment as help text

```rust
/// Enable verbose output
#[arg(short, long)]
pub verbose: bool,
```

Shows: `Options: -v, --verbose    Enable verbose output`

### Pattern 3: Multi-line help with examples

```rust
/// Prompt template: auto (detect), raw (no formatting), instruct (Q&A), llama3-chat (LLaMA-3)
///
/// # Examples
/// - auto: Detect from GGUF metadata and tokenizer hints
/// - raw: Raw text completion (no special formatting)
/// - instruct: Simple Q&A format
/// - llama3-chat: LLaMA-3 chat format with <|start_header_id|> and <|eot_id|>
#[arg(long, value_name = "TEMPLATE", default_value = "auto")]
pub prompt_template: String,
```

### Pattern 4: Long help at command level

```rust
#[command(long_about = r#"BitNet.rs CLI — one-shot generation and chat with strict receipts

QUICK EXAMPLES:

  # Deterministic math sanity check
  RUST_LOG=warn bitnet run --model model.gguf --prompt "2+2=" \
    --max-tokens 1 --temperature 0.0 --greedy

  # Q&A with instruct template
  RUST_LOG=warn bitnet run --model model.gguf \
    --prompt "What is 2+2?" --max-tokens 16

LOGGING:
  Set RUST_LOG=warn (default: info) to reduce log noise.
  Options: error, warn, info, debug, trace
"#)]
struct Cli {
    // ...
}
```

### Pattern 5: After-help text

```rust
#[command(after_help = format!(
    "CLI Interface Version: {}\nDocs: https://docs.rs/bitnet\nIssues: https://github.com/...",
    INTERFACE_VERSION
))]
struct Cli {
    // ...
}
```

---

## Part 6: Default Value Patterns

### String Defaults

```rust
// Literal string
#[arg(long, default_value = "auto")]
pub template: String,

// Dynamic/constant
#[arg(long, default_value = DEFAULT_MODEL_ID)]
pub id: String,
```

### Numeric Defaults

```rust
// For typed values (use `default_value_t`)
#[arg(long, default_value_t = 128)]
pub tokens: usize,

#[arg(long, default_value_t = 0.7)]
pub temperature: f32,
```

### Boolean Defaults

```rust
// For bools (use `default_value_t`)
#[arg(long, default_value_t = false)]
pub verbose: bool,

// For inverted bools (--no-warnings suppresses warnings)
#[arg(long, default_value_t = false)]
pub no_warnings: bool,
```

### Optional Fields (no default)

```rust
#[arg(long)]
pub custom_path: Option<PathBuf>,

#[arg(long)]
pub seed: Option<u64>,
```

### Collections (Vec with repeatable flag)

```rust
#[arg(long, value_delimiter = ',', default_values = ["1", "4", "8"])]
pub batch_sizes: Vec<usize>,

// Or repeatable without default
#[arg(long = "stop", value_name = "SEQ")]
pub stop: Vec<String>,  // --stop a --stop b --stop c
```

---

## Part 7: Template for New CLI Additions

### Complete Working Example: Add `--dump-cpp-ids` Flag

**File 1: Define in xtask enum** (`xtask/src/main.rs`)

```rust
#[derive(Subcommand)]
enum Cmd {
    // ... existing commands ...

    /// Compare Rust vs C++ logits (find first diverging token)
    ///
    /// Runs inference with both implementations and compares logits per token,
    /// optionally dumping C++ token IDs for debugging divergence.
    ///
    /// Example:
    ///   cargo run -p xtask -- crossval-per-token \
    ///     --model models/model.gguf \
    ///     --tokenizer models/tokenizer.json \
    ///     --prompt "Hello world" \
    ///     --max-tokens 4 \
    ///     --dump-cpp-ids
    #[cfg(feature = "inference")]
    #[command(name = "crossval-per-token")]
    CrossvalPerToken {
        /// Path to GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// Path to tokenizer file
        #[arg(long)]
        tokenizer: PathBuf,

        /// Input prompt to process
        #[arg(long)]
        prompt: String,

        /// Maximum tokens to generate (excluding prompt)
        #[arg(long, default_value_t = 4)]
        max_tokens: usize,

        /// Cosine similarity tolerance (0.0-1.0)
        #[arg(long, default_value_t = 0.999)]
        cos_tol: f32,

        /// Output format: "text" or "json"
        #[arg(long, default_value = "text")]
        format: String,

        /// Dump C++ token IDs for comparison (debugging)
        #[arg(long, default_value_t = false)]
        dump_cpp_ids: bool,
    },
}
```

**File 2: Handle in main command dispatcher** (`xtask/src/main.rs` main function)

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        // ... other commands ...
        
        #[cfg(feature = "inference")]
        Cmd::CrossvalPerToken {
            model,
            tokenizer,
            prompt,
            max_tokens,
            cos_tol,
            format,
            dump_cpp_ids,
        } => {
            crossval_per_token(
                model,
                tokenizer,
                prompt,
                max_tokens,
                cos_tol,
                &format,
                dump_cpp_ids,
            )
            .await?
        }
        
        // ... handle other commands ...
    }

    Ok(())
}
```

**File 3: Implement handler function**

```rust
async fn crossval_per_token(
    model: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_tokens: usize,
    cos_tol: f32,
    format: &str,
    dump_cpp_ids: bool,
) -> Result<()> {
    // ... validation ...
    
    // Run Rust inference
    let rust_logits = run_rust_inference(&model, &tokenizer, &prompt, max_tokens)?;
    
    // Run C++ inference if available
    let cpp_logits = if dump_cpp_ids {
        // Dump C++ IDs during inference
        run_cpp_inference_with_dump(&model, &tokenizer, &prompt, max_tokens)?
    } else {
        run_cpp_inference(&model, &tokenizer, &prompt, max_tokens)?
    };
    
    // Compare and report
    let divergence = compare_logits(&rust_logits, &cpp_logits, cos_tol)?;
    
    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&divergence)?);
    } else {
        println!("Divergence at token: {:?}", divergence);
    }
    
    Ok(())
}
```

**File 4: Test** (in tests or inline)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_cpp_ids_parsing() {
        // Parse command line
        let args = vec![
            "xtask",
            "crossval-per-token",
            "--model", "test.gguf",
            "--tokenizer", "tokenizer.json",
            "--prompt", "test",
            "--dump-cpp-ids",
        ];
        
        let cli = Cli::try_parse_from(&args).unwrap();
        
        match cli.cmd {
            Cmd::CrossvalPerToken { dump_cpp_ids, .. } => {
                assert!(dump_cpp_ids);
            }
            _ => panic!("Wrong command"),
        }
    }
}
```

---

## Part 8: Advanced Patterns

### Pattern 1: Feature-Gated Commands

```rust
#[cfg(feature = "inference")]
#[command(name = "crossval-per-token")]
CrossvalPerToken {
    // ... args ...
},
```

Only available when `--features inference` or `--features crossval-all`.

### Pattern 2: Box<T> for Large Args

```rust
#[derive(Subcommand)]
enum Commands {
    #[command(name = "inference")]
    Inference(Box<InferenceCommand>),  // Avoid stack bloat
    Chat(Box<InferenceCommand>),
}
```

InferenceCommand is ~300 bytes; boxing saves stack space.

### Pattern 3: Marker Flags for Behavior Bundling

```rust
/// Q&A mode: bundle Q&A-friendly defaults
/// (auto template, temp=0.7, top-p=0.95, top-k=50)
#[arg(long)]
pub qa: bool,
```

**Handler implementation:**

```rust
if cmd.qa {
    // Override defaults when --qa is set
    if cmd.prompt_template == "auto" {
        cmd.prompt_template = "instruct".to_string();
    }
    if cmd.temperature == 0.7 {  // Check if unchanged from default
        cmd.temperature = 0.7;
    }
    if !cmd.top_p.is_some() {
        cmd.top_p = Some(0.95);
    }
}
```

### Pattern 4: Dynamic Version Function

```rust
fn bitnet_version() -> &'static str {
    use std::sync::OnceLock;
    static VERSION_STRING: OnceLock<String> = OnceLock::new();

    VERSION_STRING.get_or_init(|| {
        let features = compiled_features();
        let features_line = if features.is_empty() {
            "features: none".to_string()
        } else {
            format!("features: {}", features.join(", "))
        };

        format!("{}\n{}", env!("CARGO_PKG_VERSION"), features_line)
    })
}

#[command(version = bitnet_version())]
struct Cli {
    // ...
}
```

---

## Summary: Quick Reference

| Task | Pattern | Example |
|------|---------|---------|
| Add simple flag | `#[arg(long)]` | `--verbose` |
| Add flag with default | `#[arg(long, default_value = "X")]` | `--template "auto"` |
| Add boolean flag | `#[arg(long, default_value_t = false)]` | `--dump-ids` |
| Add optional arg | `#[arg(long)]` + `Option<T>` | `--seed: Option<u64>` |
| Add repeatable arg | `#[arg(long)]` + `Vec<T>` | `--stop: Vec<String>` |
| Add env fallback | `#[arg(long, env = "VAR")]` | `env = "BITNET_ALLOW_MOCK"` |
| Add aliases | `visible_aliases = [...]` | `["max-new-tokens", "n-predict"]` |
| Custom enum | `FromStr + Display impls` | `TemplateType` |
| Validate | `.parse().with_context(...)` | Template parsing |
| Conflicting args | `conflicts_with = "..."` | `text` vs `file` |
| Help text | Doc comment | `/// Explanation` |
| Multi-line help | Raw string in `#[command(...)]` | `long_about = r#"..."#` |

---

## Key Takeaways

1. **Always use `#[derive(Args)]` for reusable command groups** (e.g., InferenceCommand)
2. **Use `FromStr + Display` for custom enums** with `.parse()` in handlers
3. **Keep flags as strings where possible** to allow flexible parsing
4. **Use `visible_aliases`** for backward compatibility
5. **Test all flag combinations** (clap validates at parse time, not run time)
6. **Document via docstrings** for help text
7. **Use `value_name` consistently** for clarity in help
8. **Validate with `.with_context()`** for actionable error messages
9. **Prefer `default_value_t` over `default_value`** for typed defaults
10. **Box large command structs** to avoid stack overhead

