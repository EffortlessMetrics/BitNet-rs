# BitNet-rs CLI Patterns - Quick Reference

## File Locations for Copy-Paste Examples

| Pattern | File | Lines |
|---------|------|-------|
| Main CLI structure | `crates/bitnet-cli/src/main.rs` | 85-166 |
| Run command | `crates/bitnet-cli/src/main.rs` | 194-303 |
| Inference args | `crates/bitnet-cli/src/commands/inference.rs` | 131-299 |
| TemplateType enum | `crates/bitnet-inference/src/prompt_template.rs` | (see enum impl) |
| Xtask commands | `xtask/src/main.rs` | 187-540 |

---

## The 5 CLI Patterns You'll Use

### 1. Simple Boolean Flag
```rust
/// Enable verbose output
#[arg(long, default_value_t = false)]
pub verbose: bool,
```
Usage: `--verbose`

### 2. String Flag with Default
```rust
/// Output format (text, json, csv)
#[arg(long, default_value = "text")]
pub format: String,
```
Usage: `--format json`

### 3. Numeric Flag
```rust
/// Number of tokens to generate
#[arg(long, default_value_t = 128)]
pub tokens: usize,
```
Usage: `--tokens 256`

### 4. Optional Argument
```rust
/// Seed for reproducibility
#[arg(long)]
pub seed: Option<u64>,
```
Usage: `--seed 42` or omit for None

### 5. Repeatable Argument
```rust
/// Stop sequences (repeatable)
#[arg(long = "stop")]
pub stop: Vec<String>,
```
Usage: `--stop "a" --stop "b"` or `--stop a,b` (with `value_delimiter = ','`)

---

## Enum-Based Flags (The Right Way)

### Define the enum:
```rust
pub enum TemplateType {
    Raw,
    Instruct,
    Llama3Chat,
}

impl std::str::FromStr for TemplateType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "raw" => Ok(Self::Raw),
            "instruct" => Ok(Self::Instruct),
            "llama3-chat" | "llama3_chat" => Ok(Self::Llama3Chat),
            _ => bail!("Unknown template type: {}", s),
        }
    }
}

impl std::fmt::Display for TemplateType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Instruct => write!(f, "instruct"),
            Self::Llama3Chat => write!(f, "llama3-chat"),
        }
    }
}
```

### Use in CLI:
```rust
/// Prompt template selection
#[arg(long, default_value = "auto")]
pub prompt_template: String,
```

### Parse in handler:
```rust
let template = prompt_template.parse::<TemplateType>()?;
```

---

## Common Patterns Cheat Sheet

### With Aliases
```rust
#[arg(
    long = "max-tokens",
    visible_aliases = ["max-new-tokens", "n-predict"],
    default_value = "512"
)]
pub max_tokens: usize,
```

### With Environment Variable Fallback
```rust
#[arg(long, env = "BITNET_ALLOW_MOCK", default_value_t = false)]
pub allow_mock: bool,
```

### Path Arguments
```rust
#[arg(short, long)]
pub model: PathBuf,

#[arg(long)]
pub tokenizer: Option<PathBuf>,
```

### Help Text
```rust
/// Short help text
#[arg(long)]
pub flag: bool,

/// Long help with details
///
/// # Examples
/// - Use this when you need X
/// - Use that when you need Y
#[arg(long)]
pub complex_flag: String,
```

---

## Adding a New Flag: Checklist

- [ ] 1. Add field to struct: `pub field_name: Type`
- [ ] 2. Add clap attribute: `#[arg(long, ...)]`
- [ ] 3. Add doc comment for help text: `/// Description`
- [ ] 4. Add default if needed: `default_value = "..."` or `default_value_t = ...`
- [ ] 5. Handle in command dispatcher or main function
- [ ] 6. Add tests (parse and validation)
- [ ] 7. Document in help text or examples

---

## Error Handling Pattern

```rust
let parsed = flag_string.parse::<MyEnum>()
    .with_context(|| format!(
        "Invalid value '{}'. Supported: option1, option2",
        flag_string
    ))?;
```

This provides context-aware error messages like:
```
Error: Invalid value 'badvalue'. Supported: option1, option2
```

---

## Testing Flags

```rust
#[test]
fn test_flag_parsing() {
    let args = vec!["prog", "--flag", "value"];
    let cli = Cli::try_parse_from(args).unwrap();
    assert_eq!(cli.flag, "value");
}

#[test]
fn test_enum_parsing() {
    assert!("raw".parse::<TemplateType>().is_ok());
    assert!("invalid".parse::<TemplateType>().is_err());
}
```

---

## Pro Tips

1. **Use `value_name = "..."` for clarity in help:**
   ```rust
   #[arg(long, value_name = "PATH")]
   pub model: PathBuf,
   ```
   Shows: `--model <PATH>`

2. **Box large command structs:**
   ```rust
   #[command(subcommand)]
   command: Some(Commands::Inference(Box::new(cmd)))
   ```

3. **Use `#[command(alias = "...")]` for backward compat:**
   ```rust
   #[command(alias = "generate")]
   Run { /* ... */ }
   ```

4. **Feature-gate commands:**
   ```rust
   #[cfg(feature = "inference")]
   #[command(name = "crossval-per-token")]
   CrossvalPerToken { /* ... */ }
   ```

5. **Validate mutually exclusive flags:**
   ```rust
   #[arg(long, conflicts_with = "file")]
   pub text: Option<String>,
   
   #[arg(long, conflicts_with = "text")]
   pub file: Option<PathBuf>,
   ```

---

## Real Examples from BitNet-rs

### Example 1: `--prompt-template` (from bitnet-cli)
```rust
#[arg(long, value_name = "TEMPLATE", default_value = "auto")]
pub prompt_template: String,
```

### Example 2: `--max-tokens` (with aliases)
```rust
#[arg(
    long = "max-tokens",
    visible_aliases = ["max-new-tokens", "n-predict"],
    default_value = "512"
)]
pub max_tokens: usize,
```

### Example 3: `--stop` (repeatable)
```rust
#[arg(
    long = "stop",
    visible_alias = "stop-sequence",
    visible_alias = "stop_sequences"
)]
pub stop: Vec<String>,
```

### Example 4: `--system-prompt` (optional)
```rust
#[arg(long, value_name = "TEXT")]
pub system_prompt: Option<String>,
```

### Example 5: `--allow-mock` (env fallback)
```rust
#[arg(long, env = "BITNET_ALLOW_MOCK", default_value_t = false)]
pub allow_mock: bool,
```

---

## Structure Patterns to Copy

### For reusable command args:
Use `#[derive(Args)]`:
```rust
#[derive(Args)]
pub struct InferenceCommand {
    #[arg(short, long)]
    pub model: PathBuf,
    // ... more args
}
```

### For top-level CLI:
Use `#[derive(Parser)]`:
```rust
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}
```

### For subcommand enums:
Use `#[derive(Subcommand)]`:
```rust
#[derive(Subcommand)]
enum Commands {
    Run { /* inline args */ },
    Inference(Box<InferenceCommand>),
}
```

---

## Common Mistakes to Avoid

1. **Wrong default syntax for types:**
   - WRONG: `#[arg(default_value = 128)]` for usize
   - RIGHT: `#[arg(default_value_t = 128)]`

2. **Forgetting `Option<T>` for optional args:**
   - WRONG: `pub seed: u64,` with no default
   - RIGHT: `pub seed: Option<u64>,`

3. **Not documenting enums:**
   - WRONG: `enum Backend { Cpu, Gpu }` (no FromStr)
   - RIGHT: `impl FromStr for Backend { /* ... */ }`

4. **Mixing `default_value` and `default_value_t`:**
   - Use ONE per flag, not both

5. **Forgetting to parse in handler:**
   - CLI parses strings, you parse to typed values

---

## See Also

- Full report: `CLI_ARGUMENT_PARSING_PATTERNS.md` (984 lines)
- Clap docs: https://docs.rs/clap/latest/clap/
- Examples: Check `crates/bitnet-cli/src/main.rs` (production examples)

