# BitNet.rs CLI Documentation Index

## Overview

This directory contains three comprehensive documents about CLI argument parsing patterns in BitNet.rs, generated from an exploration of the codebase on 2025-10-25.

## Documents

### 1. REPORT_SUMMARY.txt (Executive Summary)
**Size:** 476 lines | **Read time:** 15-20 minutes

Start here for a high-level overview. Contains:
- Key findings about the codebase structure
- Answers to specific questions about requested flags
- Validation patterns discovered
- Environment variable integration examples
- Testing patterns used
- Structure overview of the 3-level CLI hierarchy
- Common clap attributes quick reference
- Key takeaways for contributors

**Best for:** Getting a quick understanding of how CLI works in BitNet.rs

### 2. CLI_PATTERNS_QUICK_REFERENCE.md (Cheat Sheet)
**Size:** 341 lines | **Read time:** 10-15 minutes

Quick lookup guide. Contains:
- File locations for copy-paste examples
- 5 most common CLI patterns
- Enum-based flags explained (the right way)
- Common patterns cheat sheet
- Help text patterns
- Testing flags examples
- Pro tips and best practices
- Real examples from BitNet.rs codebase
- Structure patterns to copy
- Common mistakes to avoid
- See also section with links

**Best for:** Coding while referencing - keep this open in your IDE

### 3. CLI_ARGUMENT_PARSING_PATTERNS.md (Complete Reference)
**Size:** 984 lines | **Read time:** 45-60 minutes (or use as reference)

Comprehensive guide. Contains:
- Part 1: Core Clap Structures (Parser/Args/Subcommand patterns)
- Part 2: Enum-Based Flags (FromStr + Display pattern)
- Part 3: Adding New Flags (5 complete examples with step-by-step)
  - Example 1: Add --cpp-backend flag
  - Example 2: Add --prompt-template flag
  - Example 3: Add --system-prompt flag
  - Example 4: Add --dump-ids flag
  - Example 5: Add --dump-cpp-ids flag
- Part 4: Validation Patterns (5 validation strategies)
- Part 5: Help Text Patterns (5 documentation approaches)
- Part 6: Default Value Patterns (strings/numbers/bools/optional/collections)
- Part 7: Template for New CLI Additions (complete working example)
- Part 8: Advanced Patterns (feature gating, boxing, marker flags, dynamic version)
- Summary Quick Reference Table

**Best for:** Deep understanding and reference for complex patterns

## Quick Start Guide

### I want to...

**Add a simple boolean flag** (5 minutes)
1. Read: QUICK_REFERENCE.md > "The 5 CLI Patterns You'll Use" > Pattern 1
2. Copy the template
3. Add to your struct with proper doc comment
4. Done!

**Add an enum-based flag** (20 minutes)
1. Read: QUICK_REFERENCE.md > "Enum-Based Flags (The Right Way)"
2. Read: CLI_PATTERNS_QUICK_REFERENCE.md > "Enum-Based Flags"
3. Copy the FromStr + Display pattern
4. Use in CLI struct
5. Parse in handler with `.with_context()`
6. Add tests

**Add a complex flag with validation** (45 minutes)
1. Read: QUICK_REFERENCE.md > "Adding a New Flag: Checklist"
2. Read: FULL_REPORT.md > Part 3 (matching your use case)
3. Read: FULL_REPORT.md > Part 4 (validation patterns)
4. Read: FULL_REPORT.md > Part 7 (complete example)
5. Implement using the checklist
6. Test thoroughly

**Understand the current CLI architecture** (20 minutes)
1. Read: REPORT_SUMMARY.txt > "Structure Overview" and "FILE STRUCTURE"
2. Read: FULL_REPORT.md > Part 1
3. Reference: QUICK_REFERENCE.md > "Structure Patterns to Copy"

**Find existing pattern usage** (5 minutes)
1. Check: QUICK_REFERENCE.md > "File Locations for Copy-Paste Examples"
2. Jump to the file and line numbers
3. Copy the exact pattern

## Key Code Locations

| Pattern | File | Lines |
|---------|------|-------|
| Main CLI struct | `crates/bitnet-cli/src/main.rs` | 85-166 |
| Run subcommand | `crates/bitnet-cli/src/main.rs` | 194-303 |
| Inference command args | `crates/bitnet-cli/src/commands/inference.rs` | 131-299 |
| TemplateType enum impl | `crates/bitnet-inference/src/prompt_template.rs` | (see impl block) |
| Xtask commands | `xtask/src/main.rs` | 187-540 |

## Answering Your Questions

The original request asked about adding 5 specific flags. Here's where each is documented:

### Flag 1: --cpp-backend {bitnet\|llama}
- **Status:** Example pattern provided
- **Location:** See FULL_REPORT.md Part 3, Example 1
- **Time to implement:** 20-30 minutes
- **Complexity:** Medium (requires enum with FromStr)

### Flag 2: --prompt-template
- **Status:** Already exists in bitnet-cli
- **Location:** See FULL_REPORT.md Part 3, Example 2
- **Implementation file:** `crates/bitnet-cli/src/main.rs:273`
- **Handler location:** `crates/bitnet-cli/src/main.rs:906-920`

### Flag 3: --system-prompt
- **Status:** Already exists in bitnet-cli
- **Location:** See FULL_REPORT.md Part 3, Example 3
- **Implementation file:** `crates/bitnet-cli/src/main.rs:278`
- **Applied via:** `TemplateType::apply()` method

### Flag 4: --dump-ids
- **Status:** Already exists in bitnet-cli
- **Location:** See FULL_REPORT.md Part 3, Example 4
- **Implementation file:** `crates/bitnet-cli/src/main.rs:254`
- **Handler location:** `crates/bitnet-cli/src/main.rs:1439-1441`

### Flag 5: --dump-cpp-ids
- **Status:** Example pattern provided for xtask
- **Location:** See FULL_REPORT.md Part 7 (complete working example)
- **Pattern:** Feature-gated xtask command
- **Time to implement:** 20-30 minutes
- **Complexity:** Medium (requires feature gating)

## Common Workflows

### Workflow 1: Add a new simple flag
```
1. Open: QUICK_REFERENCE.md
2. Find: Pattern matching your flag type
3. Copy: The template code
4. Add: Doc comment and attributes
5. Parse: In handler function
6. Test: With try_parse_from
```

### Workflow 2: Add a new enum-based flag
```
1. Define: Enum in library crate (e.g., bitnet-inference/src/lib.rs)
2. Implement: FromStr trait with match statement
3. Implement: Display trait
4. Add: As String field in CLI struct
5. Parse: In handler with .parse::<Type>()?
6. Test: Parse success and error cases
```

### Workflow 3: Understand existing flag behavior
```
1. Search: QUICK_REFERENCE.md for similar existing flag
2. Find: File location in table
3. Read: The implementation in source file
4. Check: How it's parsed in handler
5. Look: For tests in same file
```

## Design Principles Found

1. **Strings over Enums in CLI struct** - Keep flags as String, parse in handlers
2. **FromStr + Display for custom types** - Standard Rust pattern, enables flexibility
3. **Doc comments for help text** - Auto-generates help, shows in IDE
4. **Three-level hierarchy** - Parser → Subcommand → Args for modularity
5. **Global flags with global=true** - Apply to all subcommands consistently
6. **Aliases for compatibility** - visible_aliases for backward compatibility
7. **Environment variable fallback** - #[arg(long, env = "VAR")]
8. **Validation in handlers** - .with_context(|| format!(...))? for context

## Testing Patterns

### Testing Flag Parsing
```rust
#[test]
fn test_flag_parsing() {
    let args = vec!["prog", "--flag", "value"];
    let cli = Cli::try_parse_from(&args).unwrap();
    assert_eq!(cli.flag, "value");
}
```

### Testing Enum Parsing
```rust
#[test]
fn test_enum_parsing() {
    assert!("raw".parse::<TemplateType>().is_ok());
    assert!("invalid".parse::<TemplateType>().is_err());
}
```

## Dependencies Found

The project uses:
- **clap** v4: CLI parsing with derive macros
- **anyhow**: Error handling with context
- **serde**: JSON serialization
- **std::str::FromStr**: Standard trait for string parsing

## Validation Checklist

Before submitting a new flag:
- [ ] Flag has clear, unambiguous name
- [ ] Default value makes sense
- [ ] Help text explains purpose
- [ ] Implementation is in correct struct
- [ ] Parser validates input
- [ ] Error messages are actionable
- [ ] Tests cover success and error cases
- [ ] Aliases documented (if applicable)

## Document Statistics

| Document | Lines | Size | Read Time |
|----------|-------|------|-----------|
| REPORT_SUMMARY.txt | 476 | 16 KB | 15-20 min |
| CLI_PATTERNS_QUICK_REFERENCE.md | 341 | 7 KB | 10-15 min |
| CLI_ARGUMENT_PARSING_PATTERNS.md | 984 | 25 KB | 45-60 min |
| **Total** | **1,801** | **48 KB** | **90 min** |

## Tips for Using These Documents

1. **On First Read:** Start with REPORT_SUMMARY.txt for overview
2. **While Coding:** Keep QUICK_REFERENCE.md open in IDE tab
3. **For Deep Dives:** Reference FULL_REPORT.md sections
4. **For Specific Patterns:** Use file locations table to jump directly
5. **For Testing:** See Testing Patterns section above
6. **For Examples:** All documents contain real BitNet.rs code

## Additional Resources

- Clap documentation: https://docs.rs/clap/latest/clap/
- Rust FromStr trait: https://doc.rust-lang.org/std/str/trait.FromStr.html
- Anyhow error handling: https://docs.rs/anyhow/latest/anyhow/
- Source files:
  - `crates/bitnet-cli/src/main.rs` (production CLI)
  - `xtask/src/main.rs` (developer tools)
  - `crates/bitnet-inference/src/prompt_template.rs` (enum example)

## Questions?

If you have questions about CLI patterns in BitNet.rs:

1. Check the "Answering Your Questions" section above
2. Search QUICK_REFERENCE.md for similar patterns
3. Look up specific file in "Key Code Locations" table
4. Read relevant section in FULL_REPORT.md
5. Check the actual source file (always the most authoritative)

---

**Last Updated:** 2025-10-25
**Branch:** feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2
**Explored By:** Claude Code (claude-haiku-4-5-20251001)

