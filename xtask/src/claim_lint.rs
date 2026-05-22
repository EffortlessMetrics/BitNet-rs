use anyhow::{Context, Result, bail};
use clap::Args;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct ClaimLintArgs {
    /// Claim-lint scope to check.
    #[arg(long, default_value = "apple-m4")]
    pub scope: String,
    /// Check mode: fail if any unsupported claim-boundary wording is present.
    #[arg(long, default_value_t = false)]
    pub check: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    UnsupportedAppleSilicon,
    MacBook,
    FullMetal,
    NeuralEngine,
    MpsGraph,
    Qk256,
    DenseAsBitNet,
    BroadQuality,
    PerformanceOrSpeedup,
}

impl Rule {
    fn label(self) -> &'static str {
        match self {
            Self::UnsupportedAppleSilicon => "unsupported-apple-silicon",
            Self::MacBook => "macbook",
            Self::FullMetal => "full-metal",
            Self::NeuralEngine => "neural-engine",
            Self::MpsGraph => "mpsgraph",
            Self::Qk256 => "qk256",
            Self::DenseAsBitNet => "dense-as-bitnet",
            Self::BroadQuality => "broad-quality",
            Self::PerformanceOrSpeedup => "performance-or-speedup",
        }
    }

    fn allowed(self, lower: &str) -> bool {
        match self {
            Self::FullMetal => {
                is_boundary_context(lower)
                    || (is_metal_receipt_context(lower) && !is_full_model_metal_claim(lower))
            }
            Self::MpsGraph => is_boundary_context(lower) || is_mpsgraph_receipt_context(lower),
            Self::UnsupportedAppleSilicon
            | Self::MacBook
            | Self::NeuralEngine
            | Self::Qk256
            | Self::DenseAsBitNet
            | Self::BroadQuality
            | Self::PerformanceOrSpeedup => is_boundary_context(lower),
        }
    }
}

#[derive(Debug)]
struct Issue {
    path: PathBuf,
    line: usize,
    rule: Rule,
    text: String,
}

pub fn run(args: ClaimLintArgs) -> Result<()> {
    if args.scope != "apple-m4" {
        bail!("unsupported claim-lint scope `{}`; expected `apple-m4`", args.scope);
    }

    let root = std::env::current_dir().context("resolve current directory")?;
    let targets = apple_m4_targets(&root)?;
    let issues = scan_targets(&root, &targets)?;

    if issues.is_empty() {
        let mode = if args.check { "check" } else { "scan" };
        println!("claim lint passed: scope=apple-m4 mode={mode} files={}", targets.len());
        return Ok(());
    }

    let status = if args.check { "failed" } else { "found issues" };
    eprintln!(
        "claim lint {status}: scope=apple-m4 issues={} files={}",
        issues.len(),
        targets.len()
    );
    for issue in &issues {
        let display = issue.path.strip_prefix(&root).unwrap_or(&issue.path);
        eprintln!(
            "{}:{}: {}: {}",
            display.display(),
            issue.line,
            issue.rule.label(),
            issue.text.trim()
        );
    }
    if args.check {
        bail!("apple-m4 claim lint found unsupported claim-boundary wording");
    }
    Ok(())
}

fn apple_m4_targets(root: &Path) -> Result<Vec<PathBuf>> {
    let mut targets = BTreeSet::new();

    let slm_dir = root.join("docs/slm");
    if slm_dir.exists() {
        for entry in
            fs::read_dir(&slm_dir).with_context(|| format!("read {}", slm_dir.display()))?
        {
            let entry = entry.with_context(|| format!("read entry in {}", slm_dir.display()))?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("apple-m4") && name.ends_with(".md") {
                targets.insert(path);
            }
        }
    }

    for relative in [
        "docs/tracking/campaigns/apple-m4-inference-excellence/generated/status.md",
        "crates/bitnet-cli/src/mac.rs",
    ] {
        let path = root.join(relative);
        if path.exists() {
            targets.insert(path);
        }
    }

    Ok(targets.into_iter().collect())
}

fn scan_targets(root: &Path, targets: &[PathBuf]) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();
    for path in targets {
        let contents =
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        issues.extend(scan_file(root, path, &contents));
    }
    Ok(issues)
}

fn scan_file(root: &Path, path: &Path, contents: &str) -> Vec<Issue> {
    if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
        return contents
            .lines()
            .enumerate()
            .flat_map(|(index, line)| scan_rust_line(root, path, index + 1, line))
            .collect();
    }
    scan_markdown_blocks(path, contents)
}

fn scan_rust_line(_root: &Path, path: &Path, line: usize, text: &str) -> Vec<Issue> {
    let is_rejection_message = text.contains("bail!")
        || text.contains("ensure!")
        || text.contains("context(")
        || text.contains("error")
        || text.contains("blockers.push");
    extract_rust_strings(text)
        .into_iter()
        .filter(|text| !is_identifier_like(text))
        .flat_map(|text| {
            let scan = if is_rejection_message {
                format!("claim lint rejects {text}")
            } else {
                text.to_string()
            };
            scan_text(path, line, text, &scan)
        })
        .collect()
}

fn scan_markdown_blocks(path: &Path, contents: &str) -> Vec<Issue> {
    let mut issues = Vec::new();
    let mut block_start = 1;
    let mut block = String::new();
    let mut in_fence = false;
    let mut negative_claim_list = false;

    for (index, line) in contents.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            flush_block(path, block_start, &mut block, &mut issues);
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let normalized_line = normalize(trimmed);
        if trimmed.starts_with('#') {
            negative_claim_list = false;
        }
        if contains_any(
            &normalized_line,
            &[
                " must not claim ",
                " must not include ",
                " does not claim ",
                " do not claim ",
                " does not own ",
                " not allowed now ",
                " should reject ",
                " rejects unsupported ",
                " forbidden claims ",
                " out of scope ",
                " remain out of scope ",
                " remains out of scope ",
                " remain deferred ",
                " remains deferred ",
            ],
        ) {
            negative_claim_list = true;
        }
        if trimmed.is_empty() {
            flush_block(path, block_start, &mut block, &mut issues);
            continue;
        }
        if block.is_empty() {
            block_start = line_number;
        } else {
            block.push(' ');
        }
        if negative_claim_list && trimmed.starts_with("- ") && block.is_empty() {
            block.push_str("must not claim ");
        }
        block.push_str(trimmed);
    }

    flush_block(path, block_start, &mut block, &mut issues);
    issues
}

fn flush_block(path: &Path, line: usize, block: &mut String, issues: &mut Vec<Issue>) {
    if block.is_empty() {
        return;
    }
    issues.extend(scan_text(path, line, block, block));
    block.clear();
}

fn scan_text(path: &Path, line: usize, display_text: &str, text: &str) -> Vec<Issue> {
    let lower = normalize(text);
    rules_for_line(&lower)
        .into_iter()
        .filter(|rule| !rule.allowed(&lower))
        .map(|rule| Issue { path: path.to_path_buf(), line, rule, text: display_text.to_string() })
        .collect()
}

fn rules_for_line(lower: &str) -> Vec<Rule> {
    let mut rules = Vec::new();

    if (contains_any(lower, &["broad apple silicon", "general apple silicon"])
        || (lower.contains("apple silicon")
            && contains_any(
                lower,
                &[
                    " support",
                    " benchmark",
                    " performance",
                    " quality",
                    " speedup",
                    " acceleration",
                    " proof",
                ],
            )))
        && !lower.contains("m4 mac mini")
    {
        rules.push(Rule::UnsupportedAppleSilicon);
    }

    if lower.contains("macbook") {
        rules.push(Rule::MacBook);
    }

    if contains_any(
        lower,
        &[
            "full metal",
            "full apple-m4-metal",
            "full apple m4 metal",
            "native metal",
            "end-to-end metal",
            "end to end metal",
            "apple metal",
            "apple m4 metal",
            "complete metal",
            "metal inference",
            "metal model inference",
            "metal bitnet inference",
            "metal slm inference",
            "metal backend",
            "metal route",
            "metal support",
            "metal performance",
            "metal speedup",
            "metal acceleration",
        ],
    ) {
        rules.push(Rule::FullMetal);
    }

    if lower.contains("neural engine") || contains_token(lower, "ane") {
        rules.push(Rule::NeuralEngine);
    }

    if lower.contains("mpsgraph") {
        rules.push(Rule::MpsGraph);
    }

    if lower.contains("qk256") {
        rules.push(Rule::Qk256);
    }

    if mentions_dense_as_bitnet(lower) {
        rules.push(Rule::DenseAsBitNet);
    }

    if contains_any(
        lower,
        &[
            "broad quality",
            "general quality",
            "broad answer quality",
            "general chat quality",
            "quality claim",
            "quality claims",
        ],
    ) {
        rules.push(Rule::BroadQuality);
    }

    if contains_any(
        lower,
        &[
            "broad performance",
            "general performance",
            "platform performance",
            "performance claim",
            "performance claims",
            "speedup",
            "faster than",
        ],
    ) {
        rules.push(Rule::PerformanceOrSpeedup);
    }

    rules
}

fn normalize(text: &str) -> String {
    let mut lower = String::with_capacity(text.len() + 2);
    lower.push(' ');
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            lower.push(character.to_ascii_lowercase());
        } else {
            lower.push(' ');
        }
    }
    lower.push(' ');
    lower
}

fn is_boundary_context(lower: &str) -> bool {
    contains_any(
        lower,
        &[
            " not ",
            " no ",
            " without ",
            " does not ",
            " do not ",
            " cannot ",
            " can not ",
            " must not ",
            " excludes ",
            " disabled ",
            " unsupported ",
            " not proven ",
            " not a ",
            " false ",
            " separated ",
            " separation ",
            " gated ",
            " fail ",
            " fails ",
            " forbid ",
            " forbids ",
            " forbidden ",
            " reject ",
            " rejects ",
            " rejected ",
            " separate from ",
            " separate gated ",
            " separate proof ",
            " require ",
            " requires ",
            " required ",
            " keep ",
            " keeps ",
            " keeping ",
            " preserve ",
            " preserves ",
            " preserving ",
            " remains ",
            " remain ",
            " limited ",
            " scoped ",
            " bounded ",
            " until ",
            " unless ",
            " before ",
            " tied to ",
            " plausible ",
            " candidate ",
            " candidate only ",
            " diagnostic only ",
            " reference only ",
            " unchanged ",
            " outside scope ",
            " pending ",
            " advisory ",
            " retired ",
            " only when ",
            " only after ",
            " matching accepted receipt ",
            " receipt backed ",
        ],
    )
}

fn is_metal_receipt_context(lower: &str) -> bool {
    contains_any(
        lower,
        &[
            " receipt ",
            " gate ",
            " proof ",
            " evidence ",
            " parity ",
            " smoke ",
            " probe ",
            " visibility ",
            " diagnostic ",
            " candidate ",
        ],
    )
}

fn is_mpsgraph_receipt_context(lower: &str) -> bool {
    contains_any(
        lower,
        &[" receipt ", " smoke ", " reference ", " graph reference ", " probe ", " diagnostic "],
    )
}

fn is_full_model_metal_claim(lower: &str) -> bool {
    contains_any(
        lower,
        &[
            " full metal ",
            " full apple m4 metal ",
            " end to end metal ",
            " complete metal ",
            " metal model inference ",
            " metal support ",
            " metal performance ",
            " metal speedup ",
            " metal acceleration ",
        ],
    )
}

fn mentions_dense_as_bitnet(lower: &str) -> bool {
    if !(lower.contains("dense") && lower.contains("bitnet")) {
        return false;
    }
    let dense_first = lower.find("dense").unwrap_or(usize::MAX);
    let bitnet_first = lower.find("bitnet").unwrap_or(usize::MAX);
    let gap = dense_first.abs_diff(bitnet_first);
    gap <= 120
        && contains_any(
            lower,
            &[
                " dense evidence as bitnet ",
                " dense slm evidence as bitnet ",
                " dense receipt as bitnet ",
                " dense slm receipt as bitnet ",
                " dense evidence proves bitnet ",
                " dense slm evidence proves bitnet ",
                " dense evidence supports bitnet ",
                " dense slm evidence supports bitnet ",
                " dense evidence counts as bitnet ",
                " dense slm evidence counts as bitnet ",
                " dense quality ",
                " dense slm quality ",
                " bitnet quality ",
            ],
        )
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn contains_token(haystack: &str, token: &str) -> bool {
    haystack.split_whitespace().any(|part| part == token)
}

fn extract_rust_strings(line: &str) -> Vec<&str> {
    let mut strings = Vec::new();
    let mut start = None;
    let mut escaped = false;

    for (index, character) in line.char_indices() {
        if start.is_some() {
            if escaped {
                escaped = false;
                continue;
            }
            if character == '\\' {
                escaped = true;
                continue;
            }
            if character == '"' {
                if let Some(start_index) = start.take() {
                    strings.push(&line[start_index..index]);
                }
            }
            continue;
        }
        if character == '"' {
            start = Some(index + character.len_utf8());
        }
    }

    strings
}

fn is_identifier_like(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && trimmed.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | '/')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_rules(text: &str) -> Vec<Rule> {
        let lower = normalize(text);
        rules_for_line(&lower).into_iter().filter(|rule| !rule.allowed(&lower)).collect()
    }

    #[test]
    fn rejects_positive_unsupported_claims() {
        assert_eq!(line_rules("Neural Engine execution is enabled"), vec![Rule::NeuralEngine]);
        assert_eq!(
            line_rules("The route now has broad Apple Silicon support"),
            vec![Rule::UnsupportedAppleSilicon]
        );
        assert_eq!(
            line_rules("Dense SLM evidence proves BitNet quality"),
            vec![Rule::DenseAsBitNet]
        );
        assert_eq!(line_rules("The M4 path has speedup claims"), vec![Rule::PerformanceOrSpeedup]);
    }

    #[test]
    fn allows_boundary_and_receipt_limited_wording() {
        assert!(line_rules("Neural Engine execution is not proven by this bundle").is_empty());
        assert!(line_rules("Tiny MPSGraph reference smoke receipt is present").is_empty());
        assert!(line_rules("The Metal parity receipt is evidence only").is_empty());
        assert!(line_rules("The lifecycle policy does not broaden Apple Silicon, Metal, QK256, Neural Engine, MPSGraph, MacBook, quality, performance, or speedup claims.").is_empty());
        assert!(line_rules("\"qk256_apple_claimed\": false,").is_empty());
        assert!(
            line_rules("M4 claim lint rejects broad Apple Silicon or QK256 wording.").is_empty()
        );
    }

    #[test]
    fn rust_identifier_like_strings_are_not_public_claim_text() {
        assert!(is_identifier_like("qk256_neural_engine_mpsgraph_macbook_broad_apple_silicon"));
        assert!(!is_identifier_like("Neural Engine execution is enabled"));
    }
}
