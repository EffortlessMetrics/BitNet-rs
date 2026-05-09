use anyhow::{Context, Result, bail};
use std::{fs, path::Path, process::Command};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixLockedMode {
    Apply,
    DryRun,
    Check,
}

pub(crate) fn cmd_fix_locked(_root: &Path, mode: FixLockedMode, files: Vec<String>) -> Result<()> {
    if files.is_empty() {
        print_usage();
        bail!("missing input files");
    }

    let mut changes_detected = false;

    for file in files {
        let path = Path::new(&file);
        if !path.is_file() {
            eprintln!("Warning: File not found: {file}");
            continue;
        }

        let original = fs::read_to_string(path).with_context(|| format!("reading {file}"))?;
        let updated = add_locked_flags(&original);
        if original == updated {
            if mode == FixLockedMode::DryRun {
                println!("No changes: {file}");
            }
            continue;
        }

        changes_detected = true;
        match mode {
            FixLockedMode::Apply => {
                fs::write(path, updated).with_context(|| format!("writing {file}"))?;
                println!("✓ Updated: {file}");
            }
            FixLockedMode::DryRun => {
                println!("Would update: {file}");
                println!("--- Diff ---");
                print_unified_diff(path, &updated)?;
                println!();
            }
            FixLockedMode::Check => {
                eprintln!("Changes needed in: {file}");
            }
        }
    }

    match mode {
        FixLockedMode::Apply => println!("✓ Applied --locked where missing"),
        FixLockedMode::DryRun => {
            if changes_detected {
                println!("⚠ Changes would be made (see diffs above)");
            } else {
                println!("✓ No changes needed (all files already have --locked)");
            }
        }
        FixLockedMode::Check => {
            if changes_detected {
                eprintln!("❌ Some files are missing --locked flags");
                eprintln!("Run: scripts/fix-locked.sh .github/workflows/*.yml");
                bail!("missing --locked flags");
            }
            println!("✓ All cargo commands have --locked flags");
        }
    }

    Ok(())
}

fn print_usage() {
    eprintln!("Usage: scripts/fix-locked.sh [--dry-run|--check] <file1> [file2 ...]");
    eprintln!();
    eprintln!("Modes:");
    eprintln!("  (default)  Apply changes in-place");
    eprintln!("  --dry-run  Show what would be changed (no modifications)");
    eprintln!("  --check    Exit with non-zero if changes would be made (CI mode)");
}

fn print_unified_diff(path: &Path, updated: &str) -> Result<()> {
    let temp_dir = tempfile::Builder::new()
        .prefix("bitnet-task-fix-locked-")
        .tempdir()
        .context("creating temporary diff directory")?;
    let new_path = temp_dir.path().join("updated");
    fs::write(&new_path, updated).context("writing temporary diff file")?;

    let status = Command::new("diff")
        .arg("-u")
        .arg(path)
        .arg(&new_path)
        .status()
        .context("running diff -u")?;

    if !status.success() && status.code() != Some(1) {
        bail!("diff failed with {status}");
    }
    Ok(())
}

pub(crate) fn add_locked_flags(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for line in input.split_inclusive('\n') {
        let (line_without_newline, newline) = match line.strip_suffix('\n') {
            Some(stripped) => (stripped, "\n"),
            None => (line, ""),
        };
        output.push_str(&add_locked_to_line(line_without_newline));
        output.push_str(newline);
    }
    output
}

fn add_locked_to_line(line: &str) -> String {
    if is_comment(line) || has_locked(line) || !is_target_command_line(line) {
        return line.to_string();
    }

    let (mut command, comment) = split_inline_comment(line);
    let (without_backslash, backslash) = strip_trailing_backslash(command);
    command = without_backslash;

    let mut updated = if let Some(index) = find_double_dash_separator(command) {
        let (prefix, suffix) = command.split_at(index);
        format!("{prefix} --locked{suffix}")
    } else {
        format!("{} --locked", command.trim_end())
    };

    if backslash {
        updated.push_str(" \\");
    }
    if let Some(comment) = comment {
        updated.push_str(comment);
    }
    updated
}

fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

fn has_locked(line: &str) -> bool {
    line.split_whitespace().any(|token| token == "--locked")
}

fn is_target_command_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    is_cargo_or_cross_invocation(trimmed)
        || trimmed
            .strip_prefix("run:")
            .is_some_and(|command| is_cargo_or_cross_invocation(command.trim_start()))
}

fn is_cargo_or_cross_invocation(command: &str) -> bool {
    let mut words = command.split_whitespace();
    while words.clone().next().is_some_and(is_shell_assignment) {
        words.next();
    }
    matches!(words.next(), Some("cargo" | "cross"))
        && matches!(words.next(), Some("build" | "test" | "run" | "bench" | "clippy"))
}

fn is_shell_assignment(word: &str) -> bool {
    let Some((name, _)) = word.split_once('=') else {
        return false;
    };
    let mut chars = name.chars();
    matches!(chars.next(), Some('_') | Some('A'..='Z') | Some('a'..='z'))
        && chars.all(|ch| matches!(ch, '_' | 'A'..='Z' | 'a'..='z' | '0'..='9'))
}

fn split_inline_comment(line: &str) -> (&str, Option<&str>) {
    match line.find('#') {
        Some(hash_index) => {
            let comment_start = line[..hash_index]
                .char_indices()
                .rev()
                .find_map(|(index, ch)| (!ch.is_whitespace()).then_some(index + ch.len_utf8()))
                .unwrap_or(0);
            (&line[..comment_start], Some(&line[comment_start..]))
        }
        None => (line, None),
    }
}

fn strip_trailing_backslash(line: &str) -> (&str, bool) {
    let trimmed = line.trim_end();
    match trimmed.strip_suffix('\\') {
        Some(stripped) => (stripped.trim_end(), true),
        None => (line, false),
    }
}

fn find_double_dash_separator(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    bytes.windows(4).position(|window| {
        window[0].is_ascii_whitespace()
            && window[1] == b'-'
            && window[2] == b'-'
            && window[3].is_ascii_whitespace()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("scripts/tests/fixtures")
            .join(name)
    }

    #[test]
    fn matches_existing_fix_locked_fixtures() {
        for fixture in [
            "01-simple",
            "02-multiline",
            "03-with-comments",
            "04-double-dash",
            "05-already-locked",
            "06-cross-tool",
            "07-cargo-run-with-args",
            "08-non-cargo",
        ] {
            let input = fs::read_to_string(fixture_path(&format!("{fixture}.yml"))).unwrap();
            let expected =
                fs::read_to_string(fixture_path(&format!("{fixture}.expected.yml"))).unwrap();
            assert_eq!(add_locked_flags(&input), expected, "fixture {fixture}");
        }
    }

    #[test]
    fn preserves_no_trailing_newline() {
        assert_eq!(add_locked_flags("run: cargo test"), "run: cargo test --locked");
    }

    #[test]
    fn handles_env_prefixed_cargo_commands() {
        assert_eq!(
            add_locked_flags("RUSTFLAGS=-Dwarnings cargo test --workspace"),
            "RUSTFLAGS=-Dwarnings cargo test --workspace --locked"
        );
        assert_eq!(
            add_locked_flags("run: CARGO_TERM_COLOR=always cargo clippy -- -D warnings"),
            "run: CARGO_TERM_COLOR=always cargo clippy --locked -- -D warnings"
        );
    }
}
