use anyhow::{Result, anyhow};
use std::{collections::HashSet, path::Path};
use walkdir::WalkDir;

pub fn lint_workflows() -> Result<()> {
    let workflows_dir = Path::new(".github/workflows");

    if !workflows_dir.exists() {
        return Err(anyhow!(".github/workflows directory not found"));
    }

    let mut files: Vec<_> = WalkDir::new(workflows_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "yml"))
        .map(|e| e.path().to_path_buf())
        .collect();

    files.sort();

    let mut failed = false;

    for path in files {
        match check_file(&path) {
            Ok(()) => println!("✓ {}", path.display()),
            Err(e) => {
                eprintln!("❌ {}: {}", path.display(), e);
                failed = true;
            }
        }
    }

    if failed {
        return Err(anyhow!("Some workflows have validation errors"));
    }

    println!("\n✓ All workflows valid (no duplicate keys)");
    Ok(())
}

fn check_file(path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    check_duplicate_keys(&content)?;
    let _: serde_yaml::Value =
        serde_yaml::from_str(&content).map_err(|e| anyhow!("YAML parse error: {}", e))?;
    Ok(())
}

fn check_duplicate_keys(content: &str) -> Result<()> {
    let mut frames = Vec::<MappingFrame>::new();
    let mut block_scalar_indent = None;

    for (line_index, raw_line) in content.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim_end_matches('\r');
        let Some(indent) = line.find(|ch| ch != ' ') else {
            continue;
        };
        let trimmed = &line[indent..];

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(block_indent) = block_scalar_indent {
            if indent > block_indent {
                continue;
            }
            block_scalar_indent = None;
        }

        let (key_indent, key_source, starts_sequence_item) =
            if let Some(rest) = trimmed.strip_prefix("- ") {
                (indent + 2, rest, true)
            } else {
                (indent, trimmed, false)
            };

        let Some((key, value)) = parse_mapping_key(key_source) else {
            continue;
        };

        frames.retain(|frame| frame.indent <= key_indent);
        if starts_sequence_item {
            frames.retain(|frame| frame.indent < key_indent);
        }

        let frame_index = match frames.iter().position(|frame| frame.indent == key_indent) {
            Some(index) => index,
            None => {
                frames.push(MappingFrame::new(key_indent));
                frames.len() - 1
            }
        };

        if !frames[frame_index].keys.insert(key.to_string()) {
            return Err(anyhow!("duplicate key '{}' at line {}", key, line_number));
        }

        if value.trim_start().starts_with(['|', '>']) {
            block_scalar_indent = Some(key_indent);
        }
    }

    Ok(())
}

fn parse_mapping_key(line: &str) -> Option<(&str, &str)> {
    let colon_index = if line.starts_with('"') || line.starts_with('\'') {
        quoted_key_end(line).and_then(|end| {
            line[end..].char_indices().find_map(|(offset, ch)| (ch == ':').then_some(end + offset))
        })?
    } else {
        line.find(':')?
    };

    let after_colon = &line[colon_index + 1..];
    if !after_colon.is_empty()
        && !after_colon.starts_with(char::is_whitespace)
        && !after_colon.starts_with(['|', '>', '#'])
    {
        return None;
    }

    let key = line[..colon_index].trim().trim_matches(['"', '\'']);
    (!key.is_empty()).then_some((key, after_colon))
}

fn quoted_key_end(line: &str) -> Option<usize> {
    let mut chars = line.char_indices();
    let (_, quote) = chars.next()?;
    let mut escaped = false;

    for (index, ch) in chars {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == '"' && ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == quote {
            return Some(index + ch.len_utf8());
        }
    }

    None
}

struct MappingFrame {
    indent: usize,
    keys: HashSet<String>,
}

impl MappingFrame {
    fn new(indent: usize) -> Self {
        Self { indent, keys: HashSet::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::check_duplicate_keys;

    #[test]
    fn rejects_duplicate_top_level_keys() {
        let err = check_duplicate_keys("name: CI\non: push\non: pull_request\n").unwrap_err();
        assert!(err.to_string().contains("duplicate key 'on' at line 3"));
    }

    #[test]
    fn rejects_duplicate_nested_keys() {
        let err = check_duplicate_keys(
            "jobs:\n  build:\n    runs-on: ubuntu-22.04\n    runs-on: ubuntu-24.04\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate key 'runs-on' at line 4"));
    }

    #[test]
    fn allows_same_key_in_separate_sequence_items() {
        check_duplicate_keys("steps:\n  - name: Checkout\n    uses: actions/checkout@v4\n  - name: Test\n    run: cargo test\n")
            .unwrap();
    }

    #[test]
    fn rejects_duplicate_key_in_same_sequence_item() {
        let err =
            check_duplicate_keys("steps:\n  - name: Checkout\n    name: Duplicate\n").unwrap_err();
        assert!(err.to_string().contains("duplicate key 'name' at line 3"));
    }

    #[test]
    fn ignores_mapping_like_lines_inside_block_scalars() {
        check_duplicate_keys("jobs:\n  build:\n    steps:\n      - run: |\n          echo 'on: push'\n          echo 'on: pull_request'\n        shell: bash\n")
            .unwrap();
    }
}
