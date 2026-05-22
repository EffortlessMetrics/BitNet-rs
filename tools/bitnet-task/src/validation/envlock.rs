use super::*;

const ALLOWED_ENVLOCK_PATHS: [&str; 3] =
    ["tests/common/env.rs", "tests/static_initialization_tests.rs", "tests/support/env_guard.rs"];

pub(super) fn run(root: &Path) -> Result<()> {
    let violations = find_duplicate_envlock_definitions(root)?;

    if !violations.is_empty() {
        println!("❌ Found duplicate env lock definitions; use common::env_guard()");
        for violation in violations {
            println!("{violation}");
        }
        bail!("duplicate env lock definitions found");
    }

    println!("✅ No duplicate env locks found");
    Ok(())
}

fn find_duplicate_envlock_definitions(root: &Path) -> Result<Vec<String>> {
    let mut violations = Vec::new();
    for path in collect_rust_files(root.join("tests"))? {
        if ALLOWED_ENVLOCK_PATHS.iter().any(|allowed| path.ends_with(allowed)) {
            continue;
        }
        if has_envlock_definition(root, &path)? {
            let line_no = first_envlock_line(&path)?;
            violations.push(format!("{}:{line_no}", path.display()));
        }
    }
    Ok(violations)
}

fn has_envlock_definition(root: &Path, path: &Path) -> Result<bool> {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", rel.display()))?;

    Ok(content
        .lines()
        .map(|line| line.split("//").next().unwrap_or("").trim())
        .filter(|code| !code.is_empty())
        .map(|code| code.chars().filter(|c| !c.is_whitespace()).collect::<String>())
        .any(|compact| compact.contains("OnceLock<Mutex<()>>")))
}

fn first_envlock_line(path: &Path) -> Result<usize> {
    let content = fs::read_to_string(path)?;
    for (line_no, line) in content.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        let compact = code.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        if compact.contains("OnceLock<Mutex<()>>") {
            return Ok(line_no + 1);
        }
    }
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envlock_detection_ignores_comment_only_mentions() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sample.rs");
        fs::write(&path, "// static ENV: OnceLock<Mutex<()>> = OnceLock::new();\n")?;

        assert!(!has_envlock_definition(dir.path(), &path)?);
        Ok(())
    }

    #[test]
    fn envlock_line_reports_code_not_comment() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sample.rs");
        fs::write(
            &path,
            "// static ENV: OnceLock<Mutex<()>> = OnceLock::new();\n\nstatic ENV: OnceLock<Mutex<()>> = OnceLock::new();\n",
        )?;

        assert_eq!(first_envlock_line(&path)?, 3);
        Ok(())
    }
}
