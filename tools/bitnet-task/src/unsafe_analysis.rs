use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct UnsafeOccurrence {
    line: usize,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct FileUnsafeAnalysis {
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    unsafe_blocks: Vec<UnsafeOccurrence>,
    unsafe_functions: Vec<UnsafeOccurrence>,
    unsafe_traits: Vec<UnsafeOccurrence>,
    unsafe_impls: Vec<UnsafeOccurrence>,
}

#[derive(Debug, Clone, Copy, Default)]
struct UnsafeTotals {
    unsafe_blocks: usize,
    unsafe_functions: usize,
    unsafe_traits: usize,
    unsafe_impls: usize,
}

impl UnsafeTotals {
    fn add_analysis(&mut self, analysis: &FileUnsafeAnalysis) {
        self.unsafe_blocks += analysis.unsafe_blocks.len();
        self.unsafe_functions += analysis.unsafe_functions.len();
        self.unsafe_traits += analysis.unsafe_traits.len();
        self.unsafe_impls += analysis.unsafe_impls.len();
    }
}

pub(crate) fn cmd_analyze_unsafe(root: &Path) -> Result<()> {
    eprintln!("Analyzing unsafe code usage...");

    let rust_files = find_rust_files(root)?;
    eprintln!("Found {} Rust files to analyze", rust_files.len());

    let crate_info = get_crate_info(root);
    let analyses =
        rust_files.iter().map(|file| analyze_unsafe_usage(root, file)).collect::<Vec<_>>();
    let report = generate_unsafe_report(&analyses, crate_info);

    println!("{}", serde_json::to_string_pretty(&report)?);
    print_summary(&report);
    Ok(())
}

fn find_rust_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut rust_files = Vec::new();
    collect_rust_files(&root.join("src"), &mut rust_files)?;
    collect_rust_files(&root.join("crates"), &mut rust_files)?;

    rust_files.retain(|path| {
        let display = path.to_string_lossy();
        !display.contains("target/") && !display.ends_with("test.rs")
    });
    rust_files.sort();
    Ok(rust_files)
}

fn collect_rust_files(path: &Path, rust_files: &mut Vec<PathBuf>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
        let entry = entry.with_context(|| format!("reading entry under {}", path.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("reading file type for {}", path.display()))?;

        if file_type.is_dir() {
            collect_rust_files(&path, rust_files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            rust_files.push(path);
        }
    }

    Ok(())
}

fn analyze_unsafe_usage(root: &Path, file_path: &Path) -> FileUnsafeAnalysis {
    let relative = relative_path(root, file_path);
    let content = match fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(error) => {
            return FileUnsafeAnalysis {
                file: relative,
                error: Some(error.to_string()),
                unsafe_blocks: Vec::new(),
                unsafe_functions: Vec::new(),
                unsafe_traits: Vec::new(),
                unsafe_impls: Vec::new(),
            };
        }
    };

    let lines = content.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    let mut analysis = FileUnsafeAnalysis {
        file: relative,
        error: None,
        unsafe_blocks: Vec::new(),
        unsafe_functions: Vec::new(),
        unsafe_traits: Vec::new(),
        unsafe_impls: Vec::new(),
    };

    for (index, line) in lines.iter().enumerate() {
        let line_num = index + 1;
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("/*") {
            continue;
        }

        if contains_word_followed_by(line, "unsafe", "{") {
            analysis.unsafe_blocks.push(UnsafeOccurrence {
                line: line_num,
                content: line.trim().to_string(),
                name: None,
                context: Some(context_lines(&lines, index)),
            });
        }

        if let Some(name) = name_after_unsafe_keyword(line, "fn") {
            analysis.unsafe_functions.push(UnsafeOccurrence {
                line: line_num,
                content: line.trim().to_string(),
                name: Some(name),
                context: None,
            });
        }

        if let Some(name) = name_after_unsafe_keyword(line, "trait") {
            analysis.unsafe_traits.push(UnsafeOccurrence {
                line: line_num,
                content: line.trim().to_string(),
                name: Some(name),
                context: None,
            });
        }

        if contains_word_followed_by(line, "unsafe", "impl") {
            analysis.unsafe_impls.push(UnsafeOccurrence {
                line: line_num,
                content: line.trim().to_string(),
                name: None,
                context: None,
            });
        }
    }

    analysis
}

fn context_lines(lines: &[String], unsafe_index: usize) -> Vec<String> {
    let start = unsafe_index.saturating_sub(3);
    let end = (unsafe_index + 3).min(lines.len());
    lines[start..end].to_vec()
}

fn contains_word_followed_by(line: &str, word: &str, next: &str) -> bool {
    let mut remainder = line;
    while let Some(offset) = remainder.find(word) {
        let absolute = line.len() - remainder.len() + offset;
        let before = line[..absolute].chars().next_back();
        let after_word = absolute + word.len();
        let after = line[after_word..].chars().next();
        let word_boundary_before = before.is_none_or(|ch| !is_ident_char(ch));
        let word_boundary_after = after.is_none_or(|ch| !is_ident_char(ch));

        if word_boundary_before && word_boundary_after {
            let following = line[after_word..].trim_start();
            if following.starts_with(next) {
                return true;
            }
        }
        remainder = &line[after_word..];
    }
    false
}

fn name_after_unsafe_keyword(line: &str, keyword: &str) -> Option<String> {
    let unsafe_pos = find_word(line, "unsafe")?;
    let after_unsafe = line[unsafe_pos + "unsafe".len()..].trim_start();
    let keyword_pos = find_word(after_unsafe, keyword)?;
    if keyword_pos != 0 {
        return None;
    }
    let after_keyword = after_unsafe[keyword.len()..].trim_start();
    let name = after_keyword.chars().take_while(|ch| is_ident_char(*ch)).collect::<String>();
    (!name.is_empty()).then_some(name)
}

fn find_word(line: &str, word: &str) -> Option<usize> {
    let mut remainder = line;
    while let Some(offset) = remainder.find(word) {
        let absolute = line.len() - remainder.len() + offset;
        let before = line[..absolute].chars().next_back();
        let after_word = absolute + word.len();
        let after = line[after_word..].chars().next();
        if before.is_none_or(|ch| !is_ident_char(ch)) && after.is_none_or(|ch| !is_ident_char(ch)) {
            return Some(absolute);
        }
        remainder = &line[after_word..];
    }
    None
}

fn is_ident_char(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

fn get_crate_info(root: &Path) -> Value {
    match Command::new("cargo")
        .current_dir(root)
        .args(["metadata", "--format-version", "1"])
        .output()
    {
        Ok(output) if output.status.success() => {
            match serde_json::from_slice::<Value>(&output.stdout) {
                Ok(metadata) => {
                    let packages = metadata["packages"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|package| package["source"].is_null())
                        .map(|package| {
                            json!({
                                "name": package["name"],
                                "version": package["version"],
                                "manifest_path": package["manifest_path"],
                            })
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "workspace_root": metadata["workspace_root"],
                        "packages": packages,
                    })
                }
                Err(error) => crate_info_error(root, error.to_string()),
            }
        }
        Ok(output) => {
            crate_info_error(root, String::from_utf8_lossy(&output.stderr).trim().to_string())
        }
        Err(error) => crate_info_error(root, error.to_string()),
    }
}

fn crate_info_error(root: &Path, error: String) -> Value {
    json!({
        "error": error,
        "workspace_root": root.display().to_string(),
        "packages": [],
    })
}

fn generate_unsafe_report(analyses: &[FileUnsafeAnalysis], crate_info: Value) -> Value {
    let mut totals = UnsafeTotals::default();
    let mut files_with_unsafe = 0usize;
    let mut by_crate = serde_json::Map::new();

    for analysis in analyses {
        totals.add_analysis(analysis);
        if has_unsafe(analysis) {
            files_with_unsafe += 1;
        }

        let crate_name = crate_name_for_file(&analysis.file);
        let entry = by_crate.entry(crate_name).or_insert_with(|| {
            json!({
                "files": [],
                "unsafe_blocks": 0,
                "unsafe_functions": 0,
                "unsafe_traits": 0,
                "unsafe_impls": 0,
            })
        });
        if let Some(object) = entry.as_object_mut() {
            if let Some(files) = object.get_mut("files").and_then(Value::as_array_mut) {
                files.push(json!(analysis));
            }
            increment_field(object, "unsafe_blocks", analysis.unsafe_blocks.len());
            increment_field(object, "unsafe_functions", analysis.unsafe_functions.len());
            increment_field(object, "unsafe_traits", analysis.unsafe_traits.len());
            increment_field(object, "unsafe_impls", analysis.unsafe_impls.len());
        }
    }

    let unsafe_percentage = if analyses.is_empty() {
        0.0
    } else {
        files_with_unsafe as f64 / analyses.len() as f64 * 100.0
    };

    json!({
        "timestamp": unix_timestamp(),
        "crate_info": crate_info,
        "summary": {
            "total_files_analyzed": analyses.len(),
            "files_with_unsafe": files_with_unsafe,
            "total_unsafe_blocks": totals.unsafe_blocks,
            "total_unsafe_functions": totals.unsafe_functions,
            "total_unsafe_traits": totals.unsafe_traits,
            "total_unsafe_impls": totals.unsafe_impls,
            "unsafe_percentage": unsafe_percentage,
        },
        "by_crate": by_crate,
        "detailed_analysis": analyses,
        "recommendations": generate_recommendations(analyses),
    })
}

fn increment_field(object: &mut serde_json::Map<String, Value>, key: &str, increment: usize) {
    let current = object.get(key).and_then(Value::as_u64).unwrap_or(0);
    object.insert(key.to_string(), json!(current + increment as u64));
}

fn has_unsafe(analysis: &FileUnsafeAnalysis) -> bool {
    !(analysis.unsafe_blocks.is_empty()
        && analysis.unsafe_functions.is_empty()
        && analysis.unsafe_traits.is_empty()
        && analysis.unsafe_impls.is_empty())
}

fn crate_name_for_file(file_path: &str) -> String {
    file_path
        .split_once("crates/")
        .and_then(|(_, rest)| rest.split('/').next())
        .filter(|name| !name.is_empty())
        .unwrap_or("root")
        .to_string()
}

fn generate_recommendations(analyses: &[FileUnsafeAnalysis]) -> Vec<String> {
    let total_unsafe_blocks =
        analyses.iter().map(|analysis| analysis.unsafe_blocks.len()).sum::<usize>();
    let files_with_unsafe = analyses
        .iter()
        .filter(|analysis| {
            !analysis.unsafe_blocks.is_empty() || !analysis.unsafe_functions.is_empty()
        })
        .count();

    let mut recommendations = Vec::new();
    if total_unsafe_blocks == 0 {
        recommendations.push("✅ No unsafe blocks found - excellent memory safety!".to_string());
    } else if total_unsafe_blocks < 10 {
        recommendations.push("✅ Low unsafe code usage - good safety practices".to_string());
    } else if total_unsafe_blocks < 50 {
        recommendations.push("⚠️ Moderate unsafe code usage - consider safety review".to_string());
    } else {
        recommendations
            .push("❌ High unsafe code usage - requires thorough safety audit".to_string());
    }

    if files_with_unsafe > 0 {
        recommendations.push(format!("📋 Review {files_with_unsafe} files containing unsafe code"));
        recommendations.push("📝 Ensure all unsafe blocks have safety comments".to_string());
        recommendations.push("🧪 Add comprehensive tests for unsafe code paths".to_string());
        recommendations.push("👥 Consider peer review for all unsafe code changes".to_string());
    }

    recommendations
}

fn print_summary(report: &Value) {
    let summary = &report["summary"];
    eprintln!("\n=== Unsafe Code Analysis Summary ===");
    eprintln!("Files analyzed: {}", summary["total_files_analyzed"]);
    eprintln!("Files with unsafe code: {}", summary["files_with_unsafe"]);
    eprintln!("Unsafe blocks: {}", summary["total_unsafe_blocks"]);
    eprintln!("Unsafe functions: {}", summary["total_unsafe_functions"]);
    eprintln!("Unsafe traits: {}", summary["total_unsafe_traits"]);
    eprintln!("Unsafe impls: {}", summary["total_unsafe_impls"]);
    eprintln!(
        "Unsafe percentage: {:.1}%",
        summary["unsafe_percentage"].as_f64().unwrap_or_default()
    );

    eprintln!("\nRecommendations:");
    for recommendation in report["recommendations"].as_array().into_iter().flatten() {
        if let Some(recommendation) = recommendation.as_str() {
            eprintln!("  {recommendation}");
        }
    }
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/")
}

fn unix_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_unsafe_constructs_without_counting_comments()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let src = root.path().join("src");
        fs::create_dir_all(&src)?;
        let file = src.join("lib.rs");
        fs::write(
            &file,
            r#"
// unsafe { commented_out(); }
unsafe trait Marker {}
unsafe impl Marker for usize {}
pub unsafe fn unchecked() {}
pub fn call() {
    unsafe { unchecked(); }
}
"#,
        )?;

        let analysis = analyze_unsafe_usage(root.path(), &file);
        assert_eq!(analysis.unsafe_blocks.len(), 1);
        assert_eq!(analysis.unsafe_functions[0].name.as_deref(), Some("unchecked"));
        assert_eq!(analysis.unsafe_traits[0].name.as_deref(), Some("Marker"));
        assert_eq!(analysis.unsafe_impls.len(), 1);
        Ok(())
    }

    #[test]
    fn report_groups_crates_and_summarizes_totals() {
        let analyses = vec![FileUnsafeAnalysis {
            file: "crates/bitnet-example/src/lib.rs".to_string(),
            error: None,
            unsafe_blocks: vec![UnsafeOccurrence {
                line: 10,
                content: "unsafe { work(); }".to_string(),
                name: None,
                context: Some(vec!["unsafe { work(); }".to_string()]),
            }],
            unsafe_functions: Vec::new(),
            unsafe_traits: Vec::new(),
            unsafe_impls: Vec::new(),
        }];

        let report = generate_unsafe_report(&analyses, json!({"packages": []}));
        assert_eq!(report["summary"]["total_files_analyzed"], 1);
        assert_eq!(report["summary"]["files_with_unsafe"], 1);
        assert_eq!(report["summary"]["total_unsafe_blocks"], 1);
        assert!(report["by_crate"].get("bitnet-example").is_some());
    }
}
