//! Centralized output configuration for CLI commands.
//!
//! Provides [`OutputConfig`] which controls how CLI output is rendered:
//! - `--format json` emits machine-readable JSON to stdout
//! - `--quiet` suppresses all non-essential output
//! - `--verbose` enables debug-level messages
//! - `--no-color` disables ANSI color codes

use serde::Serialize;
use std::io::Write;

/// Output format for CLI commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable text (default).
    #[default]
    Text,
    /// Machine-readable JSON.
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(format!("unknown format '{other}'. Expected one of: text, json")),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
        }
    }
}

/// Global output configuration derived from CLI flags.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub quiet: bool,
    pub verbose: bool,
    pub color: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self { format: OutputFormat::Text, quiet: false, verbose: false, color: true }
    }
}

impl OutputConfig {
    /// Print a status message (suppressed in quiet mode and JSON mode).
    pub fn status(&self, msg: &str) {
        if self.quiet || self.format == OutputFormat::Json {
            return;
        }
        eprintln!("{}", msg);
    }

    /// Print a verbose/debug message (only when `--verbose` is active).
    pub fn debug(&self, msg: &str) {
        if !self.verbose || self.format == OutputFormat::Json {
            return;
        }
        eprintln!("[debug] {}", msg);
    }

    /// Print a warning (shown unless quiet).
    pub fn warn(&self, msg: &str) {
        if self.quiet {
            return;
        }
        if self.format == OutputFormat::Json {
            return;
        }
        eprintln!("{}", msg);
    }

    /// Emit a final result value. In JSON mode it is serialized to stdout;
    /// in text mode `text_fn` is called to render human output.
    pub fn emit_result<T: Serialize>(
        &self,
        value: &T,
        text_fn: impl FnOnce(&T),
    ) -> anyhow::Result<()> {
        match self.format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(value)?;
                let mut stdout = std::io::stdout().lock();
                writeln!(stdout, "{json}")?;
            }
            OutputFormat::Text => {
                text_fn(value);
            }
        }
        Ok(())
    }

    /// Return the tracing log level implied by the flags.
    ///
    /// `--quiet` forces `error`, `--verbose` forces `debug`,
    /// otherwise returns `None` (use existing default).
    pub fn log_level_override(&self) -> Option<&'static str> {
        if self.quiet {
            Some("error")
        } else if self.verbose {
            Some("debug")
        } else {
            None
        }
    }
}

/// Summary of the loaded model, printed at startup.
#[derive(Debug, Clone, Serialize)]
pub struct ModelSummary {
    pub path: String,
    pub format: String,
    pub quantization: String,
    pub parameters: Option<String>,
    pub device: String,
    pub vocab_size: Option<u32>,
    pub context_length: Option<u32>,
}

impl std::fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Model:          {}", self.path)?;
        writeln!(f, "  Format:         {}", self.format)?;
        writeln!(f, "  Quantization:   {}", self.quantization)?;
        if let Some(p) = &self.parameters {
            writeln!(f, "  Parameters:     {p}")?;
        }
        writeln!(f, "  Device:         {}", self.device)?;
        if let Some(v) = self.vocab_size {
            writeln!(f, "  Vocab size:     {v}")?;
        }
        if let Some(c) = self.context_length {
            writeln!(f, "  Context length: {c}")?;
        }
        Ok(())
    }
}

/// Build a [`ModelSummary`] from a GGUF file path without loading tensors.
pub fn summarize_model(model_path: &std::path::Path, device: &str) -> Option<ModelSummary> {
    use bitnet_models::GgufReader;

    let data = std::fs::read(model_path).ok()?;
    let reader = GgufReader::new(&data).ok()?;

    let quantization =
        reader.get_string_metadata("general.quantization_type").unwrap_or_else(|| {
            reader
                .get_quantization_type()
                .map(|q| format!("{q:?}"))
                .unwrap_or_else(|| "unknown".into())
        });

    let vocab_size = reader
        .get_u32_metadata("llama.vocab_size")
        .or_else(|| reader.get_u32_metadata("tokenizer.ggml.tokens"));

    let context_length = reader.get_u32_metadata("llama.context_length");

    // Estimate parameter count from model name heuristics
    let parameters = reader
        .get_string_metadata("general.name")
        .and_then(|name| {
            let name_upper = name.to_uppercase();
            for suffix in ["B", "M"] {
                for part in name_upper.split(|c: char| !c.is_alphanumeric()) {
                    if let Some(num_str) = part.strip_suffix(suffix)
                        && let Ok(n) = num_str.parse::<f64>()
                    {
                        let count = if suffix == "B" { n * 1e9 } else { n * 1e6 };
                        return Some(format_param_count(count as u64));
                    }
                }
            }
            None
        })
        .or_else(|| {
            let tc = reader.tensor_count();
            if tc > 0 { Some(format!("{tc} tensors")) } else { None }
        });

    Some(ModelSummary {
        path: model_path.display().to_string(),
        format: "GGUF".into(),
        quantization,
        parameters,
        device: device.into(),
        vocab_size,
        context_length,
    })
}

fn format_param_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.0}M", count as f64 / 1e6)
    } else {
        format!("{count}")
    }
}

/// Suggest corrections for an unknown CLI flag.
///
/// Returns up to 3 candidates sorted by edit distance.
pub fn suggest_flag(unknown: &str, known: &[&str]) -> Vec<String> {
    let mut scored: Vec<(&str, usize)> = known
        .iter()
        .filter_map(|k| {
            let d = edit_distance(unknown, k);
            if d <= 3 && d < k.len() { Some((*k, d)) } else { None }
        })
        .collect();
    scored.sort_by_key(|&(_, d)| d);
    scored.into_iter().take(3).map(|(k, _)| k.to_string()).collect()
}

/// Simple Levenshtein edit distance.
#[allow(clippy::needless_range_loop)]
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_parse() {
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert!("xml".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Text.to_string(), "text");
        assert_eq!(OutputFormat::Json.to_string(), "json");
    }

    #[test]
    fn test_suggest_flag_close_match() {
        let known = &["--max-tokens", "--max-new-tokens", "--temperature", "--top-k"];
        let suggestions = suggest_flag("--max-token", known);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0], "--max-tokens");
    }

    #[test]
    fn test_suggest_flag_no_match() {
        let known = &["--max-tokens", "--temperature"];
        let suggestions = suggest_flag("--zzzzzzzzz", known);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_edit_distance_one_off() {
        assert_eq!(edit_distance("abc", "abd"), 1);
        assert_eq!(edit_distance("abc", "abcd"), 1);
    }

    #[test]
    fn test_output_config_log_level() {
        let quiet = OutputConfig { quiet: true, ..Default::default() };
        assert_eq!(quiet.log_level_override(), Some("error"));

        let verbose = OutputConfig { verbose: true, ..Default::default() };
        assert_eq!(verbose.log_level_override(), Some("debug"));

        let default = OutputConfig::default();
        assert_eq!(default.log_level_override(), None);
    }

    #[test]
    fn test_output_config_status_suppressed_in_quiet() {
        let config = OutputConfig { quiet: true, ..Default::default() };
        config.status("should be suppressed");
    }

    #[test]
    fn test_output_config_emit_json() {
        let config = OutputConfig { format: OutputFormat::Json, ..Default::default() };
        let value = serde_json::json!({"key": "value"});
        assert!(config.emit_result(&value, |_| {}).is_ok());
    }

    #[test]
    fn test_model_summary_display() {
        let summary = ModelSummary {
            path: "model.gguf".into(),
            format: "GGUF".into(),
            quantization: "I2_S".into(),
            parameters: Some("2.0B".into()),
            device: "cpu".into(),
            vocab_size: Some(32000),
            context_length: Some(2048),
        };
        let display = format!("{summary}");
        assert!(display.contains("model.gguf"));
        assert!(display.contains("I2_S"));
        assert!(display.contains("2.0B"));
    }

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(2_000_000_000), "2.0B");
        assert_eq!(format_param_count(7_500_000_000), "7.5B");
        assert_eq!(format_param_count(350_000_000), "350M");
        assert_eq!(format_param_count(1000), "1000");
    }
}
