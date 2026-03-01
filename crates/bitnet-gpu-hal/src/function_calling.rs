//! Function calling framework for structured tool use in LLM inference.
//!
//! Provides registration, parsing, validation, execution, and chaining
//! of function calls within a generation context.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors produced by function calling operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionCallError {
    /// Function not found in the registry.
    FunctionNotFound(String),
    /// Argument validation failed.
    ValidationError(String),
    /// Failed to parse function call from model output.
    ParseError(String),
    /// Maximum retry attempts exceeded.
    MaxRetriesExceeded { function: String, attempts: u32 },
    /// Chain step limit exceeded.
    ChainLimitExceeded { limit: usize },
    /// Duplicate function name in registry.
    DuplicateFunction(String),
    /// Required parameter missing.
    MissingParameter { function: String, parameter: String },
    /// Parameter type mismatch.
    TypeMismatch { parameter: String, expected: String, actual: String },
}

impl fmt::Display for FunctionCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FunctionNotFound(name) => {
                write!(f, "function not found: {name}")
            }
            Self::ValidationError(msg) => {
                write!(f, "validation error: {msg}")
            }
            Self::ParseError(msg) => {
                write!(f, "parse error: {msg}")
            }
            Self::MaxRetriesExceeded { function, attempts } => {
                write!(
                    f,
                    "max retries exceeded for {function} after \
                     {attempts} attempts"
                )
            }
            Self::ChainLimitExceeded { limit } => {
                write!(f, "chain step limit exceeded: {limit}")
            }
            Self::DuplicateFunction(name) => {
                write!(f, "duplicate function: {name}")
            }
            Self::MissingParameter { function, parameter } => {
                write!(
                    f,
                    "missing required parameter '{parameter}' \
                     for function '{function}'"
                )
            }
            Self::TypeMismatch { parameter, expected, actual } => {
                write!(
                    f,
                    "type mismatch for '{parameter}': \
                     expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for FunctionCallError {}

// ── Core types ────────────────────────────────────────────────────────────

/// JSON Schema type for a function parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

impl fmt::Display for SchemaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::Number => write!(f, "number"),
            Self::Integer => write!(f, "integer"),
            Self::Boolean => write!(f, "boolean"),
            Self::Array => write!(f, "array"),
            Self::Object => write!(f, "object"),
        }
    }
}

/// A single parameter definition within a function's schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDef {
    pub name: String,
    pub schema_type: SchemaType,
    pub description: String,
    #[serde(default)]
    pub enum_values: Option<Vec<String>>,
}

/// Definition of a callable function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ParameterDef>,
    pub required_params: Vec<String>,
}

/// A model's request to call a function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

/// The result of executing a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResult {
    pub call: FunctionCall,
    pub output: FunctionOutput,
}

/// Output variants from a function execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FunctionOutput {
    /// Successful result with JSON value.
    Success(Value),
    /// Function returned an error.
    Error(String),
}

/// How the model should choose which function to call.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToolChoice {
    /// Model decides whether to call a function.
    #[default]
    Auto,
    /// Model must not call any function.
    None,
    /// Model must call at least one function.
    Required,
    /// Model must call the named function.
    Specific(String),
}

// ── FunctionRegistry ──────────────────────────────────────────────────────

/// Registry of callable functions with metadata.
pub struct FunctionRegistry {
    functions: HashMap<String, FunctionDef>,
    tool_choice: ToolChoice,
}

impl FunctionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { functions: HashMap::new(), tool_choice: ToolChoice::Auto }
    }

    /// Register a function definition.
    pub fn register(&mut self, def: FunctionDef) -> Result<(), FunctionCallError> {
        if self.functions.contains_key(&def.name) {
            return Err(FunctionCallError::DuplicateFunction(def.name));
        }
        self.functions.insert(def.name.clone(), def);
        Ok(())
    }

    /// Remove a function by name.
    pub fn unregister(&mut self, name: &str) -> Option<FunctionDef> {
        self.functions.remove(name)
    }

    /// Look up a function by name.
    pub fn get(&self, name: &str) -> Option<&FunctionDef> {
        self.functions.get(name)
    }

    /// List all registered function names.
    pub fn function_names(&self) -> Vec<&str> {
        self.functions.keys().map(String::as_str).collect()
    }

    /// Number of registered functions.
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Set the tool choice policy.
    pub fn set_tool_choice(&mut self, choice: ToolChoice) {
        self.tool_choice = choice;
    }

    /// Get the current tool choice policy.
    pub const fn tool_choice(&self) -> &ToolChoice {
        &self.tool_choice
    }

    /// Get all function definitions.
    pub fn definitions(&self) -> Vec<&FunctionDef> {
        self.functions.values().collect()
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── FunctionCallParser ────────────────────────────────────────────────────

/// Parse model output to extract function calls.
///
/// Supports a simple JSON-based format:
/// `<function_call>{"name":"fn","arguments":{...}}</function_call>`
pub struct FunctionCallParser;

const TAG_OPEN: &str = "<function_call>";
const TAG_CLOSE: &str = "</function_call>";

impl FunctionCallParser {
    /// Extract all function calls from model output text.
    pub fn parse(text: &str) -> Result<Vec<FunctionCall>, FunctionCallError> {
        let mut calls = Vec::new();
        let mut search_from = 0;

        while let Some(start) = text[search_from..].find(TAG_OPEN) {
            let abs_start = search_from + start + TAG_OPEN.len();
            let rest = &text[abs_start..];
            let end = rest.find(TAG_CLOSE).ok_or_else(|| {
                FunctionCallError::ParseError("unclosed <function_call> tag".into())
            })?;
            let json_str = &rest[..end].trim();
            let call: FunctionCall = serde_json::from_str(json_str)
                .map_err(|e| FunctionCallError::ParseError(format!("invalid JSON: {e}")))?;
            calls.push(call);
            search_from = abs_start + end + TAG_CLOSE.len();
        }
        Ok(calls)
    }

    /// Check whether text contains any function call markers.
    pub fn contains_call(text: &str) -> bool {
        text.contains(TAG_OPEN)
    }
}

// ── FunctionCallFormatter ─────────────────────────────────────────────────

/// Format function definitions for injection into the model context.
pub struct FunctionCallFormatter;

impl FunctionCallFormatter {
    /// Render function definitions as a system prompt block.
    pub fn format_definitions(defs: &[&FunctionDef]) -> String {
        let mut out = String::from("Available functions:\n\n");
        for def in defs {
            let _ = writeln!(out, "### {}", def.name);
            let _ = writeln!(out, "{}", def.description);
            out.push_str("Parameters:\n");
            for p in &def.parameters {
                let req = if def.required_params.contains(&p.name) { " (required)" } else { "" };
                let _ =
                    writeln!(out, "  - {}: {} — {}{}", p.name, p.schema_type, p.description, req);
                if let Some(vals) = &p.enum_values {
                    let _ = writeln!(out, "    Allowed values: {}", vals.join(", "));
                }
            }
            out.push('\n');
        }
        out
    }

    /// Render a function result for inclusion in the conversation.
    pub fn format_result(result: &FunctionResult) -> String {
        match &result.output {
            FunctionOutput::Success(v) => {
                format!("<function_result name=\"{}\">{}</function_result>", result.call.name, v,)
            }
            FunctionOutput::Error(e) => {
                format!("<function_error name=\"{}\">{}</function_error>", result.call.name, e,)
            }
        }
    }
}

// ── FunctionCallValidator ─────────────────────────────────────────────────

/// Validate function call arguments against the function's schema.
pub struct FunctionCallValidator;

impl FunctionCallValidator {
    /// Validate a function call against its definition.
    pub fn validate(call: &FunctionCall, def: &FunctionDef) -> Result<(), FunctionCallError> {
        // Check required parameters are present.
        for req in &def.required_params {
            if !call.arguments.contains_key(req) {
                return Err(FunctionCallError::MissingParameter {
                    function: def.name.clone(),
                    parameter: req.clone(),
                });
            }
        }
        // Check argument types match the schema.
        for (key, value) in &call.arguments {
            if let Some(param) = def.parameters.iter().find(|p| &p.name == key) {
                Self::check_type(value, &param.schema_type, key)?;
                if let Some(allowed) = &param.enum_values {
                    Self::check_enum(value, allowed, key)?;
                }
            }
            // Extra arguments are allowed (lenient validation).
        }
        Ok(())
    }

    fn check_type(
        value: &Value,
        expected: &SchemaType,
        param_name: &str,
    ) -> Result<(), FunctionCallError> {
        let ok = match expected {
            SchemaType::String => value.is_string(),
            SchemaType::Number => value.is_f64() || value.is_i64(),
            SchemaType::Integer => value.is_i64(),
            SchemaType::Boolean => value.is_boolean(),
            SchemaType::Array => value.is_array(),
            SchemaType::Object => value.is_object(),
        };
        if !ok {
            return Err(FunctionCallError::TypeMismatch {
                parameter: param_name.to_string(),
                expected: expected.to_string(),
                actual: json_type_name(value).to_string(),
            });
        }
        Ok(())
    }

    fn check_enum(
        value: &Value,
        allowed: &[String],
        param_name: &str,
    ) -> Result<(), FunctionCallError> {
        if let Some(s) = value.as_str()
            && !allowed.iter().any(|a| a == s)
        {
            return Err(FunctionCallError::ValidationError(format!(
                "parameter '{param_name}' value '{s}' not in \
                 allowed values: {}",
                allowed.join(", "),
            )));
        }
        Ok(())
    }
}

/// Get a human-readable type name for a JSON value.
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ── ParallelFunctions ─────────────────────────────────────────────────────

/// Batch of function calls to be executed concurrently.
#[derive(Debug, Clone)]
pub struct ParallelFunctions {
    calls: Vec<FunctionCall>,
}

impl ParallelFunctions {
    /// Create from a set of calls.
    pub const fn new(calls: Vec<FunctionCall>) -> Self {
        Self { calls }
    }

    /// Number of calls in the batch.
    pub const fn len(&self) -> usize {
        self.calls.len()
    }

    /// Whether the batch is empty.
    pub const fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Get the calls.
    pub fn calls(&self) -> &[FunctionCall] {
        &self.calls
    }

    /// Consume into individual calls.
    pub fn into_calls(self) -> Vec<FunctionCall> {
        self.calls
    }

    /// Validate all calls against the registry.
    pub fn validate_all(&self, registry: &FunctionRegistry) -> Result<(), FunctionCallError> {
        for call in &self.calls {
            let def = registry
                .get(&call.name)
                .ok_or_else(|| FunctionCallError::FunctionNotFound(call.name.clone()))?;
            FunctionCallValidator::validate(call, def)?;
        }
        Ok(())
    }
}

// ── RetryPolicy ───────────────────────────────────────────────────────────

/// Retry policy for failed function calls.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Base delay in milliseconds (doubled each retry).
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self { max_retries: 3, base_delay_ms: 100, max_delay_ms: 5000 }
    }
}

impl RetryPolicy {
    /// Compute the delay in ms for the given attempt number (0-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let delay = self.base_delay_ms.saturating_mul(1u64 << attempt);
        delay.min(self.max_delay_ms)
    }

    /// Check whether another retry is allowed.
    pub const fn can_retry(&self, attempt: u32) -> bool {
        attempt < self.max_retries
    }
}

/// Track retry state for a single function call.
#[derive(Debug, Clone)]
pub struct RetryState {
    pub function_name: String,
    pub attempts: u32,
    pub last_error: Option<String>,
    pub policy: RetryPolicy,
}

impl RetryState {
    /// Create a new retry state for the given function.
    pub const fn new(function_name: String, policy: RetryPolicy) -> Self {
        Self { function_name, attempts: 0, last_error: None, policy }
    }

    /// Record a failed attempt and return the delay before next retry,
    /// or an error if retries are exhausted.
    pub fn record_failure(&mut self, error: String) -> Result<u64, FunctionCallError> {
        self.last_error = Some(error);
        self.attempts += 1;
        if self.policy.can_retry(self.attempts) {
            Ok(self.policy.delay_for_attempt(self.attempts - 1))
        } else {
            Err(FunctionCallError::MaxRetriesExceeded {
                function: self.function_name.clone(),
                attempts: self.attempts,
            })
        }
    }

    /// Record a successful attempt.
    pub fn record_success(&mut self) {
        self.last_error = None;
    }
}

// ── FunctionCallChain ─────────────────────────────────────────────────────

/// A step in a multi-step function call chain.
#[derive(Debug, Clone)]
pub struct ChainStep {
    pub call: FunctionCall,
    pub result: Option<FunctionResult>,
}

/// Multi-step function call sequence.
pub struct FunctionCallChain {
    steps: Vec<ChainStep>,
    max_steps: usize,
}

impl FunctionCallChain {
    /// Create a new chain with the given step limit.
    pub const fn new(max_steps: usize) -> Self {
        Self { steps: Vec::new(), max_steps }
    }

    /// Append a call to the chain. Returns error if limit exceeded.
    pub fn push_call(&mut self, call: FunctionCall) -> Result<usize, FunctionCallError> {
        if self.steps.len() >= self.max_steps {
            return Err(FunctionCallError::ChainLimitExceeded { limit: self.max_steps });
        }
        let idx = self.steps.len();
        self.steps.push(ChainStep { call, result: None });
        Ok(idx)
    }

    /// Record the result for a step.
    pub fn set_result(
        &mut self,
        index: usize,
        result: FunctionResult,
    ) -> Result<(), FunctionCallError> {
        if index >= self.steps.len() {
            return Err(FunctionCallError::ValidationError(format!(
                "chain index {index} out of bounds (len={})",
                self.steps.len(),
            )));
        }
        self.steps[index].result = Some(result);
        Ok(())
    }

    /// Number of steps so far.
    pub const fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the chain has any steps.
    pub const fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Maximum steps allowed.
    pub const fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Get all steps.
    pub fn steps(&self) -> &[ChainStep] {
        &self.steps
    }

    /// Check whether the last step has a result (chain is ready for
    /// the next call or is complete).
    pub fn is_current_step_complete(&self) -> bool {
        self.steps.last().is_some_and(|s| s.result.is_some())
    }

    /// Build a context string summarizing the chain so far.
    pub fn context_summary(&self) -> String {
        let mut out = String::new();
        for (i, step) in self.steps.iter().enumerate() {
            let _ = write!(out, "Step {}: {}(", i + 1, step.call.name);
            let args: Vec<String> =
                step.call.arguments.iter().map(|(k, v)| format!("{k}={v}")).collect();
            out.push_str(&args.join(", "));
            out.push(')');
            if let Some(ref res) = step.result {
                match &res.output {
                    FunctionOutput::Success(v) => {
                        let _ = write!(out, " -> {v}");
                    }
                    FunctionOutput::Error(e) => {
                        let _ = write!(out, " -> ERROR: {e}");
                    }
                }
            }
            out.push('\n');
        }
        out
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────

    fn weather_def() -> FunctionDef {
        FunctionDef {
            name: "get_weather".into(),
            description: "Get current weather for a location".into(),
            parameters: vec![
                ParameterDef {
                    name: "location".into(),
                    schema_type: SchemaType::String,
                    description: "City name".into(),
                    enum_values: None,
                },
                ParameterDef {
                    name: "units".into(),
                    schema_type: SchemaType::String,
                    description: "Temperature units".into(),
                    enum_values: Some(vec!["celsius".into(), "fahrenheit".into()]),
                },
            ],
            required_params: vec!["location".into()],
        }
    }

    fn search_def() -> FunctionDef {
        FunctionDef {
            name: "search".into(),
            description: "Search the web".into(),
            parameters: vec![
                ParameterDef {
                    name: "query".into(),
                    schema_type: SchemaType::String,
                    description: "Search query".into(),
                    enum_values: None,
                },
                ParameterDef {
                    name: "max_results".into(),
                    schema_type: SchemaType::Integer,
                    description: "Max results".into(),
                    enum_values: None,
                },
            ],
            required_params: vec!["query".into()],
        }
    }

    fn make_call(name: &str, args: Value) -> FunctionCall {
        let arguments = match args {
            Value::Object(map) => map.into_iter().collect::<HashMap<String, Value>>(),
            _ => HashMap::new(),
        };
        FunctionCall { name: name.into(), arguments }
    }

    // ── FunctionRegistry ────────────────────────────────────────

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = FunctionRegistry::new();
        reg.register(weather_def()).unwrap();
        assert!(reg.get("get_weather").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_duplicate_rejected() {
        let mut reg = FunctionRegistry::new();
        reg.register(weather_def()).unwrap();
        let err = reg.register(weather_def()).unwrap_err();
        assert!(matches!(err, FunctionCallError::DuplicateFunction(_)));
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = FunctionRegistry::new();
        reg.register(weather_def()).unwrap();
        assert_eq!(reg.len(), 1);
        let removed = reg.unregister("get_weather");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_unregister_nonexistent() {
        let mut reg = FunctionRegistry::new();
        assert!(reg.unregister("nope").is_none());
    }

    #[test]
    fn test_registry_function_names() {
        let mut reg = FunctionRegistry::new();
        reg.register(weather_def()).unwrap();
        reg.register(search_def()).unwrap();
        let mut names = reg.function_names();
        names.sort();
        assert_eq!(names, vec!["get_weather", "search"]);
    }

    #[test]
    fn test_registry_len_and_empty() {
        let mut reg = FunctionRegistry::new();
        assert!(reg.is_empty());
        reg.register(weather_def()).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_default_tool_choice() {
        let reg = FunctionRegistry::new();
        assert_eq!(*reg.tool_choice(), ToolChoice::Auto);
    }

    #[test]
    fn test_registry_set_tool_choice() {
        let mut reg = FunctionRegistry::new();
        reg.set_tool_choice(ToolChoice::Required);
        assert_eq!(*reg.tool_choice(), ToolChoice::Required);
    }

    #[test]
    fn test_registry_definitions() {
        let mut reg = FunctionRegistry::new();
        reg.register(weather_def()).unwrap();
        reg.register(search_def()).unwrap();
        assert_eq!(reg.definitions().len(), 2);
    }

    // ── ToolChoice ──────────────────────────────────────────────

    #[test]
    fn test_tool_choice_default_is_auto() {
        assert_eq!(ToolChoice::default(), ToolChoice::Auto);
    }

    #[test]
    fn test_tool_choice_specific() {
        let tc = ToolChoice::Specific("get_weather".into());
        assert_eq!(tc, ToolChoice::Specific("get_weather".into()));
    }

    // ── FunctionCallParser ──────────────────────────────────────

    #[test]
    fn test_parse_single_call() {
        let text = r#"Let me check that. <function_call>{"name":"get_weather","arguments":{"location":"London"}}</function_call> Done."#;
        let calls = FunctionCallParser::parse(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "London");
    }

    #[test]
    fn test_parse_multiple_calls() {
        let text = concat!(
            r#"<function_call>{"name":"a","arguments":{}}</function_call>"#,
            r#" then <function_call>{"name":"b","arguments":{}}</function_call>"#,
        );
        let calls = FunctionCallParser::parse(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn test_parse_no_calls() {
        let calls = FunctionCallParser::parse("Just some text").unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_unclosed_tag() {
        let text = "<function_call>{\"name\":\"x\"}";
        let err = FunctionCallParser::parse(text).unwrap_err();
        assert!(matches!(err, FunctionCallError::ParseError(_)));
    }

    #[test]
    fn test_parse_invalid_json() {
        let text = "<function_call>not json</function_call>";
        let err = FunctionCallParser::parse(text).unwrap_err();
        assert!(matches!(err, FunctionCallError::ParseError(_)));
    }

    #[test]
    fn test_contains_call_true() {
        assert!(FunctionCallParser::contains_call("prefix <function_call> suffix"));
    }

    #[test]
    fn test_contains_call_false() {
        assert!(!FunctionCallParser::contains_call("no calls here"));
    }

    // ── FunctionCallFormatter ───────────────────────────────────

    #[test]
    fn test_format_definitions_includes_name() {
        let def = weather_def();
        let text = FunctionCallFormatter::format_definitions(&[&def]);
        assert!(text.contains("### get_weather"));
        assert!(text.contains("City name"));
        assert!(text.contains("(required)"));
    }

    #[test]
    fn test_format_definitions_enum_values() {
        let def = weather_def();
        let text = FunctionCallFormatter::format_definitions(&[&def]);
        assert!(text.contains("celsius, fahrenheit"));
    }

    #[test]
    fn test_format_result_success() {
        let result = FunctionResult {
            call: make_call("get_weather", serde_json::json!({"location": "London"})),
            output: FunctionOutput::Success(serde_json::json!({"temp": 20})),
        };
        let text = FunctionCallFormatter::format_result(&result);
        assert!(text.contains("<function_result"));
        assert!(text.contains("get_weather"));
    }

    #[test]
    fn test_format_result_error() {
        let result = FunctionResult {
            call: make_call("get_weather", serde_json::json!({})),
            output: FunctionOutput::Error("timeout".into()),
        };
        let text = FunctionCallFormatter::format_result(&result);
        assert!(text.contains("<function_error"));
        assert!(text.contains("timeout"));
    }

    // ── FunctionCallValidator ───────────────────────────────────

    #[test]
    fn test_validate_valid_call() {
        let def = weather_def();
        let call =
            make_call("get_weather", serde_json::json!({"location": "Paris", "units": "celsius"}));
        assert!(FunctionCallValidator::validate(&call, &def).is_ok());
    }

    #[test]
    fn test_validate_missing_required() {
        let def = weather_def();
        let call = make_call("get_weather", serde_json::json!({"units": "celsius"}));
        let err = FunctionCallValidator::validate(&call, &def).unwrap_err();
        assert!(matches!(err, FunctionCallError::MissingParameter { .. }));
    }

    #[test]
    fn test_validate_type_mismatch() {
        let def = search_def();
        let call = make_call("search", serde_json::json!({"query": "rust", "max_results": "five"}));
        let err = FunctionCallValidator::validate(&call, &def).unwrap_err();
        assert!(matches!(err, FunctionCallError::TypeMismatch { .. }));
    }

    #[test]
    fn test_validate_enum_valid() {
        let def = weather_def();
        let call =
            make_call("get_weather", serde_json::json!({"location": "Rome", "units": "celsius"}));
        assert!(FunctionCallValidator::validate(&call, &def).is_ok());
    }

    #[test]
    fn test_validate_enum_invalid() {
        let def = weather_def();
        let call =
            make_call("get_weather", serde_json::json!({"location": "Rome", "units": "kelvin"}));
        let err = FunctionCallValidator::validate(&call, &def).unwrap_err();
        assert!(matches!(err, FunctionCallError::ValidationError(_)));
    }

    #[test]
    fn test_validate_extra_args_allowed() {
        let def = weather_def();
        let call = make_call("get_weather", serde_json::json!({"location": "Berlin", "extra": 42}));
        assert!(FunctionCallValidator::validate(&call, &def).is_ok());
    }

    #[test]
    fn test_validate_boolean_type() {
        let def = FunctionDef {
            name: "toggle".into(),
            description: "Toggle a flag".into(),
            parameters: vec![ParameterDef {
                name: "enabled".into(),
                schema_type: SchemaType::Boolean,
                description: "Flag".into(),
                enum_values: None,
            }],
            required_params: vec!["enabled".into()],
        };
        let good = make_call("toggle", serde_json::json!({"enabled": true}));
        assert!(FunctionCallValidator::validate(&good, &def).is_ok());
        let bad = make_call("toggle", serde_json::json!({"enabled": "yes"}));
        assert!(FunctionCallValidator::validate(&bad, &def).is_err());
    }

    #[test]
    fn test_validate_number_accepts_float() {
        let def = FunctionDef {
            name: "set_temp".into(),
            description: "Set temperature".into(),
            parameters: vec![ParameterDef {
                name: "value".into(),
                schema_type: SchemaType::Number,
                description: "Temp value".into(),
                enum_values: None,
            }],
            required_params: vec!["value".into()],
        };
        let call = make_call("set_temp", serde_json::json!({"value": 36.6}));
        assert!(FunctionCallValidator::validate(&call, &def).is_ok());
    }

    #[test]
    fn test_validate_array_type() {
        let def = FunctionDef {
            name: "batch".into(),
            description: "Batch operation".into(),
            parameters: vec![ParameterDef {
                name: "items".into(),
                schema_type: SchemaType::Array,
                description: "Items".into(),
                enum_values: None,
            }],
            required_params: vec!["items".into()],
        };
        let good = make_call("batch", serde_json::json!({"items": [1, 2, 3]}));
        assert!(FunctionCallValidator::validate(&good, &def).is_ok());
        let bad = make_call("batch", serde_json::json!({"items": "nope"}));
        assert!(FunctionCallValidator::validate(&bad, &def).is_err());
    }

    #[test]
    fn test_validate_object_type() {
        let def = FunctionDef {
            name: "config".into(),
            description: "Set config".into(),
            parameters: vec![ParameterDef {
                name: "settings".into(),
                schema_type: SchemaType::Object,
                description: "Settings object".into(),
                enum_values: None,
            }],
            required_params: vec!["settings".into()],
        };
        let good = make_call("config", serde_json::json!({"settings": {"a": 1}}));
        assert!(FunctionCallValidator::validate(&good, &def).is_ok());
    }

    // ── ParallelFunctions ───────────────────────────────────────

    #[test]
    fn test_parallel_basic() {
        let calls = vec![
            make_call("get_weather", serde_json::json!({"location": "A"})),
            make_call("get_weather", serde_json::json!({"location": "B"})),
        ];
        let pf = ParallelFunctions::new(calls);
        assert_eq!(pf.len(), 2);
        assert!(!pf.is_empty());
    }

    #[test]
    fn test_parallel_empty() {
        let pf = ParallelFunctions::new(vec![]);
        assert!(pf.is_empty());
        assert_eq!(pf.len(), 0);
    }

    #[test]
    fn test_parallel_into_calls() {
        let calls =
            vec![make_call("a", serde_json::json!({})), make_call("b", serde_json::json!({}))];
        let pf = ParallelFunctions::new(calls);
        let recovered = pf.into_calls();
        assert_eq!(recovered.len(), 2);
    }

    #[test]
    fn test_parallel_validate_all_success() {
        let mut reg = FunctionRegistry::new();
        reg.register(weather_def()).unwrap();
        let calls = vec![
            make_call("get_weather", serde_json::json!({"location": "A"})),
            make_call("get_weather", serde_json::json!({"location": "B"})),
        ];
        let pf = ParallelFunctions::new(calls);
        assert!(pf.validate_all(&reg).is_ok());
    }

    #[test]
    fn test_parallel_validate_unknown_function() {
        let reg = FunctionRegistry::new();
        let calls = vec![make_call("unknown", serde_json::json!({}))];
        let pf = ParallelFunctions::new(calls);
        let err = pf.validate_all(&reg).unwrap_err();
        assert!(matches!(err, FunctionCallError::FunctionNotFound(_)));
    }

    #[test]
    fn test_parallel_validate_invalid_args() {
        let mut reg = FunctionRegistry::new();
        reg.register(weather_def()).unwrap();
        let calls = vec![make_call("get_weather", serde_json::json!({"units": "celsius"}))];
        let pf = ParallelFunctions::new(calls);
        let err = pf.validate_all(&reg).unwrap_err();
        assert!(matches!(err, FunctionCallError::MissingParameter { .. }));
    }

    // ── RetryPolicy ─────────────────────────────────────────────

    #[test]
    fn test_retry_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert!(policy.can_retry(0));
        assert!(policy.can_retry(2));
        assert!(!policy.can_retry(3));
    }

    #[test]
    fn test_retry_exponential_backoff() {
        let policy = RetryPolicy { max_retries: 5, base_delay_ms: 100, max_delay_ms: 5000 };
        assert_eq!(policy.delay_for_attempt(0), 100);
        assert_eq!(policy.delay_for_attempt(1), 200);
        assert_eq!(policy.delay_for_attempt(2), 400);
        assert_eq!(policy.delay_for_attempt(3), 800);
    }

    #[test]
    fn test_retry_delay_capped() {
        let policy = RetryPolicy { max_retries: 10, base_delay_ms: 1000, max_delay_ms: 5000 };
        assert_eq!(policy.delay_for_attempt(5), 5000);
        assert_eq!(policy.delay_for_attempt(10), 5000);
    }

    #[test]
    fn test_retry_state_success_after_failures() {
        let policy = RetryPolicy::default();
        let mut state = RetryState::new("test_fn".into(), policy);
        let delay = state.record_failure("err1".into()).unwrap();
        assert!(delay > 0);
        assert_eq!(state.attempts, 1);
        state.record_success();
        assert!(state.last_error.is_none());
    }

    #[test]
    fn test_retry_state_exhausted() {
        let policy = RetryPolicy { max_retries: 2, base_delay_ms: 10, max_delay_ms: 100 };
        let mut state = RetryState::new("fn".into(), policy);
        state.record_failure("e1".into()).unwrap();
        let err = state.record_failure("e2".into()).unwrap_err();
        assert!(matches!(err, FunctionCallError::MaxRetriesExceeded { .. }));
    }

    // ── FunctionCallChain ───────────────────────────────────────

    #[test]
    fn test_chain_basic() {
        let mut chain = FunctionCallChain::new(5);
        assert!(chain.is_empty());
        let idx =
            chain.push_call(make_call("search", serde_json::json!({"query": "rust"}))).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_chain_limit_exceeded() {
        let mut chain = FunctionCallChain::new(1);
        chain.push_call(make_call("a", serde_json::json!({}))).unwrap();
        let err = chain.push_call(make_call("b", serde_json::json!({}))).unwrap_err();
        assert!(matches!(err, FunctionCallError::ChainLimitExceeded { limit: 1 }));
    }

    #[test]
    fn test_chain_set_result() {
        let mut chain = FunctionCallChain::new(5);
        chain.push_call(make_call("a", serde_json::json!({}))).unwrap();
        let result = FunctionResult {
            call: make_call("a", serde_json::json!({})),
            output: FunctionOutput::Success(serde_json::json!(42)),
        };
        chain.set_result(0, result).unwrap();
        assert!(chain.is_current_step_complete());
    }

    #[test]
    fn test_chain_set_result_out_of_bounds() {
        let mut chain = FunctionCallChain::new(5);
        let result = FunctionResult {
            call: make_call("a", serde_json::json!({})),
            output: FunctionOutput::Success(Value::Null),
        };
        let err = chain.set_result(0, result).unwrap_err();
        assert!(matches!(err, FunctionCallError::ValidationError(_)));
    }

    #[test]
    fn test_chain_current_step_incomplete() {
        let mut chain = FunctionCallChain::new(5);
        chain.push_call(make_call("a", serde_json::json!({}))).unwrap();
        assert!(!chain.is_current_step_complete());
    }

    #[test]
    fn test_chain_context_summary() {
        let mut chain = FunctionCallChain::new(5);
        chain.push_call(make_call("search", serde_json::json!({"query": "hello"}))).unwrap();
        let result = FunctionResult {
            call: make_call("search", serde_json::json!({"query": "hello"})),
            output: FunctionOutput::Success(serde_json::json!(["result1"])),
        };
        chain.set_result(0, result).unwrap();
        let summary = chain.context_summary();
        assert!(summary.contains("Step 1: search"));
        assert!(summary.contains("query"));
    }

    #[test]
    fn test_chain_context_summary_error() {
        let mut chain = FunctionCallChain::new(5);
        chain.push_call(make_call("fail", serde_json::json!({}))).unwrap();
        let result = FunctionResult {
            call: make_call("fail", serde_json::json!({})),
            output: FunctionOutput::Error("boom".into()),
        };
        chain.set_result(0, result).unwrap();
        let summary = chain.context_summary();
        assert!(summary.contains("ERROR: boom"));
    }

    #[test]
    fn test_chain_max_steps() {
        let chain = FunctionCallChain::new(10);
        assert_eq!(chain.max_steps(), 10);
    }

    #[test]
    fn test_chain_multi_step() {
        let mut chain = FunctionCallChain::new(3);
        for i in 0..3 {
            let name = format!("step{i}");
            chain.push_call(make_call(&name, serde_json::json!({}))).unwrap();
        }
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.steps().len(), 3);
    }

    // ── Error display ───────────────────────────────────────────

    #[test]
    fn test_error_display_function_not_found() {
        let e = FunctionCallError::FunctionNotFound("foo".into());
        assert_eq!(e.to_string(), "function not found: foo");
    }

    #[test]
    fn test_error_display_missing_param() {
        let e = FunctionCallError::MissingParameter { function: "f".into(), parameter: "p".into() };
        let s = e.to_string();
        assert!(s.contains("missing required parameter"));
    }

    #[test]
    fn test_error_display_type_mismatch() {
        let e = FunctionCallError::TypeMismatch {
            parameter: "x".into(),
            expected: "integer".into(),
            actual: "string".into(),
        };
        let s = e.to_string();
        assert!(s.contains("type mismatch"));
    }

    // ── SchemaType display ──────────────────────────────────────

    #[test]
    fn test_schema_type_display() {
        assert_eq!(SchemaType::String.to_string(), "string");
        assert_eq!(SchemaType::Number.to_string(), "number");
        assert_eq!(SchemaType::Integer.to_string(), "integer");
        assert_eq!(SchemaType::Boolean.to_string(), "boolean");
        assert_eq!(SchemaType::Array.to_string(), "array");
        assert_eq!(SchemaType::Object.to_string(), "object");
    }

    // ── Serde roundtrip ─────────────────────────────────────────

    #[test]
    fn test_function_def_serde_roundtrip() {
        let def = weather_def();
        let json = serde_json::to_string(&def).unwrap();
        let restored: FunctionDef = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, def.name);
        assert_eq!(restored.parameters.len(), def.parameters.len());
    }

    #[test]
    fn test_function_call_serde_roundtrip() {
        let call = make_call("get_weather", serde_json::json!({"location": "Tokyo"}));
        let json = serde_json::to_string(&call).unwrap();
        let restored: FunctionCall = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, "get_weather");
        assert_eq!(restored.arguments["location"], "Tokyo");
    }

    #[test]
    fn test_tool_choice_serde_roundtrip() {
        let choices = vec![
            ToolChoice::Auto,
            ToolChoice::None,
            ToolChoice::Required,
            ToolChoice::Specific("fn".into()),
        ];
        for choice in choices {
            let json = serde_json::to_string(&choice).unwrap();
            let restored: ToolChoice = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, choice);
        }
    }

    #[test]
    fn test_function_result_serde_roundtrip() {
        let result = FunctionResult {
            call: make_call("f", serde_json::json!({})),
            output: FunctionOutput::Success(serde_json::json!(42)),
        };
        let json = serde_json::to_string(&result).unwrap();
        let restored: FunctionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(
            serde_json::to_string(&restored.output).unwrap(),
            serde_json::to_string(&result.output).unwrap(),
        );
    }
}
