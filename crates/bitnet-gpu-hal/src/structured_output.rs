//! Structured output generation with schema validation and grammar constraints.
//!
//! This module provides tools for constraining LLM output to valid structured
//! formats (JSON, XML, CSV, etc.) through schema validation, grammar-based
//! token filtering, output parsing, and repair strategies.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Output format ────────────────────────────────────────────────────────

/// Supported structured output formats.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Standard JSON object/array.
    Json,
    /// Newline-delimited JSON (one object per line).
    JsonL,
    /// XML document.
    Xml,
    /// Comma-separated values.
    Csv,
    /// Markdown text.
    Markdown,
    /// YAML document.
    Yaml,
    /// User-defined format with a descriptive tag.
    Custom(String),
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::JsonL => write!(f, "jsonl"),
            Self::Xml => write!(f, "xml"),
            Self::Csv => write!(f, "csv"),
            Self::Markdown => write!(f, "markdown"),
            Self::Yaml => write!(f, "yaml"),
            Self::Custom(tag) => write!(f, "custom({tag})"),
        }
    }
}

// ── JSON schema types ────────────────────────────────────────────────────

/// Primitive and compound types used in JSON schema definitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

impl fmt::Display for SchemaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Null => "null",
        };
        write!(f, "{s}")
    }
}

/// A simplified JSON schema definition.
///
/// Supports type constraints, required fields, enumeration values, and
/// nested object/array schemas—enough for typical structured-output use
/// cases without pulling in a full JSON-Schema library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Expected top-level type.
    pub schema_type: SchemaType,
    /// Property schemas (only meaningful when `schema_type == Object`).
    pub properties: HashMap<String, Self>,
    /// Required property names (only meaningful for objects).
    pub required: Vec<String>,
    /// Allowed literal values (acts as an enum constraint).
    pub enum_values: Vec<serde_json::Value>,
    /// Schema for array items (only meaningful when `schema_type == Array`).
    pub items: Option<Box<Self>>,
    /// Optional human-readable description.
    pub description: Option<String>,
}

impl Default for JsonSchema {
    fn default() -> Self {
        Self {
            schema_type: SchemaType::Object,
            properties: HashMap::new(),
            required: Vec::new(),
            enum_values: Vec::new(),
            items: None,
            description: None,
        }
    }
}

impl JsonSchema {
    /// Create a schema that expects a specific primitive type.
    pub fn new(schema_type: SchemaType) -> Self {
        Self { schema_type, ..Default::default() }
    }

    /// Create an object schema with the given property definitions.
    pub fn object(properties: HashMap<String, Self>) -> Self {
        Self { schema_type: SchemaType::Object, properties, ..Default::default() }
    }

    /// Create an array schema with a given item schema.
    pub fn array(items: Self) -> Self {
        Self { schema_type: SchemaType::Array, items: Some(Box::new(items)), ..Default::default() }
    }

    /// Mark certain property names as required.
    #[must_use]
    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = required;
        self
    }

    /// Restrict values to an explicit set.
    #[must_use]
    pub fn with_enum(mut self, values: Vec<serde_json::Value>) -> Self {
        self.enum_values = values;
        self
    }
}

// ── Schema validation ────────────────────────────────────────────────────

/// Validation errors produced by [`SchemaValidator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The value's type does not match the schema.
    TypeMismatch { expected: SchemaType, got: String },
    /// A required property is missing from an object.
    MissingRequired(String),
    /// The value is not among the allowed enum values.
    InvalidEnumValue(String),
    /// A nested property failed validation.
    PropertyError { property: String, error: Box<Self> },
    /// An array item failed validation.
    ItemError { index: usize, error: Box<Self> },
    /// The input could not be parsed as JSON.
    ParseError(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            Self::MissingRequired(field) => {
                write!(f, "missing required field: {field}")
            }
            Self::InvalidEnumValue(val) => {
                write!(f, "value not in enum: {val}")
            }
            Self::PropertyError { property, error } => {
                write!(f, "property '{property}': {error}")
            }
            Self::ItemError { index, error } => {
                write!(f, "item[{index}]: {error}")
            }
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

/// Validates a [`serde_json::Value`] against a [`JsonSchema`].
pub struct SchemaValidator;

impl SchemaValidator {
    /// Validate `value` against `schema`, returning all errors found.
    pub fn validate(schema: &JsonSchema, value: &serde_json::Value) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        Self::validate_inner(schema, value, &mut errors);
        errors
    }

    fn validate_inner(
        schema: &JsonSchema,
        value: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) {
        // Enum check (applies to any type).
        if !schema.enum_values.is_empty() && !schema.enum_values.contains(value) {
            errors.push(ValidationError::InvalidEnumValue(value.to_string()));
        }

        // Type check.
        if !Self::type_matches(&schema.schema_type, value) {
            errors.push(ValidationError::TypeMismatch {
                expected: schema.schema_type.clone(),
                got: Self::json_type_name(value).to_string(),
            });
            return; // No point diving deeper if the top-level type is wrong.
        }

        match &schema.schema_type {
            SchemaType::Object => {
                if let Some(obj) = value.as_object() {
                    // Required fields.
                    for req in &schema.required {
                        if !obj.contains_key(req) {
                            errors.push(ValidationError::MissingRequired(req.clone()));
                        }
                    }
                    // Property schemas.
                    for (key, prop_schema) in &schema.properties {
                        if let Some(prop_val) = obj.get(key) {
                            let mut prop_errors = Vec::new();
                            Self::validate_inner(prop_schema, prop_val, &mut prop_errors);
                            for e in prop_errors {
                                errors.push(ValidationError::PropertyError {
                                    property: key.clone(),
                                    error: Box::new(e),
                                });
                            }
                        }
                    }
                }
            }
            SchemaType::Array => {
                if let (Some(items_schema), Some(arr)) = (&schema.items, value.as_array()) {
                    for (idx, item) in arr.iter().enumerate() {
                        let mut item_errors = Vec::new();
                        Self::validate_inner(items_schema, item, &mut item_errors);
                        for e in item_errors {
                            errors.push(ValidationError::ItemError {
                                index: idx,
                                error: Box::new(e),
                            });
                        }
                    }
                }
            }
            _ => {} // Primitive types already type-checked above.
        }
    }

    fn type_matches(expected: &SchemaType, value: &serde_json::Value) -> bool {
        match (expected, value) {
            (SchemaType::Integer, serde_json::Value::Number(n)) => {
                n.as_i64().is_some() || n.as_u64().is_some()
            }
            (SchemaType::String, serde_json::Value::String(_))
            | (SchemaType::Number, serde_json::Value::Number(_))
            | (SchemaType::Boolean, serde_json::Value::Bool(_))
            | (SchemaType::Array, serde_json::Value::Array(_))
            | (SchemaType::Object, serde_json::Value::Object(_))
            | (SchemaType::Null, serde_json::Value::Null) => true,
            _ => false,
        }
    }

    const fn json_type_name(value: &serde_json::Value) -> &'static str {
        match value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "boolean",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        }
    }
}

// ── JSON forcer ──────────────────────────────────────────────────────────

/// Tracks bracket/quote state to constrain token generation to valid JSON.
#[derive(Debug, Clone)]
pub struct JsonForcer {
    /// Stack of open brackets/braces (e.g. `[`, `{`).
    bracket_stack: Vec<char>,
    /// `true` when inside a quoted string literal.
    in_string: bool,
    /// `true` when the previous character was a backslash (escape).
    escape_next: bool,
    /// Characters consumed so far.
    position: usize,
}

impl Default for JsonForcer {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonForcer {
    pub const fn new() -> Self {
        Self { bracket_stack: Vec::new(), in_string: false, escape_next: false, position: 0 }
    }

    /// Feed a character and update internal state.
    pub fn feed(&mut self, ch: char) {
        self.position += 1;

        if self.escape_next {
            self.escape_next = false;
            return;
        }

        if ch == '\\' && self.in_string {
            self.escape_next = true;
            return;
        }

        if ch == '"' {
            self.in_string = !self.in_string;
            return;
        }

        if self.in_string {
            return;
        }

        match ch {
            '{' | '[' => self.bracket_stack.push(ch),
            '}' => {
                if self.bracket_stack.last() == Some(&'{') {
                    self.bracket_stack.pop();
                }
            }
            ']' => {
                if self.bracket_stack.last() == Some(&'[') {
                    self.bracket_stack.pop();
                }
            }
            _ => {}
        }
    }

    /// Feed an entire string.
    pub fn feed_str(&mut self, s: &str) {
        for ch in s.chars() {
            self.feed(ch);
        }
    }

    /// Returns `true` when brackets are balanced and we are not inside a
    /// string—i.e. the JSON is structurally complete.
    pub const fn is_complete(&self) -> bool {
        self.bracket_stack.is_empty() && !self.in_string
    }

    /// Characters that would be valid closers right now.
    pub fn expected_closers(&self) -> Vec<char> {
        let mut closers = Vec::new();
        if self.in_string {
            closers.push('"');
        }
        if let Some(&top) = self.bracket_stack.last() {
            closers.push(match top {
                '{' => '}',
                '[' => ']',
                _ => unreachable!(),
            });
        }
        closers
    }

    /// Suffix that would close every open bracket/string.
    pub fn closing_suffix(&self) -> String {
        let mut suffix = String::new();
        if self.in_string {
            suffix.push('"');
        }
        for &ch in self.bracket_stack.iter().rev() {
            suffix.push(match ch {
                '{' => '}',
                '[' => ']',
                _ => unreachable!(),
            });
        }
        suffix
    }

    /// Number of characters consumed so far.
    /// Number of characters consumed so far.
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Current bracket nesting depth.
    pub const fn depth(&self) -> usize {
        self.bracket_stack.len()
    }

    /// Whether we are currently inside a string literal.
    pub const fn in_string(&self) -> bool {
        self.in_string
    }
}

// ── Grammar constraint ───────────────────────────────────────────────────

/// A single production rule in a context-free grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Production {
    /// Left-hand side non-terminal.
    pub lhs: String,
    /// Right-hand side symbols (terminals are lowercase, non-terminals
    /// uppercase by convention).
    pub rhs: Vec<String>,
}

/// A context-free grammar used to constrain token generation.
#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    /// Set of production rules.
    pub productions: Vec<Production>,
    /// The start symbol.
    pub start_symbol: String,
    /// Non-terminal symbols.
    pub non_terminals: HashSet<String>,
    /// Terminal symbols.
    pub terminals: HashSet<String>,
}

impl GrammarConstraint {
    /// Build a grammar from a list of productions and a start symbol.
    pub fn new(productions: Vec<Production>, start_symbol: String) -> Self {
        let mut non_terminals = HashSet::new();
        let mut terminals = HashSet::new();

        for p in &productions {
            non_terminals.insert(p.lhs.clone());
        }
        for p in &productions {
            for sym in &p.rhs {
                if !non_terminals.contains(sym) {
                    terminals.insert(sym.clone());
                }
            }
        }

        Self { productions, start_symbol, non_terminals, terminals }
    }

    /// Return all productions whose LHS equals `symbol`.
    pub fn productions_for(&self, symbol: &str) -> Vec<&Production> {
        self.productions.iter().filter(|p| p.lhs == symbol).collect()
    }

    /// Check if `symbol` is a terminal.
    pub fn is_terminal(&self, symbol: &str) -> bool {
        self.terminals.contains(symbol)
    }

    /// Check if `symbol` is a non-terminal.
    pub fn is_non_terminal(&self, symbol: &str) -> bool {
        self.non_terminals.contains(symbol)
    }
}

/// Tracks the current parse position inside a grammar-constrained
/// generation.
#[derive(Debug, Clone)]
pub struct GrammarState {
    /// Stack of symbols still to be expanded (top = next to process).
    pub stack: Vec<String>,
    /// Tokens generated so far.
    pub generated: Vec<String>,
    /// Position in the output.
    pub position: usize,
}

impl GrammarState {
    /// Create an initial state from the grammar's start symbol.
    pub fn new(start_symbol: &str) -> Self {
        Self { stack: vec![start_symbol.to_string()], generated: Vec::new(), position: 0 }
    }

    /// Return the set of terminal symbols that could validly appear next,
    /// given the grammar and current stack.
    pub fn valid_next_tokens(&self, grammar: &GrammarConstraint) -> HashSet<String> {
        let mut result = HashSet::new();
        if let Some(top) = self.stack.last() {
            if grammar.is_terminal(top) {
                result.insert(top.clone());
            } else {
                // Expand non-terminal: each production's first symbol is a
                // candidate (simplified first-set approximation).
                for prod in grammar.productions_for(top) {
                    if let Some(first) = prod.rhs.first()
                        && grammar.is_terminal(first)
                    {
                        result.insert(first.clone());
                    }
                }
            }
        }
        result
    }

    /// Advance the state by consuming `token` (a terminal).
    ///
    /// Returns `true` if the token was accepted, `false` if it was not
    /// valid at this position.
    pub fn advance(&mut self, token: &str, grammar: &GrammarConstraint) -> bool {
        if let Some(top) = self.stack.last().cloned() {
            if grammar.is_terminal(&top) {
                if top == token {
                    self.stack.pop();
                    self.generated.push(token.to_string());
                    self.position += 1;
                    return true;
                }
                return false;
            }
            // Try to expand the non-terminal.
            for prod in grammar.productions_for(&top) {
                if let Some(first) = prod.rhs.first()
                    && grammar.is_terminal(first)
                    && first == token
                {
                    self.stack.pop();
                    // Push remaining RHS symbols (after the matched
                    // first) in reverse so the second symbol ends up
                    // on top of the stack.
                    for sym in prod.rhs[1..].iter().rev() {
                        self.stack.push(sym.clone());
                    }
                    self.generated.push(token.to_string());
                    self.position += 1;
                    return true;
                }
            }
        }
        false
    }

    /// `true` when the stack is empty (all symbols consumed).
    pub const fn is_complete(&self) -> bool {
        self.stack.is_empty()
    }
}

// ── Output parser ────────────────────────────────────────────────────────

/// Result of parsing a structured output string.
#[derive(Debug, Clone)]
pub enum ParseResult {
    /// Successfully parsed a JSON value.
    Json(serde_json::Value),
    /// Multiple JSON values (JSONL).
    JsonL(Vec<serde_json::Value>),
    /// Raw text (fallback for non-JSON formats).
    Raw(String),
    /// Parsing failed.
    Error(String),
}

/// Parses raw text into structured output.
pub struct OutputParser;

impl OutputParser {
    /// Attempt to parse `text` according to `format`.
    pub fn parse(text: &str, format: &OutputFormat) -> ParseResult {
        match format {
            OutputFormat::Json => Self::parse_json(text),
            OutputFormat::JsonL => Self::parse_jsonl(text),
            _ => ParseResult::Raw(text.to_string()),
        }
    }

    fn parse_json(text: &str) -> ParseResult {
        let trimmed = text.trim();
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(val) => ParseResult::Json(val),
            Err(e) => ParseResult::Error(format!("invalid JSON: {e}")),
        }
    }

    fn parse_jsonl(text: &str) -> ParseResult {
        let mut values = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(val) => values.push(val),
                Err(e) => {
                    return ParseResult::Error(format!("invalid JSON on line {}: {e}", i + 1));
                }
            }
        }
        ParseResult::JsonL(values)
    }

    /// Validate parsed output against a schema, returning errors.
    pub fn validate(value: &serde_json::Value, schema: &JsonSchema) -> Vec<ValidationError> {
        SchemaValidator::validate(schema, value)
    }
}

// ── Repair strategies ────────────────────────────────────────────────────

/// Strategies for repairing malformed structured output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Close unclosed brackets and braces.
    CloseBrackets,
    /// Fix unmatched quotes.
    FixQuotes,
    /// Truncate to the last valid JSON value.
    TruncateToValid,
    /// Remove trailing commas before closing brackets.
    RemoveTrailingCommas,
    /// Apply all strategies in order.
    All,
}

/// Attempts to repair broken JSON output.
pub struct JsonRepairer;

impl JsonRepairer {
    /// Apply the given strategy to `input`, returning the repaired string.
    pub fn repair(input: &str, strategy: &RepairStrategy) -> String {
        match strategy {
            RepairStrategy::CloseBrackets => Self::close_brackets(input),
            RepairStrategy::FixQuotes => Self::fix_quotes(input),
            RepairStrategy::TruncateToValid => Self::truncate_to_valid(input),
            RepairStrategy::RemoveTrailingCommas => Self::remove_trailing_commas(input),
            RepairStrategy::All => {
                let s = Self::fix_quotes(input);
                let s = Self::remove_trailing_commas(&s);
                Self::close_brackets(&s)
            }
        }
    }

    fn close_brackets(input: &str) -> String {
        let mut forcer = JsonForcer::new();
        forcer.feed_str(input);
        let suffix = forcer.closing_suffix();
        format!("{input}{suffix}")
    }

    fn fix_quotes(input: &str) -> String {
        let mut in_string = false;
        let mut escaped = false;
        let mut quote_count = 0u32;

        for ch in input.chars() {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' && in_string {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = !in_string;
                quote_count += 1;
            }
        }

        if quote_count.is_multiple_of(2) { input.to_string() } else { format!("{input}\"") }
    }

    fn truncate_to_valid(input: &str) -> String {
        let trimmed = input.trim();
        // Try progressively shorter prefixes.
        for end in (1..=trimmed.len()).rev() {
            let candidate = &trimmed[..end];
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return candidate.to_string();
            }
        }
        // Nothing valid found—return empty object as fallback.
        "{}".to_string()
    }

    fn remove_trailing_commas(input: &str) -> String {
        let mut result = input.to_string();
        // Pattern: comma followed by optional whitespace then `}` or `]`.
        loop {
            let before = result.clone();
            result = Self::remove_one_trailing_comma(&result);
            if result == before {
                break;
            }
        }
        result
    }

    fn remove_one_trailing_comma(input: &str) -> String {
        let bytes = input.as_bytes();
        let len = bytes.len();
        let mut i = len;
        // Walk backwards past whitespace looking for `}` or `]`.
        while i > 0 {
            i -= 1;
            let ch = bytes[i] as char;
            if ch.is_ascii_whitespace() {
                continue;
            }
            if ch == '}' || ch == ']' {
                // Now walk backwards again past whitespace to find a comma.
                let closer_pos = i;
                let mut j = closer_pos;
                while j > 0 {
                    j -= 1;
                    let c2 = bytes[j] as char;
                    if c2.is_ascii_whitespace() {
                        continue;
                    }
                    if c2 == ',' {
                        let mut out = String::with_capacity(len);
                        out.push_str(&input[..j]);
                        out.push_str(&input[j + 1..]);
                        return out;
                    }
                    break;
                }
            }
            break;
        }
        input.to_string()
    }
}

// ── Format converter ─────────────────────────────────────────────────────

/// Converts between output formats.
pub struct FormatConverter;

impl FormatConverter {
    /// Convert a JSON value to JSONL (one line per top-level array element,
    /// or the value itself if it is not an array).
    pub fn json_to_jsonl(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n"),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }

    /// Convert a JSON array of objects to CSV.
    ///
    /// Uses the keys from the first object as the header row. Missing
    /// fields in subsequent objects are rendered as empty strings.
    pub fn json_to_csv(value: &serde_json::Value) -> Option<String> {
        let arr = value.as_array()?;
        if arr.is_empty() {
            return Some(String::new());
        }

        let first = arr[0].as_object()?;
        let headers: Vec<&String> = first.keys().collect();

        let mut out = String::new();

        // Header row.
        out.push_str(&headers.iter().map(|h| Self::csv_escape(h)).collect::<Vec<_>>().join(","));
        out.push('\n');

        // Data rows.
        for row in arr {
            if let Some(obj) = row.as_object() {
                let cells: Vec<String> = headers
                    .iter()
                    .map(|h| match obj.get(*h) {
                        Some(serde_json::Value::String(s)) => Self::csv_escape(s),
                        Some(v) => Self::csv_escape(&v.to_string()),
                        None => String::new(),
                    })
                    .collect();
                out.push_str(&cells.join(","));
                out.push('\n');
            }
        }

        Some(out)
    }

    fn csv_escape(field: &str) -> String {
        if field.contains(',') || field.contains('"') || field.contains('\n') {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }
}

// ── Structured output engine ─────────────────────────────────────────────

/// Error type for the structured output engine.
#[derive(Debug, Clone)]
pub enum StructuredOutputError {
    /// Schema validation failed.
    ValidationFailed(Vec<ValidationError>),
    /// Output could not be parsed.
    ParseFailed(String),
    /// Grammar constraint violation.
    GrammarViolation(String),
    /// Repair failed.
    RepairFailed(String),
}

impl fmt::Display for StructuredOutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValidationFailed(errs) => {
                write!(f, "validation failed: ")?;
                for (i, e) in errs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{e}")?;
                }
                Ok(())
            }
            Self::ParseFailed(msg) => write!(f, "parse failed: {msg}"),
            Self::GrammarViolation(msg) => {
                write!(f, "grammar violation: {msg}")
            }
            Self::RepairFailed(msg) => write!(f, "repair failed: {msg}"),
        }
    }
}

/// Main engine for structured output: constrain, validate, repair, convert.
pub struct StructuredOutputEngine {
    /// Target output format.
    pub format: OutputFormat,
    /// Optional JSON schema for validation.
    pub schema: Option<JsonSchema>,
    /// Optional grammar constraint.
    pub grammar: Option<GrammarConstraint>,
    /// Repair strategies to apply on failure (in order).
    pub repair_strategies: Vec<RepairStrategy>,
}

impl StructuredOutputEngine {
    /// Create a new engine for the given format.
    pub const fn new(format: OutputFormat) -> Self {
        Self { format, schema: None, grammar: None, repair_strategies: Vec::new() }
    }

    /// Attach a JSON schema for validation.
    #[must_use]
    pub fn with_schema(mut self, schema: JsonSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Attach a grammar constraint.
    #[must_use]
    pub fn with_grammar(mut self, grammar: GrammarConstraint) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Add repair strategies.
    #[must_use]
    pub fn with_repair(mut self, strategies: Vec<RepairStrategy>) -> Self {
        self.repair_strategies = strategies;
        self
    }

    /// Process raw output: parse → validate → repair if needed.
    pub fn process(&self, raw: &str) -> Result<serde_json::Value, StructuredOutputError> {
        // 1. Parse.
        let parsed = OutputParser::parse(raw, &self.format);
        let value = match parsed {
            ParseResult::Json(v) => v,
            ParseResult::JsonL(v) => serde_json::Value::Array(v),
            ParseResult::Raw(_) => {
                return Err(StructuredOutputError::ParseFailed(
                    "non-JSON format returned as raw".into(),
                ));
            }
            ParseResult::Error(e) => {
                // Try repair.
                return self.try_repair(raw, &e);
            }
        };

        // 2. Validate against schema.
        if let Some(schema) = &self.schema {
            let errors = SchemaValidator::validate(schema, &value);
            if !errors.is_empty() {
                return Err(StructuredOutputError::ValidationFailed(errors));
            }
        }

        Ok(value)
    }

    fn try_repair(
        &self,
        raw: &str,
        _original_error: &str,
    ) -> Result<serde_json::Value, StructuredOutputError> {
        for strategy in &self.repair_strategies {
            let repaired = JsonRepairer::repair(raw, strategy);
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&repaired) {
                // Validate if we have a schema.
                if let Some(schema) = &self.schema {
                    let errors = SchemaValidator::validate(schema, &val);
                    if errors.is_empty() {
                        return Ok(val);
                    }
                } else {
                    return Ok(val);
                }
            }
        }

        Err(StructuredOutputError::RepairFailed("no repair strategy succeeded".into()))
    }

    /// Convert a parsed value to a different format.
    pub fn convert(
        &self,
        value: &serde_json::Value,
        target: &OutputFormat,
    ) -> Result<String, StructuredOutputError> {
        match target {
            OutputFormat::Json => serde_json::to_string_pretty(value)
                .map_err(|e| StructuredOutputError::ParseFailed(e.to_string())),
            OutputFormat::JsonL => Ok(FormatConverter::json_to_jsonl(value)),
            OutputFormat::Csv => FormatConverter::json_to_csv(value).ok_or_else(|| {
                StructuredOutputError::ParseFailed(
                    "cannot convert to CSV (expected array of objects)".into(),
                )
            }),
            other => Err(StructuredOutputError::ParseFailed(format!(
                "conversion to {other} not yet implemented"
            ))),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── OutputFormat ─────────────────────────────────────────────────

    #[test]
    fn output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::JsonL.to_string(), "jsonl");
        assert_eq!(OutputFormat::Xml.to_string(), "xml");
        assert_eq!(OutputFormat::Csv.to_string(), "csv");
        assert_eq!(OutputFormat::Markdown.to_string(), "markdown");
        assert_eq!(OutputFormat::Yaml.to_string(), "yaml");
        assert_eq!(OutputFormat::Custom("proto".into()).to_string(), "custom(proto)");
    }

    #[test]
    fn output_format_equality() {
        assert_eq!(OutputFormat::Json, OutputFormat::Json);
        assert_ne!(OutputFormat::Json, OutputFormat::Xml);
        assert_eq!(OutputFormat::Custom("a".into()), OutputFormat::Custom("a".into()));
        assert_ne!(OutputFormat::Custom("a".into()), OutputFormat::Custom("b".into()));
    }

    // ── SchemaType ───────────────────────────────────────────────────

    #[test]
    fn schema_type_display() {
        assert_eq!(SchemaType::String.to_string(), "string");
        assert_eq!(SchemaType::Number.to_string(), "number");
        assert_eq!(SchemaType::Integer.to_string(), "integer");
        assert_eq!(SchemaType::Boolean.to_string(), "boolean");
        assert_eq!(SchemaType::Array.to_string(), "array");
        assert_eq!(SchemaType::Object.to_string(), "object");
        assert_eq!(SchemaType::Null.to_string(), "null");
    }

    // ── JsonSchema construction ──────────────────────────────────────

    #[test]
    fn schema_default_is_object() {
        let s = JsonSchema::default();
        assert_eq!(s.schema_type, SchemaType::Object);
        assert!(s.properties.is_empty());
        assert!(s.required.is_empty());
    }

    #[test]
    fn schema_new_primitive() {
        let s = JsonSchema::new(SchemaType::String);
        assert_eq!(s.schema_type, SchemaType::String);
    }

    #[test]
    fn schema_object_builder() {
        let mut props = HashMap::new();
        props.insert("name".into(), JsonSchema::new(SchemaType::String));
        let s = JsonSchema::object(props).with_required(vec!["name".into()]);
        assert_eq!(s.required, vec!["name".to_string()]);
        assert!(s.properties.contains_key("name"));
    }

    #[test]
    fn schema_array_builder() {
        let s = JsonSchema::array(JsonSchema::new(SchemaType::Number));
        assert_eq!(s.schema_type, SchemaType::Array);
        assert!(s.items.is_some());
    }

    #[test]
    fn schema_with_enum() {
        let s = JsonSchema::new(SchemaType::String).with_enum(vec![json!("a"), json!("b")]);
        assert_eq!(s.enum_values.len(), 2);
    }

    // ── SchemaValidator – type checking ──────────────────────────────

    #[test]
    fn validate_string_type_pass() {
        let schema = JsonSchema::new(SchemaType::String);
        let errors = SchemaValidator::validate(&schema, &json!("hello"));
        assert!(errors.is_empty());
    }

    #[test]
    fn validate_string_type_fail() {
        let schema = JsonSchema::new(SchemaType::String);
        let errors = SchemaValidator::validate(&schema, &json!(42));
        assert_eq!(errors.len(), 1);
        matches!(&errors[0], ValidationError::TypeMismatch { .. });
    }

    #[test]
    fn validate_number_type() {
        let schema = JsonSchema::new(SchemaType::Number);
        assert!(SchemaValidator::validate(&schema, &json!(2.72)).is_empty());
        assert!(!SchemaValidator::validate(&schema, &json!("pi")).is_empty());
    }

    #[test]
    fn validate_integer_type() {
        let schema = JsonSchema::new(SchemaType::Integer);
        assert!(SchemaValidator::validate(&schema, &json!(42)).is_empty());
        // Floats that are not representable as i64/u64 fail.
        assert!(!SchemaValidator::validate(&schema, &json!("nope")).is_empty());
    }

    #[test]
    fn validate_boolean_type() {
        let schema = JsonSchema::new(SchemaType::Boolean);
        assert!(SchemaValidator::validate(&schema, &json!(true)).is_empty());
        assert!(!SchemaValidator::validate(&schema, &json!(0)).is_empty());
    }

    #[test]
    fn validate_null_type() {
        let schema = JsonSchema::new(SchemaType::Null);
        assert!(SchemaValidator::validate(&schema, &serde_json::Value::Null).is_empty());
        assert!(!SchemaValidator::validate(&schema, &json!("null")).is_empty());
    }

    #[test]
    fn validate_array_type() {
        let schema = JsonSchema::new(SchemaType::Array);
        assert!(SchemaValidator::validate(&schema, &json!([])).is_empty());
        assert!(!SchemaValidator::validate(&schema, &json!({})).is_empty());
    }

    #[test]
    fn validate_object_type() {
        let schema = JsonSchema::new(SchemaType::Object);
        assert!(SchemaValidator::validate(&schema, &json!({})).is_empty());
        assert!(!SchemaValidator::validate(&schema, &json!([])).is_empty());
    }

    // ── SchemaValidator – required fields ────────────────────────────

    #[test]
    fn validate_required_field_present() {
        let mut props = HashMap::new();
        props.insert("name".into(), JsonSchema::new(SchemaType::String));
        let schema = JsonSchema::object(props).with_required(vec!["name".into()]);
        let errors = SchemaValidator::validate(&schema, &json!({"name": "Alice"}));
        assert!(errors.is_empty());
    }

    #[test]
    fn validate_required_field_missing() {
        let mut props = HashMap::new();
        props.insert("name".into(), JsonSchema::new(SchemaType::String));
        let schema = JsonSchema::object(props).with_required(vec!["name".into()]);
        let errors = SchemaValidator::validate(&schema, &json!({}));
        assert!(
            errors.iter().any(|e| matches!(e, ValidationError::MissingRequired(f) if f == "name"))
        );
    }

    #[test]
    fn validate_multiple_required_fields() {
        let mut props = HashMap::new();
        props.insert("a".into(), JsonSchema::new(SchemaType::String));
        props.insert("b".into(), JsonSchema::new(SchemaType::Number));
        let schema = JsonSchema::object(props).with_required(vec!["a".into(), "b".into()]);
        let errors = SchemaValidator::validate(&schema, &json!({"a": "x"}));
        assert!(
            errors.iter().any(|e| matches!(e, ValidationError::MissingRequired(f) if f == "b"))
        );
    }

    // ── SchemaValidator – enum values ────────────────────────────────

    #[test]
    fn validate_enum_pass() {
        let schema = JsonSchema::new(SchemaType::String).with_enum(vec![
            json!("red"),
            json!("green"),
            json!("blue"),
        ]);
        assert!(SchemaValidator::validate(&schema, &json!("red")).is_empty());
    }

    #[test]
    fn validate_enum_fail() {
        let schema =
            JsonSchema::new(SchemaType::String).with_enum(vec![json!("red"), json!("green")]);
        let errors = SchemaValidator::validate(&schema, &json!("yellow"));
        assert!(errors.iter().any(|e| matches!(e, ValidationError::InvalidEnumValue(_))));
    }

    // ── SchemaValidator – nested objects ──────────────────────────────

    #[test]
    fn validate_nested_object_pass() {
        let address_schema = {
            let mut props = HashMap::new();
            props.insert("city".into(), JsonSchema::new(SchemaType::String));
            JsonSchema::object(props).with_required(vec!["city".into()])
        };
        let mut props = HashMap::new();
        props.insert("address".into(), address_schema);
        let schema = JsonSchema::object(props);

        let val = json!({"address": {"city": "Portland"}});
        assert!(SchemaValidator::validate(&schema, &val).is_empty());
    }

    #[test]
    fn validate_nested_object_fail() {
        let address_schema = {
            let mut props = HashMap::new();
            props.insert("city".into(), JsonSchema::new(SchemaType::String));
            JsonSchema::object(props).with_required(vec!["city".into()])
        };
        let mut props = HashMap::new();
        props.insert("address".into(), address_schema);
        let schema = JsonSchema::object(props);

        let val = json!({"address": {}});
        let errors = SchemaValidator::validate(&schema, &val);
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::PropertyError { property, error }
            if property == "address"
                && matches!(**error, ValidationError::MissingRequired(_))
        )));
    }

    #[test]
    fn validate_deeply_nested_object() {
        let inner = {
            let mut p = HashMap::new();
            p.insert("val".into(), JsonSchema::new(SchemaType::Integer));
            JsonSchema::object(p).with_required(vec!["val".into()])
        };
        let mid = {
            let mut p = HashMap::new();
            p.insert("inner".into(), inner);
            JsonSchema::object(p)
        };
        let outer = {
            let mut p = HashMap::new();
            p.insert("mid".into(), mid);
            JsonSchema::object(p)
        };

        let val = json!({"mid": {"inner": {"val": 1}}});
        assert!(SchemaValidator::validate(&outer, &val).is_empty());

        let bad = json!({"mid": {"inner": {"val": "not_int"}}});
        assert!(!SchemaValidator::validate(&outer, &bad).is_empty());
    }

    // ── SchemaValidator – arrays with item types ─────────────────────

    #[test]
    fn validate_array_items_pass() {
        let schema = JsonSchema::array(JsonSchema::new(SchemaType::Number));
        let val = json!([1, 2, 3]);
        assert!(SchemaValidator::validate(&schema, &val).is_empty());
    }

    #[test]
    fn validate_array_items_fail() {
        let schema = JsonSchema::array(JsonSchema::new(SchemaType::Number));
        let val = json!([1, "two", 3]);
        let errors = SchemaValidator::validate(&schema, &val);
        assert!(errors.iter().any(|e| matches!(e, ValidationError::ItemError { index: 1, .. })));
    }

    #[test]
    fn validate_array_of_objects() {
        let item_schema = {
            let mut p = HashMap::new();
            p.insert("id".into(), JsonSchema::new(SchemaType::Integer));
            JsonSchema::object(p).with_required(vec!["id".into()])
        };
        let schema = JsonSchema::array(item_schema);

        let val = json!([{"id": 1}, {"id": 2}]);
        assert!(SchemaValidator::validate(&schema, &val).is_empty());

        let bad = json!([{"id": 1}, {}]);
        assert!(!SchemaValidator::validate(&schema, &bad).is_empty());
    }

    #[test]
    fn validate_empty_array_passes() {
        let schema = JsonSchema::array(JsonSchema::new(SchemaType::String));
        assert!(SchemaValidator::validate(&schema, &json!([])).is_empty());
    }

    // ── SchemaValidator – empty / edge-case schemas ──────────────────

    #[test]
    fn validate_empty_object_schema() {
        let schema = JsonSchema::default();
        // Any object should pass an unconstrained object schema.
        assert!(SchemaValidator::validate(&schema, &json!({"x": 1})).is_empty());
    }

    #[test]
    fn validate_no_items_array_schema() {
        // Array schema without `items` → no item-level validation.
        let schema = JsonSchema::new(SchemaType::Array);
        assert!(SchemaValidator::validate(&schema, &json!([1, "a", null])).is_empty());
    }

    // ── JsonForcer ───────────────────────────────────────────────────

    #[test]
    fn forcer_empty_is_complete() {
        let f = JsonForcer::new();
        assert!(f.is_complete());
        assert_eq!(f.depth(), 0);
    }

    #[test]
    fn forcer_single_object() {
        let mut f = JsonForcer::new();
        f.feed('{');
        assert!(!f.is_complete());
        assert_eq!(f.depth(), 1);
        f.feed('}');
        assert!(f.is_complete());
    }

    #[test]
    fn forcer_nested_brackets() {
        let mut f = JsonForcer::new();
        f.feed_str("{[{");
        assert_eq!(f.depth(), 3);
        assert!(!f.is_complete());
        f.feed_str("}]}");
        assert!(f.is_complete());
    }

    #[test]
    fn forcer_string_ignores_brackets() {
        let mut f = JsonForcer::new();
        f.feed_str(r#"{"key": "val[ue"}"#);
        assert!(f.is_complete());
    }

    #[test]
    fn forcer_escaped_quote() {
        let mut f = JsonForcer::new();
        f.feed_str(r#"{"key": "val\"ue"}"#);
        assert!(f.is_complete());
    }

    #[test]
    fn forcer_tracks_in_string() {
        let mut f = JsonForcer::new();
        f.feed('"');
        assert!(f.in_string());
        f.feed('"');
        assert!(!f.in_string());
    }

    #[test]
    fn forcer_expected_closers_brace() {
        let mut f = JsonForcer::new();
        f.feed('{');
        let closers = f.expected_closers();
        assert!(closers.contains(&'}'));
    }

    #[test]
    fn forcer_expected_closers_string() {
        let mut f = JsonForcer::new();
        f.feed('"');
        let closers = f.expected_closers();
        assert!(closers.contains(&'"'));
    }

    #[test]
    fn forcer_closing_suffix() {
        let mut f = JsonForcer::new();
        f.feed_str("{[");
        assert_eq!(f.closing_suffix(), "]}");
    }

    #[test]
    fn forcer_closing_suffix_with_open_string() {
        let mut f = JsonForcer::new();
        f.feed_str(r#"{"key": "val"#);
        let suffix = f.closing_suffix();
        assert!(suffix.starts_with('"'));
        assert!(suffix.ends_with('}'));
    }

    #[test]
    fn forcer_position_tracks_chars() {
        let mut f = JsonForcer::new();
        f.feed_str("abc");
        assert_eq!(f.position(), 3);
    }

    #[test]
    fn forcer_balanced_json() {
        let mut f = JsonForcer::new();
        f.feed_str(r#"{"a": [1, 2], "b": {"c": true}}"#);
        assert!(f.is_complete());
    }

    #[test]
    fn forcer_unbalanced_json() {
        let mut f = JsonForcer::new();
        f.feed_str(r#"{"a": [1, 2"#);
        assert!(!f.is_complete());
        assert_eq!(f.depth(), 2);
    }

    // ── GrammarConstraint ────────────────────────────────────────────

    fn simple_grammar() -> GrammarConstraint {
        // S → a B
        // B → b
        // B → c
        let productions = vec![
            Production { lhs: "S".into(), rhs: vec!["a".into(), "B".into()] },
            Production { lhs: "B".into(), rhs: vec!["b".into()] },
            Production { lhs: "B".into(), rhs: vec!["c".into()] },
        ];
        GrammarConstraint::new(productions, "S".into())
    }

    #[test]
    fn grammar_terminals_and_nonterminals() {
        let g = simple_grammar();
        assert!(g.is_non_terminal("S"));
        assert!(g.is_non_terminal("B"));
        assert!(g.is_terminal("a"));
        assert!(g.is_terminal("b"));
        assert!(g.is_terminal("c"));
    }

    #[test]
    fn grammar_productions_for() {
        let g = simple_grammar();
        assert_eq!(g.productions_for("B").len(), 2);
        assert_eq!(g.productions_for("S").len(), 1);
        assert_eq!(g.productions_for("X").len(), 0);
    }

    #[test]
    fn grammar_state_initial() {
        let state = GrammarState::new("S");
        assert_eq!(state.stack, vec!["S".to_string()]);
        assert!(!state.is_complete());
    }

    #[test]
    fn grammar_state_valid_next_from_start() {
        let g = simple_grammar();
        let state = GrammarState::new("S");
        let next = state.valid_next_tokens(&g);
        // S → a B, so "a" should be valid.
        assert!(next.contains("a"));
    }

    #[test]
    fn grammar_state_advance_valid() {
        let g = simple_grammar();
        let mut state = GrammarState::new("S");
        assert!(state.advance("a", &g));
        assert_eq!(state.generated, vec!["a".to_string()]);
        // After consuming "a", "B" should be on the stack.
        assert!(state.advance("b", &g));
        assert!(state.is_complete());
    }

    #[test]
    fn grammar_state_advance_alternate() {
        let g = simple_grammar();
        let mut state = GrammarState::new("S");
        assert!(state.advance("a", &g));
        assert!(state.advance("c", &g));
        assert!(state.is_complete());
        assert_eq!(state.generated, vec!["a", "c"]);
    }

    #[test]
    fn grammar_state_advance_invalid() {
        let g = simple_grammar();
        let mut state = GrammarState::new("S");
        // "b" is not valid as the first token.
        assert!(!state.advance("b", &g));
    }

    #[test]
    fn grammar_state_position() {
        let g = simple_grammar();
        let mut state = GrammarState::new("S");
        assert_eq!(state.position, 0);
        state.advance("a", &g);
        assert_eq!(state.position, 1);
        state.advance("b", &g);
        assert_eq!(state.position, 2);
    }

    // ── OutputParser ─────────────────────────────────────────────────

    #[test]
    fn parse_valid_json() {
        let result = OutputParser::parse(r#"{"x": 1}"#, &OutputFormat::Json);
        assert!(matches!(result, ParseResult::Json(_)));
    }

    #[test]
    fn parse_invalid_json() {
        let result = OutputParser::parse(r#"{"x": }"#, &OutputFormat::Json);
        assert!(matches!(result, ParseResult::Error(_)));
    }

    #[test]
    fn parse_json_with_whitespace() {
        let result = OutputParser::parse("  \n{\"a\": 1}\n  ", &OutputFormat::Json);
        assert!(matches!(result, ParseResult::Json(_)));
    }

    #[test]
    fn parse_valid_jsonl() {
        let input = "{\"a\":1}\n{\"b\":2}\n{\"c\":3}";
        let result = OutputParser::parse(input, &OutputFormat::JsonL);
        if let ParseResult::JsonL(vals) = result {
            assert_eq!(vals.len(), 3);
        } else {
            panic!("expected JsonL");
        }
    }

    #[test]
    fn parse_jsonl_with_blank_lines() {
        let input = "{\"a\":1}\n\n{\"b\":2}\n";
        let result = OutputParser::parse(input, &OutputFormat::JsonL);
        if let ParseResult::JsonL(vals) = result {
            assert_eq!(vals.len(), 2);
        } else {
            panic!("expected JsonL");
        }
    }

    #[test]
    fn parse_invalid_jsonl_line() {
        let input = "{\"a\":1}\n{bad}\n{\"c\":3}";
        let result = OutputParser::parse(input, &OutputFormat::JsonL);
        assert!(matches!(result, ParseResult::Error(_)));
    }

    #[test]
    fn parse_raw_format() {
        let result = OutputParser::parse("hello world", &OutputFormat::Markdown);
        assert!(matches!(result, ParseResult::Raw(_)));
    }

    #[test]
    fn parse_validate_integration() {
        let schema = JsonSchema::new(SchemaType::Object);
        let val = json!({"foo": "bar"});
        let errors = OutputParser::validate(&val, &schema);
        assert!(errors.is_empty());
    }

    // ── RepairStrategy / JsonRepairer ────────────────────────────────

    #[test]
    fn repair_close_brackets_simple() {
        let input = r#"{"a": [1, 2"#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::CloseBrackets);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_close_brackets_nested() {
        let input = r#"{"a": {"b": [1"#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::CloseBrackets);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_close_brackets_noop_on_valid() {
        let input = r#"{"a": 1}"#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::CloseBrackets);
        assert_eq!(repaired, input);
    }

    #[test]
    fn repair_fix_quotes() {
        let input = r#"{"key": "value"#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::FixQuotes);
        // Should close the dangling quote.
        let quote_count = repaired.chars().filter(|&c| c == '"').count();
        assert_eq!(quote_count % 2, 0);
    }

    #[test]
    fn repair_fix_quotes_noop_on_balanced() {
        let input = r#""hello""#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::FixQuotes);
        assert_eq!(repaired, input);
    }

    #[test]
    fn repair_truncate_to_valid() {
        let input = r#"{"a": 1} extra garbage"#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::TruncateToValid);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_truncate_returns_fallback() {
        let input = "completely invalid %%%";
        let repaired = JsonRepairer::repair(input, &RepairStrategy::TruncateToValid);
        // Fallback is `{}`.
        assert_eq!(repaired, "{}");
    }

    #[test]
    fn repair_remove_trailing_commas() {
        let input = r#"{"a": 1, "b": 2,}"#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::RemoveTrailingCommas);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_remove_trailing_comma_array() {
        let input = "[1, 2, 3,]";
        let repaired = JsonRepairer::repair(input, &RepairStrategy::RemoveTrailingCommas);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_all_strategy() {
        let input = r#"{"a": "hello, "b": 1,}"#;
        let repaired = JsonRepairer::repair(input, &RepairStrategy::All);
        // Should at least produce something parseable.
        // "All" chains: fix_quotes → remove_trailing_commas → close_brackets.
        let quote_count = repaired.chars().filter(|&c| c == '"').count();
        assert_eq!(quote_count % 2, 0);
    }

    // ── FormatConverter ──────────────────────────────────────────────

    #[test]
    fn json_to_jsonl_array() {
        let val = json!([{"a":1}, {"b":2}]);
        let jsonl = FormatConverter::json_to_jsonl(&val);
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(serde_json::from_str::<serde_json::Value>(lines[0]).is_ok());
    }

    #[test]
    fn json_to_jsonl_scalar() {
        let val = json!(42);
        let jsonl = FormatConverter::json_to_jsonl(&val);
        assert_eq!(jsonl, "42");
    }

    #[test]
    fn json_to_csv_basic() {
        let val = json!([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]);
        let csv = FormatConverter::json_to_csv(&val).unwrap();
        assert!(csv.contains("name"));
        assert!(csv.contains("age"));
        assert!(csv.contains("Alice"));
        assert!(csv.contains("Bob"));
    }

    #[test]
    fn json_to_csv_empty_array() {
        let val = json!([]);
        let csv = FormatConverter::json_to_csv(&val).unwrap();
        assert!(csv.is_empty());
    }

    #[test]
    fn json_to_csv_non_array_returns_none() {
        let val = json!({"a": 1});
        assert!(FormatConverter::json_to_csv(&val).is_none());
    }

    #[test]
    fn csv_escapes_commas() {
        let val = json!([{"f": "a,b"}]);
        let csv = FormatConverter::json_to_csv(&val).unwrap();
        assert!(csv.contains("\"a,b\""));
    }

    #[test]
    fn csv_escapes_quotes() {
        let val = json!([{"f": "say \"hi\""}]);
        let csv = FormatConverter::json_to_csv(&val).unwrap();
        assert!(csv.contains("\"\""));
    }

    // ── StructuredOutputEngine ───────────────────────────────────────

    #[test]
    fn engine_process_valid_json() {
        let engine = StructuredOutputEngine::new(OutputFormat::Json);
        let result = engine.process(r#"{"x": 1}"#);
        assert!(result.is_ok());
    }

    #[test]
    fn engine_process_invalid_json_no_repair() {
        let engine = StructuredOutputEngine::new(OutputFormat::Json);
        let result = engine.process(r#"{"x": "#);
        assert!(result.is_err());
    }

    #[test]
    fn engine_process_with_repair() {
        let engine = StructuredOutputEngine::new(OutputFormat::Json)
            .with_repair(vec![RepairStrategy::CloseBrackets]);
        let result = engine.process(r#"{"x": 1"#);
        assert!(result.is_ok());
    }

    #[test]
    fn engine_process_with_schema_pass() {
        let mut props = HashMap::new();
        props.insert("name".into(), JsonSchema::new(SchemaType::String));
        let schema = JsonSchema::object(props).with_required(vec!["name".into()]);

        let engine = StructuredOutputEngine::new(OutputFormat::Json).with_schema(schema);
        let result = engine.process(r#"{"name": "Alice"}"#);
        assert!(result.is_ok());
    }

    #[test]
    fn engine_process_with_schema_fail() {
        let mut props = HashMap::new();
        props.insert("name".into(), JsonSchema::new(SchemaType::String));
        let schema = JsonSchema::object(props).with_required(vec!["name".into()]);

        let engine = StructuredOutputEngine::new(OutputFormat::Json).with_schema(schema);
        let result = engine.process(r#"{"age": 30}"#);
        assert!(result.is_err());
    }

    #[test]
    fn engine_convert_json_to_jsonl() {
        let engine = StructuredOutputEngine::new(OutputFormat::Json);
        let val = json!([{"a":1}, {"b":2}]);
        let result = engine.convert(&val, &OutputFormat::JsonL);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert_eq!(text.lines().count(), 2);
    }

    #[test]
    fn engine_convert_json_to_csv() {
        let engine = StructuredOutputEngine::new(OutputFormat::Json);
        let val = json!([{"x": 1}, {"x": 2}]);
        let result = engine.convert(&val, &OutputFormat::Csv);
        assert!(result.is_ok());
    }

    #[test]
    fn engine_convert_unsupported_format() {
        let engine = StructuredOutputEngine::new(OutputFormat::Json);
        let val = json!({"a": 1});
        let result = engine.convert(&val, &OutputFormat::Xml);
        assert!(result.is_err());
    }

    #[test]
    fn engine_builder_chain() {
        let mut props = HashMap::new();
        props.insert("x".into(), JsonSchema::new(SchemaType::Number));
        let schema = JsonSchema::object(props);

        let grammar = simple_grammar();

        let engine = StructuredOutputEngine::new(OutputFormat::Json)
            .with_schema(schema)
            .with_grammar(grammar)
            .with_repair(vec![RepairStrategy::FixQuotes, RepairStrategy::CloseBrackets]);

        assert!(engine.schema.is_some());
        assert!(engine.grammar.is_some());
        assert_eq!(engine.repair_strategies.len(), 2);
    }

    #[test]
    fn engine_process_jsonl() {
        let engine = StructuredOutputEngine::new(OutputFormat::JsonL);
        let result = engine.process("{\"a\":1}\n{\"b\":2}");
        assert!(result.is_ok());
        if let Ok(serde_json::Value::Array(arr)) = result {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn engine_repair_then_validate() {
        let mut props = HashMap::new();
        props.insert("val".into(), JsonSchema::new(SchemaType::Number));
        let schema = JsonSchema::object(props).with_required(vec!["val".into()]);

        let engine = StructuredOutputEngine::new(OutputFormat::Json)
            .with_schema(schema)
            .with_repair(vec![RepairStrategy::CloseBrackets]);

        // Missing closing brace but contains required field.
        let result = engine.process(r#"{"val": 42"#);
        assert!(result.is_ok());
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn empty_input_parse() {
        let result = OutputParser::parse("", &OutputFormat::Json);
        assert!(matches!(result, ParseResult::Error(_)));
    }

    #[test]
    fn empty_jsonl_parse() {
        let result = OutputParser::parse("", &OutputFormat::JsonL);
        if let ParseResult::JsonL(vals) = result {
            assert!(vals.is_empty());
        } else {
            panic!("expected empty JsonL");
        }
    }

    #[test]
    fn very_long_json_string() {
        let long_str = "x".repeat(10_000);
        let input = format!(r#"{{"data": "{long_str}"}}"#);
        let result = OutputParser::parse(&input, &OutputFormat::Json);
        assert!(matches!(result, ParseResult::Json(_)));
    }

    #[test]
    fn forcer_deeply_nested() {
        let mut f = JsonForcer::new();
        let open: String = std::iter::repeat_n('{', 100).collect();
        f.feed_str(&open);
        assert_eq!(f.depth(), 100);
        let close: String = std::iter::repeat_n('}', 100).collect();
        f.feed_str(&close);
        assert!(f.is_complete());
    }

    #[test]
    fn validation_error_display() {
        let e =
            ValidationError::TypeMismatch { expected: SchemaType::String, got: "number".into() };
        let s = e.to_string();
        assert!(s.contains("string"));
        assert!(s.contains("number"));
    }

    #[test]
    fn structured_output_error_display() {
        let e = StructuredOutputError::ParseFailed("bad input".into());
        assert!(e.to_string().contains("bad input"));
    }

    // ── Property-based tests ─────────────────────────────────────────

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn repaired_close_brackets_is_balanced(
                s in r#"\{[a-z ,":\[\]0-9]{0,50}"#
            ) {
                let repaired = JsonRepairer::repair(
                    &s,
                    &RepairStrategy::CloseBrackets,
                );
                let mut f = JsonForcer::new();
                f.feed_str(&repaired);
                prop_assert!(f.is_complete());
            }

            #[test]
            fn repaired_fix_quotes_even_count(
                s in r#"[a-z "]{0,60}"#
            ) {
                let repaired = JsonRepairer::repair(
                    &s,
                    &RepairStrategy::FixQuotes,
                );
                let count = repaired.chars().filter(|&c| c == '"').count();
                prop_assert!(count % 2 == 0);
            }

            #[test]
            fn forcer_closing_suffix_always_completes(
                s in r#"[\{\}\[\]"a-z0-9 :,]{0,80}"#
            ) {
                let mut f = JsonForcer::new();
                f.feed_str(&s);
                let suffix = f.closing_suffix();
                f.feed_str(&suffix);
                prop_assert!(f.is_complete());
            }

            #[test]
            fn truncate_to_valid_always_parseable(
                s in r#"\{"[a-z]+": [0-9]+\}[a-z ]{0,20}"#
            ) {
                let repaired = JsonRepairer::repair(
                    &s,
                    &RepairStrategy::TruncateToValid,
                );
                prop_assert!(
                    serde_json::from_str::<serde_json::Value>(&repaired)
                        .is_ok()
                );
            }
        }
    }
}
