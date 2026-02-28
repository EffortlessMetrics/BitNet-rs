//! Guided Generation / Grammar-based Decoding.
//!
//! Provides grammar-constrained decoding for structured output generation.
//! Supports context-free grammars, JSON schema constraints, and regex
//! patterns.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

// ── Core Grammar Types ─────────────────────────────────────────────

/// A symbol in a grammar production rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GrammarSymbol {
    /// Matches a literal string token.
    Terminal(String),
    /// References a named grammar rule.
    NonTerminal(String),
}

/// A production is a sequence of grammar symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Production {
    pub symbols: Vec<GrammarSymbol>,
}

impl Production {
    pub const fn new(symbols: Vec<GrammarSymbol>) -> Self {
        Self { symbols }
    }

    pub const fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

/// A grammar rule defines how a symbol can be expanded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarRule {
    /// A non-terminal with named alternatives.
    NonTerminal { name: String, productions: Vec<Production> },
    /// A terminal pattern that matches token text.
    Terminal { pattern: String },
}

impl GrammarRule {
    /// Return the name (non-terminal) or pattern (terminal).
    pub fn name(&self) -> &str {
        match self {
            Self::NonTerminal { name, .. } => name,
            Self::Terminal { pattern } => pattern,
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────────

/// Errors during grammar operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    /// Referenced non-terminal not found in rules.
    UndefinedSymbol(String),
    /// Start symbol not found.
    MissingStartSymbol(String),
    /// Grammar has no rules.
    EmptyGrammar,
    /// Rule name conflicts.
    DuplicateRule(String),
    /// Grammar not compiled yet.
    NotCompiled,
    /// Invalid JSON schema.
    InvalidSchema(String),
    /// Invalid regex pattern.
    InvalidRegex(String),
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndefinedSymbol(s) => {
                write!(f, "undefined symbol: {s}")
            }
            Self::MissingStartSymbol(s) => {
                write!(f, "missing start symbol: {s}")
            }
            Self::EmptyGrammar => write!(f, "grammar has no rules"),
            Self::DuplicateRule(s) => write!(f, "duplicate rule: {s}"),
            Self::NotCompiled => write!(f, "grammar not compiled"),
            Self::InvalidSchema(s) => {
                write!(f, "invalid schema: {s}")
            }
            Self::InvalidRegex(s) => {
                write!(f, "invalid regex: {s}")
            }
        }
    }
}

impl std::error::Error for GrammarError {}

// ── Grammar ────────────────────────────────────────────────────────

/// A context-free grammar for constrained decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    pub rules: HashMap<String, GrammarRule>,
    pub start_symbol: String,
    compiled: bool,
}

impl Grammar {
    pub fn new(start_symbol: String) -> Self {
        Self { rules: HashMap::new(), start_symbol, compiled: false }
    }

    /// Add a rule to the grammar.
    ///
    /// # Errors
    /// Returns `DuplicateRule` if the name already exists.
    pub fn add_rule(&mut self, name: String, rule: GrammarRule) -> Result<(), GrammarError> {
        if self.rules.contains_key(&name) {
            return Err(GrammarError::DuplicateRule(name));
        }
        self.rules.insert(name, rule);
        self.compiled = false;
        Ok(())
    }

    /// Compile and validate the grammar.
    ///
    /// # Errors
    /// Returns errors for empty grammar, missing start symbol, or
    /// undefined non-terminal references.
    pub fn compile(&mut self) -> Result<(), GrammarError> {
        if self.rules.is_empty() {
            return Err(GrammarError::EmptyGrammar);
        }
        if !self.rules.contains_key(&self.start_symbol) {
            return Err(GrammarError::MissingStartSymbol(self.start_symbol.clone()));
        }

        // Validate all non-terminal references.
        for rule in self.rules.values() {
            if let GrammarRule::NonTerminal { productions, .. } = rule {
                for prod in productions {
                    for sym in &prod.symbols {
                        if let GrammarSymbol::NonTerminal(n) = sym
                            && !self.rules.contains_key(n)
                        {
                            return Err(GrammarError::UndefinedSymbol(n.clone()));
                        }
                    }
                }
            }
        }

        self.compiled = true;
        Ok(())
    }

    pub const fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Compute the FIRST set for a symbol.
    pub fn first_set(&self, symbol: &GrammarSymbol) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        self.first_set_recursive(symbol, &mut result, &mut visited);
        result
    }

    /// Compute the FIRST set for a production's symbol sequence.
    pub fn production_first_set(&self, production: &Production) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        self.first_set_for_sequence(&production.symbols, &mut result, &mut visited);
        result
    }

    fn first_set_recursive(
        &self,
        symbol: &GrammarSymbol,
        result: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) {
        match symbol {
            GrammarSymbol::Terminal(t) => {
                result.insert(t.clone());
            }
            GrammarSymbol::NonTerminal(name) => {
                if visited.contains(name) {
                    return;
                }
                visited.insert(name.clone());
                if let Some(rule) = self.rules.get(name) {
                    match rule {
                        GrammarRule::NonTerminal { productions, .. } => {
                            for prod in productions {
                                self.first_set_for_sequence(&prod.symbols, result, visited);
                            }
                        }
                        GrammarRule::Terminal { pattern } => {
                            result.insert(pattern.clone());
                        }
                    }
                }
            }
        }
    }

    fn first_set_for_sequence(
        &self,
        symbols: &[GrammarSymbol],
        result: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) {
        for symbol in symbols {
            self.first_set_recursive(symbol, result, visited);
            if !self.can_derive_epsilon(symbol) {
                break;
            }
        }
    }

    fn can_derive_epsilon(&self, symbol: &GrammarSymbol) -> bool {
        match symbol {
            GrammarSymbol::Terminal(_) => false,
            GrammarSymbol::NonTerminal(name) => {
                if let Some(GrammarRule::NonTerminal { productions, .. }) = self.rules.get(name) {
                    productions.iter().any(|p| p.symbols.is_empty())
                } else {
                    false
                }
            }
        }
    }
}

// ── Grammar State ──────────────────────────────────────────────────

/// Current parse state during grammar-constrained decoding.
#[derive(Debug, Clone)]
pub struct GrammarState {
    /// Stack of expected symbols (last element = next to match).
    pub stack: Vec<GrammarSymbol>,
    /// Number of tokens matched so far.
    pub position: usize,
    /// Whether parsing has completed successfully.
    pub completed: bool,
}

impl GrammarState {
    /// Create an initial state from a grammar's start symbol.
    pub fn new(start_symbol: &str) -> Self {
        Self {
            stack: vec![GrammarSymbol::NonTerminal(start_symbol.to_string())],
            position: 0,
            completed: false,
        }
    }

    /// Check if the parse is complete (stack empty).
    pub const fn is_complete(&self) -> bool {
        self.stack.is_empty() || self.completed
    }

    /// Peek at the next expected symbol.
    pub fn peek(&self) -> Option<&GrammarSymbol> {
        self.stack.last()
    }

    /// Advance the state by matching a terminal.
    ///
    /// # Errors
    /// Returns `NotCompiled` if the grammar has not been compiled.
    pub fn advance(&mut self, token_text: &str, grammar: &Grammar) -> Result<(), GrammarError> {
        if !grammar.is_compiled() {
            return Err(GrammarError::NotCompiled);
        }

        self.expand_matching(token_text, grammar);

        if let Some(GrammarSymbol::Terminal(expected)) = self.stack.last()
            && expected == token_text
        {
            self.stack.pop();
            self.position += 1;
        }

        if self.stack.is_empty() {
            self.completed = true;
        }

        Ok(())
    }

    /// Expand non-terminals on top of stack, choosing the production
    /// whose FIRST set contains `token_text`.
    fn expand_matching(&mut self, token_text: &str, grammar: &Grammar) {
        let mut depth = 0;
        while depth < 100 {
            let Some(GrammarSymbol::NonTerminal(top)) = self.stack.last().cloned() else { break };

            let Some(rule) = grammar.rules.get(&top) else {
                break;
            };

            match rule {
                GrammarRule::NonTerminal { productions, .. } => {
                    let matching = productions.iter().find(|p| {
                        if p.symbols.is_empty() {
                            return false;
                        }
                        grammar.production_first_set(p).contains(token_text)
                    });

                    if let Some(prod) = matching {
                        self.stack.pop();
                        for sym in prod.symbols.iter().rev() {
                            self.stack.push(sym.clone());
                        }
                        depth += 1;
                        continue;
                    }

                    // Try epsilon production.
                    let has_eps = productions.iter().any(|p| p.symbols.is_empty());
                    if has_eps {
                        self.stack.pop();
                        depth += 1;
                        continue;
                    }

                    break;
                }
                GrammarRule::Terminal { pattern } => {
                    self.stack.pop();
                    self.stack.push(GrammarSymbol::Terminal(pattern.clone()));
                    depth += 1;
                }
            }
        }
    }
}

// ── Grammar Constrainer ────────────────────────────────────────────

/// Computes allowed tokens and masks invalid logits.
#[derive(Debug)]
pub struct GrammarConstrainer {
    grammar: Grammar,
    vocab: Vec<String>,
}

impl GrammarConstrainer {
    /// Create a constrainer from a compiled grammar and vocabulary.
    ///
    /// # Errors
    /// Returns `NotCompiled` if the grammar has not been compiled.
    pub fn new(grammar: Grammar, vocab: Vec<String>) -> Result<Self, GrammarError> {
        if !grammar.is_compiled() {
            return Err(GrammarError::NotCompiled);
        }
        Ok(Self { grammar, vocab })
    }

    /// Compute which token IDs are allowed at the current state.
    #[allow(clippy::cast_possible_truncation)]
    pub fn allowed_tokens(&self, state: &GrammarState) -> Vec<u32> {
        if state.is_complete() {
            return Vec::new();
        }

        let allowed_terminals = self.compute_allowed_terminals(state);

        let mut allowed = Vec::new();
        for (id, text) in self.vocab.iter().enumerate() {
            if allowed_terminals.contains(text) {
                allowed.push(id as u32);
            }
        }
        allowed
    }

    /// Mask logits for invalid tokens (set to `NEG_INFINITY`).
    pub fn apply_constraint(&self, logits: &mut [f32], state: &GrammarState) {
        let allowed = self.allowed_tokens(state);
        let allowed_set: HashSet<u32> = allowed.into_iter().collect();

        for (i, logit) in logits.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            if !allowed_set.contains(&(i as u32)) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    pub const fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    fn compute_allowed_terminals(&self, state: &GrammarState) -> HashSet<String> {
        let mut result = HashSet::new();

        for symbol in state.stack.iter().rev() {
            let firsts = self.grammar.first_set(symbol);
            result.extend(firsts);

            if !self.grammar.can_derive_epsilon(symbol) {
                break;
            }
        }

        result
    }
}

// ── JSON Schema Grammar ────────────────────────────────────────────

/// Converts a JSON schema into a grammar for constrained decoding.
pub struct JsonSchemaGrammar;

impl JsonSchemaGrammar {
    /// Convert a JSON schema to a compiled [`Grammar`].
    ///
    /// # Errors
    /// Returns errors for unsupported types or invalid schemas.
    pub fn from_schema(schema: &serde_json::Value) -> Result<Grammar, GrammarError> {
        let mut grammar = Grammar::new("root".to_string());
        Self::add_schema_rules(&mut grammar, "root", schema)?;
        grammar.compile()?;
        Ok(grammar)
    }

    fn add_schema_rules(
        grammar: &mut Grammar,
        name: &str,
        schema: &serde_json::Value,
    ) -> Result<(), GrammarError> {
        if let Some(enum_values) = schema.get("enum") {
            return Self::add_enum_rules(grammar, name, enum_values);
        }

        let schema_type = schema.get("type").and_then(|t| t.as_str()).unwrap_or("object");

        match schema_type {
            "object" => Self::add_object_rules(grammar, name, schema),
            "array" => Self::add_array_rules(grammar, name, schema),
            "string" => Self::add_string_rules(grammar, name),
            "number" | "integer" => Self::add_number_rules(grammar, name),
            "boolean" => Self::add_boolean_rules(grammar, name),
            "null" => Self::add_null_rules(grammar, name),
            other => Err(GrammarError::InvalidSchema(format!("unsupported type: {other}"))),
        }
    }

    fn add_object_rules(
        grammar: &mut Grammar,
        name: &str,
        schema: &serde_json::Value,
    ) -> Result<(), GrammarError> {
        let mut productions = Vec::new();

        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            let mut symbols = vec![GrammarSymbol::Terminal("{".to_string())];
            let prop_count = properties.len();

            for (i, (prop_name, prop_schema)) in properties.iter().enumerate() {
                let prop_rule = format!("{name}_prop_{prop_name}");
                symbols.push(GrammarSymbol::Terminal(format!("\"{prop_name}\"")));
                symbols.push(GrammarSymbol::Terminal(":".to_string()));
                Self::add_schema_rules(grammar, &prop_rule, prop_schema)?;
                symbols.push(GrammarSymbol::NonTerminal(prop_rule));
                if i + 1 < prop_count {
                    symbols.push(GrammarSymbol::Terminal(",".to_string()));
                }
            }
            symbols.push(GrammarSymbol::Terminal("}".to_string()));
            productions.push(Production::new(symbols));
        } else {
            productions.push(Production::new(vec![
                GrammarSymbol::Terminal("{".to_string()),
                GrammarSymbol::Terminal("}".to_string()),
            ]));
        }

        grammar.add_rule(
            name.to_string(),
            GrammarRule::NonTerminal { name: name.to_string(), productions },
        )
    }

    fn add_array_rules(
        grammar: &mut Grammar,
        name: &str,
        schema: &serde_json::Value,
    ) -> Result<(), GrammarError> {
        let items_rule = format!("{name}_item");
        if let Some(items) = schema.get("items") {
            Self::add_schema_rules(grammar, &items_rule, items)?;
        } else {
            Self::add_string_rules(grammar, &items_rule)?;
        }

        let elements_rule = format!("{name}_elements");
        grammar.add_rule(
            elements_rule.clone(),
            GrammarRule::NonTerminal {
                name: elements_rule.clone(),
                productions: vec![
                    Production::new(vec![GrammarSymbol::NonTerminal(items_rule.clone())]),
                    Production::new(vec![
                        GrammarSymbol::NonTerminal(items_rule),
                        GrammarSymbol::Terminal(",".to_string()),
                        GrammarSymbol::NonTerminal(elements_rule.clone()),
                    ]),
                ],
            },
        )?;

        grammar.add_rule(
            name.to_string(),
            GrammarRule::NonTerminal {
                name: name.to_string(),
                productions: vec![
                    Production::new(vec![
                        GrammarSymbol::Terminal("[".to_string()),
                        GrammarSymbol::NonTerminal(elements_rule),
                        GrammarSymbol::Terminal("]".to_string()),
                    ]),
                    Production::new(vec![
                        GrammarSymbol::Terminal("[".to_string()),
                        GrammarSymbol::Terminal("]".to_string()),
                    ]),
                ],
            },
        )
    }

    fn add_string_rules(grammar: &mut Grammar, name: &str) -> Result<(), GrammarError> {
        grammar.add_rule(
            name.to_string(),
            GrammarRule::NonTerminal {
                name: name.to_string(),
                productions: vec![Production::new(vec![
                    GrammarSymbol::Terminal("\"".to_string()),
                    GrammarSymbol::Terminal("text".to_string()),
                    GrammarSymbol::Terminal("\"".to_string()),
                ])],
            },
        )
    }

    fn add_number_rules(grammar: &mut Grammar, name: &str) -> Result<(), GrammarError> {
        grammar.add_rule(
            name.to_string(),
            GrammarRule::NonTerminal {
                name: name.to_string(),
                productions: vec![
                    Production::new(vec![GrammarSymbol::Terminal("number".to_string())]),
                    Production::new(vec![
                        GrammarSymbol::Terminal("-".to_string()),
                        GrammarSymbol::Terminal("number".to_string()),
                    ]),
                ],
            },
        )
    }

    fn add_boolean_rules(grammar: &mut Grammar, name: &str) -> Result<(), GrammarError> {
        grammar.add_rule(
            name.to_string(),
            GrammarRule::NonTerminal {
                name: name.to_string(),
                productions: vec![
                    Production::new(vec![GrammarSymbol::Terminal("true".to_string())]),
                    Production::new(vec![GrammarSymbol::Terminal("false".to_string())]),
                ],
            },
        )
    }

    fn add_null_rules(grammar: &mut Grammar, name: &str) -> Result<(), GrammarError> {
        grammar.add_rule(
            name.to_string(),
            GrammarRule::NonTerminal {
                name: name.to_string(),
                productions: vec![Production::new(vec![GrammarSymbol::Terminal(
                    "null".to_string(),
                )])],
            },
        )
    }

    fn add_enum_rules(
        grammar: &mut Grammar,
        name: &str,
        values: &serde_json::Value,
    ) -> Result<(), GrammarError> {
        let arr = values
            .as_array()
            .ok_or_else(|| GrammarError::InvalidSchema("enum must be an array".to_string()))?;
        if arr.is_empty() {
            return Err(GrammarError::InvalidSchema("empty enum".to_string()));
        }

        let productions: Vec<Production> = arr
            .iter()
            .map(|v| {
                let text = match v {
                    serde_json::Value::String(s) => {
                        format!("\"{s}\"")
                    }
                    other => other.to_string(),
                };
                Production::new(vec![GrammarSymbol::Terminal(text)])
            })
            .collect();

        grammar.add_rule(
            name.to_string(),
            GrammarRule::NonTerminal { name: name.to_string(), productions },
        )
    }
}

// ── Regex Grammar ──────────────────────────────────────────────────

/// Converts simple regex patterns into grammars.
pub struct RegexGrammar;

impl RegexGrammar {
    /// Convert a simple regex pattern to a compiled [`Grammar`].
    ///
    /// Supports: literals, `.`, `*`, `+`, `?`, `[...]`, `|`, `(...)`,
    /// and `\` escapes.
    ///
    /// # Errors
    /// Returns errors for empty patterns or syntax errors.
    pub fn from_pattern(pattern: &str) -> Result<Grammar, GrammarError> {
        if pattern.is_empty() {
            return Err(GrammarError::InvalidRegex("empty pattern".to_string()));
        }

        let mut grammar = Grammar::new("root".to_string());
        let mut productions = Vec::new();
        let mut current: Vec<GrammarSymbol> = Vec::new();
        let mut counter = 0_u32;

        let chars: Vec<char> = pattern.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                '|' => {
                    if !current.is_empty() {
                        productions.push(Production::new(std::mem::take(&mut current)));
                    }
                    i += 1;
                }
                '[' => {
                    i = Self::parse_char_class(
                        &chars,
                        i,
                        &mut grammar,
                        &mut current,
                        &mut counter,
                    )?;
                }
                '(' => {
                    i = Self::parse_group(&chars, i, &mut grammar, &mut current, &mut counter)?;
                }
                '.' => {
                    current.push(GrammarSymbol::Terminal(".".to_string()));
                    i += 1;
                }
                '*' | '+' | '?' => {
                    Self::parse_quantifier(chars[i], &mut grammar, &mut current, &mut counter)?;
                    i += 1;
                }
                '\\' if i + 1 < chars.len() => {
                    i += 1;
                    current.push(GrammarSymbol::Terminal(chars[i].to_string()));
                    i += 1;
                }
                c => {
                    current.push(GrammarSymbol::Terminal(c.to_string()));
                    i += 1;
                }
            }
        }

        if !current.is_empty() {
            productions.push(Production::new(std::mem::take(&mut current)));
        }
        if productions.is_empty() {
            productions.push(Production::new(vec![]));
        }

        grammar.add_rule(
            "root".to_string(),
            GrammarRule::NonTerminal { name: "root".to_string(), productions },
        )?;
        grammar.compile()?;
        Ok(grammar)
    }

    fn parse_char_class(
        chars: &[char],
        pos: usize,
        grammar: &mut Grammar,
        current: &mut Vec<GrammarSymbol>,
        counter: &mut u32,
    ) -> Result<usize, GrammarError> {
        let start = pos + 1;
        let mut end = start;
        while end < chars.len() && chars[end] != ']' {
            end += 1;
        }
        if end >= chars.len() {
            return Err(GrammarError::InvalidRegex("unclosed character class".to_string()));
        }
        let class_name = format!("char_class_{counter}");
        *counter += 1;
        let class_prods: Vec<Production> = (start..end)
            .map(|j| Production::new(vec![GrammarSymbol::Terminal(chars[j].to_string())]))
            .collect();
        grammar.add_rule(
            class_name.clone(),
            GrammarRule::NonTerminal { name: class_name.clone(), productions: class_prods },
        )?;
        current.push(GrammarSymbol::NonTerminal(class_name));
        Ok(end + 1)
    }

    fn parse_group(
        chars: &[char],
        pos: usize,
        grammar: &mut Grammar,
        current: &mut Vec<GrammarSymbol>,
        counter: &mut u32,
    ) -> Result<usize, GrammarError> {
        let start = pos + 1;
        let mut depth = 1_u32;
        let mut end = start;
        while end < chars.len() && depth > 0 {
            match chars[end] {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            if depth > 0 {
                end += 1;
            }
        }
        if depth != 0 {
            return Err(GrammarError::InvalidRegex("unclosed parenthesis".to_string()));
        }
        let group_name = format!("group_{counter}");
        *counter += 1;
        let sub: String = chars[start..end].iter().collect();
        let sub_grammar = Self::from_pattern(&sub)?;
        for (key, rule) in &sub_grammar.rules {
            let pfx =
                if key == "root" { group_name.clone() } else { format!("{group_name}_{key}") };
            if !grammar.rules.contains_key(&pfx) {
                let remapped = Self::remap_rule(rule, &group_name);
                grammar.add_rule(pfx, remapped)?;
            }
        }
        current.push(GrammarSymbol::NonTerminal(group_name));
        Ok(end + 1)
    }

    fn parse_quantifier(
        quantifier: char,
        grammar: &mut Grammar,
        current: &mut Vec<GrammarSymbol>,
        counter: &mut u32,
    ) -> Result<(), GrammarError> {
        if let Some(last) = current.pop() {
            let rep_name = format!("rep_{counter}");
            *counter += 1;
            let rep_sym = GrammarSymbol::NonTerminal(rep_name.clone());
            let prods = match quantifier {
                '*' => vec![Production::new(vec![]), Production::new(vec![last, rep_sym])],
                '+' => {
                    vec![Production::new(vec![last.clone()]), Production::new(vec![last, rep_sym])]
                }
                '?' => vec![Production::new(vec![]), Production::new(vec![last])],
                _ => unreachable!(),
            };
            grammar.add_rule(
                rep_name.clone(),
                GrammarRule::NonTerminal { name: rep_name.clone(), productions: prods },
            )?;
            current.push(GrammarSymbol::NonTerminal(rep_name));
        }
        Ok(())
    }

    fn remap_rule(rule: &GrammarRule, group_name: &str) -> GrammarRule {
        match rule {
            GrammarRule::Terminal { pattern } => GrammarRule::Terminal { pattern: pattern.clone() },
            GrammarRule::NonTerminal { productions, .. } => {
                let new_name = group_name.to_string();
                let new_prods = productions
                    .iter()
                    .map(|p| {
                        let syms = p
                            .symbols
                            .iter()
                            .map(|s| match s {
                                GrammarSymbol::NonTerminal(n) if n == "root" => {
                                    GrammarSymbol::NonTerminal(group_name.to_string())
                                }
                                GrammarSymbol::NonTerminal(n) => {
                                    GrammarSymbol::NonTerminal(format!("{group_name}_{n}"))
                                }
                                other @ GrammarSymbol::Terminal(_) => other.clone(),
                            })
                            .collect();
                        Production::new(syms)
                    })
                    .collect();
                GrammarRule::NonTerminal { name: new_name, productions: new_prods }
            }
        }
    }
}

// ── Guided Decoder ─────────────────────────────────────────────────

/// Wraps a decoding step with grammar constraints.
pub struct GuidedDecoder {
    constrainer: GrammarConstrainer,
    state: GrammarState,
    metrics: GuidedGenerationMetrics,
}

impl GuidedDecoder {
    /// Create a decoder with grammar constraints.
    ///
    /// # Errors
    /// Returns `NotCompiled` if the grammar has not been compiled.
    pub fn new(grammar: Grammar, vocab: Vec<String>) -> Result<Self, GrammarError> {
        let start = grammar.start_symbol.clone();
        let constrainer = GrammarConstrainer::new(grammar, vocab)?;
        Ok(Self {
            constrainer,
            state: GrammarState::new(&start),
            metrics: GuidedGenerationMetrics::new(),
        })
    }

    /// Mask invalid logits and return allowed token IDs.
    pub fn constrain_logits(&mut self, logits: &mut [f32]) -> Vec<u32> {
        self.metrics.grammar_evaluations += 1;
        let allowed = self.constrainer.allowed_tokens(&self.state);

        #[allow(clippy::cast_possible_truncation)]
        {
            self.metrics.tokens_constrained += logits.len() as u64 - allowed.len() as u64;
            self.metrics.tokens_allowed_total += allowed.len() as u64;
        }
        self.metrics.steps += 1;

        self.constrainer.apply_constraint(logits, &self.state);
        allowed
    }

    /// Advance state after selecting a token.
    ///
    /// # Errors
    /// Returns `NotCompiled` (should not happen in normal usage).
    pub fn advance(&mut self, token_id: u32) -> Result<(), GrammarError> {
        let text = self.constrainer.vocab().get(token_id as usize).cloned();
        if let Some(text) = text {
            self.state.advance(&text, self.constrainer.grammar())?;
        }
        Ok(())
    }

    /// Check if generation should stop (grammar complete).
    pub const fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    pub const fn metrics(&self) -> &GuidedGenerationMetrics {
        &self.metrics
    }

    pub const fn state(&self) -> &GrammarState {
        &self.state
    }

    /// Reset the decoder to initial state.
    pub fn reset(&mut self) {
        let start = self.constrainer.grammar().start_symbol.clone();
        self.state = GrammarState::new(&start);
    }
}

// ── Grammar Cache ──────────────────────────────────────────────────

/// Caches compiled grammars by schema hash for reuse.
pub struct GrammarCache {
    cache: HashMap<u64, Grammar>,
    hits: u64,
    misses: u64,
}

impl GrammarCache {
    pub fn new() -> Self {
        Self { cache: HashMap::new(), hits: 0, misses: 0 }
    }

    /// Get a cached grammar or compile one from a JSON schema.
    ///
    /// # Errors
    /// Returns grammar compilation errors on cache miss.
    pub fn get_or_compile(&mut self, schema: &serde_json::Value) -> Result<Grammar, GrammarError> {
        let hash = Self::hash_schema(schema);
        if let Some(grammar) = self.cache.get(&hash) {
            self.hits += 1;
            return Ok(grammar.clone());
        }
        self.misses += 1;
        let grammar = JsonSchemaGrammar::from_schema(schema)?;
        self.cache.insert(hash, grammar.clone());
        Ok(grammar)
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub const fn hits(&self) -> u64 {
        self.hits
    }

    pub const fn misses(&self) -> u64 {
        self.misses
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    fn hash_schema(schema: &serde_json::Value) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        schema.to_string().hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for GrammarCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Metrics ────────────────────────────────────────────────────────

/// Metrics for guided generation performance tracking.
#[derive(Debug, Clone)]
pub struct GuidedGenerationMetrics {
    /// Tokens masked (constrained out) across all steps.
    pub tokens_constrained: u64,
    /// Total allowed tokens across all steps.
    pub tokens_allowed_total: u64,
    /// Number of grammar evaluation steps.
    pub grammar_evaluations: u64,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of decoding steps.
    pub steps: u64,
}

impl GuidedGenerationMetrics {
    pub const fn new() -> Self {
        Self {
            tokens_constrained: 0,
            tokens_allowed_total: 0,
            grammar_evaluations: 0,
            cache_hits: 0,
            steps: 0,
        }
    }

    /// Average allowed tokens per step.
    pub fn tokens_allowed_avg(&self) -> f64 {
        if self.steps == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.tokens_allowed_total as f64 / self.steps as f64
        }
    }

    /// Reset all metrics.
    pub const fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for GuidedGenerationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_vocab(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| (*s).to_string()).collect()
    }

    fn simple_seq_grammar() -> Grammar {
        let mut g = Grammar::new("root".to_string());
        g.add_rule(
            "root".to_string(),
            GrammarRule::NonTerminal {
                name: "root".to_string(),
                productions: vec![Production::new(vec![
                    GrammarSymbol::Terminal("hello".to_string()),
                    GrammarSymbol::Terminal("world".to_string()),
                ])],
            },
        )
        .unwrap();
        g.compile().unwrap();
        g
    }

    fn bool_grammar() -> Grammar {
        let mut g = Grammar::new("root".to_string());
        g.add_rule(
            "root".to_string(),
            GrammarRule::NonTerminal {
                name: "root".to_string(),
                productions: vec![
                    Production::new(vec![GrammarSymbol::Terminal("true".to_string())]),
                    Production::new(vec![GrammarSymbol::Terminal("false".to_string())]),
                ],
            },
        )
        .unwrap();
        g.compile().unwrap();
        g
    }

    // ── Grammar compilation ────────────────────────────────────

    #[test]
    fn test_compile_empty_grammar() {
        let mut g = Grammar::new("root".to_string());
        assert_eq!(g.compile(), Err(GrammarError::EmptyGrammar));
    }

    #[test]
    fn test_compile_missing_start_symbol() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule("other".to_string(), GrammarRule::Terminal { pattern: "x".to_string() })
            .unwrap();
        assert_eq!(g.compile(), Err(GrammarError::MissingStartSymbol("root".to_string())));
    }

    #[test]
    fn test_compile_undefined_symbol() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule(
            "root".to_string(),
            GrammarRule::NonTerminal {
                name: "root".to_string(),
                productions: vec![Production::new(vec![GrammarSymbol::NonTerminal(
                    "missing".to_string(),
                )])],
            },
        )
        .unwrap();
        assert_eq!(g.compile(), Err(GrammarError::UndefinedSymbol("missing".to_string())));
    }

    #[test]
    fn test_compile_success() {
        let g = simple_seq_grammar();
        assert!(g.is_compiled());
    }

    #[test]
    fn test_compile_duplicate_rule() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule("root".to_string(), GrammarRule::Terminal { pattern: "a".to_string() }).unwrap();
        let err = g
            .add_rule("root".to_string(), GrammarRule::Terminal { pattern: "b".to_string() })
            .unwrap_err();
        assert_eq!(err, GrammarError::DuplicateRule("root".to_string()));
    }

    #[test]
    fn test_compile_terminal_rule() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule("root".to_string(), GrammarRule::Terminal { pattern: "hello".to_string() })
            .unwrap();
        assert!(g.compile().is_ok());
    }

    #[test]
    fn test_compile_multiple_productions() {
        let g = bool_grammar();
        assert!(g.is_compiled());
        if let Some(GrammarRule::NonTerminal { productions, .. }) = g.rules.get("root") {
            assert_eq!(productions.len(), 2);
        } else {
            panic!("expected non-terminal rule");
        }
    }

    #[test]
    fn test_compiled_flag() {
        let mut g = Grammar::new("root".to_string());
        assert!(!g.is_compiled());
        g.add_rule("root".to_string(), GrammarRule::Terminal { pattern: "x".to_string() }).unwrap();
        assert!(!g.is_compiled());
        g.compile().unwrap();
        assert!(g.is_compiled());
    }

    // ── Symbol and Production ──────────────────────────────────

    #[test]
    fn test_terminal_symbol_eq() {
        let a = GrammarSymbol::Terminal("x".to_string());
        let b = GrammarSymbol::Terminal("x".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn test_nonterminal_symbol_eq() {
        let a = GrammarSymbol::NonTerminal("A".to_string());
        let b = GrammarSymbol::NonTerminal("A".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn test_symbol_not_eq() {
        let t = GrammarSymbol::Terminal("x".to_string());
        let n = GrammarSymbol::NonTerminal("x".to_string());
        assert_ne!(t, n);
    }

    #[test]
    fn test_production_new() {
        let p = Production::new(vec![GrammarSymbol::Terminal("a".to_string())]);
        assert_eq!(p.symbols.len(), 1);
    }

    #[test]
    fn test_production_empty_check() {
        assert!(Production::new(vec![]).is_empty());
        assert!(!Production::new(vec![GrammarSymbol::Terminal("a".to_string())]).is_empty());
    }

    #[test]
    fn test_rule_name_nonterminal() {
        let r = GrammarRule::NonTerminal { name: "foo".to_string(), productions: vec![] };
        assert_eq!(r.name(), "foo");
    }

    #[test]
    fn test_rule_name_terminal() {
        let r = GrammarRule::Terminal { pattern: "bar".to_string() };
        assert_eq!(r.name(), "bar");
    }

    // ── FIRST set ──────────────────────────────────────────────

    #[test]
    fn test_first_set_terminal() {
        let g = simple_seq_grammar();
        let first = g.first_set(&GrammarSymbol::Terminal("hello".to_string()));
        assert_eq!(first.len(), 1);
        assert!(first.contains("hello"));
    }

    #[test]
    fn test_first_set_nonterminal() {
        let g = simple_seq_grammar();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("hello"));
    }

    #[test]
    fn test_first_set_alternatives() {
        let g = bool_grammar();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("true"));
        assert!(first.contains("false"));
        assert_eq!(first.len(), 2);
    }

    #[test]
    fn test_first_set_recursive_stops() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule(
            "root".to_string(),
            GrammarRule::NonTerminal {
                name: "root".to_string(),
                productions: vec![
                    Production::new(vec![
                        GrammarSymbol::NonTerminal("root".to_string()),
                        GrammarSymbol::Terminal("x".to_string()),
                    ]),
                    Production::new(vec![GrammarSymbol::Terminal("y".to_string())]),
                ],
            },
        )
        .unwrap();
        g.compile().unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("y"));
    }

    // ── Grammar state ──────────────────────────────────────────

    #[test]
    fn test_state_new() {
        let s = GrammarState::new("root");
        assert_eq!(s.position, 0);
        assert!(!s.completed);
        assert_eq!(s.stack.len(), 1);
    }

    #[test]
    fn test_state_not_complete_initially() {
        let s = GrammarState::new("root");
        assert!(!s.is_complete());
    }

    #[test]
    fn test_state_peek() {
        let s = GrammarState::new("root");
        assert_eq!(s.peek(), Some(&GrammarSymbol::NonTerminal("root".to_string())));
    }

    #[test]
    fn test_state_advance_simple() {
        let g = bool_grammar();
        let mut s = GrammarState::new("root");
        s.advance("true", &g).unwrap();
        assert!(s.is_complete());
        assert_eq!(s.position, 1);
    }

    #[test]
    fn test_state_advance_seq() {
        let g = simple_seq_grammar();
        let mut s = GrammarState::new("root");
        s.advance("hello", &g).unwrap();
        assert!(!s.is_complete());
        s.advance("world", &g).unwrap();
        assert!(s.is_complete());
        assert_eq!(s.position, 2);
    }

    #[test]
    fn test_state_advance_not_compiled() {
        let g = Grammar::new("root".to_string());
        let mut s = GrammarState::new("root");
        assert_eq!(s.advance("x", &g), Err(GrammarError::NotCompiled));
    }

    #[test]
    fn test_state_completion() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule(
            "root".to_string(),
            GrammarRule::NonTerminal {
                name: "root".to_string(),
                productions: vec![Production::new(vec![GrammarSymbol::Terminal("a".to_string())])],
            },
        )
        .unwrap();
        g.compile().unwrap();

        let mut s = GrammarState::new("root");
        s.advance("a", &g).unwrap();
        assert!(s.is_complete());
    }

    // ── Constrainer ────────────────────────────────────────────

    #[test]
    fn test_constrainer_needs_compiled() {
        let g = Grammar::new("root".to_string());
        let result = GrammarConstrainer::new(g, make_vocab(&[]));
        assert_eq!(result.unwrap_err(), GrammarError::NotCompiled);
    }

    #[test]
    fn test_constrainer_allowed_basic() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false", "other"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        let state = GrammarState::new("root");
        let allowed = c.allowed_tokens(&state);
        assert_eq!(allowed.len(), 2);
        assert!(allowed.contains(&0));
        assert!(allowed.contains(&1));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_constrainer_mask_invalid() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false", "other"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        let state = GrammarState::new("root");
        let mut logits = vec![1.0, 2.0, 3.0];
        c.apply_constraint(&mut logits, &state);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_constrainer_complete_state() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        let state = GrammarState { stack: vec![], position: 1, completed: true };
        assert!(c.allowed_tokens(&state).is_empty());
    }

    #[test]
    fn test_constrainer_no_match() {
        let g = bool_grammar();
        let vocab = make_vocab(&["x", "y", "z"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        let state = GrammarState::new("root");
        assert!(c.allowed_tokens(&state).is_empty());
    }

    #[test]
    fn test_constrainer_vocab_access() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        assert_eq!(c.vocab().len(), 2);
        assert_eq!(c.vocab()[0], "true");
    }

    #[test]
    fn test_constrainer_grammar_access() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        assert_eq!(c.grammar().start_symbol, "root");
    }

    // ── JSON schema grammar ────────────────────────────────────

    #[test]
    fn test_json_string_type() {
        let schema = json!({"type": "string"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        assert!(g.is_compiled());
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("\""));
    }

    #[test]
    fn test_json_number_type() {
        let schema = json!({"type": "number"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("number"));
        assert!(first.contains("-"));
    }

    #[test]
    fn test_json_integer_type() {
        let schema = json!({"type": "integer"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("number"));
    }

    #[test]
    fn test_json_boolean_type() {
        let schema = json!({"type": "boolean"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("true"));
        assert!(first.contains("false"));
    }

    #[test]
    fn test_json_null_type() {
        let schema = json!({"type": "null"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("null"));
        assert_eq!(first.len(), 1);
    }

    #[test]
    fn test_json_empty_object() {
        let schema = json!({"type": "object"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("{"));
    }

    #[test]
    fn test_json_object_with_props() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        assert!(g.rules.contains_key("root"));
        assert!(g.rules.contains_key("root_prop_name"));
    }

    #[test]
    fn test_json_array_basic() {
        let schema = json!({"type": "array"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("["));
    }

    #[test]
    fn test_json_array_with_items() {
        let schema = json!({
            "type": "array",
            "items": {"type": "number"}
        });
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        assert!(g.rules.contains_key("root_item"));
        assert!(g.rules.contains_key("root_elements"));
    }

    #[test]
    fn test_json_enum_strings() {
        let schema = json!({"enum": ["red", "green", "blue"]});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("\"red\""));
        assert!(first.contains("\"green\""));
        assert!(first.contains("\"blue\""));
    }

    #[test]
    fn test_json_enum_mixed() {
        let schema = json!({"enum": ["a", 1, true]});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("\"a\""));
        assert!(first.contains("1"));
        assert!(first.contains("true"));
    }

    #[test]
    fn test_json_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "val": {"type": "number"}
                    }
                }
            }
        });
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        assert!(g.rules.contains_key("root_prop_inner_prop_val"));
    }

    #[test]
    fn test_json_invalid_type() {
        let schema = json!({"type": "custom"});
        let err = JsonSchemaGrammar::from_schema(&schema).unwrap_err();
        assert!(matches!(err, GrammarError::InvalidSchema(_)));
    }

    #[test]
    fn test_json_empty_enum() {
        let schema = json!({"enum": []});
        let err = JsonSchemaGrammar::from_schema(&schema).unwrap_err();
        assert!(matches!(err, GrammarError::InvalidSchema(_)));
    }

    #[test]
    fn test_json_boolean_first_set() {
        let schema = json!({"type": "boolean"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert_eq!(first.len(), 2);
    }

    // ── Regex grammar ──────────────────────────────────────────

    #[test]
    fn test_regex_single_char() {
        let g = RegexGrammar::from_pattern("a").unwrap();
        assert!(g.is_compiled());
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a"));
    }

    #[test]
    fn test_regex_literal_seq() {
        let g = RegexGrammar::from_pattern("ab").unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a"));
        assert!(!first.contains("b"));
    }

    #[test]
    fn test_regex_alternation() {
        let g = RegexGrammar::from_pattern("a|b").unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a"));
        assert!(first.contains("b"));
    }

    #[test]
    fn test_regex_char_class() {
        let g = RegexGrammar::from_pattern("[abc]").unwrap();
        assert!(g.rules.contains_key("char_class_0"));
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a"));
        assert!(first.contains("b"));
        assert!(first.contains("c"));
    }

    #[test]
    fn test_regex_star() {
        let g = RegexGrammar::from_pattern("a*").unwrap();
        assert!(g.rules.contains_key("rep_0"));
    }

    #[test]
    fn test_regex_plus() {
        let g = RegexGrammar::from_pattern("a+").unwrap();
        assert!(g.rules.contains_key("rep_0"));
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a"));
    }

    #[test]
    fn test_regex_optional() {
        let g = RegexGrammar::from_pattern("a?").unwrap();
        assert!(g.rules.contains_key("rep_0"));
    }

    #[test]
    fn test_regex_grouping() {
        let g = RegexGrammar::from_pattern("(ab)").unwrap();
        assert!(g.rules.contains_key("group_0"));
    }

    #[test]
    fn test_regex_escaped() {
        let g = RegexGrammar::from_pattern("a\\|b").unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a"));
        // "|" should be a literal, not alternation
        assert!(!first.contains("b"));
    }

    #[test]
    fn test_regex_dot() {
        let g = RegexGrammar::from_pattern(".").unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("."));
    }

    #[test]
    fn test_regex_empty_fails() {
        assert_eq!(
            RegexGrammar::from_pattern(""),
            Err(GrammarError::InvalidRegex("empty pattern".to_string()))
        );
    }

    #[test]
    fn test_regex_unclosed_bracket() {
        assert!(matches!(RegexGrammar::from_pattern("[abc"), Err(GrammarError::InvalidRegex(_))));
    }

    #[test]
    fn test_regex_unclosed_paren() {
        assert!(matches!(RegexGrammar::from_pattern("(abc"), Err(GrammarError::InvalidRegex(_))));
    }

    #[test]
    fn test_regex_complex() {
        let g = RegexGrammar::from_pattern("a[bc]+").unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a"));
    }

    // ── Guided decoder ─────────────────────────────────────────

    #[test]
    fn test_decoder_new() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false"]);
        let d = GuidedDecoder::new(g, vocab).unwrap();
        assert!(!d.is_complete());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_decoder_constrain() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false", "null"]);
        let mut d = GuidedDecoder::new(g, vocab).unwrap();
        let mut logits = vec![1.0, 2.0, 3.0];
        let allowed = d.constrain_logits(&mut logits);
        assert!(allowed.contains(&0));
        assert!(allowed.contains(&1));
        assert!(!allowed.contains(&2));
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_decoder_advance() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false"]);
        let mut d = GuidedDecoder::new(g, vocab).unwrap();
        d.advance(0).unwrap();
        assert!(d.is_complete());
    }

    #[test]
    fn test_decoder_complete() {
        let g = simple_seq_grammar();
        let vocab = make_vocab(&["hello", "world", "x"]);
        let mut d = GuidedDecoder::new(g, vocab).unwrap();
        assert!(!d.is_complete());
        d.advance(0).unwrap(); // "hello"
        assert!(!d.is_complete());
        d.advance(1).unwrap(); // "world"
        assert!(d.is_complete());
    }

    #[test]
    fn test_decoder_reset() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false"]);
        let mut d = GuidedDecoder::new(g, vocab).unwrap();
        d.advance(0).unwrap();
        assert!(d.is_complete());
        d.reset();
        assert!(!d.is_complete());
    }

    #[test]
    fn test_decoder_metrics_update() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false", "x"]);
        let mut d = GuidedDecoder::new(g, vocab).unwrap();
        let mut logits = vec![1.0, 2.0, 3.0];
        d.constrain_logits(&mut logits);
        assert_eq!(d.metrics().grammar_evaluations, 1);
        assert_eq!(d.metrics().steps, 1);
        assert_eq!(d.metrics().tokens_constrained, 1);
        assert_eq!(d.metrics().tokens_allowed_total, 2);
    }

    // ── Grammar cache ──────────────────────────────────────────

    #[test]
    fn test_cache_new_empty() {
        let c = GrammarCache::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_cache_miss_stores() {
        let mut c = GrammarCache::new();
        let schema = json!({"type": "boolean"});
        c.get_or_compile(&schema).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c.misses(), 1);
        assert_eq!(c.hits(), 0);
    }

    #[test]
    fn test_cache_hit() {
        let mut c = GrammarCache::new();
        let schema = json!({"type": "boolean"});
        c.get_or_compile(&schema).unwrap();
        c.get_or_compile(&schema).unwrap();
        assert_eq!(c.hits(), 1);
        assert_eq!(c.misses(), 1);
    }

    #[test]
    fn test_cache_different_schemas() {
        let mut c = GrammarCache::new();
        c.get_or_compile(&json!({"type": "boolean"})).unwrap();
        c.get_or_compile(&json!({"type": "null"})).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.misses(), 2);
    }

    #[test]
    fn test_cache_clear() {
        let mut c = GrammarCache::new();
        c.get_or_compile(&json!({"type": "boolean"})).unwrap();
        assert_eq!(c.len(), 1);
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn test_cache_default() {
        let c = GrammarCache::default();
        assert!(c.is_empty());
        assert_eq!(c.hits(), 0);
    }

    // ── Metrics ────────────────────────────────────────────────

    #[test]
    fn test_metrics_new_zeros() {
        let m = GuidedGenerationMetrics::new();
        assert_eq!(m.tokens_constrained, 0);
        assert_eq!(m.tokens_allowed_total, 0);
        assert_eq!(m.grammar_evaluations, 0);
        assert_eq!(m.cache_hits, 0);
        assert_eq!(m.steps, 0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_metrics_avg_zero_steps() {
        let m = GuidedGenerationMetrics::new();
        assert_eq!(m.tokens_allowed_avg(), 0.0);
    }

    #[test]
    fn test_metrics_avg_computed() {
        let mut m = GuidedGenerationMetrics::new();
        m.tokens_allowed_total = 100;
        m.steps = 10;
        assert!((m.tokens_allowed_avg() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_reset() {
        let mut m = GuidedGenerationMetrics::new();
        m.tokens_constrained = 42;
        m.steps = 7;
        m.reset();
        assert_eq!(m.tokens_constrained, 0);
        assert_eq!(m.steps, 0);
    }

    #[test]
    fn test_metrics_default() {
        let m = GuidedGenerationMetrics::default();
        assert_eq!(m.steps, 0);
    }

    // ── Edge cases ─────────────────────────────────────────────

    #[test]
    fn test_single_terminal_grammar() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule("root".to_string(), GrammarRule::Terminal { pattern: "only".to_string() })
            .unwrap();
        g.compile().unwrap();

        let vocab = make_vocab(&["only", "other"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        let state = GrammarState::new("root");
        let allowed = c.allowed_tokens(&state);
        assert_eq!(allowed, vec![0]);
    }

    #[test]
    fn test_grammar_error_display() {
        let e = GrammarError::EmptyGrammar;
        assert_eq!(e.to_string(), "grammar has no rules");

        let e = GrammarError::UndefinedSymbol("X".into());
        assert_eq!(e.to_string(), "undefined symbol: X");

        let e = GrammarError::NotCompiled;
        assert_eq!(e.to_string(), "grammar not compiled");

        let e = GrammarError::MissingStartSymbol("s".into());
        assert_eq!(e.to_string(), "missing start symbol: s");

        let e = GrammarError::DuplicateRule("r".into());
        assert_eq!(e.to_string(), "duplicate rule: r");

        let e = GrammarError::InvalidSchema("bad".into());
        assert_eq!(e.to_string(), "invalid schema: bad");

        let e = GrammarError::InvalidRegex("re".into());
        assert_eq!(e.to_string(), "invalid regex: re");
    }

    #[test]
    fn test_constrained_json_generation() {
        let schema = json!({"type": "boolean"});
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let vocab = make_vocab(&["true", "false", "null", "{", "}"]);
        let mut d = GuidedDecoder::new(g, vocab).unwrap();

        let mut logits = vec![1.0; 5];
        let allowed = d.constrain_logits(&mut logits);
        assert!(allowed.contains(&0)); // "true"
        assert!(allowed.contains(&1)); // "false"
        assert!(!allowed.contains(&2)); // "null"

        d.advance(0).unwrap(); // pick "true"
        assert!(d.is_complete());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_large_vocab_masking() {
        let g = bool_grammar();
        let mut vocab: Vec<String> = (0..1000).map(|i| format!("tok_{i}")).collect();
        vocab[42] = "true".to_string();
        vocab[777] = "false".to_string();

        let c = GrammarConstrainer::new(g, vocab).unwrap();
        let state = GrammarState::new("root");

        let mut logits = vec![1.0_f32; 1000];
        c.apply_constraint(&mut logits, &state);

        assert_eq!(logits[42], 1.0);
        assert_eq!(logits[777], 1.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[999], f32::NEG_INFINITY);

        let finite = logits.iter().filter(|l| l.is_finite()).count();
        assert_eq!(finite, 2);
    }

    #[test]
    fn test_production_first_set() {
        let g = simple_seq_grammar();
        let prod = Production::new(vec![
            GrammarSymbol::Terminal("hello".to_string()),
            GrammarSymbol::Terminal("world".to_string()),
        ]);
        let first = g.production_first_set(&prod);
        assert!(first.contains("hello"));
        assert!(!first.contains("world"));
    }

    #[test]
    fn test_epsilon_first_set() {
        let mut g = Grammar::new("root".to_string());
        g.add_rule(
            "maybe".to_string(),
            GrammarRule::NonTerminal {
                name: "maybe".to_string(),
                productions: vec![
                    Production::new(vec![]),
                    Production::new(vec![GrammarSymbol::Terminal("x".to_string())]),
                ],
            },
        )
        .unwrap();
        g.add_rule(
            "root".to_string(),
            GrammarRule::NonTerminal {
                name: "root".to_string(),
                productions: vec![Production::new(vec![
                    GrammarSymbol::NonTerminal("maybe".to_string()),
                    GrammarSymbol::Terminal("y".to_string()),
                ])],
            },
        )
        .unwrap();
        g.compile().unwrap();

        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("x"));
        assert!(first.contains("y"));
    }

    #[test]
    fn test_decoder_state_access() {
        let g = bool_grammar();
        let vocab = make_vocab(&["true", "false"]);
        let d = GuidedDecoder::new(g, vocab).unwrap();
        assert_eq!(d.state().position, 0);
    }

    #[test]
    fn test_constrainer_with_sequence() {
        let g = simple_seq_grammar();
        let vocab = make_vocab(&["hello", "world", "x"]);
        let c = GrammarConstrainer::new(g, vocab).unwrap();
        let state = GrammarState::new("root");

        let allowed = c.allowed_tokens(&state);
        assert!(allowed.contains(&0)); // "hello"
        assert!(!allowed.contains(&1)); // "world" not yet
    }

    #[test]
    fn test_json_object_constrained_generation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"}
            }
        });
        let g = JsonSchemaGrammar::from_schema(&schema).unwrap();
        let vocab = make_vocab(&["{", "}", "\"ok\"", ":", "true", "false", ","]);
        let mut d = GuidedDecoder::new(g, vocab).unwrap();

        // Step 1: must be "{"
        let mut logits = vec![1.0; 7];
        d.constrain_logits(&mut logits);
        assert!(logits[0].is_finite()); // "{"
        assert!(!logits[1].is_finite()); // "}" masked
        d.advance(0).unwrap(); // "{"

        // Step 2: must be "\"ok\""
        let mut logits = vec![1.0; 7];
        d.constrain_logits(&mut logits);
        assert!(logits[2].is_finite()); // "\"ok\""
        d.advance(2).unwrap();

        // Step 3: must be ":"
        let mut logits = vec![1.0; 7];
        d.constrain_logits(&mut logits);
        assert!(logits[3].is_finite()); // ":"
        d.advance(3).unwrap();

        // Step 4: must be "true" or "false"
        let mut logits = vec![1.0; 7];
        let allowed = d.constrain_logits(&mut logits);
        assert!(allowed.contains(&4) || allowed.contains(&5));
        d.advance(4).unwrap(); // "true"

        // Step 5: must be "}"
        let mut logits = vec![1.0; 7];
        d.constrain_logits(&mut logits);
        assert!(logits[1].is_finite()); // "}"
        d.advance(1).unwrap();

        assert!(d.is_complete());
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_grammar_clone() {
        let g = bool_grammar();
        let g2 = g.clone();
        assert_eq!(g2.start_symbol, "root");
        assert!(g2.is_compiled());
    }

    #[test]
    #[allow(clippy::redundant_clone)]
    fn test_grammar_state_clone() {
        let s = GrammarState::new("root");
        let s2 = s.clone();
        assert_eq!(s2.position, 0);
        assert!(!s2.completed);
    }

    #[test]
    fn test_regex_with_nested_group() {
        let g = RegexGrammar::from_pattern("(a|b)c").unwrap();
        let first = g.first_set(&GrammarSymbol::NonTerminal("root".to_string()));
        assert!(first.contains("a") || first.contains("b"));
    }
}
