//! WASM-compatible OpenCL kernel validation shim.
//!
//! OpenCL kernel **source code** can be validated (parsed, syntax-checked)
//! even without OpenCL hardware.  This module provides:
//!
//! 1. [`parse_kernel_signatures`] — extract function signatures from kernel source
//! 2. [`MockOpenClContext`] — a mock context for testing kernel argument setup
//! 3. [`KernelSignature`] / [`KernelArg`] — parsed kernel metadata
//!
//! Everything here is pure Rust with zero FFI, so it compiles on `wasm32`.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// OpenCL address-space qualifier for a kernel argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgQualifier {
    Global,
    Local,
    Private,
    Constant,
}

impl std::fmt::Display for ArgQualifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgQualifier::Global => write!(f, "__global"),
            ArgQualifier::Local => write!(f, "__local"),
            ArgQualifier::Private => write!(f, "__private"),
            ArgQualifier::Constant => write!(f, "__constant"),
        }
    }
}

/// A single kernel function argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArg {
    pub name: String,
    pub qualifier: ArgQualifier,
    pub type_name: String,
    pub is_pointer: bool,
}

/// A parsed OpenCL kernel function signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSignature {
    pub name: String,
    pub args: Vec<KernelArg>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse an OpenCL kernel source string and extract all `__kernel` function
/// signatures.
///
/// This is a lightweight heuristic parser — not a full C99 front-end. It
/// handles the common patterns emitted by bitnet-rs kernel generators.
pub fn parse_kernel_signatures(source: &str) -> Vec<KernelSignature> {
    let mut signatures = Vec::new();

    // Strip C-style block comments
    let stripped = strip_block_comments(source);
    // Strip line comments
    let stripped = strip_line_comments(&stripped);

    // Find `__kernel void name(...)` patterns
    let kernel_prefix = "__kernel";
    let mut search_from = 0;

    while let Some(kw_pos) = stripped[search_from..].find(kernel_prefix) {
        let abs_pos = search_from + kw_pos;
        let after_keyword = &stripped[abs_pos + kernel_prefix.len()..];

        if let Some(sig) = try_parse_kernel(after_keyword) {
            signatures.push(sig);
        }

        search_from = abs_pos + kernel_prefix.len();
    }

    signatures
}

/// Try to parse a kernel signature from text following the `__kernel` keyword.
fn try_parse_kernel(text: &str) -> Option<KernelSignature> {
    let text = text.trim_start();

    // Expect a return type (usually `void`)
    let (_, rest) = split_first_word(text)?;
    let rest = rest.trim_start();

    // Kernel function name
    let (name, rest) = split_first_word(rest)?;
    let rest = rest.trim_start();

    // Opening paren
    if !rest.starts_with('(') {
        return None;
    }
    let paren_end = find_matching_paren(rest)?;
    let args_str = &rest[1..paren_end];

    let args = parse_args(args_str);

    Some(KernelSignature { name: name.to_string(), args })
}

/// Parse a comma-separated argument list.
fn parse_args(args_str: &str) -> Vec<KernelArg> {
    if args_str.trim().is_empty() {
        return Vec::new();
    }

    args_str.split(',').filter_map(|arg| parse_single_arg(arg.trim())).collect()
}

/// Parse a single kernel argument like `__global float* input`.
fn parse_single_arg(arg: &str) -> Option<KernelArg> {
    if arg.is_empty() {
        return None;
    }

    let tokens: Vec<&str> = arg.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }

    let mut idx = 0;

    // Parse qualifier
    let qualifier = match tokens.get(idx).copied() {
        Some("__global" | "global") => {
            idx += 1;
            ArgQualifier::Global
        }
        Some("__local" | "local") => {
            idx += 1;
            ArgQualifier::Local
        }
        Some("__constant" | "constant") => {
            idx += 1;
            ArgQualifier::Constant
        }
        Some("__private" | "private") => {
            idx += 1;
            ArgQualifier::Private
        }
        _ => ArgQualifier::Private,
    };

    // Remaining tokens form `type_name [*] name` or `type_name* name` etc.
    let remaining: Vec<&str> = tokens[idx..].to_vec();
    if remaining.is_empty() {
        return None;
    }

    // Rebuild the type+name string to handle `float*`, `float *`, `float * name`
    let joined = remaining.join(" ");
    let is_pointer = joined.contains('*');
    let cleaned = joined.replace('*', " ");
    let parts: Vec<&str> = cleaned.split_whitespace().collect();

    if parts.is_empty() {
        return None;
    }

    let (type_name, name) = if parts.len() == 1 {
        // Only type, no name — synthesize one
        (parts[0].to_string(), format!("arg{}", 0))
    } else {
        // Last token is the name, everything before is the type
        let name = parts[parts.len() - 1];
        let type_parts = &parts[..parts.len() - 1];
        (type_parts.join(" "), name.to_string())
    };

    Some(KernelArg { name, qualifier, type_name, is_pointer })
}

fn split_first_word(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    let end = s.find(|c: char| c.is_whitespace() || c == '(').unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((&s[..end], &s[end..]))
}

fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_block_comments(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            // Find closing */
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i += 2; // skip */
            result.push(' ');
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    result
}

fn strip_line_comments(s: &str) -> String {
    s.lines()
        .map(|line| if let Some(pos) = line.find("//") { &line[..pos] } else { line })
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

/// Errors detected during kernel source validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelValidationError {
    /// No kernel functions found in source.
    NoKernelsFound,
    /// A kernel has no arguments (likely a mistake).
    EmptyArgList { kernel_name: String },
    /// Argument count mismatch between expected and parsed.
    ArgCountMismatch { kernel_name: String, expected: usize, actual: usize },
    /// Duplicate kernel names in source.
    DuplicateKernelName { name: String },
}

impl std::fmt::Display for KernelValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelValidationError::NoKernelsFound => {
                write!(f, "no __kernel functions found in source")
            }
            KernelValidationError::EmptyArgList { kernel_name } => {
                write!(f, "kernel \'{kernel_name}\' has no arguments")
            }
            KernelValidationError::ArgCountMismatch { kernel_name, expected, actual } => {
                write!(f, "kernel \'{kernel_name}\': expected {expected} args, found {actual}")
            }
            KernelValidationError::DuplicateKernelName { name } => {
                write!(f, "duplicate kernel name: \'{name}\'")
            }
        }
    }
}

impl std::error::Error for KernelValidationError {}

/// Validate kernel source: checks for common issues.
pub fn validate_kernel_source(source: &str) -> Result<Vec<KernelSignature>, KernelValidationError> {
    let sigs = parse_kernel_signatures(source);
    if sigs.is_empty() {
        return Err(KernelValidationError::NoKernelsFound);
    }

    // Check for duplicates
    let mut seen = HashMap::new();
    for sig in &sigs {
        if let Some(_prev) = seen.insert(&sig.name, ()) {
            return Err(KernelValidationError::DuplicateKernelName { name: sig.name.clone() });
        }
    }

    Ok(sigs)
}

// ---------------------------------------------------------------------------
// MockOpenClContext
// ---------------------------------------------------------------------------

/// Mock OpenCL context for testing kernel argument setup without FFI.
///
/// Programs are "compiled" by parsing their kernel source, and kernels can
/// have arguments set for later assertion.
pub struct MockOpenClContext {
    programs: HashMap<String, Vec<KernelSignature>>,
    kernel_args: HashMap<String, Vec<Option<MockArgValue>>>,
}

/// A mock kernel argument value for testing.
#[derive(Debug, Clone, PartialEq)]
pub enum MockArgValue {
    Buffer { size: usize },
    Scalar(f64),
    Int(i64),
    LocalMem { size: usize },
}

/// Errors from mock context operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockError {
    CompileError(String),
    KernelNotFound(String),
    ArgIndexOutOfRange { kernel: String, index: usize, max: usize },
    ProgramNotFound(String),
}

impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MockError::CompileError(msg) => write!(f, "compile error: {msg}"),
            MockError::KernelNotFound(name) => write!(f, "kernel not found: \'{name}\'"),
            MockError::ArgIndexOutOfRange { kernel, index, max } => {
                write!(f, "kernel \'{kernel}\': arg index {index} out of range (max {max})")
            }
            MockError::ProgramNotFound(name) => write!(f, "program not found: \'{name}\'"),
        }
    }
}

impl std::error::Error for MockError {}

impl MockOpenClContext {
    /// Create a new empty mock context.
    pub fn new() -> Self {
        Self { programs: HashMap::new(), kernel_args: HashMap::new() }
    }

    /// "Compile" an OpenCL program from source, parsing kernel signatures.
    pub fn compile_program(&mut self, program_name: &str, source: &str) -> Result<(), MockError> {
        let sigs =
            validate_kernel_source(source).map_err(|e| MockError::CompileError(e.to_string()))?;

        for sig in &sigs {
            let n_args = sig.args.len();
            self.kernel_args.insert(sig.name.clone(), vec![None; n_args]);
        }

        self.programs.insert(program_name.to_string(), sigs);
        Ok(())
    }

    /// List all kernel names in a compiled program.
    pub fn kernel_names(&self, program_name: &str) -> Result<Vec<String>, MockError> {
        let sigs = self
            .programs
            .get(program_name)
            .ok_or_else(|| MockError::ProgramNotFound(program_name.to_string()))?;
        Ok(sigs.iter().map(|s| s.name.clone()).collect())
    }

    /// Get the parsed signature for a kernel.
    pub fn kernel_signature(&self, kernel_name: &str) -> Result<&KernelSignature, MockError> {
        for sigs in self.programs.values() {
            if let Some(sig) = sigs.iter().find(|s| s.name == kernel_name) {
                return Ok(sig);
            }
        }
        Err(MockError::KernelNotFound(kernel_name.to_string()))
    }

    /// Set a kernel argument (mock — just records the value).
    pub fn set_kernel_arg(
        &mut self,
        kernel_name: &str,
        index: usize,
        value: MockArgValue,
    ) -> Result<(), MockError> {
        let args = self
            .kernel_args
            .get_mut(kernel_name)
            .ok_or_else(|| MockError::KernelNotFound(kernel_name.to_string()))?;

        if index >= args.len() {
            return Err(MockError::ArgIndexOutOfRange {
                kernel: kernel_name.to_string(),
                index,
                max: args.len().saturating_sub(1),
            });
        }

        args[index] = Some(value);
        Ok(())
    }

    /// Check whether all arguments for a kernel have been set.
    pub fn all_args_set(&self, kernel_name: &str) -> Result<bool, MockError> {
        let args = self
            .kernel_args
            .get(kernel_name)
            .ok_or_else(|| MockError::KernelNotFound(kernel_name.to_string()))?;
        Ok(args.iter().all(|a| a.is_some()))
    }

    /// Get the number of programs loaded.
    pub fn program_count(&self) -> usize {
        self.programs.len()
    }
}

impl Default for MockOpenClContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compile-time validation (const-friendly where possible)
// ---------------------------------------------------------------------------

/// Validate that a kernel source string contains at least one `__kernel`
/// function. Suitable for build-script or `const` context validation.
pub fn source_contains_kernel(source: &str) -> bool {
    source.contains("__kernel")
}

// ---------------------------------------------------------------------------
// Real OpenCL FFI — only on non-WASM targets
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    //! Placeholder for native OpenCL FFI bindings.
    //!
    //! This module would contain the real `cl_context`, `cl_program`, etc.
    //! wrappers.  It is excluded from `wasm32` builds.

    /// Marker that native OpenCL is available on this target.
    pub const NATIVE_OPENCL_AVAILABLE: bool = true;
}

#[cfg(target_arch = "wasm32")]
pub mod native {
    //! Stub for wasm32 — native OpenCL is not available.
    pub const NATIVE_OPENCL_AVAILABLE: bool = false;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_KERNEL: &str = "
        __kernel void vector_add(
            __global const float* a,
            __global const float* b,
            __global float* result,
            const int n
        ) {
            int gid = get_global_id(0);
            if (gid < n) {
                result[gid] = a[gid] + b[gid];
            }
        }
    ";

    const MULTI_KERNEL: &str = "
        __kernel void mat_mul(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int M,
            const int N,
            const int K
        ) {
        }

        __kernel void relu_activate(
            __global float* data,
            const int size
        ) {
        }
    ";

    // --- Parsing tests ---

    #[test]
    fn parse_single_kernel() {
        let sigs = parse_kernel_signatures(SAMPLE_KERNEL);
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].name, "vector_add");
        assert_eq!(sigs[0].args.len(), 4);
    }

    #[test]
    fn parse_kernel_arg_qualifiers() {
        let sigs = parse_kernel_signatures(SAMPLE_KERNEL);
        let args = &sigs[0].args;

        assert_eq!(args[0].qualifier, ArgQualifier::Global);
        assert_eq!(args[0].type_name, "const float");
        assert!(args[0].is_pointer);
        assert_eq!(args[0].name, "a");

        assert_eq!(args[3].qualifier, ArgQualifier::Private);
        assert_eq!(args[3].type_name, "const int");
        assert!(!args[3].is_pointer);
        assert_eq!(args[3].name, "n");
    }

    #[test]
    fn parse_multiple_kernels() {
        let sigs = parse_kernel_signatures(MULTI_KERNEL);
        assert_eq!(sigs.len(), 2);
        assert_eq!(sigs[0].name, "mat_mul");
        assert_eq!(sigs[0].args.len(), 6);
        assert_eq!(sigs[1].name, "relu_activate");
        assert_eq!(sigs[1].args.len(), 2);
    }

    #[test]
    fn parse_empty_source() {
        let sigs = parse_kernel_signatures("");
        assert!(sigs.is_empty());
    }

    #[test]
    fn parse_source_without_kernels() {
        let source = "void helper_function(float* data) { }";
        let sigs = parse_kernel_signatures(source);
        assert!(sigs.is_empty());
    }

    #[test]
    fn parse_kernel_with_block_comments() {
        let source = "
            /* This is a block comment */
            __kernel void add(
                __global float* a,
                __global float* b
            ) { }
        ";
        let sigs = parse_kernel_signatures(source);
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].name, "add");
        assert_eq!(sigs[0].args.len(), 2);
    }

    #[test]
    fn parse_kernel_local_arg() {
        let source = "
            __kernel void reduce(
                __global float* input,
                __local float* scratch,
                const int n
            ) { }
        ";
        let sigs = parse_kernel_signatures(source);
        assert_eq!(sigs[0].args[1].qualifier, ArgQualifier::Local);
        assert!(sigs[0].args[1].is_pointer);
    }

    #[test]
    fn parse_kernel_constant_arg() {
        let source = "
            __kernel void lookup(
                __constant float* table,
                __global float* output
            ) { }
        ";
        let sigs = parse_kernel_signatures(source);
        assert_eq!(sigs[0].args[0].qualifier, ArgQualifier::Constant);
    }

    #[test]
    fn parse_no_args_kernel() {
        let source = "__kernel void empty_kernel() { }";
        let sigs = parse_kernel_signatures(source);
        assert_eq!(sigs.len(), 1);
        assert_eq!(sigs[0].name, "empty_kernel");
        assert!(sigs[0].args.is_empty());
    }

    // --- Validation tests ---

    #[test]
    fn validate_valid_source() {
        let result = validate_kernel_source(SAMPLE_KERNEL);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_empty_source_fails() {
        let result = validate_kernel_source("");
        assert_eq!(result, Err(KernelValidationError::NoKernelsFound));
    }

    #[test]
    fn validate_duplicate_kernel_names() {
        let source = "
            __kernel void dup(__global float* a) { }
            __kernel void dup(__global float* b) { }
        ";
        let result = validate_kernel_source(source);
        assert!(matches!(result, Err(KernelValidationError::DuplicateKernelName { .. })));
    }

    // --- MockOpenClContext tests ---

    #[test]
    fn mock_context_compile_and_list_kernels() {
        let mut ctx = MockOpenClContext::new();
        ctx.compile_program("prog", MULTI_KERNEL).unwrap();

        let names = ctx.kernel_names("prog").unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"mat_mul".to_string()));
        assert!(names.contains(&"relu_activate".to_string()));
    }

    #[test]
    fn mock_context_compile_invalid_source() {
        let mut ctx = MockOpenClContext::new();
        let result = ctx.compile_program("bad", "no kernels here");
        assert!(matches!(result, Err(MockError::CompileError(_))));
    }

    #[test]
    fn mock_context_kernel_signature() {
        let mut ctx = MockOpenClContext::new();
        ctx.compile_program("prog", SAMPLE_KERNEL).unwrap();

        let sig = ctx.kernel_signature("vector_add").unwrap();
        assert_eq!(sig.args.len(), 4);
        assert_eq!(sig.args[0].name, "a");
    }

    #[test]
    fn mock_context_kernel_not_found() {
        let mut ctx = MockOpenClContext::new();
        ctx.compile_program("prog", SAMPLE_KERNEL).unwrap();

        let result = ctx.kernel_signature("nonexistent");
        assert!(matches!(result, Err(MockError::KernelNotFound(_))));
    }

    #[test]
    fn mock_context_set_kernel_args() {
        let mut ctx = MockOpenClContext::new();
        ctx.compile_program("prog", SAMPLE_KERNEL).unwrap();

        assert!(!ctx.all_args_set("vector_add").unwrap());

        ctx.set_kernel_arg("vector_add", 0, MockArgValue::Buffer { size: 1024 }).unwrap();
        ctx.set_kernel_arg("vector_add", 1, MockArgValue::Buffer { size: 1024 }).unwrap();
        ctx.set_kernel_arg("vector_add", 2, MockArgValue::Buffer { size: 1024 }).unwrap();
        ctx.set_kernel_arg("vector_add", 3, MockArgValue::Int(256)).unwrap();

        assert!(ctx.all_args_set("vector_add").unwrap());
    }

    #[test]
    fn mock_context_arg_out_of_range() {
        let mut ctx = MockOpenClContext::new();
        ctx.compile_program("prog", SAMPLE_KERNEL).unwrap();

        let result = ctx.set_kernel_arg("vector_add", 99, MockArgValue::Int(0));
        assert!(matches!(result, Err(MockError::ArgIndexOutOfRange { .. })));
    }

    #[test]
    fn mock_context_program_not_found() {
        let ctx = MockOpenClContext::new();
        let result = ctx.kernel_names("nonexistent");
        assert!(matches!(result, Err(MockError::ProgramNotFound(_))));
    }

    #[test]
    fn mock_context_multiple_programs() {
        let mut ctx = MockOpenClContext::new();
        ctx.compile_program("prog1", SAMPLE_KERNEL).unwrap();
        ctx.compile_program("prog2", MULTI_KERNEL).unwrap();
        assert_eq!(ctx.program_count(), 2);
    }

    #[test]
    fn mock_context_default_trait() {
        let ctx = MockOpenClContext::default();
        assert_eq!(ctx.program_count(), 0);
    }

    // --- source_contains_kernel ---

    #[test]
    fn source_contains_kernel_positive() {
        assert!(source_contains_kernel("__kernel void foo() {}"));
    }

    #[test]
    fn source_contains_kernel_negative() {
        assert!(!source_contains_kernel("void foo() {}"));
    }

    // --- ArgQualifier display ---

    #[test]
    fn arg_qualifier_display() {
        assert_eq!(ArgQualifier::Global.to_string(), "__global");
        assert_eq!(ArgQualifier::Local.to_string(), "__local");
        assert_eq!(ArgQualifier::Private.to_string(), "__private");
        assert_eq!(ArgQualifier::Constant.to_string(), "__constant");
    }

    // --- Error display ---

    #[test]
    fn validation_error_display() {
        let err = KernelValidationError::NoKernelsFound;
        assert!(err.to_string().contains("no __kernel"));

        let err = KernelValidationError::ArgCountMismatch {
            kernel_name: "foo".into(),
            expected: 3,
            actual: 2,
        };
        assert!(err.to_string().contains("expected 3"));
    }

    #[test]
    fn mock_error_display() {
        let err = MockError::KernelNotFound("foo".into());
        assert!(err.to_string().contains("foo"));

        let err = MockError::ArgIndexOutOfRange { kernel: "bar".into(), index: 5, max: 3 };
        assert!(err.to_string().contains("5"));
    }
}
