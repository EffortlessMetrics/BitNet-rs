//! SPIR-V compiler with optimization passes and caching.
//!
//! Provides a CPU-reference implementation of SPIR-V compilation, validation,
//! optimization and caching for the `bitnet-gpu-hal` abstraction layer.
//!
//! # Overview
//!
//! The pipeline translates `OpenCL` C or GLSL compute shader source into
//! SPIR-V binary, validates the result, applies optimisation passes and
//! stores the output in a content-addressed cache keyed by source hash.
//!
//! All heavy lifting is simulated on the CPU so that the module can be
//! tested without a real GPU driver or external toolchain.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

// ── SPIR-V constants ────────────────────────────────────────────────

/// SPIR-V magic number (little-endian).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

// ── Target environment ──────────────────────────────────────────────

/// `Vulkan` version targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VulkanVersion {
    V1_0,
    V1_1,
    V1_2,
}

/// `OpenCL` version targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenCLVersion {
    V1_2,
    V2_0,
    V3_0,
}

/// Target environment for SPIR-V generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetEnvironment {
    Vulkan(VulkanVersion),
    OpenCL(OpenCLVersion),
}

impl fmt::Display for TargetEnvironment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(v) => write!(f, "Vulkan {v:?}"),
            Self::OpenCL(v) => write!(f, "OpenCL {v:?}"),
        }
    }
}

// ── Optimisation level ──────────────────────────────────────────────

/// Optimisation level applied during SPIR-V compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationLevel {
    None,
    Size,
    Performance,
}

// ── SPIRVConfig ─────────────────────────────────────────────────────

/// Configuration for a SPIR-V compilation run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SPIRVConfig {
    pub target: TargetEnvironment,
    pub optimization: OptimizationLevel,
    pub debug_info: bool,
}

impl SPIRVConfig {
    #[must_use]
    pub const fn new(target: TargetEnvironment, optimization: OptimizationLevel) -> Self {
        Self { target, optimization, debug_info: false }
    }

    #[must_use]
    pub const fn with_debug(mut self) -> Self {
        self.debug_info = true;
        self
    }
}

impl Default for SPIRVConfig {
    fn default() -> Self {
        Self::new(TargetEnvironment::Vulkan(VulkanVersion::V1_2), OptimizationLevel::Performance)
    }
}

// ── SPIRVHeader ─────────────────────────────────────────────────────

/// Parsed SPIR-V module header (first 5 words).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SPIRVHeader {
    pub magic: u32,
    pub version: u32,
    pub generator: u32,
    pub bound: u32,
    pub instruction_schema: u32,
}

impl SPIRVHeader {
    /// Size of the header in 32-bit words.
    pub const WORD_COUNT: usize = 5;

    #[must_use]
    pub const fn new(version: u32, generator: u32, bound: u32) -> Self {
        Self { magic: SPIRV_MAGIC, version, generator, bound, instruction_schema: 0 }
    }

    /// Encode header into a word stream.
    #[must_use]
    pub fn encode(&self) -> Vec<u32> {
        vec![self.magic, self.version, self.generator, self.bound, self.instruction_schema]
    }

    /// Decode header from a word stream. Returns `None` when the slice
    /// is too short or the magic number does not match.
    #[must_use]
    pub fn decode(words: &[u32]) -> Option<Self> {
        if words.len() < Self::WORD_COUNT {
            return None;
        }
        if words[0] != SPIRV_MAGIC {
            return None;
        }
        Some(Self {
            magic: words[0],
            version: words[1],
            generator: words[2],
            bound: words[3],
            instruction_schema: words[4],
        })
    }
}

// ── SPIRVInstruction ────────────────────────────────────────────────

/// Kind of operand carried by an instruction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperandKind {
    /// SSA type id.
    TypeId(u32),
    /// SSA result id.
    ResultId(u32),
    /// Literal 32-bit constant.
    Literal(u32),
    /// Reference to another SSA id.
    IdRef(u32),
}

/// A single SPIR-V instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SPIRVInstruction {
    pub opcode: u16,
    pub word_count: u16,
    pub operands: Vec<OperandKind>,
}

impl SPIRVInstruction {
    #[must_use]
    pub fn new(opcode: u16, operands: Vec<OperandKind>) -> Self {
        let word_count = 1u16.saturating_add(u16::try_from(operands.len()).unwrap_or(u16::MAX));
        Self { opcode, word_count, operands }
    }

    /// Encode instruction into a word stream.
    #[must_use]
    pub fn encode(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.word_count as usize);
        words.push((u32::from(self.word_count) << 16) | u32::from(self.opcode));
        for op in &self.operands {
            words.push(match op {
                OperandKind::TypeId(v)
                | OperandKind::ResultId(v)
                | OperandKind::Literal(v)
                | OperandKind::IdRef(v) => *v,
            });
        }
        words
    }

    /// Decode a single instruction from `words`. Returns
    /// `(instruction, words_consumed)` or `None` on failure.
    #[must_use]
    pub fn decode(words: &[u32]) -> Option<(Self, usize)> {
        if words.is_empty() {
            return None;
        }
        let first = words[0];
        let word_count = (first >> 16) as u16;
        let opcode = (first & 0xFFFF) as u16;
        if (word_count as usize) > words.len() || word_count == 0 {
            return None;
        }
        let operands =
            words[1..word_count as usize].iter().map(|&w| OperandKind::Literal(w)).collect();
        Some((Self { opcode, word_count, operands }, word_count as usize))
    }

    /// Returns the result id carried by this instruction, if any.
    #[must_use]
    pub fn result_id(&self) -> Option<u32> {
        self.operands
            .iter()
            .find_map(|op| if let OperandKind::ResultId(id) = op { Some(*id) } else { None })
    }
}

// ── SPIRVModule ─────────────────────────────────────────────────────

/// A parsed SPIR-V binary module.
#[derive(Debug, Clone)]
pub struct SPIRVModule {
    pub header: SPIRVHeader,
    pub instructions: Vec<SPIRVInstruction>,
}

impl SPIRVModule {
    #[must_use]
    pub const fn new(header: SPIRVHeader, instructions: Vec<SPIRVInstruction>) -> Self {
        Self { header, instructions }
    }

    /// Total size in 32-bit words.
    #[must_use]
    pub fn word_count(&self) -> usize {
        SPIRVHeader::WORD_COUNT
            + self.instructions.iter().map(|i| i.word_count as usize).sum::<usize>()
    }

    /// Encode the entire module into a `Vec<u32>`.
    #[must_use]
    pub fn encode(&self) -> Vec<u32> {
        let mut words = self.header.encode();
        for inst in &self.instructions {
            words.extend(inst.encode());
        }
        words
    }

    /// Decode a module from a word stream.
    pub fn decode(words: &[u32]) -> Result<Self, SPIRVError> {
        let header = SPIRVHeader::decode(words).ok_or(SPIRVError::InvalidHeader)?;
        let mut offset = SPIRVHeader::WORD_COUNT;
        let mut instructions = Vec::new();
        while offset < words.len() {
            let (inst, consumed) = SPIRVInstruction::decode(&words[offset..])
                .ok_or(SPIRVError::MalformedInstruction { offset })?;
            instructions.push(inst);
            offset += consumed;
        }
        Ok(Self { header, instructions })
    }

    /// Encode the module into a byte vector (little-endian).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.encode().iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Decode a module from a little-endian byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SPIRVError> {
        if !bytes.len().is_multiple_of(4) {
            return Err(SPIRVError::InvalidBinaryLength { length: bytes.len() });
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Self::decode(&words)
    }
}

// ── Errors ──────────────────────────────────────────────────────────

/// Errors produced by the SPIR-V pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SPIRVError {
    InvalidHeader,
    MalformedInstruction { offset: usize },
    InvalidBinaryLength { length: usize },
    ValidationFailed { reason: String },
    CompilationFailed { reason: String },
    UnsupportedSource { reason: String },
    EmptySource,
}

impl fmt::Display for SPIRVError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeader => write!(f, "invalid SPIR-V header"),
            Self::MalformedInstruction { offset } => {
                write!(f, "malformed instruction at word offset {offset}")
            }
            Self::InvalidBinaryLength { length } => {
                write!(f, "binary length {length} is not a multiple of 4")
            }
            Self::ValidationFailed { reason } => {
                write!(f, "validation failed: {reason}")
            }
            Self::CompilationFailed { reason } => {
                write!(f, "compilation failed: {reason}")
            }
            Self::UnsupportedSource { reason } => {
                write!(f, "unsupported source: {reason}")
            }
            Self::EmptySource => write!(f, "empty source string"),
        }
    }
}

impl std::error::Error for SPIRVError {}

// ── SPIRVValidator ──────────────────────────────────────────────────

/// Validates structural and semantic properties of a [`SPIRVModule`].
pub struct SPIRVValidator {
    max_bound: u32,
    require_entry_point: bool,
}

/// Opcodes used during validation / optimisation.
mod opcodes {
    pub const OP_NOP: u16 = 0;
    pub const OP_ENTRY_POINT: u16 = 15;
    pub const OP_CONSTANT: u16 = 43;
    pub const OP_FUNCTION: u16 = 54;
    pub const OP_FUNCTION_END: u16 = 56;
    pub const OP_FUNCTION_CALL: u16 = 57;
    pub const OP_CAPABILITY: u16 = 17;
    pub const OP_MEMORY_MODEL: u16 = 14;
    pub const OP_TYPE_VOID: u16 = 19;
    pub const OP_TYPE_INT: u16 = 21;
    pub const OP_TYPE_FLOAT: u16 = 22;
    pub const OP_TYPE_VECTOR: u16 = 23;
    pub const OP_TYPE_FUNCTION: u16 = 33;
    pub const OP_VARIABLE: u16 = 59;
    pub const OP_STORE: u16 = 62;
    pub const OP_LOAD: u16 = 61;
    pub const OP_RETURN: u16 = 253;
    pub const OP_RETURN_VALUE: u16 = 254;
    pub const OP_LABEL: u16 = 248;
}

impl SPIRVValidator {
    #[must_use]
    pub const fn new() -> Self {
        Self { max_bound: 1 << 22, require_entry_point: false }
    }

    #[must_use]
    pub const fn with_max_bound(mut self, bound: u32) -> Self {
        self.max_bound = bound;
        self
    }

    #[must_use]
    pub const fn require_entry_point(mut self) -> Self {
        self.require_entry_point = true;
        self
    }

    /// Run all validation passes on `module`.
    pub fn validate(&self, module: &SPIRVModule) -> Result<(), SPIRVError> {
        self.validate_header(module)?;
        Self::validate_instructions(module)?;
        if self.require_entry_point {
            Self::validate_entry_point(module)?;
        }
        Ok(())
    }

    fn validate_header(&self, module: &SPIRVModule) -> Result<(), SPIRVError> {
        if module.header.magic != SPIRV_MAGIC {
            return Err(SPIRVError::ValidationFailed {
                reason: format!(
                    "bad magic: expected {SPIRV_MAGIC:#010x}, got {:#010x}",
                    module.header.magic
                ),
            });
        }
        if module.header.bound > self.max_bound {
            return Err(SPIRVError::ValidationFailed {
                reason: format!("bound {} exceeds maximum {}", module.header.bound, self.max_bound),
            });
        }
        if module.header.instruction_schema != 0 {
            return Err(SPIRVError::ValidationFailed {
                reason: "instruction_schema must be 0".into(),
            });
        }
        Ok(())
    }

    fn validate_instructions(module: &SPIRVModule) -> Result<(), SPIRVError> {
        for (i, inst) in module.instructions.iter().enumerate() {
            if inst.word_count == 0 {
                return Err(SPIRVError::ValidationFailed {
                    reason: format!("instruction {i} has zero word count"),
                });
            }
            let expected =
                1u16.saturating_add(u16::try_from(inst.operands.len()).unwrap_or(u16::MAX));
            if inst.word_count != expected {
                return Err(SPIRVError::ValidationFailed {
                    reason: format!(
                        "instruction {i}: word_count {} != expected {expected}",
                        inst.word_count,
                    ),
                });
            }
        }
        Ok(())
    }

    fn validate_entry_point(module: &SPIRVModule) -> Result<(), SPIRVError> {
        let has_ep = module.instructions.iter().any(|i| i.opcode == opcodes::OP_ENTRY_POINT);
        if !has_ep {
            return Err(SPIRVError::ValidationFailed { reason: "no OpEntryPoint found".into() });
        }
        Ok(())
    }
}

impl Default for SPIRVValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Optimisation passes ─────────────────────────────────────────────

/// Available optimisation passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationPass {
    DeadCodeElimination,
    ConstantFolding,
    InlineExpansion,
}

impl fmt::Display for OptimizationPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeadCodeElimination => {
                write!(f, "dead-code-elimination")
            }
            Self::ConstantFolding => write!(f, "constant-folding"),
            Self::InlineExpansion => write!(f, "inline-expansion"),
        }
    }
}

/// Statistics emitted by a single optimisation pass.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    pub instructions_removed: usize,
    pub instructions_replaced: usize,
    pub functions_inlined: usize,
}

/// Applies optimisation passes to a [`SPIRVModule`].
pub struct SPIRVOptimizer {
    passes: Vec<OptimizationPass>,
}

impl SPIRVOptimizer {
    #[must_use]
    pub const fn new() -> Self {
        Self { passes: Vec::new() }
    }

    #[must_use]
    pub fn with_passes(mut self, passes: &[OptimizationPass]) -> Self {
        self.passes.extend_from_slice(passes);
        self
    }

    /// Build the default pass pipeline for a given [`OptimizationLevel`].
    #[must_use]
    pub fn for_level(level: OptimizationLevel) -> Self {
        let passes = match level {
            OptimizationLevel::None => vec![],
            OptimizationLevel::Size => {
                vec![OptimizationPass::DeadCodeElimination, OptimizationPass::ConstantFolding]
            }
            OptimizationLevel::Performance => {
                vec![
                    OptimizationPass::DeadCodeElimination,
                    OptimizationPass::ConstantFolding,
                    OptimizationPass::InlineExpansion,
                ]
            }
        };
        Self { passes }
    }

    /// Apply all configured passes sequentially and return aggregate
    /// stats.
    pub fn optimize(&self, module: &mut SPIRVModule) -> Vec<(OptimizationPass, PassStats)> {
        let mut results = Vec::with_capacity(self.passes.len());
        for &pass in &self.passes {
            let stats = match pass {
                OptimizationPass::DeadCodeElimination => Self::dead_code_elimination(module),
                OptimizationPass::ConstantFolding => Self::constant_folding(module),
                OptimizationPass::InlineExpansion => Self::inline_expansion(module),
            };
            results.push((pass, stats));
        }
        results
    }

    /// Remove `OpNop` and unreferenced instructions.
    fn dead_code_elimination(module: &mut SPIRVModule) -> PassStats {
        let before = module.instructions.len();
        let referenced: std::collections::HashSet<u32> = module
            .instructions
            .iter()
            .flat_map(|inst| inst.operands.iter())
            .filter_map(|op| if let OperandKind::IdRef(id) = op { Some(*id) } else { None })
            .collect();
        module.instructions.retain(|inst| {
            if inst.opcode == opcodes::OP_NOP {
                return false;
            }
            inst.result_id().is_none_or(|rid| {
                let is_structural = matches!(
                    inst.opcode,
                    opcodes::OP_ENTRY_POINT
                        | opcodes::OP_FUNCTION
                        | opcodes::OP_FUNCTION_END
                        | opcodes::OP_CAPABILITY
                        | opcodes::OP_MEMORY_MODEL
                        | opcodes::OP_TYPE_VOID
                        | opcodes::OP_TYPE_INT
                        | opcodes::OP_TYPE_FLOAT
                        | opcodes::OP_TYPE_VECTOR
                        | opcodes::OP_TYPE_FUNCTION
                        | opcodes::OP_LABEL
                        | opcodes::OP_RETURN
                        | opcodes::OP_RETURN_VALUE
                        | opcodes::OP_VARIABLE
                        | opcodes::OP_STORE
                        | opcodes::OP_LOAD
                );
                is_structural || referenced.contains(&rid)
            })
        });
        let removed = before.saturating_sub(module.instructions.len());
        PassStats { instructions_removed: removed, ..Default::default() }
    }

    /// Fold constant expressions (simulated — replaces duplicate
    /// `OpConstant` instructions carrying identical type + value).
    fn constant_folding(module: &SPIRVModule) -> PassStats {
        let mut seen: HashMap<(u16, Vec<OperandKind>), usize> = HashMap::new();
        let mut replaced = 0usize;
        for (i, inst) in module.instructions.iter().enumerate() {
            if inst.opcode == opcodes::OP_CONSTANT {
                let key = (inst.opcode, inst.operands.clone());
                if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
                    e.insert(i);
                } else {
                    replaced += 1;
                }
            }
        }
        PassStats { instructions_replaced: replaced, ..Default::default() }
    }

    /// Inline single-call functions (simulated — counts eligible
    /// candidates).
    fn inline_expansion(module: &SPIRVModule) -> PassStats {
        let call_count: HashMap<u32, usize> = module
            .instructions
            .iter()
            .filter(|i| i.opcode == opcodes::OP_FUNCTION_CALL)
            .filter_map(|i| {
                i.operands
                    .iter()
                    .find_map(|op| if let OperandKind::IdRef(id) = op { Some(*id) } else { None })
            })
            .fold(HashMap::new(), |mut map, id| {
                *map.entry(id).or_insert(0) += 1;
                map
            });
        let inlined = call_count.values().filter(|&&c| c == 1).count();
        PassStats { functions_inlined: inlined, ..Default::default() }
    }
}

impl Default for SPIRVOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Source language helpers ──────────────────────────────────────────

/// Source language tag embedded in compilation output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceLanguage {
    OpenCLC,
    GLSL,
}

// ── OpenCLToSPIRV ───────────────────────────────────────────────────

/// Simulated translator from `OpenCL` C source to SPIR-V.
pub struct OpenCLToSPIRV {
    target: OpenCLVersion,
}

impl OpenCLToSPIRV {
    #[must_use]
    pub const fn new(target: OpenCLVersion) -> Self {
        Self { target }
    }

    /// Translate `source` into a [`SPIRVModule`].
    ///
    /// This is a *simulated* translation: the resulting module contains
    /// a synthetic instruction stream derived from hashing the source.
    pub fn translate(&self, source: &str) -> Result<SPIRVModule, SPIRVError> {
        if source.is_empty() {
            return Err(SPIRVError::EmptySource);
        }
        if !source.contains("__kernel") && !source.contains("kernel ") {
            return Err(SPIRVError::UnsupportedSource {
                reason: "OpenCL source must contain a kernel function".into(),
            });
        }
        let version = match self.target {
            OpenCLVersion::V1_2 => 0x0001_0200,
            OpenCLVersion::V2_0 => 0x0002_0000,
            OpenCLVersion::V3_0 => 0x0003_0000,
        };
        let hash = simple_hash(source);
        let bound = (hash % 1000) + 10;
        let header = SPIRVHeader::new(version, 0xBEEF_0001, bound);
        let instructions = Self::synthesize_instructions(hash);
        Ok(SPIRVModule::new(header, instructions))
    }

    fn synthesize_instructions(hash: u32) -> Vec<SPIRVInstruction> {
        vec![
            SPIRVInstruction::new(opcodes::OP_CAPABILITY, vec![OperandKind::Literal(0)]),
            SPIRVInstruction::new(
                opcodes::OP_MEMORY_MODEL,
                vec![OperandKind::Literal(0), OperandKind::Literal(1)],
            ),
            SPIRVInstruction::new(
                opcodes::OP_ENTRY_POINT,
                vec![
                    OperandKind::Literal(6), // GLCompute
                    OperandKind::ResultId(1),
                    OperandKind::Literal(hash),
                ],
            ),
            SPIRVInstruction::new(opcodes::OP_TYPE_VOID, vec![OperandKind::ResultId(2)]),
            SPIRVInstruction::new(
                opcodes::OP_TYPE_FUNCTION,
                vec![OperandKind::ResultId(3), OperandKind::IdRef(2)],
            ),
            SPIRVInstruction::new(
                opcodes::OP_FUNCTION,
                vec![
                    OperandKind::TypeId(2),
                    OperandKind::ResultId(4),
                    OperandKind::Literal(0),
                    OperandKind::IdRef(3),
                ],
            ),
            SPIRVInstruction::new(opcodes::OP_LABEL, vec![OperandKind::ResultId(5)]),
            SPIRVInstruction::new(opcodes::OP_RETURN, vec![]),
            SPIRVInstruction::new(opcodes::OP_FUNCTION_END, vec![]),
        ]
    }
}

// ── GLSLToSPIRV ─────────────────────────────────────────────────────

/// Simulated translator from GLSL compute shaders to SPIR-V.
pub struct GLSLToSPIRV {
    target: VulkanVersion,
}

impl GLSLToSPIRV {
    #[must_use]
    pub const fn new(target: VulkanVersion) -> Self {
        Self { target }
    }

    /// Translate `source` into a [`SPIRVModule`].
    pub fn translate(&self, source: &str) -> Result<SPIRVModule, SPIRVError> {
        if source.is_empty() {
            return Err(SPIRVError::EmptySource);
        }
        if !source.contains("#version") {
            return Err(SPIRVError::UnsupportedSource {
                reason: "GLSL source must contain a #version directive".into(),
            });
        }
        if !source.contains("void main") {
            return Err(SPIRVError::UnsupportedSource {
                reason: "GLSL source must contain void main()".into(),
            });
        }
        let version = match self.target {
            VulkanVersion::V1_0 => 0x0001_0000,
            VulkanVersion::V1_1 => 0x0001_0100,
            VulkanVersion::V1_2 => 0x0001_0200,
        };
        let hash = simple_hash(source);
        let bound = (hash % 1000) + 10;
        let header = SPIRVHeader::new(version, 0xBEEF_0002, bound);
        let instructions = Self::synthesize_instructions(hash);
        Ok(SPIRVModule::new(header, instructions))
    }

    fn synthesize_instructions(hash: u32) -> Vec<SPIRVInstruction> {
        vec![
            SPIRVInstruction::new(opcodes::OP_CAPABILITY, vec![OperandKind::Literal(1)]),
            SPIRVInstruction::new(
                opcodes::OP_MEMORY_MODEL,
                vec![OperandKind::Literal(1), OperandKind::Literal(0)],
            ),
            SPIRVInstruction::new(
                opcodes::OP_ENTRY_POINT,
                vec![
                    OperandKind::Literal(5), // GLCompute
                    OperandKind::ResultId(1),
                    OperandKind::Literal(hash),
                ],
            ),
            SPIRVInstruction::new(opcodes::OP_TYPE_VOID, vec![OperandKind::ResultId(2)]),
            SPIRVInstruction::new(
                opcodes::OP_TYPE_INT,
                vec![OperandKind::ResultId(3), OperandKind::Literal(32), OperandKind::Literal(1)],
            ),
            SPIRVInstruction::new(
                opcodes::OP_TYPE_FUNCTION,
                vec![OperandKind::ResultId(4), OperandKind::IdRef(2)],
            ),
            SPIRVInstruction::new(
                opcodes::OP_FUNCTION,
                vec![
                    OperandKind::TypeId(2),
                    OperandKind::ResultId(5),
                    OperandKind::Literal(0),
                    OperandKind::IdRef(4),
                ],
            ),
            SPIRVInstruction::new(opcodes::OP_LABEL, vec![OperandKind::ResultId(6)]),
            SPIRVInstruction::new(opcodes::OP_RETURN, vec![]),
            SPIRVInstruction::new(opcodes::OP_FUNCTION_END, vec![]),
        ]
    }
}

// ── SPIRVCache ──────────────────────────────────────────────────────

/// Content-addressed cache for compiled SPIR-V modules.
///
/// Keyed by a 64-bit hash of the source string so that identical
/// compilations are not repeated.
#[derive(Debug)]
pub struct SPIRVCache {
    entries: Arc<Mutex<HashMap<u64, CacheEntry>>>,
    max_entries: usize,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    module: SPIRVModule,
    #[allow(dead_code)]
    source_lang: SourceLanguage,
    hits: u64,
}

impl SPIRVCache {
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self { entries: Arc::new(Mutex::new(HashMap::new())), max_entries }
    }

    /// Look up a cached module by source hash.
    pub fn get(&self, source: &str) -> Option<SPIRVModule> {
        let key = hash64(source);
        let mut map = self.entries.lock().unwrap();
        if let Some(entry) = map.get_mut(&key) {
            entry.hits += 1;
            Some(entry.module.clone())
        } else {
            None
        }
    }

    /// Insert a compiled module into the cache. Evicts the
    /// least-hit entry when the cache is full.
    pub fn insert(&self, source: &str, module: SPIRVModule, lang: SourceLanguage) {
        let key = hash64(source);
        let mut map = self.entries.lock().unwrap();
        if map.len() >= self.max_entries
            && !map.contains_key(&key)
            && let Some((&evict_key, _)) = map.iter().min_by_key(|(_, e)| e.hits)
        {
            map.remove(&evict_key);
        }
        map.insert(key, CacheEntry { module, source_lang: lang, hits: 0 });
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// Hit count for a given source string (0 if absent).
    #[must_use]
    pub fn hits(&self, source: &str) -> u64 {
        let key = hash64(source);
        self.entries.lock().unwrap().get(&key).map_or(0, |e| e.hits)
    }
}

impl Default for SPIRVCache {
    fn default() -> Self {
        Self::new(256)
    }
}

// ── SPIRVCompiler (orchestrator) ────────────────────────────────────

/// Result of a successful compilation.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    pub module: SPIRVModule,
    pub source_language: SourceLanguage,
    pub optimization_stats: Vec<(OptimizationPass, PassStats)>,
    pub cached: bool,
}

/// Full SPIR-V compilation pipeline.
///
/// ```text
/// source → translate → validate → optimise → cache → result
/// ```
pub struct SPIRVCompiler {
    config: SPIRVConfig,
    cache: SPIRVCache,
    validator: SPIRVValidator,
    optimizer: SPIRVOptimizer,
    compilations: u64,
}

impl SPIRVCompiler {
    #[must_use]
    pub fn new(config: SPIRVConfig) -> Self {
        let optimizer = SPIRVOptimizer::for_level(config.optimization);
        Self {
            config,
            cache: SPIRVCache::default(),
            validator: SPIRVValidator::new(),
            optimizer,
            compilations: 0,
        }
    }

    #[must_use]
    pub fn with_cache(mut self, cache: SPIRVCache) -> Self {
        self.cache = cache;
        self
    }

    #[must_use]
    pub const fn with_validator(mut self, validator: SPIRVValidator) -> Self {
        self.validator = validator;
        self
    }

    #[must_use]
    pub fn with_optimizer(mut self, optimizer: SPIRVOptimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Compile `OpenCL` C source through the full pipeline.
    pub fn compile_opencl(&mut self, source: &str) -> Result<CompilationResult, SPIRVError> {
        if let Some(module) = self.cache.get(source) {
            return Ok(CompilationResult {
                module,
                source_language: SourceLanguage::OpenCLC,
                optimization_stats: vec![],
                cached: true,
            });
        }
        let cl_version = match self.config.target {
            TargetEnvironment::OpenCL(v) => v,
            TargetEnvironment::Vulkan(_) => OpenCLVersion::V1_2,
        };
        let translator = OpenCLToSPIRV::new(cl_version);
        let mut module = translator.translate(source)?;
        self.validator.validate(&module)?;
        let stats = self.optimizer.optimize(&mut module);
        self.cache.insert(source, module.clone(), SourceLanguage::OpenCLC);
        self.compilations += 1;
        Ok(CompilationResult {
            module,
            source_language: SourceLanguage::OpenCLC,
            optimization_stats: stats,
            cached: false,
        })
    }

    /// Compile GLSL compute-shader source through the full pipeline.
    pub fn compile_glsl(&mut self, source: &str) -> Result<CompilationResult, SPIRVError> {
        if let Some(module) = self.cache.get(source) {
            return Ok(CompilationResult {
                module,
                source_language: SourceLanguage::GLSL,
                optimization_stats: vec![],
                cached: true,
            });
        }
        let vk_version = match self.config.target {
            TargetEnvironment::Vulkan(v) => v,
            TargetEnvironment::OpenCL(_) => VulkanVersion::V1_2,
        };
        let translator = GLSLToSPIRV::new(vk_version);
        let mut module = translator.translate(source)?;
        self.validator.validate(&module)?;
        let stats = self.optimizer.optimize(&mut module);
        self.cache.insert(source, module.clone(), SourceLanguage::GLSL);
        self.compilations += 1;
        Ok(CompilationResult {
            module,
            source_language: SourceLanguage::GLSL,
            optimization_stats: stats,
            cached: false,
        })
    }

    /// Number of non-cached compilations performed so far.
    #[must_use]
    pub const fn compilation_count(&self) -> u64 {
        self.compilations
    }

    /// Reference to the inner cache.
    #[must_use]
    pub const fn cache(&self) -> &SPIRVCache {
        &self.cache
    }

    /// Reference to the active config.
    #[must_use]
    pub const fn config(&self) -> &SPIRVConfig {
        &self.config
    }
}

impl Default for SPIRVCompiler {
    fn default() -> Self {
        Self::new(SPIRVConfig::default())
    }
}

// ── Hashing helpers ─────────────────────────────────────────────────

/// Deterministic 32-bit FNV-1a hash (for synthesising instruction
/// operands).
fn simple_hash(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Deterministic 64-bit FNV-1a hash (for cache keys).
fn hash64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper fixtures ─────────────────────────────────────────

    fn sample_opencl() -> &'static str {
        "__kernel void add(__global float* a, __global float* b) { }"
    }

    fn sample_glsl() -> &'static str {
        "#version 450\nlayout(local_size_x=64) in;\nvoid main() { }"
    }

    fn minimal_module() -> SPIRVModule {
        let header = SPIRVHeader::new(0x0001_0200, 0, 10);
        let instructions = vec![
            SPIRVInstruction::new(opcodes::OP_CAPABILITY, vec![OperandKind::Literal(0)]),
            SPIRVInstruction::new(
                opcodes::OP_MEMORY_MODEL,
                vec![OperandKind::Literal(0), OperandKind::Literal(1)],
            ),
        ];
        SPIRVModule::new(header, instructions)
    }

    fn module_with_entry_point() -> SPIRVModule {
        let header = SPIRVHeader::new(0x0001_0200, 0, 10);
        let instructions = vec![
            SPIRVInstruction::new(opcodes::OP_CAPABILITY, vec![OperandKind::Literal(0)]),
            SPIRVInstruction::new(
                opcodes::OP_MEMORY_MODEL,
                vec![OperandKind::Literal(0), OperandKind::Literal(1)],
            ),
            SPIRVInstruction::new(
                opcodes::OP_ENTRY_POINT,
                vec![OperandKind::Literal(5), OperandKind::ResultId(1), OperandKind::Literal(0)],
            ),
        ];
        SPIRVModule::new(header, instructions)
    }

    // ── SPIRVConfig ─────────────────────────────────────────────

    #[test]
    fn config_default_is_vulkan_1_2_performance() {
        let c = SPIRVConfig::default();
        assert_eq!(c.target, TargetEnvironment::Vulkan(VulkanVersion::V1_2));
        assert_eq!(c.optimization, OptimizationLevel::Performance);
        assert!(!c.debug_info);
    }

    #[test]
    fn config_new_sets_fields() {
        let c = SPIRVConfig::new(
            TargetEnvironment::OpenCL(OpenCLVersion::V2_0),
            OptimizationLevel::Size,
        );
        assert_eq!(c.target, TargetEnvironment::OpenCL(OpenCLVersion::V2_0));
        assert_eq!(c.optimization, OptimizationLevel::Size);
    }

    #[test]
    fn config_with_debug() {
        let c = SPIRVConfig::default().with_debug();
        assert!(c.debug_info);
    }

    #[test]
    fn config_clone_eq() {
        let a = SPIRVConfig::default();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── TargetEnvironment Display ───────────────────────────────

    #[test]
    fn target_env_display_vulkan() {
        let t = TargetEnvironment::Vulkan(VulkanVersion::V1_1);
        assert!(format!("{t}").contains("Vulkan"));
    }

    #[test]
    fn target_env_display_opencl() {
        let t = TargetEnvironment::OpenCL(OpenCLVersion::V3_0);
        assert!(format!("{t}").contains("OpenCL"));
    }

    // ── SPIRVHeader ─────────────────────────────────────────────

    #[test]
    fn header_new_sets_magic() {
        let h = SPIRVHeader::new(1, 2, 3);
        assert_eq!(h.magic, SPIRV_MAGIC);
        assert_eq!(h.version, 1);
        assert_eq!(h.generator, 2);
        assert_eq!(h.bound, 3);
        assert_eq!(h.instruction_schema, 0);
    }

    #[test]
    fn header_encode_decode_roundtrip() {
        let h = SPIRVHeader::new(0x0001_0500, 42, 100);
        let words = h.encode();
        assert_eq!(words.len(), SPIRVHeader::WORD_COUNT);
        let h2 = SPIRVHeader::decode(&words).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn header_decode_too_short() {
        assert!(SPIRVHeader::decode(&[SPIRV_MAGIC, 1]).is_none());
    }

    #[test]
    fn header_decode_bad_magic() {
        assert!(SPIRVHeader::decode(&[0xDEAD, 0, 0, 0, 0]).is_none());
    }

    // ── SPIRVInstruction ────────────────────────────────────────

    #[test]
    fn instruction_new_word_count() {
        let inst = SPIRVInstruction::new(opcodes::OP_NOP, vec![OperandKind::Literal(1)]);
        assert_eq!(inst.word_count, 2);
    }

    #[test]
    fn instruction_encode_decode_roundtrip() {
        let inst = SPIRVInstruction::new(opcodes::OP_CAPABILITY, vec![OperandKind::Literal(5)]);
        let words = inst.encode();
        let (decoded, consumed) = SPIRVInstruction::decode(&words).unwrap();
        assert_eq!(consumed, words.len());
        assert_eq!(decoded.opcode, inst.opcode);
        assert_eq!(decoded.word_count, inst.word_count);
    }

    #[test]
    fn instruction_decode_empty_fails() {
        assert!(SPIRVInstruction::decode(&[]).is_none());
    }

    #[test]
    fn instruction_decode_zero_word_count_fails() {
        assert!(SPIRVInstruction::decode(&[0x0000_0000]).is_none());
    }

    #[test]
    fn instruction_result_id_present() {
        let inst = SPIRVInstruction::new(opcodes::OP_TYPE_VOID, vec![OperandKind::ResultId(42)]);
        assert_eq!(inst.result_id(), Some(42));
    }

    #[test]
    fn instruction_result_id_absent() {
        let inst = SPIRVInstruction::new(opcodes::OP_NOP, vec![]);
        assert_eq!(inst.result_id(), None);
    }

    #[test]
    fn instruction_encode_no_operands() {
        let inst = SPIRVInstruction::new(opcodes::OP_RETURN, vec![]);
        let words = inst.encode();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0] >> 16, 1); // word_count = 1
    }

    #[test]
    fn instruction_multiple_operands() {
        let inst = SPIRVInstruction::new(
            opcodes::OP_FUNCTION,
            vec![
                OperandKind::TypeId(2),
                OperandKind::ResultId(4),
                OperandKind::Literal(0),
                OperandKind::IdRef(3),
            ],
        );
        assert_eq!(inst.word_count, 5);
        assert_eq!(inst.encode().len(), 5);
    }

    // ── SPIRVModule ─────────────────────────────────────────────

    #[test]
    fn module_word_count() {
        let m = minimal_module();
        // header (5) + 2-word inst + 3-word inst = 10
        assert_eq!(m.word_count(), 5 + 2 + 3);
    }

    #[test]
    fn module_encode_decode_roundtrip() {
        let m = minimal_module();
        let words = m.encode();
        let m2 = SPIRVModule::decode(&words).unwrap();
        assert_eq!(m.header, m2.header);
        assert_eq!(m.instructions.len(), m2.instructions.len());
    }

    #[test]
    fn module_to_bytes_from_bytes_roundtrip() {
        let m = minimal_module();
        let bytes = m.to_bytes();
        let m2 = SPIRVModule::from_bytes(&bytes).unwrap();
        assert_eq!(m.header, m2.header);
        assert_eq!(m.instructions.len(), m2.instructions.len());
    }

    #[test]
    fn module_from_bytes_bad_length() {
        let err = SPIRVModule::from_bytes(&[1, 2, 3]).unwrap_err();
        assert!(matches!(err, SPIRVError::InvalidBinaryLength { length: 3 }));
    }

    #[test]
    fn module_decode_bad_magic() {
        let words = vec![0xDEAD_BEEF, 0, 0, 0, 0];
        assert!(matches!(SPIRVModule::decode(&words), Err(SPIRVError::InvalidHeader)));
    }

    #[test]
    fn module_decode_malformed_instruction() {
        let mut words = SPIRVHeader::new(1, 0, 5).encode();
        words.push((99u32 << 16) | 1);
        assert!(matches!(
            SPIRVModule::decode(&words),
            Err(SPIRVError::MalformedInstruction { .. })
        ));
    }

    #[test]
    fn module_empty_instructions() {
        let header = SPIRVHeader::new(1, 0, 1);
        let m = SPIRVModule::new(header, vec![]);
        assert_eq!(m.word_count(), SPIRVHeader::WORD_COUNT);
        let words = m.encode();
        let m2 = SPIRVModule::decode(&words).unwrap();
        assert_eq!(m2.instructions.len(), 0);
    }

    // ── SPIRVError Display ──────────────────────────────────────

    #[test]
    fn error_display_invalid_header() {
        let e = SPIRVError::InvalidHeader;
        assert_eq!(format!("{e}"), "invalid SPIR-V header");
    }

    #[test]
    fn error_display_malformed_instruction() {
        let e = SPIRVError::MalformedInstruction { offset: 7 };
        assert!(format!("{e}").contains('7'));
    }

    #[test]
    fn error_display_validation_failed() {
        let e = SPIRVError::ValidationFailed { reason: "bad".into() };
        assert!(format!("{e}").contains("bad"));
    }

    #[test]
    fn error_display_compilation_failed() {
        let e = SPIRVError::CompilationFailed { reason: "oops".into() };
        assert!(format!("{e}").contains("oops"));
    }

    #[test]
    fn error_display_unsupported_source() {
        let e = SPIRVError::UnsupportedSource { reason: "nope".into() };
        assert!(format!("{e}").contains("nope"));
    }

    #[test]
    fn error_display_empty_source() {
        let e = SPIRVError::EmptySource;
        assert!(format!("{e}").contains("empty"));
    }

    #[test]
    fn error_display_invalid_binary_length() {
        let e = SPIRVError::InvalidBinaryLength { length: 7 };
        assert!(format!("{e}").contains('7'));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(SPIRVError::EmptySource);
        assert!(!e.to_string().is_empty());
    }

    // ── SPIRVValidator ──────────────────────────────────────────

    #[test]
    fn validator_accepts_minimal_module() {
        let v = SPIRVValidator::new();
        assert!(v.validate(&minimal_module()).is_ok());
    }

    #[test]
    fn validator_rejects_bad_magic() {
        let mut m = minimal_module();
        m.header.magic = 0;
        assert!(matches!(
            SPIRVValidator::new().validate(&m),
            Err(SPIRVError::ValidationFailed { .. })
        ));
    }

    #[test]
    fn validator_rejects_high_bound() {
        let mut m = minimal_module();
        m.header.bound = u32::MAX;
        assert!(SPIRVValidator::new().validate(&m).is_err());
    }

    #[test]
    fn validator_with_max_bound() {
        let mut m = minimal_module();
        m.header.bound = 50;
        let v = SPIRVValidator::new().with_max_bound(49);
        assert!(v.validate(&m).is_err());
        let v2 = SPIRVValidator::new().with_max_bound(50);
        assert!(v2.validate(&m).is_ok());
    }

    #[test]
    fn validator_rejects_bad_schema() {
        let mut m = minimal_module();
        m.header.instruction_schema = 1;
        assert!(SPIRVValidator::new().validate(&m).is_err());
    }

    #[test]
    fn validator_rejects_zero_word_count() {
        let mut m = minimal_module();
        m.instructions.push(SPIRVInstruction { opcode: 0, word_count: 0, operands: vec![] });
        assert!(SPIRVValidator::new().validate(&m).is_err());
    }

    #[test]
    fn validator_rejects_mismatched_word_count() {
        let mut m = minimal_module();
        m.instructions.push(SPIRVInstruction {
            opcode: 0,
            word_count: 5,
            operands: vec![OperandKind::Literal(1)],
        });
        assert!(SPIRVValidator::new().validate(&m).is_err());
    }

    #[test]
    fn validator_require_entry_point_missing() {
        let v = SPIRVValidator::new().require_entry_point();
        assert!(v.validate(&minimal_module()).is_err());
    }

    #[test]
    fn validator_require_entry_point_present() {
        let v = SPIRVValidator::new().require_entry_point();
        assert!(v.validate(&module_with_entry_point()).is_ok());
    }

    #[test]
    fn validator_default_does_not_require_entry_point() {
        let v = SPIRVValidator::default();
        assert!(v.validate(&minimal_module()).is_ok());
    }

    // ── OptimizationPass Display ────────────────────────────────

    #[test]
    fn optimization_pass_display() {
        assert_eq!(format!("{}", OptimizationPass::DeadCodeElimination), "dead-code-elimination");
        assert_eq!(format!("{}", OptimizationPass::ConstantFolding), "constant-folding");
        assert_eq!(format!("{}", OptimizationPass::InlineExpansion), "inline-expansion");
    }

    // ── SPIRVOptimizer ──────────────────────────────────────────

    #[test]
    fn optimizer_none_level_no_passes() {
        let o = SPIRVOptimizer::for_level(OptimizationLevel::None);
        let mut m = minimal_module();
        let stats = o.optimize(&mut m);
        assert!(stats.is_empty());
    }

    #[test]
    fn optimizer_size_level_two_passes() {
        let o = SPIRVOptimizer::for_level(OptimizationLevel::Size);
        let mut m = minimal_module();
        let stats = o.optimize(&mut m);
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn optimizer_performance_level_three_passes() {
        let o = SPIRVOptimizer::for_level(OptimizationLevel::Performance);
        let mut m = minimal_module();
        let stats = o.optimize(&mut m);
        assert_eq!(stats.len(), 3);
    }

    #[test]
    fn optimizer_dce_removes_nops() {
        let mut m = minimal_module();
        let before = m.instructions.len();
        m.instructions.push(SPIRVInstruction::new(opcodes::OP_NOP, vec![]));
        m.instructions.push(SPIRVInstruction::new(opcodes::OP_NOP, vec![]));
        let o = SPIRVOptimizer::new().with_passes(&[OptimizationPass::DeadCodeElimination]);
        let stats = o.optimize(&mut m);
        assert_eq!(m.instructions.len(), before);
        assert_eq!(stats[0].1.instructions_removed, 2);
    }

    #[test]
    fn optimizer_constant_folding_counts_duplicates() {
        let mut m = minimal_module();
        let dup = SPIRVInstruction::new(
            opcodes::OP_CONSTANT,
            vec![OperandKind::TypeId(1), OperandKind::ResultId(2), OperandKind::Literal(42)],
        );
        m.instructions.push(dup.clone());
        m.instructions.push(dup);
        let o = SPIRVOptimizer::new().with_passes(&[OptimizationPass::ConstantFolding]);
        let stats = o.optimize(&mut m);
        assert_eq!(stats[0].1.instructions_replaced, 1);
    }

    #[test]
    fn optimizer_inline_expansion_counts_single_calls() {
        let mut m = minimal_module();
        m.instructions.push(SPIRVInstruction::new(
            opcodes::OP_FUNCTION_CALL,
            vec![OperandKind::TypeId(1), OperandKind::ResultId(10), OperandKind::IdRef(99)],
        ));
        let o = SPIRVOptimizer::new().with_passes(&[OptimizationPass::InlineExpansion]);
        let stats = o.optimize(&mut m);
        assert_eq!(stats[0].1.functions_inlined, 1);
    }

    #[test]
    fn optimizer_inline_skips_multi_call() {
        let mut m = minimal_module();
        for rid in [10, 11] {
            m.instructions.push(SPIRVInstruction::new(
                opcodes::OP_FUNCTION_CALL,
                vec![OperandKind::TypeId(1), OperandKind::ResultId(rid), OperandKind::IdRef(99)],
            ));
        }
        let o = SPIRVOptimizer::new().with_passes(&[OptimizationPass::InlineExpansion]);
        let stats = o.optimize(&mut m);
        assert_eq!(stats[0].1.functions_inlined, 0);
    }

    #[test]
    fn optimizer_default_is_empty() {
        let o = SPIRVOptimizer::default();
        let mut m = minimal_module();
        assert!(o.optimize(&mut m).is_empty());
    }

    #[test]
    fn optimizer_with_passes_appends() {
        let o = SPIRVOptimizer::new()
            .with_passes(&[OptimizationPass::DeadCodeElimination])
            .with_passes(&[OptimizationPass::ConstantFolding]);
        let mut m = minimal_module();
        assert_eq!(o.optimize(&mut m).len(), 2);
    }

    // ── OpenCLToSPIRV ───────────────────────────────────────────

    #[test]
    fn opencl_translate_basic() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V1_2);
        let m = t.translate(sample_opencl()).unwrap();
        assert_eq!(m.header.magic, SPIRV_MAGIC);
        assert!(!m.instructions.is_empty());
    }

    #[test]
    fn opencl_translate_empty_fails() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V1_2);
        assert!(matches!(t.translate(""), Err(SPIRVError::EmptySource)));
    }

    #[test]
    fn opencl_translate_no_kernel_fails() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V1_2);
        assert!(matches!(t.translate("void foo() {}"), Err(SPIRVError::UnsupportedSource { .. })));
    }

    #[test]
    fn opencl_v2_sets_version() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V2_0);
        let m = t.translate(sample_opencl()).unwrap();
        assert_eq!(m.header.version, 0x0002_0000);
    }

    #[test]
    fn opencl_v3_sets_version() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V3_0);
        let m = t.translate(sample_opencl()).unwrap();
        assert_eq!(m.header.version, 0x0003_0000);
    }

    #[test]
    fn opencl_deterministic_output() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V1_2);
        let a = t.translate(sample_opencl()).unwrap().encode();
        let b = t.translate(sample_opencl()).unwrap().encode();
        assert_eq!(a, b);
    }

    #[test]
    fn opencl_module_validates() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V1_2);
        let m = t.translate(sample_opencl()).unwrap();
        SPIRVValidator::new().validate(&m).unwrap();
    }

    #[test]
    fn opencl_kernel_keyword_variant() {
        let t = OpenCLToSPIRV::new(OpenCLVersion::V1_2);
        let src = "kernel void k() {}";
        assert!(t.translate(src).is_ok());
    }

    // ── GLSLToSPIRV ────────────────────────────────────────────

    #[test]
    fn glsl_translate_basic() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_2);
        let m = t.translate(sample_glsl()).unwrap();
        assert_eq!(m.header.magic, SPIRV_MAGIC);
        assert!(!m.instructions.is_empty());
    }

    #[test]
    fn glsl_translate_empty_fails() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_2);
        assert!(matches!(t.translate(""), Err(SPIRVError::EmptySource)));
    }

    #[test]
    fn glsl_translate_no_version_fails() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_2);
        assert!(matches!(t.translate("void main() {}"), Err(SPIRVError::UnsupportedSource { .. })));
    }

    #[test]
    fn glsl_translate_no_main_fails() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_2);
        assert!(matches!(
            t.translate("#version 450\nvoid foo() {}"),
            Err(SPIRVError::UnsupportedSource { .. })
        ));
    }

    #[test]
    fn glsl_v1_0_sets_version() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_0);
        let m = t.translate(sample_glsl()).unwrap();
        assert_eq!(m.header.version, 0x0001_0000);
    }

    #[test]
    fn glsl_v1_1_sets_version() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_1);
        let m = t.translate(sample_glsl()).unwrap();
        assert_eq!(m.header.version, 0x0001_0100);
    }

    #[test]
    fn glsl_deterministic_output() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_2);
        let a = t.translate(sample_glsl()).unwrap().encode();
        let b = t.translate(sample_glsl()).unwrap().encode();
        assert_eq!(a, b);
    }

    #[test]
    fn glsl_module_validates() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_2);
        let m = t.translate(sample_glsl()).unwrap();
        SPIRVValidator::new().validate(&m).unwrap();
    }

    #[test]
    fn glsl_contains_type_int() {
        let t = GLSLToSPIRV::new(VulkanVersion::V1_2);
        let m = t.translate(sample_glsl()).unwrap();
        assert!(m.instructions.iter().any(|i| i.opcode == opcodes::OP_TYPE_INT));
    }

    // ── SPIRVCache ──────────────────────────────────────────────

    #[test]
    fn cache_miss_returns_none() {
        let c = SPIRVCache::new(16);
        assert!(c.get("missing").is_none());
    }

    #[test]
    fn cache_insert_and_get() {
        let c = SPIRVCache::new(16);
        let m = minimal_module();
        c.insert("key", m.clone(), SourceLanguage::OpenCLC);
        let m2 = c.get("key").unwrap();
        assert_eq!(m.header, m2.header);
    }

    #[test]
    fn cache_hit_increments_counter() {
        let c = SPIRVCache::new(16);
        c.insert("k", minimal_module(), SourceLanguage::GLSL);
        c.get("k");
        c.get("k");
        assert_eq!(c.hits("k"), 2);
    }

    #[test]
    fn cache_len_and_is_empty() {
        let c = SPIRVCache::new(16);
        assert!(c.is_empty());
        c.insert("a", minimal_module(), SourceLanguage::GLSL);
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());
    }

    #[test]
    fn cache_clear() {
        let c = SPIRVCache::new(16);
        c.insert("a", minimal_module(), SourceLanguage::GLSL);
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn cache_evicts_when_full() {
        let c = SPIRVCache::new(2);
        c.insert("a", minimal_module(), SourceLanguage::GLSL);
        c.insert("b", minimal_module(), SourceLanguage::GLSL);
        // Access "b" so it has more hits than "a".
        c.get("b");
        c.insert("c", minimal_module(), SourceLanguage::GLSL);
        // "a" (least hits) should have been evicted.
        assert_eq!(c.len(), 2);
        assert!(c.get("a").is_none());
    }

    #[test]
    fn cache_no_evict_on_duplicate_key() {
        let c = SPIRVCache::new(1);
        c.insert("a", minimal_module(), SourceLanguage::GLSL);
        c.insert("a", minimal_module(), SourceLanguage::GLSL);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn cache_default_capacity() {
        let c = SPIRVCache::default();
        assert!(c.is_empty());
    }

    #[test]
    fn cache_hits_missing_key() {
        let c = SPIRVCache::new(16);
        assert_eq!(c.hits("nope"), 0);
    }

    // ── SPIRVCompiler ───────────────────────────────────────────

    #[test]
    fn compiler_default_config() {
        let c = SPIRVCompiler::default();
        assert_eq!(c.config().target, TargetEnvironment::Vulkan(VulkanVersion::V1_2));
    }

    #[test]
    fn compiler_compile_opencl() {
        let mut c = SPIRVCompiler::default();
        let r = c.compile_opencl(sample_opencl()).unwrap();
        assert_eq!(r.source_language, SourceLanguage::OpenCLC);
        assert!(!r.cached);
        assert_eq!(c.compilation_count(), 1);
    }

    #[test]
    fn compiler_compile_glsl() {
        let mut c = SPIRVCompiler::default();
        let r = c.compile_glsl(sample_glsl()).unwrap();
        assert_eq!(r.source_language, SourceLanguage::GLSL);
        assert!(!r.cached);
    }

    #[test]
    fn compiler_opencl_cached_on_repeat() {
        let mut c = SPIRVCompiler::default();
        c.compile_opencl(sample_opencl()).unwrap();
        let r2 = c.compile_opencl(sample_opencl()).unwrap();
        assert!(r2.cached);
        assert_eq!(c.compilation_count(), 1);
    }

    #[test]
    fn compiler_glsl_cached_on_repeat() {
        let mut c = SPIRVCompiler::default();
        c.compile_glsl(sample_glsl()).unwrap();
        let r2 = c.compile_glsl(sample_glsl()).unwrap();
        assert!(r2.cached);
    }

    #[test]
    fn compiler_opencl_empty_source_error() {
        let mut c = SPIRVCompiler::default();
        assert!(matches!(c.compile_opencl(""), Err(SPIRVError::EmptySource)));
    }

    #[test]
    fn compiler_glsl_empty_source_error() {
        let mut c = SPIRVCompiler::default();
        assert!(matches!(c.compile_glsl(""), Err(SPIRVError::EmptySource)));
    }

    #[test]
    fn compiler_with_cache() {
        let cache = SPIRVCache::new(4);
        let mut c = SPIRVCompiler::default().with_cache(cache);
        c.compile_opencl(sample_opencl()).unwrap();
        assert_eq!(c.cache().len(), 1);
    }

    #[test]
    fn compiler_with_validator() {
        let v = SPIRVValidator::new().with_max_bound(u32::MAX);
        let mut c = SPIRVCompiler::default().with_validator(v);
        assert!(c.compile_opencl(sample_opencl()).is_ok());
    }

    #[test]
    fn compiler_with_optimizer() {
        let o = SPIRVOptimizer::for_level(OptimizationLevel::None);
        let mut c = SPIRVCompiler::default().with_optimizer(o);
        let r = c.compile_opencl(sample_opencl()).unwrap();
        assert!(r.optimization_stats.is_empty());
    }

    #[test]
    fn compiler_opencl_with_opencl_target() {
        let cfg = SPIRVConfig::new(
            TargetEnvironment::OpenCL(OpenCLVersion::V3_0),
            OptimizationLevel::Size,
        );
        let mut c = SPIRVCompiler::new(cfg);
        let r = c.compile_opencl(sample_opencl()).unwrap();
        assert_eq!(r.module.header.version, 0x0003_0000);
    }

    #[test]
    fn compiler_glsl_with_vulkan_target() {
        let cfg = SPIRVConfig::new(
            TargetEnvironment::Vulkan(VulkanVersion::V1_0),
            OptimizationLevel::None,
        );
        let mut c = SPIRVCompiler::new(cfg);
        let r = c.compile_glsl(sample_glsl()).unwrap();
        assert_eq!(r.module.header.version, 0x0001_0000);
    }

    #[test]
    fn compiler_performance_stats_populated() {
        let mut c = SPIRVCompiler::new(SPIRVConfig::new(
            TargetEnvironment::Vulkan(VulkanVersion::V1_2),
            OptimizationLevel::Performance,
        ));
        let r = c.compile_opencl(sample_opencl()).unwrap();
        assert_eq!(r.optimization_stats.len(), 3);
    }

    #[test]
    fn compiler_different_sources_different_modules() {
        let mut c = SPIRVCompiler::default();
        let a = c.compile_opencl(sample_opencl()).unwrap();
        let b = c.compile_glsl(sample_glsl()).unwrap();
        assert_ne!(a.module.header.generator, b.module.header.generator);
    }

    // ── Hashing helpers ─────────────────────────────────────────

    #[test]
    fn simple_hash_deterministic() {
        assert_eq!(simple_hash("hello"), simple_hash("hello"));
    }

    #[test]
    fn simple_hash_different_inputs() {
        assert_ne!(simple_hash("a"), simple_hash("b"));
    }

    #[test]
    fn hash64_deterministic() {
        assert_eq!(hash64("world"), hash64("world"));
    }

    #[test]
    fn hash64_different_inputs() {
        assert_ne!(hash64("x"), hash64("y"));
    }

    // ── OperandKind ─────────────────────────────────────────────

    #[test]
    fn operand_kind_type_id_equality() {
        assert_eq!(OperandKind::TypeId(1), OperandKind::TypeId(1));
        assert_ne!(OperandKind::TypeId(1), OperandKind::TypeId(2));
    }

    #[test]
    fn operand_kind_variants_not_equal() {
        assert_ne!(OperandKind::TypeId(1), OperandKind::ResultId(1));
        assert_ne!(OperandKind::Literal(1), OperandKind::IdRef(1));
    }

    #[test]
    fn operand_kind_debug_format() {
        let op = OperandKind::Literal(42);
        assert!(format!("{op:?}").contains("42"));
    }

    // ── Integration-style tests ─────────────────────────────────

    #[test]
    fn end_to_end_opencl_pipeline() {
        let cfg = SPIRVConfig::new(
            TargetEnvironment::OpenCL(OpenCLVersion::V2_0),
            OptimizationLevel::Performance,
        );
        let mut compiler = SPIRVCompiler::new(cfg);
        let result = compiler.compile_opencl(sample_opencl()).unwrap();
        assert!(!result.cached);
        assert_eq!(result.source_language, SourceLanguage::OpenCLC);
        let bytes = result.module.to_bytes();
        let decoded = SPIRVModule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.header, result.module.header);
        assert_eq!(decoded.instructions.len(), result.module.instructions.len());
    }

    #[test]
    fn end_to_end_glsl_pipeline() {
        let cfg = SPIRVConfig::new(
            TargetEnvironment::Vulkan(VulkanVersion::V1_1),
            OptimizationLevel::Size,
        );
        let mut compiler = SPIRVCompiler::new(cfg);
        let result = compiler.compile_glsl(sample_glsl()).unwrap();
        assert!(!result.cached);
        let bytes = result.module.to_bytes();
        let decoded = SPIRVModule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.header, result.module.header);
    }

    #[test]
    fn end_to_end_cache_hit_path() {
        let mut compiler = SPIRVCompiler::default();
        let first = compiler.compile_glsl(sample_glsl()).unwrap();
        assert!(!first.cached);
        let second = compiler.compile_glsl(sample_glsl()).unwrap();
        assert!(second.cached);
        assert_eq!(first.module.header, second.module.header);
    }

    #[test]
    fn end_to_end_validate_then_optimize() {
        let translator = OpenCLToSPIRV::new(OpenCLVersion::V1_2);
        let mut module = translator.translate(sample_opencl()).unwrap();
        SPIRVValidator::new().validate(&module).unwrap();
        let optimizer = SPIRVOptimizer::for_level(OptimizationLevel::Performance);
        let stats = optimizer.optimize(&mut module);
        assert_eq!(stats.len(), 3);
    }

    #[test]
    fn module_survives_full_encode_decode_cycle() {
        let glsl = GLSLToSPIRV::new(VulkanVersion::V1_2);
        let original = glsl.translate(sample_glsl()).unwrap();
        let words = original.encode();
        let decoded = SPIRVModule::decode(&words).unwrap();
        let bytes = decoded.to_bytes();
        let final_mod = SPIRVModule::from_bytes(&bytes).unwrap();
        assert_eq!(original.header, final_mod.header);
        assert_eq!(original.instructions.len(), final_mod.instructions.len());
    }
}
