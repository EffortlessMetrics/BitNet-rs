#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal GPU error handling tests for Apple Silicon.
//!
//! Validates graceful failure handling for Metal GPU compute operations including
//! buffer allocation, shader compilation, command encoding, pipeline state,
//! resource limits, error recovery, timeout handling, and memory pressure.
//!
//! All types are LOCAL to this test file — no imports from `metal_compute`
//! (which is feature-gated behind `metal`). Tests exercise error-handling logic
//! without requiring real Metal hardware (except one explicitly `#[ignore]`d test).

#![cfg(feature = "cpu")]

// ── Local error types mirroring Metal patterns ──────────────────────────────

/// Errors that can occur during Metal GPU compute operations.
#[derive(Debug, Clone, PartialEq)]
enum MetalError {
    BufferAllocationFailed { size: usize, reason: String },
    ShaderCompilationFailed { source: String, error: String },
    InvalidDispatch { dimensions: [u32; 3], reason: String },
    PipelineCreationFailed { function: String, error: String },
    ResourceLimitExceeded { resource: String, requested: usize, limit: usize },
    Timeout { operation: String, elapsed_ms: u64 },
    OutOfMemory { requested: usize, available: usize },
    CommandEncodingFailed { stage: String, error: String },
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferAllocationFailed { size, reason } => {
                write!(f, "buffer allocation failed: {size} bytes — {reason}")
            }
            Self::ShaderCompilationFailed { source: _, error } => {
                write!(f, "shader compilation failed: {error}")
            }
            Self::InvalidDispatch { dimensions, reason } => {
                write!(
                    f,
                    "invalid dispatch [{}, {}, {}]: {reason}",
                    dimensions[0], dimensions[1], dimensions[2]
                )
            }
            Self::PipelineCreationFailed { function, error } => {
                write!(f, "pipeline creation failed for '{function}': {error}")
            }
            Self::ResourceLimitExceeded { resource, requested, limit } => {
                write!(f, "resource limit exceeded: {resource} ({requested} > {limit})")
            }
            Self::Timeout { operation, elapsed_ms } => {
                write!(f, "timeout: {operation} after {elapsed_ms}ms")
            }
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: requested {requested}, available {available}")
            }
            Self::CommandEncodingFailed { stage, error } => {
                write!(f, "command encoding failed at '{stage}': {error}")
            }
        }
    }
}

// ── Local helper structs ────────────────────────────────────────────────────

/// Metal buffer alignment on Apple Silicon (256 bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// Maximum buffer size on Apple Silicon (shared memory, typically ≤ device RAM).
const MAX_BUFFER_SIZE: usize = 256 * 1024 * 1024 * 1024; // 256 GB upper bound

/// Maximum threadgroup memory on Apple Silicon (32 KB).
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

/// Maximum texture dimension on Apple Silicon (16384).
const MAX_TEXTURE_DIMENSION: u32 = 16384;

/// Maximum dispatch threads per dimension.
const MAX_DISPATCH_DIM: u32 = 65535;

/// Simulates Metal error conditions for testing error-handling paths.
struct MetalErrorSimulator {
    allocated_bytes: usize,
    memory_limit: usize,
    timeout_ms: u64,
    error_log: Vec<MetalError>,
}

impl MetalErrorSimulator {
    fn new(memory_limit: usize, timeout_ms: u64) -> Self {
        Self { allocated_bytes: 0, memory_limit, timeout_ms, error_log: Vec::new() }
    }

    fn allocate_buffer(&mut self, size: usize) -> Result<Vec<u8>, MetalError> {
        if size == 0 {
            let err = MetalError::BufferAllocationFailed {
                size,
                reason: "zero-size buffer not allowed".into(),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        if size % METAL_BUFFER_ALIGNMENT != 0 {
            let err = MetalError::BufferAllocationFailed {
                size,
                reason: format!("buffer size must be aligned to {METAL_BUFFER_ALIGNMENT} bytes"),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        if self.allocated_bytes + size > self.memory_limit {
            let err = MetalError::OutOfMemory {
                requested: size,
                available: self.memory_limit.saturating_sub(self.allocated_bytes),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        self.allocated_bytes += size;
        Ok(vec![0u8; size])
    }

    fn free_buffer(&mut self, size: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
    }

    fn compile_shader(&mut self, source: &str, entry_point: &str) -> Result<String, MetalError> {
        if source.is_empty() {
            let err = MetalError::ShaderCompilationFailed {
                source: source.into(),
                error: "empty shader source".into(),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        if !source.contains("kernel") && !source.contains("fn ") {
            let err = MetalError::ShaderCompilationFailed {
                source: source.into(),
                error: "no kernel function found in source".into(),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        if !source.contains(entry_point) {
            let err = MetalError::ShaderCompilationFailed {
                source: source.into(),
                error: format!("entry point '{entry_point}' not found"),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        // Detect obvious syntax errors
        let opens = source.chars().filter(|&c| c == '{').count();
        let closes = source.chars().filter(|&c| c == '}').count();
        if opens != closes {
            let err = MetalError::ShaderCompilationFailed {
                source: source.into(),
                error: "mismatched braces in shader source".into(),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        Ok(format!("compiled:{entry_point}"))
    }

    fn validate_dispatch(&self, dims: [u32; 3]) -> Result<(), MetalError> {
        for (i, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(MetalError::InvalidDispatch {
                    dimensions: dims,
                    reason: format!("dimension {i} is zero"),
                });
            }
            if d > MAX_DISPATCH_DIM {
                return Err(MetalError::InvalidDispatch {
                    dimensions: dims,
                    reason: format!("dimension {i} ({d}) exceeds max ({MAX_DISPATCH_DIM})"),
                });
            }
        }
        Ok(())
    }

    fn create_pipeline(&mut self, function: &str) -> Result<String, MetalError> {
        if function.is_empty() {
            let err = MetalError::PipelineCreationFailed {
                function: function.into(),
                error: "empty function name".into(),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        if function.contains(' ') {
            let err = MetalError::PipelineCreationFailed {
                function: function.into(),
                error: "function name contains spaces".into(),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        Ok(format!("pipeline:{function}"))
    }

    fn encode_command(
        &mut self,
        pipeline: &str,
        buffer_bindings: &[(u32, usize)],
    ) -> Result<(), MetalError> {
        if pipeline.is_empty() {
            let err = MetalError::CommandEncodingFailed {
                stage: "bind_pipeline".into(),
                error: "no pipeline set".into(),
            };
            self.error_log.push(err.clone());
            return Err(err);
        }
        for &(index, size) in buffer_bindings {
            if size == 0 {
                let err = MetalError::CommandEncodingFailed {
                    stage: format!("bind_buffer[{index}]"),
                    error: "zero-size buffer binding".into(),
                };
                self.error_log.push(err.clone());
                return Err(err);
            }
        }
        Ok(())
    }

    fn simulate_timeout(&mut self, operation: &str, duration_ms: u64) -> Result<(), MetalError> {
        if duration_ms > self.timeout_ms {
            let err = MetalError::Timeout { operation: operation.into(), elapsed_ms: duration_ms };
            self.error_log.push(err.clone());
            return Err(err);
        }
        Ok(())
    }

    fn check_resource_limit(
        &self,
        resource: &str,
        requested: usize,
        limit: usize,
    ) -> Result<(), MetalError> {
        if requested > limit {
            return Err(MetalError::ResourceLimitExceeded {
                resource: resource.into(),
                requested,
                limit,
            });
        }
        Ok(())
    }

    fn error_count(&self) -> usize {
        self.error_log.len()
    }

    fn last_error(&self) -> Option<&MetalError> {
        self.error_log.last()
    }

    fn clear_errors(&mut self) {
        self.error_log.clear();
    }

    fn available_memory(&self) -> usize {
        self.memory_limit.saturating_sub(self.allocated_bytes)
    }
}

/// Aligns a size up to the Metal buffer alignment boundary.
fn align_to_metal(size: usize) -> usize {
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Invalid buffer tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod invalid_buffer {
    use super::*;

    #[test]
    fn zero_size_buffer_rejected() {
        let mut sim = MetalErrorSimulator::new(1024 * 1024, 5000);
        let result = sim.allocate_buffer(0);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::BufferAllocationFailed { size, reason } => {
                assert_eq!(size, 0);
                assert!(reason.contains("zero-size"));
            }
            other => panic!("expected BufferAllocationFailed, got {other:?}"),
        }
    }

    #[test]
    fn misaligned_buffer_rejected() {
        let mut sim = MetalErrorSimulator::new(1024 * 1024, 5000);
        let result = sim.allocate_buffer(100); // not 256-byte aligned
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::BufferAllocationFailed { size, reason } => {
                assert_eq!(size, 100);
                assert!(reason.contains("aligned"));
            }
            other => panic!("expected BufferAllocationFailed, got {other:?}"),
        }
    }

    #[test]
    fn aligned_buffer_succeeds() {
        let mut sim = MetalErrorSimulator::new(1024 * 1024, 5000);
        let result = sim.allocate_buffer(METAL_BUFFER_ALIGNMENT);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), METAL_BUFFER_ALIGNMENT);
    }

    #[test]
    fn oversized_allocation_rejected() {
        let mut sim = MetalErrorSimulator::new(512, 5000);
        let result = sim.allocate_buffer(METAL_BUFFER_ALIGNMENT * 4); // 1024 > 512
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::OutOfMemory { requested, available } => {
                assert_eq!(requested, METAL_BUFFER_ALIGNMENT * 4);
                assert_eq!(available, 512);
            }
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
    }

    #[test]
    fn alignment_helper_rounds_up() {
        assert_eq!(align_to_metal(0), 0);
        assert_eq!(align_to_metal(1), METAL_BUFFER_ALIGNMENT);
        assert_eq!(align_to_metal(METAL_BUFFER_ALIGNMENT), METAL_BUFFER_ALIGNMENT);
        assert_eq!(align_to_metal(METAL_BUFFER_ALIGNMENT + 1), METAL_BUFFER_ALIGNMENT * 2);
    }

    #[test]
    fn multiple_misaligned_sizes_all_rejected() {
        let mut sim = MetalErrorSimulator::new(1024 * 1024, 5000);
        let bad_sizes = [1, 7, 127, 255, 257, 511, 1023];
        for &size in &bad_sizes {
            let result = sim.allocate_buffer(size);
            assert!(result.is_err(), "size {size} should be rejected as misaligned");
        }
    }

    #[test]
    fn error_logged_on_failed_allocation() {
        let mut sim = MetalErrorSimulator::new(1024 * 1024, 5000);
        assert_eq!(sim.error_count(), 0);
        let _ = sim.allocate_buffer(0);
        assert_eq!(sim.error_count(), 1);
        let _ = sim.allocate_buffer(100);
        assert_eq!(sim.error_count(), 2);
    }

    #[test]
    fn successive_aligned_allocations_track_memory() {
        let limit = METAL_BUFFER_ALIGNMENT * 4;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        // First allocation succeeds
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_ok());
        assert_eq!(sim.available_memory(), limit - METAL_BUFFER_ALIGNMENT);
        // Fill remaining
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT * 3).is_ok());
        assert_eq!(sim.available_memory(), 0);
        // Next allocation fails
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Shader compilation tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod shader_compilation {
    use super::*;

    #[test]
    fn empty_source_rejected() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.compile_shader("", "main");
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::ShaderCompilationFailed { error, .. } => {
                assert!(error.contains("empty"));
            }
            other => panic!("expected ShaderCompilationFailed, got {other:?}"),
        }
    }

    #[test]
    fn missing_entry_point_rejected() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let source = "kernel void other_func() {}";
        let result = sim.compile_shader(source, "main");
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::ShaderCompilationFailed { error, .. } => {
                assert!(error.contains("entry point"));
                assert!(error.contains("main"));
            }
            other => panic!("expected ShaderCompilationFailed, got {other:?}"),
        }
    }

    #[test]
    fn valid_shader_compiles() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let source = "kernel void main() {}";
        let result = sim.compile_shader(source, "main");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "compiled:main");
    }

    #[test]
    fn syntax_error_mismatched_braces() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let source = "kernel void main() {";
        let result = sim.compile_shader(source, "main");
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::ShaderCompilationFailed { error, .. } => {
                assert!(error.contains("braces"));
            }
            other => panic!("expected ShaderCompilationFailed, got {other:?}"),
        }
    }

    #[test]
    fn no_kernel_function_rejected() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let source = "// just a comment with no entry";
        let result = sim.compile_shader(source, "main");
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::ShaderCompilationFailed { error, .. } => {
                assert!(error.contains("no kernel function"));
            }
            other => panic!("expected ShaderCompilationFailed, got {other:?}"),
        }
    }

    #[test]
    fn compilation_error_logged() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let _ = sim.compile_shader("", "main");
        assert_eq!(sim.error_count(), 1);
        assert!(matches!(sim.last_error(), Some(MetalError::ShaderCompilationFailed { .. })));
    }

    #[test]
    fn error_display_format_includes_message() {
        let err = MetalError::ShaderCompilationFailed {
            source: "bad source".into(),
            error: "parse error at line 1".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("shader compilation failed"));
        assert!(msg.contains("parse error at line 1"));
    }

    #[test]
    fn multiple_entry_points_selects_correct() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let source = "kernel void main() {} kernel void secondary() {}";
        assert!(sim.compile_shader(source, "main").is_ok());
        assert!(sim.compile_shader(source, "secondary").is_ok());
        assert!(sim.compile_shader(source, "nonexistent").is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Command encoding tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod command_encoding {
    use super::*;

    #[test]
    fn empty_pipeline_rejected() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.encode_command("", &[(0, 256)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::CommandEncodingFailed { stage, error } => {
                assert_eq!(stage, "bind_pipeline");
                assert!(error.contains("no pipeline"));
            }
            other => panic!("expected CommandEncodingFailed, got {other:?}"),
        }
    }

    #[test]
    fn zero_size_binding_rejected() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.encode_command("pipeline:main", &[(0, 0)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::CommandEncodingFailed { stage, error } => {
                assert!(stage.contains("bind_buffer[0]"));
                assert!(error.contains("zero-size"));
            }
            other => panic!("expected CommandEncodingFailed, got {other:?}"),
        }
    }

    #[test]
    fn valid_encoding_succeeds() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.encode_command("pipeline:main", &[(0, 256), (1, 512)]);
        assert!(result.is_ok());
    }

    #[test]
    fn invalid_dispatch_zero_dimension() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.validate_dispatch([0, 1, 1]);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::InvalidDispatch { dimensions, reason } => {
                assert_eq!(dimensions, [0, 1, 1]);
                assert!(reason.contains("dimension 0 is zero"));
            }
            other => panic!("expected InvalidDispatch, got {other:?}"),
        }
    }

    #[test]
    fn invalid_dispatch_exceeds_max() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.validate_dispatch([1, MAX_DISPATCH_DIM + 1, 1]);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::InvalidDispatch { reason, .. } => {
                assert!(reason.contains("exceeds max"));
            }
            other => panic!("expected InvalidDispatch, got {other:?}"),
        }
    }

    #[test]
    fn valid_dispatch_at_max() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.validate_dispatch([MAX_DISPATCH_DIM, MAX_DISPATCH_DIM, MAX_DISPATCH_DIM]);
        assert!(result.is_ok());
    }

    #[test]
    fn multiple_zero_bindings_first_error_reported() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.encode_command("pipeline:main", &[(0, 0), (1, 0)]);
        assert!(result.is_err());
        // First zero-size binding (index 0) should be reported
        match result.unwrap_err() {
            MetalError::CommandEncodingFailed { stage, .. } => {
                assert!(stage.contains("bind_buffer[0]"));
            }
            other => panic!("expected CommandEncodingFailed, got {other:?}"),
        }
    }

    #[test]
    fn encoding_error_display_format() {
        let err = MetalError::CommandEncodingFailed {
            stage: "dispatch".into(),
            error: "invalid threadgroup size".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("command encoding failed"));
        assert!(msg.contains("dispatch"));
        assert!(msg.contains("invalid threadgroup size"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Pipeline state tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod pipeline_state {
    use super::*;

    #[test]
    fn empty_function_name_rejected() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.create_pipeline("");
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::PipelineCreationFailed { function, error } => {
                assert!(function.is_empty());
                assert!(error.contains("empty function name"));
            }
            other => panic!("expected PipelineCreationFailed, got {other:?}"),
        }
    }

    #[test]
    fn function_with_spaces_rejected() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.create_pipeline("my function");
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::PipelineCreationFailed { error, .. } => {
                assert!(error.contains("spaces"));
            }
            other => panic!("expected PipelineCreationFailed, got {other:?}"),
        }
    }

    #[test]
    fn valid_function_creates_pipeline() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.create_pipeline("matmul_kernel");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "pipeline:matmul_kernel");
    }

    #[test]
    fn pipeline_error_logged() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let _ = sim.create_pipeline("");
        assert_eq!(sim.error_count(), 1);
        assert!(matches!(sim.last_error(), Some(MetalError::PipelineCreationFailed { .. })));
    }

    #[test]
    fn pipeline_display_includes_function_name() {
        let err = MetalError::PipelineCreationFailed {
            function: "bad_kernel".into(),
            error: "library not loaded".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("bad_kernel"));
        assert!(msg.contains("library not loaded"));
    }

    #[test]
    fn multiple_pipelines_independent() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let p1 = sim.create_pipeline("kernel_a");
        let p2 = sim.create_pipeline("kernel_b");
        assert!(p1.is_ok());
        assert!(p2.is_ok());
        assert_ne!(p1.unwrap(), p2.unwrap());
    }

    #[test]
    fn pipeline_then_encode_workflow() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let pipeline = sim.create_pipeline("compute_main").unwrap();
        let result = sim.encode_command(&pipeline, &[(0, 256)]);
        assert!(result.is_ok());
    }

    #[test]
    fn failed_pipeline_blocks_encoding() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let pipeline_result = sim.create_pipeline("");
        assert!(pipeline_result.is_err());
        // Using empty string as a stand-in for "no pipeline"
        let encode_result = sim.encode_command("", &[(0, 256)]);
        assert!(encode_result.is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Resource limits tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod resource_limits {
    use super::*;

    #[test]
    fn buffer_size_within_limit() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.check_resource_limit("buffer", 1024, MAX_BUFFER_SIZE);
        assert!(result.is_ok());
    }

    #[test]
    fn buffer_size_exceeds_limit() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.check_resource_limit("buffer", MAX_BUFFER_SIZE + 1, MAX_BUFFER_SIZE);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::ResourceLimitExceeded { resource, requested, limit } => {
                assert_eq!(resource, "buffer");
                assert_eq!(requested, MAX_BUFFER_SIZE + 1);
                assert_eq!(limit, MAX_BUFFER_SIZE);
            }
            other => panic!("expected ResourceLimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn threadgroup_memory_within_limit() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result =
            sim.check_resource_limit("threadgroup_memory", 16 * 1024, MAX_THREADGROUP_MEMORY);
        assert!(result.is_ok());
    }

    #[test]
    fn threadgroup_memory_exceeds_limit() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.check_resource_limit(
            "threadgroup_memory",
            MAX_THREADGROUP_MEMORY + 1,
            MAX_THREADGROUP_MEMORY,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::ResourceLimitExceeded { resource, .. } => {
                assert_eq!(resource, "threadgroup_memory");
            }
            other => panic!("expected ResourceLimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn texture_dimension_within_limit() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.check_resource_limit(
            "texture_dimension",
            MAX_TEXTURE_DIMENSION as usize,
            MAX_TEXTURE_DIMENSION as usize,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn texture_dimension_exceeds_limit() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.check_resource_limit(
            "texture_dimension",
            (MAX_TEXTURE_DIMENSION + 1) as usize,
            MAX_TEXTURE_DIMENSION as usize,
        );
        assert!(result.is_err());
    }

    #[test]
    fn resource_limit_display_format() {
        let err = MetalError::ResourceLimitExceeded {
            resource: "buffer".into(),
            requested: 1_000_000,
            limit: 500_000,
        };
        let msg = format!("{err}");
        assert!(msg.contains("resource limit exceeded"));
        assert!(msg.contains("buffer"));
        assert!(msg.contains("1000000"));
        assert!(msg.contains("500000"));
    }

    #[test]
    fn threads_per_threadgroup_limit() {
        let sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.check_resource_limit(
            "threads_per_threadgroup",
            MAX_THREADS_PER_THREADGROUP as usize,
            MAX_THREADS_PER_THREADGROUP as usize,
        );
        assert!(result.is_ok());
        let result = sim.check_resource_limit(
            "threads_per_threadgroup",
            (MAX_THREADS_PER_THREADGROUP + 1) as usize,
            MAX_THREADS_PER_THREADGROUP as usize,
        );
        assert!(result.is_err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Error recovery tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod error_recovery {
    use super::*;

    #[test]
    fn error_propagation_chain() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        // Shader failure → pipeline failure → encoding failure chain
        let shader_err = sim.compile_shader("", "main");
        assert!(shader_err.is_err());
        let pipeline_err = sim.create_pipeline("");
        assert!(pipeline_err.is_err());
        let encode_err = sim.encode_command("", &[(0, 256)]);
        assert!(encode_err.is_err());
        assert_eq!(sim.error_count(), 3);
    }

    #[test]
    fn error_context_preserved() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let _ = sim.allocate_buffer(0);
        let err = sim.last_error().unwrap();
        match err {
            MetalError::BufferAllocationFailed { size, reason } => {
                assert_eq!(*size, 0);
                assert!(!reason.is_empty());
            }
            other => panic!("expected BufferAllocationFailed, got {other:?}"),
        }
    }

    #[test]
    fn recovery_after_allocation_failure() {
        let limit = METAL_BUFFER_ALIGNMENT * 2;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        // Fail: too large
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT * 4).is_err());
        // Succeed: within limits
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_ok());
    }

    #[test]
    fn recovery_after_shader_failure() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        // Fail first
        assert!(sim.compile_shader("", "main").is_err());
        // Then succeed
        let result = sim.compile_shader("kernel void main() {}", "main");
        assert!(result.is_ok());
    }

    #[test]
    fn error_log_accumulates() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let _ = sim.allocate_buffer(0);
        let _ = sim.compile_shader("", "x");
        let _ = sim.create_pipeline("");
        assert_eq!(sim.error_count(), 3);
    }

    #[test]
    fn clear_errors_resets_log() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let _ = sim.allocate_buffer(0);
        assert_eq!(sim.error_count(), 1);
        sim.clear_errors();
        assert_eq!(sim.error_count(), 0);
        assert!(sim.last_error().is_none());
    }

    #[test]
    fn free_buffer_enables_reallocation() {
        let limit = METAL_BUFFER_ALIGNMENT;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_ok());
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_err());
        sim.free_buffer(METAL_BUFFER_ALIGNMENT);
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_ok());
    }

    #[test]
    fn error_equality_check() {
        let err1 = MetalError::BufferAllocationFailed { size: 0, reason: "zero".into() };
        let err2 = MetalError::BufferAllocationFailed { size: 0, reason: "zero".into() };
        let err3 = MetalError::BufferAllocationFailed { size: 1, reason: "other".into() };
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Timeout handling tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod timeout_handling {
    use super::*;

    #[test]
    fn within_timeout_succeeds() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.simulate_timeout("compute_pass", 1000);
        assert!(result.is_ok());
    }

    #[test]
    fn exceeds_timeout_fails() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.simulate_timeout("compute_pass", 6000);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::Timeout { operation, elapsed_ms } => {
                assert_eq!(operation, "compute_pass");
                assert_eq!(elapsed_ms, 6000);
            }
            other => panic!("expected Timeout, got {other:?}"),
        }
    }

    #[test]
    fn exact_timeout_boundary_succeeds() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.simulate_timeout("compute_pass", 5000);
        assert!(result.is_ok());
    }

    #[test]
    fn timeout_one_ms_over_fails() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let result = sim.simulate_timeout("compute_pass", 5001);
        assert!(result.is_err());
    }

    #[test]
    fn timeout_error_logged() {
        let mut sim = MetalErrorSimulator::new(1024, 5000);
        let _ = sim.simulate_timeout("long_kernel", 10_000);
        assert_eq!(sim.error_count(), 1);
        assert!(matches!(sim.last_error(), Some(MetalError::Timeout { .. })));
    }

    #[test]
    fn timeout_display_format() {
        let err = MetalError::Timeout { operation: "matmul".into(), elapsed_ms: 8000 };
        let msg = format!("{err}");
        assert!(msg.contains("timeout"));
        assert!(msg.contains("matmul"));
        assert!(msg.contains("8000"));
    }

    #[test]
    fn zero_timeout_rejects_all() {
        let mut sim = MetalErrorSimulator::new(1024, 0);
        let result = sim.simulate_timeout("any_op", 1);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires Metal GPU"]
    fn graceful_cancellation_on_real_device() {
        // This test would exercise real Metal command buffer timeout and cancellation.
        // Requires Apple Silicon hardware with Metal support.
        panic!("requires real Metal GPU hardware");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Memory pressure tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(target_os = "macos")]
mod memory_pressure {
    use super::*;

    #[test]
    fn oom_on_exhausted_memory() {
        let limit = METAL_BUFFER_ALIGNMENT;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_ok());
        let result = sim.allocate_buffer(METAL_BUFFER_ALIGNMENT);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalError::OutOfMemory { requested, available } => {
                assert_eq!(requested, METAL_BUFFER_ALIGNMENT);
                assert_eq!(available, 0);
            }
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
    }

    #[test]
    fn allocation_failure_recovery_via_free() {
        let limit = METAL_BUFFER_ALIGNMENT * 2;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT * 2).is_ok());
        // Memory exhausted
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_err());
        // Free half
        sim.free_buffer(METAL_BUFFER_ALIGNMENT);
        // Now allocation succeeds
        assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_ok());
    }

    #[test]
    fn memory_tracking_accuracy() {
        let limit = METAL_BUFFER_ALIGNMENT * 10;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        assert_eq!(sim.available_memory(), limit);
        sim.allocate_buffer(METAL_BUFFER_ALIGNMENT * 3).unwrap();
        assert_eq!(sim.available_memory(), METAL_BUFFER_ALIGNMENT * 7);
        sim.allocate_buffer(METAL_BUFFER_ALIGNMENT * 2).unwrap();
        assert_eq!(sim.available_memory(), METAL_BUFFER_ALIGNMENT * 5);
        sim.free_buffer(METAL_BUFFER_ALIGNMENT * 3);
        assert_eq!(sim.available_memory(), METAL_BUFFER_ALIGNMENT * 8);
    }

    #[test]
    fn oom_display_format() {
        let err = MetalError::OutOfMemory { requested: 1024, available: 256 };
        let msg = format!("{err}");
        assert!(msg.contains("out of memory"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn repeated_alloc_free_cycle() {
        let limit = METAL_BUFFER_ALIGNMENT * 2;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        for _ in 0..10 {
            assert!(sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).is_ok());
            sim.free_buffer(METAL_BUFFER_ALIGNMENT);
        }
        assert_eq!(sim.available_memory(), limit);
    }

    #[test]
    fn free_more_than_allocated_saturates_at_zero() {
        let limit = METAL_BUFFER_ALIGNMENT * 4;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).unwrap();
        // Free more than allocated — should not underflow
        sim.free_buffer(METAL_BUFFER_ALIGNMENT * 10);
        assert_eq!(sim.allocated_bytes, 0);
        assert_eq!(sim.available_memory(), limit);
    }

    #[test]
    fn oom_error_count_increments() {
        let limit = METAL_BUFFER_ALIGNMENT;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        sim.allocate_buffer(METAL_BUFFER_ALIGNMENT).unwrap();
        let _ = sim.allocate_buffer(METAL_BUFFER_ALIGNMENT);
        let _ = sim.allocate_buffer(METAL_BUFFER_ALIGNMENT);
        assert_eq!(sim.error_count(), 2);
    }

    #[test]
    fn gradual_pressure_until_oom() {
        let limit = METAL_BUFFER_ALIGNMENT * 4;
        let mut sim = MetalErrorSimulator::new(limit, 5000);
        let mut allocated = 0;
        for i in 0..10 {
            match sim.allocate_buffer(METAL_BUFFER_ALIGNMENT) {
                Ok(_) => allocated += 1,
                Err(MetalError::OutOfMemory { .. }) => {
                    assert_eq!(allocated, 4, "should have allocated 4 buffers before OOM");
                    assert_eq!(i, 4);
                    return;
                }
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        panic!("should have hit OOM");
    }
}
