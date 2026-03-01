//! OpenCL Kernel Provider for Intel Arc GPU acceleration.
//!
//! This module provides a CPU-reference implementation of the OpenCL
//! kernel provider interface. When the OpenCL runtime is available,
//! it will compile and launch `.cl` kernels on Intel Arc GPUs.
//! Falls back to CPU when OpenCL is not available.

use std::collections::HashMap;
use std::fmt;

/// Configuration for the OpenCL provider.
#[derive(Debug, Clone)]
pub struct OpenClConfig {
    /// Platform index (0 = first platform)
    pub platform_index: usize,
    /// Device index within the platform
    pub device_index: usize,
    /// Enable kernel profiling
    pub enable_profiling: bool,
    /// Work group size hint (0 = auto)
    pub preferred_work_group_size: usize,
    /// Cache compiled programs
    pub cache_programs: bool,
}

impl Default for OpenClConfig {
    fn default() -> Self {
        Self {
            platform_index: 0,
            device_index: 0,
            enable_profiling: false,
            preferred_work_group_size: 0,
            cache_programs: true,
        }
    }
}

/// Represents an OpenCL buffer (CPU reference implementation).
#[derive(Debug, Clone)]
pub struct OpenClBuffer {
    /// Buffer data (CPU reference)
    data: Vec<u8>,
    /// Size in bytes
    size: usize,
    /// Whether this is read-only
    read_only: bool,
    /// Label for debugging
    label: String,
}

impl OpenClBuffer {
    /// Create a new buffer with given size.
    pub fn new(size: usize, label: &str) -> Self {
        Self { data: vec![0u8; size], size, read_only: false, label: label.to_string() }
    }

    /// Create a buffer from existing data.
    pub fn from_data(data: &[u8], label: &str) -> Self {
        Self { data: data.to_vec(), size: data.len(), read_only: false, label: label.to_string() }
    }

    /// Create a read-only buffer from data.
    pub fn from_data_readonly(data: &[u8], label: &str) -> Self {
        Self { data: data.to_vec(), size: data.len(), read_only: true, label: label.to_string() }
    }

    /// Get buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get buffer data as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable buffer data.
    pub fn as_bytes_mut(&mut self) -> Result<&mut [u8], OpenClError> {
        if self.read_only {
            return Err(OpenClError::BufferReadOnly(self.label.clone()));
        }
        Ok(&mut self.data)
    }

    /// Write data to buffer.
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), OpenClError> {
        if self.read_only {
            return Err(OpenClError::BufferReadOnly(self.label.clone()));
        }
        if offset + data.len() > self.size {
            return Err(OpenClError::BufferOverflow {
                offset,
                write_size: data.len(),
                buffer_size: self.size,
            });
        }
        self.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read data from buffer.
    pub fn read(&self, offset: usize, size: usize) -> Result<&[u8], OpenClError> {
        if offset + size > self.size {
            return Err(OpenClError::BufferOverflow {
                offset,
                write_size: size,
                buffer_size: self.size,
            });
        }
        Ok(&self.data[offset..offset + size])
    }

    /// Get buffer label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Check if buffer is read-only.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }
}

/// OpenCL-specific errors.
#[derive(Debug, Clone)]
pub enum OpenClError {
    /// Platform not found
    PlatformNotFound(usize),
    /// Device not found
    DeviceNotFound(usize),
    /// Kernel compilation error
    KernelCompileError(String),
    /// Buffer is read-only
    BufferReadOnly(String),
    /// Buffer overflow on read/write
    BufferOverflow { offset: usize, write_size: usize, buffer_size: usize },
    /// Kernel not found in program
    KernelNotFound(String),
    /// Invalid work dimensions
    InvalidWorkDimensions(String),
    /// Runtime not available
    RuntimeNotAvailable,
}

impl fmt::Display for OpenClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlatformNotFound(idx) => write!(f, "OpenCL platform {} not found", idx),
            Self::DeviceNotFound(idx) => write!(f, "OpenCL device {} not found", idx),
            Self::KernelCompileError(msg) => {
                write!(f, "Kernel compile error: {}", msg)
            }
            Self::BufferReadOnly(label) => write!(f, "Buffer '{}' is read-only", label),
            Self::BufferOverflow { offset, write_size, buffer_size } => {
                write!(
                    f,
                    "Buffer overflow: offset {} + size {} > buffer size {}",
                    offset, write_size, buffer_size
                )
            }
            Self::KernelNotFound(name) => write!(f, "Kernel '{}' not found", name),
            Self::InvalidWorkDimensions(msg) => {
                write!(f, "Invalid work dimensions: {}", msg)
            }
            Self::RuntimeNotAvailable => write!(f, "OpenCL runtime not available"),
        }
    }
}

impl std::error::Error for OpenClError {}

/// Represents a compiled kernel program (CPU reference).
#[derive(Debug, Clone)]
pub struct OpenClProgram {
    /// Source code
    source: String,
    /// Available kernel names
    kernel_names: Vec<String>,
}

impl OpenClProgram {
    /// Create a program from source (CPU reference: just parses kernel names).
    pub fn from_source(source: &str) -> Self {
        let kernel_names: Vec<String> = source
            .lines()
            .filter(|line| line.contains("__kernel"))
            .filter_map(|line| {
                // Extract kernel name from "__kernel void name("
                let after_void = line.split("void").nth(1)?;
                let name = after_void.trim().split('(').next()?.trim();
                if name.is_empty() { None } else { Some(name.to_string()) }
            })
            .collect();

        Self { source: source.to_string(), kernel_names }
    }

    /// Get available kernel names.
    pub fn kernel_names(&self) -> &[String] {
        &self.kernel_names
    }

    /// Get source code.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Check if kernel exists.
    pub fn has_kernel(&self, name: &str) -> bool {
        self.kernel_names.iter().any(|n| n == name)
    }
}

/// OpenCL Kernel Provider (CPU reference implementation).
///
/// When OpenCL runtime is available, this will compile and launch
/// kernels on the GPU. Currently provides CPU reference implementations.
#[derive(Debug)]
pub struct OpenClKernelProvider {
    config: OpenClConfig,
    programs: HashMap<String, OpenClProgram>,
    buffers: Vec<OpenClBuffer>,
}

impl OpenClKernelProvider {
    /// Create a new provider with given config.
    pub fn new(config: OpenClConfig) -> Self {
        Self { config, programs: HashMap::new(), buffers: Vec::new() }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(OpenClConfig::default())
    }

    /// Load a kernel source.
    pub fn load_program(&mut self, name: &str, source: &str) -> Result<(), OpenClError> {
        let program = OpenClProgram::from_source(source);
        if program.kernel_names().is_empty() {
            return Err(OpenClError::KernelCompileError(format!(
                "No kernels found in source '{}'",
                name
            )));
        }
        self.programs.insert(name.to_string(), program);
        Ok(())
    }

    /// Get loaded program names.
    pub fn program_names(&self) -> Vec<&String> {
        self.programs.keys().collect()
    }

    /// Check if a kernel is available.
    pub fn has_kernel(&self, program_name: &str, kernel_name: &str) -> bool {
        self.programs.get(program_name).map_or(false, |p| p.has_kernel(kernel_name))
    }

    /// Allocate a buffer.
    pub fn alloc_buffer(&mut self, size: usize, label: &str) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(OpenClBuffer::new(size, label));
        idx
    }

    /// Get buffer by index.
    pub fn get_buffer(&self, index: usize) -> Option<&OpenClBuffer> {
        self.buffers.get(index)
    }

    /// Get mutable buffer by index.
    pub fn get_buffer_mut(&mut self, index: usize) -> Option<&mut OpenClBuffer> {
        self.buffers.get_mut(index)
    }

    /// Get config.
    pub fn config(&self) -> &OpenClConfig {
        &self.config
    }

    /// Return number of loaded programs.
    pub fn program_count(&self) -> usize {
        self.programs.len()
    }

    /// Return total allocated buffer memory.
    pub fn total_buffer_memory(&self) -> usize {
        self.buffers.iter().map(|b| b.size()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- OpenClConfig tests ---

    #[test]
    fn config_default() {
        let cfg = OpenClConfig::default();
        assert_eq!(cfg.platform_index, 0);
        assert_eq!(cfg.device_index, 0);
        assert!(!cfg.enable_profiling);
        assert_eq!(cfg.preferred_work_group_size, 0);
        assert!(cfg.cache_programs);
    }

    #[test]
    fn config_custom() {
        let cfg = OpenClConfig {
            platform_index: 1,
            device_index: 2,
            enable_profiling: true,
            preferred_work_group_size: 256,
            cache_programs: false,
        };
        assert_eq!(cfg.platform_index, 1);
        assert!(cfg.enable_profiling);
    }

    // --- OpenClBuffer tests ---

    #[test]
    fn buffer_new() {
        let buf = OpenClBuffer::new(1024, "test");
        assert_eq!(buf.size(), 1024);
        assert_eq!(buf.label(), "test");
        assert!(!buf.is_read_only());
        assert_eq!(buf.as_bytes().len(), 1024);
    }

    #[test]
    fn buffer_from_data() {
        let data = vec![1u8, 2, 3, 4];
        let buf = OpenClBuffer::from_data(&data, "test");
        assert_eq!(buf.size(), 4);
        assert_eq!(buf.as_bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn buffer_from_data_readonly() {
        let data = vec![1u8, 2, 3];
        let buf = OpenClBuffer::from_data_readonly(&data, "ro");
        assert!(buf.is_read_only());
    }

    #[test]
    fn buffer_write() {
        let mut buf = OpenClBuffer::new(8, "test");
        buf.write(2, &[0xAA, 0xBB]).unwrap();
        assert_eq!(buf.as_bytes()[2], 0xAA);
        assert_eq!(buf.as_bytes()[3], 0xBB);
    }

    #[test]
    fn buffer_write_overflow() {
        let mut buf = OpenClBuffer::new(4, "test");
        let result = buf.write(3, &[1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn buffer_readonly_write_fails() {
        let mut buf = OpenClBuffer::from_data_readonly(&[1, 2], "ro");
        assert!(buf.write(0, &[3]).is_err());
        assert!(buf.as_bytes_mut().is_err());
    }

    #[test]
    fn buffer_read() {
        let buf = OpenClBuffer::from_data(&[10, 20, 30, 40], "test");
        let slice = buf.read(1, 2).unwrap();
        assert_eq!(slice, &[20, 30]);
    }

    #[test]
    fn buffer_read_overflow() {
        let buf = OpenClBuffer::new(4, "test");
        assert!(buf.read(3, 2).is_err());
    }

    // --- OpenClError tests ---

    #[test]
    fn error_display() {
        let err = OpenClError::PlatformNotFound(0);
        assert!(err.to_string().contains("platform"));

        let err = OpenClError::RuntimeNotAvailable;
        assert!(err.to_string().contains("not available"));
    }

    #[test]
    fn error_buffer_overflow_display() {
        let err = OpenClError::BufferOverflow { offset: 10, write_size: 5, buffer_size: 12 };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("12"));
    }

    // --- OpenClProgram tests ---

    #[test]
    fn program_from_source() {
        let src = "__kernel void add(__global float* a) { }";
        let prog = OpenClProgram::from_source(src);
        assert_eq!(prog.kernel_names(), &["add"]);
    }

    #[test]
    fn program_multiple_kernels() {
        let src = r#"
__kernel void foo(__global float* a) { }
__kernel void bar(__global float* b) { }
"#;
        let prog = OpenClProgram::from_source(src);
        assert_eq!(prog.kernel_names().len(), 2);
        assert!(prog.has_kernel("foo"));
        assert!(prog.has_kernel("bar"));
        assert!(!prog.has_kernel("baz"));
    }

    #[test]
    fn program_source_preserved() {
        let src = "// test\n__kernel void test() {}";
        let prog = OpenClProgram::from_source(src);
        assert!(prog.source().contains("// test"));
    }

    // --- OpenClKernelProvider tests ---

    #[test]
    fn provider_new() {
        let prov = OpenClKernelProvider::with_defaults();
        assert_eq!(prov.program_count(), 0);
        assert_eq!(prov.total_buffer_memory(), 0);
    }

    #[test]
    fn provider_load_program() {
        let mut prov = OpenClKernelProvider::with_defaults();
        prov.load_program("test", "__kernel void add() {}").unwrap();
        assert_eq!(prov.program_count(), 1);
        assert!(prov.has_kernel("test", "add"));
    }

    #[test]
    fn provider_load_empty_program_fails() {
        let mut prov = OpenClKernelProvider::with_defaults();
        let result = prov.load_program("empty", "// no kernels here");
        assert!(result.is_err());
    }

    #[test]
    fn provider_alloc_buffer() {
        let mut prov = OpenClKernelProvider::with_defaults();
        let idx = prov.alloc_buffer(1024, "weights");
        assert_eq!(idx, 0);
        assert_eq!(prov.total_buffer_memory(), 1024);

        let buf = prov.get_buffer(idx).unwrap();
        assert_eq!(buf.size(), 1024);
        assert_eq!(buf.label(), "weights");
    }

    #[test]
    fn provider_multiple_buffers() {
        let mut prov = OpenClKernelProvider::with_defaults();
        prov.alloc_buffer(100, "a");
        prov.alloc_buffer(200, "b");
        prov.alloc_buffer(300, "c");
        assert_eq!(prov.total_buffer_memory(), 600);
    }

    #[test]
    fn provider_config_access() {
        let cfg = OpenClConfig { platform_index: 3, ..Default::default() };
        let prov = OpenClKernelProvider::new(cfg);
        assert_eq!(prov.config().platform_index, 3);
    }

    #[test]
    fn provider_buffer_mut() {
        let mut prov = OpenClKernelProvider::with_defaults();
        let idx = prov.alloc_buffer(8, "test");
        let buf = prov.get_buffer_mut(idx).unwrap();
        buf.write(0, &[42]).unwrap();
        assert_eq!(prov.get_buffer(idx).unwrap().as_bytes()[0], 42);
    }

    #[test]
    fn provider_nonexistent_buffer() {
        let prov = OpenClKernelProvider::with_defaults();
        assert!(prov.get_buffer(99).is_none());
    }
}
