use std::fmt;

/// Errors that can occur during kernel runner operations.
#[derive(Debug)]
pub enum RunnerError {
    /// Failed to obtain a wgpu adapter.
    AdapterNotFound,
    /// Failed to obtain a wgpu device.
    DeviceRequest(wgpu::RequestDeviceError),
    /// Shader compilation failed.
    ShaderCompilation(String),
    /// Buffer mapping failed.
    BufferMap(wgpu::BufferAsyncError),
    /// Invalid dimensions for the operation.
    InvalidDimensions { expected: usize, actual: usize, name: &'static str },
    /// Dispatch workgroup count is zero on at least one axis.
    ZeroWorkgroup,
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterNotFound => write!(f, "no suitable wgpu adapter found"),
            Self::DeviceRequest(e) => write!(f, "wgpu device request failed: {e}"),
            Self::ShaderCompilation(msg) => write!(f, "shader compilation failed: {msg}"),
            Self::BufferMap(e) => write!(f, "buffer mapping failed: {e}"),
            Self::InvalidDimensions { expected, actual, name } => {
                write!(f, "invalid dimensions for {name}: expected {expected}, got {actual}")
            }
            Self::ZeroWorkgroup => write!(f, "workgroup count must be non-zero on all axes"),
        }
    }
}

impl std::error::Error for RunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DeviceRequest(e) => Some(e),
            Self::BufferMap(e) => Some(e),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for RunnerError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::DeviceRequest(e)
    }
}

impl From<wgpu::BufferAsyncError> for RunnerError {
    fn from(e: wgpu::BufferAsyncError) -> Self {
        Self::BufferMap(e)
    }
}

/// Helper: compute the number of workgroups needed to cover `total` items
/// with a given `group_size`.
pub fn div_ceil(total: u32, group_size: u32) -> u32 {
    assert!(group_size > 0, "group_size must be > 0");
    (total + group_size - 1) / group_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_adapter_not_found() {
        let e = RunnerError::AdapterNotFound;
        assert_eq!(e.to_string(), "no suitable wgpu adapter found");
    }

    #[test]
    fn error_display_shader_compilation() {
        let e = RunnerError::ShaderCompilation("syntax error at line 3".into());
        assert!(e.to_string().contains("syntax error at line 3"));
    }

    #[test]
    fn error_display_invalid_dimensions() {
        let e = RunnerError::InvalidDimensions { expected: 16, actual: 12, name: "matrix A" };
        let msg = e.to_string();
        assert!(msg.contains("matrix A"));
        assert!(msg.contains("16"));
        assert!(msg.contains("12"));
    }

    #[test]
    fn error_display_zero_workgroup() {
        let e = RunnerError::ZeroWorkgroup;
        assert!(e.to_string().contains("non-zero"));
    }

    #[test]
    fn error_source_adapter_is_none() {
        let e = RunnerError::AdapterNotFound;
        assert!(std::error::Error::source(&e).is_none());
    }

    #[test]
    fn div_ceil_exact() {
        assert_eq!(div_ceil(64, 8), 8);
    }

    #[test]
    fn div_ceil_remainder() {
        assert_eq!(div_ceil(65, 8), 9);
    }

    #[test]
    fn div_ceil_one() {
        assert_eq!(div_ceil(1, 256), 1);
    }

    #[test]
    fn div_ceil_zero_total() {
        assert_eq!(div_ceil(0, 8), 0);
    }

    #[test]
    #[should_panic(expected = "group_size must be > 0")]
    fn div_ceil_zero_group_panics() {
        div_ceil(10, 0);
    }
}
