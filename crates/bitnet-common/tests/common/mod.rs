// Test utilities and skip macros
//
// This module provides macros for conditional test skipping based on
// infrastructure requirements (environment variables, GPU availability, etc.)

/// Skip test unless environment variable is set and non-empty
#[macro_export]
macro_rules! skip_unless_env {
    ($name:expr) => {{
        if std::env::var($name)
            .ok()
            .filter(|v| !v.is_empty())
            .is_none()
        {
            eprintln!("skip: missing env {}", $name);
            return;
        }
    }};
}

/// Skip test unless GPU device is available at runtime
#[macro_export]
macro_rules! skip_unless_gpu {
    () => {{
        #[cfg(any(feature = "gpu", feature = "cuda"))]
        {
            if !bitnet_kernels::device_features::gpu_available_runtime() {
                eprintln!("skip: no GPU device on runner");
                return;
            }
        }
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
        {
            eprintln!("skip: GPU features not compiled");
            return;
        }
    }};
}

/// Skip test unless network connectivity is available
#[macro_export]
macro_rules! skip_unless_network {
    () => {{
        // Simple network check - try to resolve a common hostname
        if std::net::TcpStream::connect("dns.google:53").is_err() {
            eprintln!("skip: no network connectivity");
            return;
        }
    }};
}

/// Skip test if it's a slow test and BITNET_SKIP_SLOW_TESTS is set
#[macro_export]
macro_rules! skip_if_slow_tests_disabled {
    () => {{
        if std::env::var("BITNET_SKIP_SLOW_TESTS")
            .ok()
            .filter(|v| v == "1" || v.to_lowercase() == "true")
            .is_some()
        {
            eprintln!("skip: slow test disabled by BITNET_SKIP_SLOW_TESTS=1");
            return;
        }
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_skip_macros_compile() {
        // Just verify the macros compile
    }
}
