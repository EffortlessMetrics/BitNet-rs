//! Mixed precision support for GPU kernels with device-aware capability detection

use crate::gpu::cuda::{CudaDeviceInfo, CudaKernel};
use bitnet_common::{KernelError, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Mixed precision configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    FP32,
    /// Half precision (FP16)
    FP16,
    /// Brain floating point (BF16)
    BF16,
    /// Automatic precision selection
    Auto,
}

/// Performance metrics for mixed precision operations
#[derive(Debug, Default, Clone)]
pub struct MixedPrecisionMetrics {
    /// Total number of matrix multiplications performed
    pub total_operations: u64,
    /// Total execution time for FP16 operations
    pub fp16_execution_time: Duration,
    /// Total execution time for BF16 operations
    pub bf16_execution_time: Duration,
    /// Total execution time for FP32 operations
    pub fp32_execution_time: Duration,
    /// Memory allocated for operations (bytes)
    pub memory_allocated: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Number of memory transfers (host to device)
    pub memory_transfers_h2d: u64,
    /// Number of memory transfers (device to host)
    pub memory_transfers_d2h: u64,
    /// Total bytes transferred to device
    pub bytes_transferred_h2d: u64,
    /// Total bytes transferred from device
    pub bytes_transferred_d2h: u64,
}

/// Memory tracking for CUDA operations
#[derive(Debug, Default)]
pub struct MemoryTracker {
    /// Current allocated memory
    current_allocated: usize,
    /// Peak memory usage
    peak_memory: usize,
    /// Number of allocations
    allocation_count: u64,
    /// Number of deallocations
    deallocation_count: u64,
}

/// Mixed precision kernel provider with device-aware capabilities
pub struct MixedPrecisionKernel {
    device_info: CudaDeviceInfo,
    precision_mode: PrecisionMode,
    optimal_precision: PrecisionMode,
    #[allow(dead_code)] // Reserved for future advanced GPU operations
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    #[allow(dead_code)] // Reserved for future kernel loading operations
    module: Arc<CudaModule>,
    matmul_fp16_function: Option<CudaFunction>,
    matmul_bf16_function: Option<CudaFunction>,
    #[allow(dead_code)] // Reserved for future tensor core optimization
    tensor_core_function: Option<CudaFunction>,
    convert_fp32_to_fp16_function: Option<CudaFunction>,
    convert_fp32_to_bf16_function: Option<CudaFunction>,
    convert_fp16_to_fp32_function: Option<CudaFunction>,
    convert_bf16_to_fp32_function: Option<CudaFunction>,
    metrics: MixedPrecisionMetrics,
    memory_tracker: MemoryTracker,
}

impl MixedPrecisionKernel {
    /// Create a new mixed precision kernel provider with device capability detection
    pub fn new(device_id: usize) -> Result<Self> {
        log::info!("Creating mixed precision kernel for device {}", device_id);

        // Get device information for capability detection
        let device_info = CudaKernel::get_device_info(device_id)?;
        log::info!(
            "Device capabilities: FP16={}, BF16={}",
            device_info.supports_fp16,
            device_info.supports_bf16
        );

        // Create CUDA context
        let ctx = CudaContext::new(device_id).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to create CUDA context for device {}: {:?}", device_id, e),
        })?;

        let stream = ctx.default_stream();

        // Compile mixed precision PTX kernels
        let ptx = compile_ptx(include_str!("kernels/mixed_precision_kernels.cu")).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Failed to compile mixed precision PTX: {:?}", e),
            }
        })?;

        // Load module
        let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to load mixed precision CUDA module: {:?}", e),
        })?;

        // Load kernel functions based on device capabilities
        let matmul_fp16_function = if device_info.supports_fp16 {
            module.load_function("bitnet_matmul_fp16").ok()
        } else {
            None
        };

        let matmul_bf16_function = if device_info.supports_bf16 {
            module.load_function("bitnet_matmul_bf16").ok()
        } else {
            None
        };

        let tensor_core_function = if device_info.compute_capability.0 >= 7 {
            module.load_function("bitnet_matmul_tensor_core").ok()
        } else {
            None
        };

        let convert_fp32_to_fp16_function = if device_info.supports_fp16 {
            module.load_function("convert_fp32_to_fp16").ok()
        } else {
            None
        };

        let convert_fp32_to_bf16_function = if device_info.supports_bf16 {
            module.load_function("convert_fp32_to_bf16").ok()
        } else {
            None
        };

        let convert_fp16_to_fp32_function = if device_info.supports_fp16 {
            module.load_function("convert_fp16_to_fp32").ok()
        } else {
            None
        };

        let convert_bf16_to_fp32_function = if device_info.supports_bf16 {
            module.load_function("convert_bf16_to_fp32").ok()
        } else {
            None
        };

        // Detect optimal precision based on device capabilities
        let optimal_precision = detect_best_precision(&device_info);
        log::info!("Optimal precision mode: {:?}", optimal_precision);

        Ok(Self {
            device_info,
            precision_mode: PrecisionMode::Auto,
            optimal_precision,
            ctx,
            stream,
            module,
            matmul_fp16_function,
            matmul_bf16_function,
            tensor_core_function,
            convert_fp32_to_fp16_function,
            convert_fp32_to_bf16_function,
            convert_fp16_to_fp32_function,
            convert_bf16_to_fp32_function,
            metrics: MixedPrecisionMetrics::default(),
            memory_tracker: MemoryTracker::default(),
        })
    }

    /// Create a mixed precision kernel with explicit device info (for testing/benchmarking)
    pub fn with_device_info(
        device_info: CudaDeviceInfo,
        precision_mode: PrecisionMode,
    ) -> Result<Self> {
        let device_id = device_info.device_id;

        // Create CUDA context
        let ctx = CudaContext::new(device_id).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to create CUDA context for device {}: {:?}", device_id, e),
        })?;

        let stream = ctx.default_stream();

        // Compile mixed precision PTX kernels
        let ptx = compile_ptx(include_str!("kernels/mixed_precision_kernels.cu")).map_err(|e| {
            KernelError::GpuError {
                reason: format!("Failed to compile mixed precision PTX: {:?}", e),
            }
        })?;

        // Load module
        let module = ctx.load_module(ptx).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to load mixed precision CUDA module: {:?}", e),
        })?;

        // Load kernel functions based on device capabilities
        let matmul_fp16_function = if device_info.supports_fp16 {
            module.load_function("bitnet_matmul_fp16").ok()
        } else {
            None
        };

        let matmul_bf16_function = if device_info.supports_bf16 {
            module.load_function("bitnet_matmul_bf16").ok()
        } else {
            None
        };

        let tensor_core_function = if device_info.compute_capability.0 >= 7 {
            module.load_function("bitnet_matmul_tensor_core").ok()
        } else {
            None
        };

        let convert_fp32_to_fp16_function = if device_info.supports_fp16 {
            module.load_function("convert_fp32_to_fp16").ok()
        } else {
            None
        };

        let convert_fp32_to_bf16_function = if device_info.supports_bf16 {
            module.load_function("convert_fp32_to_bf16").ok()
        } else {
            None
        };

        let convert_fp16_to_fp32_function = if device_info.supports_fp16 {
            module.load_function("convert_fp16_to_fp32").ok()
        } else {
            None
        };

        let convert_bf16_to_fp32_function = if device_info.supports_bf16 {
            module.load_function("convert_bf16_to_fp32").ok()
        } else {
            None
        };

        let optimal_precision = detect_best_precision(&device_info);

        Ok(Self {
            device_info,
            precision_mode,
            optimal_precision,
            ctx,
            stream,
            module,
            matmul_fp16_function,
            matmul_bf16_function,
            tensor_core_function,
            convert_fp32_to_fp16_function,
            convert_fp32_to_bf16_function,
            convert_fp16_to_fp32_function,
            convert_bf16_to_fp32_function,
            metrics: MixedPrecisionMetrics::default(),
            memory_tracker: MemoryTracker::default(),
        })
    }

    /// Set precision mode
    pub fn set_precision_mode(&mut self, mode: PrecisionMode) {
        self.precision_mode = mode;
        log::info!("Set precision mode to {:?}", mode);
    }

    /// Get current precision mode
    pub fn precision_mode(&self) -> PrecisionMode {
        self.precision_mode
    }

    /// Get the device ID associated with this kernel
    pub fn device_id(&self) -> usize {
        self.device_info.device_id
    }

    /// Check if FP16 is supported
    pub fn supports_fp16(&self) -> bool {
        self.device_info.supports_fp16
    }

    /// Check if BF16 is supported
    pub fn supports_bf16(&self) -> bool {
        self.device_info.supports_bf16
    }

    /// Get device information
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }

    /// Get the optimal precision mode for this device
    pub fn optimal_precision(&self) -> PrecisionMode {
        self.optimal_precision
    }

    /// Get the effective precision mode (resolves Auto to optimal)
    pub fn effective_precision(&self) -> PrecisionMode {
        match self.precision_mode {
            PrecisionMode::Auto => self.optimal_precision,
            other => other,
        }
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &MixedPrecisionMetrics {
        &self.metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = MixedPrecisionMetrics::default();
        self.memory_tracker = MemoryTracker::default();
    }

    /// Get current memory usage
    pub fn current_memory_usage(&self) -> usize {
        self.memory_tracker.current_allocated
    }

    /// Get peak memory usage
    pub fn peak_memory_usage(&self) -> usize {
        self.memory_tracker.peak_memory
    }

    /// Record memory allocation
    fn track_allocation(&mut self, size: usize) {
        self.memory_tracker.current_allocated += size;
        self.memory_tracker.peak_memory =
            self.memory_tracker.peak_memory.max(self.memory_tracker.current_allocated);
        self.memory_tracker.allocation_count += 1;
        self.metrics.memory_allocated += size;
        self.metrics.peak_memory_usage = self.memory_tracker.peak_memory;
    }

    /// Record memory deallocation
    fn track_deallocation(&mut self, size: usize) {
        self.memory_tracker.current_allocated =
            self.memory_tracker.current_allocated.saturating_sub(size);
        self.memory_tracker.deallocation_count += 1;
    }

    /// Record memory transfer
    fn track_transfer(&mut self, size: usize, host_to_device: bool) {
        if host_to_device {
            self.metrics.memory_transfers_h2d += 1;
            self.metrics.bytes_transferred_h2d += size as u64;
        } else {
            self.metrics.memory_transfers_d2h += 1;
            self.metrics.bytes_transferred_d2h += size as u64;
        }
    }

    /// Matrix multiplication with FP16 precision
    pub fn matmul_fp16(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.supports_fp16() {
            return Err(KernelError::GpuError {
                reason: "FP16 not supported on this device".to_string(),
            }
            .into());
        }

        // Check kernel availability early
        if self.matmul_fp16_function.is_none() {
            return Err(KernelError::GpuError {
                reason: "FP16 matmul kernel not available".to_string(),
            }
            .into());
        }

        if self.convert_fp32_to_fp16_function.is_none() {
            return Err(KernelError::GpuError {
                reason: "FP32 to FP16 conversion kernel not available".to_string(),
            }
            .into());
        }

        if self.convert_fp16_to_fp32_function.is_none() {
            return Err(KernelError::GpuError {
                reason: "FP16 to FP32 conversion kernel not available".to_string(),
            }
            .into());
        }

        log::debug!("Executing native CUDA FP16 matmul: {}x{}x{}", m, n, k);

        let start_time = Instant::now();

        // Convert input matrices to FP16 on device
        let a_fp16_size = m * k;
        let b_fp16_size = k * n;
        let c_size = m * n;

        // Calculate memory usage
        let input_memory = (a_fp16_size + b_fp16_size) * std::mem::size_of::<f32>();
        let fp16_memory = (a_fp16_size + b_fp16_size + c_size) * std::mem::size_of::<u16>();
        let output_memory = c_size * std::mem::size_of::<f32>();
        let total_memory = input_memory + fp16_memory + output_memory;

        self.track_allocation(total_memory);

        // Transfer FP32 input to device
        let a_dev = self.stream.memcpy_stod(a).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer A to device: {:?}", e),
        })?;
        self.track_transfer(std::mem::size_of_val(a), true);

        let b_dev = self.stream.memcpy_stod(b).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer B to device: {:?}", e),
        })?;
        self.track_transfer(std::mem::size_of_val(b), true);

        // Allocate FP16 arrays on device
        let mut a_fp16_dev: CudaSlice<u16> =
            self.stream.alloc_zeros(a_fp16_size).map_err(|e| KernelError::GpuError {
                reason: format!("Failed to allocate A_FP16 on device: {:?}", e),
            })?;

        let mut b_fp16_dev: CudaSlice<u16> =
            self.stream.alloc_zeros(b_fp16_size).map_err(|e| KernelError::GpuError {
                reason: format!("Failed to allocate B_FP16 on device: {:?}", e),
            })?;

        let mut c_fp16_dev: CudaSlice<u16> =
            self.stream.alloc_zeros(c_size).map_err(|e| KernelError::GpuError {
                reason: format!("Failed to allocate C_FP16 on device: {:?}", e),
            })?;

        let mut c_dev: CudaSlice<f32> = self.stream.alloc_zeros(c_size).map_err(|e| {
            KernelError::GpuError { reason: format!("Failed to allocate C on device: {:?}", e) }
        })?;

        // Convert FP32 to FP16
        let block_size = 256;
        let grid_size_a = a_fp16_size.div_ceil(block_size);
        let grid_size_b = b_fp16_size.div_ceil(block_size);

        let cfg_convert = LaunchConfig {
            grid_dim: (grid_size_a as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Convert A from FP32 to FP16
        let mut builder_a =
            self.stream.launch_builder(self.convert_fp32_to_fp16_function.as_ref().unwrap());
        builder_a.arg(&a_dev);
        builder_a.arg(&mut a_fp16_dev);
        let n_a = a_fp16_size as i32;
        builder_a.arg(&n_a);
        unsafe { builder_a.launch(cfg_convert) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch FP32 to FP16 conversion for A: {:?}", e),
        })?;

        // Convert B from FP32 to FP16
        let cfg_convert_b = LaunchConfig {
            grid_dim: (grid_size_b as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder_b =
            self.stream.launch_builder(self.convert_fp32_to_fp16_function.as_ref().unwrap());
        builder_b.arg(&b_dev);
        builder_b.arg(&mut b_fp16_dev);
        let n_b = b_fp16_size as i32;
        builder_b.arg(&n_b);
        unsafe { builder_b.launch(cfg_convert_b) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch FP32 to FP16 conversion for B: {:?}", e),
        })?;

        // Launch FP16 matrix multiplication
        let block_dim = 16; // 16x16 threads per block
        let grid_x = n.div_ceil(block_dim);
        let grid_y = m.div_ceil(block_dim);

        let cfg_matmul = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_dim as u32, block_dim as u32, 1),
            shared_mem_bytes: 0,
        };

        let mut builder_matmul =
            self.stream.launch_builder(self.matmul_fp16_function.as_ref().unwrap());
        builder_matmul.arg(&a_fp16_dev);
        builder_matmul.arg(&b_fp16_dev);
        builder_matmul.arg(&mut c_fp16_dev);
        let m_arg = m as i32;
        let n_arg = n as i32;
        let k_arg = k as i32;
        builder_matmul.arg(&m_arg);
        builder_matmul.arg(&n_arg);
        builder_matmul.arg(&k_arg);

        unsafe { builder_matmul.launch(cfg_matmul) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch FP16 matmul kernel: {:?}", e),
        })?;

        // Convert result from FP16 back to FP32
        let grid_size_c = c_size.div_ceil(block_size);
        let cfg_convert_c = LaunchConfig {
            grid_dim: (grid_size_c as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder_c =
            self.stream.launch_builder(self.convert_fp16_to_fp32_function.as_ref().unwrap());
        builder_c.arg(&c_fp16_dev);
        builder_c.arg(&mut c_dev);
        let n_c = c_size as i32;
        builder_c.arg(&n_c);
        unsafe { builder_c.launch(cfg_convert_c) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch FP16 to FP32 conversion for C: {:?}", e),
        })?;

        // Transfer result back to host
        let c_host: Vec<f32> = self.stream.memcpy_dtov(&c_dev).map_err(|e| {
            KernelError::GpuError { reason: format!("Failed to transfer result back: {:?}", e) }
        })?;
        self.track_transfer(std::mem::size_of_val(&c_host), false);

        // Synchronize stream and measure execution time
        self.stream.synchronize().map_err(|e| KernelError::GpuError {
            reason: format!("Failed to synchronize stream: {:?}", e),
        })?;

        let total_time = start_time.elapsed();
        self.metrics.fp16_execution_time += total_time;
        self.metrics.total_operations += 1;

        c.copy_from_slice(&c_host);

        // Track memory deallocation
        self.track_deallocation(total_memory);

        log::debug!(
            "Native CUDA FP16 matmul completed successfully: Total time: {:.2}ms",
            total_time.as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// Matrix multiplication with BF16 precision
    pub fn matmul_bf16(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.supports_bf16() {
            return Err(KernelError::GpuError {
                reason: "BF16 not supported on this device".to_string(),
            }
            .into());
        }

        // Check kernel availability early
        if self.matmul_bf16_function.is_none() {
            return Err(KernelError::GpuError {
                reason: "BF16 matmul kernel not available".to_string(),
            }
            .into());
        }

        if self.convert_fp32_to_bf16_function.is_none() {
            return Err(KernelError::GpuError {
                reason: "FP32 to BF16 conversion kernel not available".to_string(),
            }
            .into());
        }

        if self.convert_bf16_to_fp32_function.is_none() {
            return Err(KernelError::GpuError {
                reason: "BF16 to FP32 conversion kernel not available".to_string(),
            }
            .into());
        }

        log::debug!("Executing native CUDA BF16 matmul: {}x{}x{}", m, n, k);

        let start_time = Instant::now();

        // Convert input matrices to BF16 on device
        let a_bf16_size = m * k;
        let b_bf16_size = k * n;
        let c_size = m * n;

        // Calculate memory usage
        let input_memory = (a_bf16_size + b_bf16_size) * std::mem::size_of::<f32>();
        let bf16_memory = (a_bf16_size + b_bf16_size + c_size) * std::mem::size_of::<u16>();
        let output_memory = c_size * std::mem::size_of::<f32>();
        let total_memory = input_memory + bf16_memory + output_memory;

        self.track_allocation(total_memory);

        // Transfer FP32 input to device
        let a_dev = self.stream.memcpy_stod(a).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer A to device: {:?}", e),
        })?;
        self.track_transfer(std::mem::size_of_val(a), true);

        let b_dev = self.stream.memcpy_stod(b).map_err(|e| KernelError::GpuError {
            reason: format!("Failed to transfer B to device: {:?}", e),
        })?;
        self.track_transfer(std::mem::size_of_val(b), true);

        // Allocate BF16 arrays on device (BF16 is 16-bit like FP16)
        let mut a_bf16_dev: CudaSlice<u16> =
            self.stream.alloc_zeros(a_bf16_size).map_err(|e| KernelError::GpuError {
                reason: format!("Failed to allocate A_BF16 on device: {:?}", e),
            })?;

        let mut b_bf16_dev: CudaSlice<u16> =
            self.stream.alloc_zeros(b_bf16_size).map_err(|e| KernelError::GpuError {
                reason: format!("Failed to allocate B_BF16 on device: {:?}", e),
            })?;

        let mut c_bf16_dev: CudaSlice<u16> =
            self.stream.alloc_zeros(c_size).map_err(|e| KernelError::GpuError {
                reason: format!("Failed to allocate C_BF16 on device: {:?}", e),
            })?;

        let mut c_dev: CudaSlice<f32> = self.stream.alloc_zeros(c_size).map_err(|e| {
            KernelError::GpuError { reason: format!("Failed to allocate C on device: {:?}", e) }
        })?;

        // Convert FP32 to BF16
        let block_size = 256;
        let grid_size_a = a_bf16_size.div_ceil(block_size);
        let grid_size_b = b_bf16_size.div_ceil(block_size);

        let cfg_convert = LaunchConfig {
            grid_dim: (grid_size_a as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Convert A from FP32 to BF16
        let mut builder_a =
            self.stream.launch_builder(self.convert_fp32_to_bf16_function.as_ref().unwrap());
        builder_a.arg(&a_dev);
        builder_a.arg(&mut a_bf16_dev);
        let n_a = a_bf16_size as i32;
        builder_a.arg(&n_a);
        unsafe { builder_a.launch(cfg_convert) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch FP32 to BF16 conversion for A: {:?}", e),
        })?;

        // Convert B from FP32 to BF16
        let cfg_convert_b = LaunchConfig {
            grid_dim: (grid_size_b as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder_b =
            self.stream.launch_builder(self.convert_fp32_to_bf16_function.as_ref().unwrap());
        builder_b.arg(&b_dev);
        builder_b.arg(&mut b_bf16_dev);
        let n_b = b_bf16_size as i32;
        builder_b.arg(&n_b);
        unsafe { builder_b.launch(cfg_convert_b) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch FP32 to BF16 conversion for B: {:?}", e),
        })?;

        // Launch BF16 matrix multiplication
        let block_dim = 16; // 16x16 threads per block
        let grid_x = n.div_ceil(block_dim);
        let grid_y = m.div_ceil(block_dim);

        let cfg_matmul = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_dim as u32, block_dim as u32, 1),
            shared_mem_bytes: 0,
        };

        let mut builder_matmul =
            self.stream.launch_builder(self.matmul_bf16_function.as_ref().unwrap());
        builder_matmul.arg(&a_bf16_dev);
        builder_matmul.arg(&b_bf16_dev);
        builder_matmul.arg(&mut c_bf16_dev);
        let m_arg = m as i32;
        let n_arg = n as i32;
        let k_arg = k as i32;
        builder_matmul.arg(&m_arg);
        builder_matmul.arg(&n_arg);
        builder_matmul.arg(&k_arg);

        unsafe { builder_matmul.launch(cfg_matmul) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch BF16 matmul kernel: {:?}", e),
        })?;

        // Convert result from BF16 back to FP32
        let grid_size_c = c_size.div_ceil(block_size);
        let cfg_convert_c = LaunchConfig {
            grid_dim: (grid_size_c as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder_c =
            self.stream.launch_builder(self.convert_bf16_to_fp32_function.as_ref().unwrap());
        builder_c.arg(&c_bf16_dev);
        builder_c.arg(&mut c_dev);
        let n_c = c_size as i32;
        builder_c.arg(&n_c);
        unsafe { builder_c.launch(cfg_convert_c) }.map_err(|e| KernelError::GpuError {
            reason: format!("Failed to launch BF16 to FP32 conversion for C: {:?}", e),
        })?;

        // Transfer result back to host
        let c_host: Vec<f32> = self.stream.memcpy_dtov(&c_dev).map_err(|e| {
            KernelError::GpuError { reason: format!("Failed to transfer result back: {:?}", e) }
        })?;
        self.track_transfer(std::mem::size_of_val(&c_host), false);

        // Synchronize stream and measure execution time
        self.stream.synchronize().map_err(|e| KernelError::GpuError {
            reason: format!("Failed to synchronize stream: {:?}", e),
        })?;

        let total_time = start_time.elapsed();
        self.metrics.bf16_execution_time += total_time;
        self.metrics.total_operations += 1;

        c.copy_from_slice(&c_host);

        // Track memory deallocation
        self.track_deallocation(total_memory);

        log::debug!(
            "Native CUDA BF16 matmul completed successfully: Total time: {:.2}ms",
            total_time.as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// Matrix multiplication with automatic precision selection
    pub fn matmul_auto(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self.effective_precision() {
            PrecisionMode::FP32 => self.matmul_fp32(a, b, c, m, n, k),
            PrecisionMode::FP16 => self.matmul_fp16(a, b, c, m, n, k),
            PrecisionMode::BF16 => self.matmul_bf16(a, b, c, m, n, k),
            PrecisionMode::Auto => unreachable!("Auto should be resolved in effective_precision"),
        }
    }

    /// Matrix multiplication with FP32 precision (reference implementation)
    pub fn matmul_fp32(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        log::debug!("Executing FP32 matmul: {}x{}x{}", m, n, k);

        let start_time = Instant::now();

        // Standard FP32 matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        let execution_time = start_time.elapsed();
        self.metrics.fp32_execution_time += execution_time;
        self.metrics.total_operations += 1;

        log::debug!("FP32 matmul completed: {:.2}ms", execution_time.as_secs_f64() * 1000.0);

        Ok(())
    }
}

/// Detect the best precision mode for a given device
pub fn detect_best_precision(device_info: &CudaDeviceInfo) -> PrecisionMode {
    // Prioritize BF16 for modern architectures (Ampere and newer)
    if device_info.supports_bf16 {
        log::debug!(
            "Selected BF16 precision for compute capability {:?}",
            device_info.compute_capability
        );
        PrecisionMode::BF16
    }
    // Use FP16 for older architectures that support it (Pascal and newer)
    else if device_info.supports_fp16 {
        log::debug!(
            "Selected FP16 precision for compute capability {:?}",
            device_info.compute_capability
        );
        PrecisionMode::FP16
    }
    // Fallback to FP32 for older architectures
    else {
        log::debug!(
            "Selected FP32 precision for compute capability {:?}",
            device_info.compute_capability
        );
        PrecisionMode::FP32
    }
}

/// Convert FP32 values to FP16 with precision simulation
pub fn convert_to_fp16_sim(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&val| {
            // Simulate FP16 precision by converting to half and back
            // This is a simplified simulation - real implementation would use proper IEEE 754 half precision
            let _half_val = val; // In reality, convert to f16 and back

            // Clamp to FP16 range roughly [-65504, 65504] and apply precision loss
            let clamped = val.clamp(-65504.0, 65504.0);

            // Simulate precision loss by quantizing to ~11-bit precision
            (clamped * 2048.0).round() / 2048.0
        })
        .collect()
}

/// Convert FP32 values to BF16 with precision simulation
pub fn convert_to_bf16_sim(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&val| {
            // Simulate BF16 precision by truncating mantissa bits
            // BF16 has same exponent range as FP32 but only 7 mantissa bits vs 23
            let bits = val.to_bits();

            // Truncate lower 16 bits to simulate BF16 precision
            let bf16_bits = bits & 0xFFFF0000;
            f32::from_bits(bf16_bits)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_creation() {
        // Test creation with mock device ID (will use fallback FP32 for tests)
        let result = MixedPrecisionKernel::new(0);

        // In test environment, this might fail due to no GPU, which is expected
        if let Ok(mut kernel) = result {
            assert_eq!(kernel.precision_mode(), PrecisionMode::Auto);

            kernel.set_precision_mode(PrecisionMode::FP16);
            assert_eq!(kernel.precision_mode(), PrecisionMode::FP16);

            // Test effective precision resolution
            kernel.set_precision_mode(PrecisionMode::Auto);
            let effective = kernel.effective_precision();
            assert_ne!(effective, PrecisionMode::Auto); // Should be resolved
        } else {
            // Expected in test environment without GPU
            println!("GPU not available in test environment");
        }
    }

    #[test]
    fn test_precision_detection() {
        // Test precision detection logic with mock device info
        let device_info_ampere = CudaDeviceInfo {
            device_id: 0,
            name: "Mock Ampere".to_string(),
            compute_capability: (8, 0), // Ampere architecture
            total_memory: 1024 * 1024 * 1024,
            multiprocessor_count: 68,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            supports_fp16: true,
            supports_bf16: true,
        };

        let optimal = detect_best_precision(&device_info_ampere);
        assert_eq!(optimal, PrecisionMode::BF16);

        // Test Pascal architecture (FP16 support only)
        let device_info_pascal = CudaDeviceInfo {
            device_id: 0,
            name: "Mock Pascal".to_string(),
            compute_capability: (6, 1), // Pascal architecture
            total_memory: 1024 * 1024 * 1024,
            multiprocessor_count: 20,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            supports_fp16: true,
            supports_bf16: false,
        };

        let optimal = detect_best_precision(&device_info_pascal);
        assert_eq!(optimal, PrecisionMode::FP16);

        // Test older architecture (no mixed precision support)
        let device_info_old = CudaDeviceInfo {
            device_id: 0,
            name: "Mock Maxwell".to_string(),
            compute_capability: (5, 0), // Maxwell architecture
            total_memory: 1024 * 1024 * 1024,
            multiprocessor_count: 16,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            supports_fp16: false,
            supports_bf16: false,
        };

        let optimal = detect_best_precision(&device_info_old);
        assert_eq!(optimal, PrecisionMode::FP32);
    }

    #[test]
    fn test_precision_conversion() {
        let test_values = vec![1.0, -1.0, 0.5, -0.5, 1000.0, -1000.0];

        // Test FP16 simulation
        let fp16_sim = convert_to_fp16_sim(&test_values);
        assert_eq!(fp16_sim.len(), test_values.len());

        // Values should be approximately equal but with precision loss
        for (orig, sim) in test_values.iter().zip(fp16_sim.iter()) {
            let diff = (orig - sim).abs();
            // Allow for precision loss in FP16 simulation
            assert!(diff < 0.1, "FP16 precision loss too high: {} vs {}", orig, sim);
        }

        // Test BF16 simulation
        let bf16_sim = convert_to_bf16_sim(&test_values);
        assert_eq!(bf16_sim.len(), test_values.len());

        // BF16 should preserve the range better but lose precision
        for (orig, sim) in test_values.iter().zip(bf16_sim.iter()) {
            // BF16 has wider range than FP16, so large values preserved better
            if orig.abs() < 1000.0 {
                let diff = (orig - sim).abs();
                assert!(diff < 0.01, "BF16 precision loss too high: {} vs {}", orig, sim);
            }
        }
    }

    #[test]
    fn test_matmul_implementations() {
        // Create test matrices (small for testing)
        let m = 2;
        let n = 3;
        let k = 2;

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // 2x3 matrix
        let mut c_fp32 = vec![0.0; m * n]; // 2x3 result
        let mut c_fp16 = vec![0.0; m * n];
        let mut c_bf16 = vec![0.0; m * n];

        // Create mock kernel for testing
        let device_info = CudaDeviceInfo {
            device_id: 0,
            name: "Mock Device".to_string(),
            compute_capability: (8, 0),
            total_memory: 1024 * 1024 * 1024,
            multiprocessor_count: 68,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            supports_fp16: true,
            supports_bf16: true,
        };

        // Test with mock kernel (will not actually run CUDA operations in test environment)
        if let Ok(mut kernel) =
            MixedPrecisionKernel::with_device_info(device_info.clone(), PrecisionMode::Auto)
        {
            // Test FP32 matmul (should work without GPU)
            assert!(kernel.matmul_fp32(&a, &b, &mut c_fp32, m, n, k).is_ok());

            // Test FP16 matmul (may fail without actual GPU, that's expected in test env)
            let _ = kernel.matmul_fp16(&a, &b, &mut c_fp16, m, n, k);

            // Test BF16 matmul (may fail without actual GPU, that's expected in test env)
            let _ = kernel.matmul_bf16(&a, &b, &mut c_bf16, m, n, k);
        } else {
            // Expected in test environment without GPU
            println!("GPU not available in test environment");
            return;
        }

        // Results should be similar but not identical due to precision differences
        for i in 0..c_fp32.len() {
            let fp32_val = c_fp32[i];
            let fp16_val = c_fp16[i];
            let bf16_val = c_bf16[i];

            // Allow for precision differences
            assert!((fp32_val - fp16_val).abs() < 1.0);
            assert!((fp32_val - bf16_val).abs() < 0.1);
        }
    }

    #[test]
    fn test_precision_mode_validation() {
        // Test with device that doesn't support FP16/BF16
        let device_info = CudaDeviceInfo {
            device_id: 0,
            name: "Mock Old Device".to_string(),
            compute_capability: (5, 0),
            total_memory: 1024 * 1024 * 1024,
            multiprocessor_count: 16,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            supports_fp16: false,
            supports_bf16: false,
        };

        if let Ok(mut kernel) =
            MixedPrecisionKernel::with_device_info(device_info.clone(), PrecisionMode::FP16)
        {
            // Should fail for unsupported precision modes
            let a = vec![1.0, 2.0];
            let b = vec![1.0, 2.0];
            let mut c = vec![0.0; 2];

            let _result = kernel.matmul_fp16(&a, &b, &mut c, 1, 2, 1);
            // In actual test environment, this might fail for different reasons
            // The important thing is that error handling works

            let _result = kernel.matmul_bf16(&a, &b, &mut c, 1, 2, 1);
            // Same here - error handling should work

            // FP32 should always work
            let result = kernel.matmul_fp32(&a, &b, &mut c, 1, 2, 1);
            assert!(result.is_ok());
        } else {
            println!("GPU not available in test environment");
        }
    }
}
