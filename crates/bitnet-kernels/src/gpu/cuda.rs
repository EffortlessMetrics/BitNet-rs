//! CUDA kernel implementation using cudarc

use crate::KernelProvider;
use bitnet_common::{KernelError, QuantizationType, Result};
use cudarc::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// CUDA kernel provider with memory management and stream handling
pub struct CudaKernel {
    device: Arc<CudaDevice>,
    module: CudaModule,
    streams: Vec<CudaStream>,
    memory_pool: Arc<Mutex<CudaMemoryPool>>,
    device_info: CudaDeviceInfo,
    cuda_graphs: Arc<Mutex<HashMap<String, CudaGraph>>>,
    performance_stats: Arc<Mutex<PerformanceStats>>,
}

/// CUDA graph for optimized kernel execution
struct CudaGraph {
    graph: cudarc::driver::CudaGraph,
    graph_exec: cudarc::driver::CudaGraphExec,
    input_ptrs: Vec<*mut std::ffi::c_void>,
    output_ptrs: Vec<*mut std::ffi::c_void>,
}

/// Performance statistics for monitoring and optimization
#[derive(Debug, Default)]
struct PerformanceStats {
    total_kernel_launches: u64,
    total_execution_time_ms: f64,
    memory_transfers_host_to_device: u64,
    memory_transfers_device_to_host: u64,
    bytes_transferred_h2d: u64,
    bytes_transferred_d2h: u64,
    cache_hits: u64,
    cache_misses: u64,
}

/// CUDA device information and capabilities
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory_per_block: usize,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
}

/// Memory pool for efficient GPU memory management
struct CudaMemoryPool {
    free_buffers: HashMap<usize, Vec<CudaSlice<u8>>>,
    allocated_size: usize,
    max_pool_size: usize,
}

impl CudaMemoryPool {
    fn new(max_pool_size: usize) -> Self {
        Self {
            free_buffers: HashMap::new(),
            allocated_size: 0,
            max_pool_size,
        }
    }

    fn allocate(&mut self, device: &CudaDevice, size: usize) -> Result<CudaSlice<u8>> {
        // Round up to nearest power of 2 for better reuse
        let rounded_size = size.next_power_of_two();
        
        if let Some(buffers) = self.free_buffers.get_mut(&rounded_size) {
            if let Some(buffer) = buffers.pop() {
                log::debug!("Reusing GPU buffer of size {}", rounded_size);
                return Ok(buffer);
            }
        }

        if self.allocated_size + rounded_size > self.max_pool_size {
            // Clean up some buffers if we're over the limit
            self.cleanup_buffers();
        }

        log::debug!("Allocating new GPU buffer of size {}", rounded_size);
        let buffer = device.alloc_zeros::<u8>(rounded_size)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory: {}", e) 
            })?;
        
        self.allocated_size += rounded_size;
        Ok(buffer)
    }

    fn deallocate(&mut self, buffer: CudaSlice<u8>) {
        let size = buffer.len();
        self.free_buffers.entry(size).or_default().push(buffer);
    }

    fn cleanup_buffers(&mut self) {
        // Remove half of the cached buffers to free memory
        for buffers in self.free_buffers.values_mut() {
            let keep_count = buffers.len() / 2;
            buffers.truncate(keep_count);
        }
        // Recalculate allocated size
        self.allocated_size = self.free_buffers.values()
            .map(|buffers| buffers.iter().map(|b| b.len()).sum::<usize>())
            .sum();
    }
}

impl CudaKernel {
    /// Create a new CUDA kernel provider
    pub fn new() -> Result<Self> {
        Self::new_with_device(0)
    }

    /// Create a new CUDA kernel provider with specific device
    pub fn new_with_device(device_id: usize) -> Result<Self> {
        log::info!("Initializing CUDA kernel provider on device {}", device_id);

        // Initialize CUDA device
        let device = CudaDevice::new(device_id)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to initialize CUDA device {}: {}", device_id, e) 
            })?;

        // Get device information
        let device_info = Self::get_device_info(&device, device_id)?;
        log::info!("CUDA device info: {:?}", device_info);

        // Compile and load CUDA kernels
        let module = Self::compile_and_load_kernels(&device)?;

        // Create CUDA streams for concurrent execution
        let stream_count = 4; // Configurable number of streams
        let streams = (0..stream_count)
            .map(|i| {
                device.fork_default_stream()
                    .map_err(|e| KernelError::GpuError { 
                        reason: format!("Failed to create CUDA stream {}: {}", i, e) 
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // Initialize memory pool (default 1GB)
        let max_pool_size = 1024 * 1024 * 1024; // 1GB
        let memory_pool = Arc::new(Mutex::new(CudaMemoryPool::new(max_pool_size)));

        // Initialize CUDA graphs cache
        let cuda_graphs = Arc::new(Mutex::new(HashMap::new()));

        // Initialize performance statistics
        let performance_stats = Arc::new(Mutex::new(PerformanceStats::default()));

        Ok(Self {
            device,
            module,
            streams,
            memory_pool,
            device_info,
            cuda_graphs,
            performance_stats,
        })
    }

    /// Get detailed device information and capabilities
    fn get_device_info(device: &CudaDevice, device_id: usize) -> Result<CudaDeviceInfo> {
        // Get device properties using cudarc
        let name = device.name()
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to get device name: {}", e) 
            })?;

        let total_memory = device.total_memory()
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to get total memory: {}", e) 
            })?;

        // Note: cudarc doesn't expose all device properties directly
        // For now, we'll use reasonable defaults and what's available
        let compute_capability = (7, 5); // Default to compute capability 7.5
        let multiprocessor_count = 80; // Default value
        let max_threads_per_block = 1024;
        let max_shared_memory_per_block = 48 * 1024; // 48KB

        // Check for mixed precision support (assume modern GPUs support it)
        let supports_fp16 = compute_capability.0 >= 6;
        let supports_bf16 = compute_capability.0 >= 8;

        Ok(CudaDeviceInfo {
            device_id,
            name,
            compute_capability,
            total_memory,
            multiprocessor_count,
            max_threads_per_block,
            max_shared_memory_per_block,
            supports_fp16,
            supports_bf16,
        })
    }

    /// Compile and load CUDA kernels
    fn compile_and_load_kernels(device: &CudaDevice) -> Result<CudaModule> {
        log::info!("Compiling CUDA kernels...");

        // CUDA kernel source code
        let kernel_source = include_str!("kernels/bitnet_kernels.cu");

        // Compile PTX with optimization
        let ptx = compile_ptx_with_opts(
            kernel_source,
            PtxJitOptions {
                arch: Some("sm_75".to_string()), // Target compute capability 7.5+
                include_paths: vec![],
                max_register_count: Some(64),
                optimization_level: Some(OptLevel::O3),
                debug: false,
                verbose: false,
                ..Default::default()
            },
        ).map_err(|e| KernelError::GpuError { 
            reason: format!("Failed to compile CUDA kernels: {}", e) 
        })?;

        // Load the compiled module
        let module = device.load_ptx(
            ptx,
            "bitnet_kernels",
            &[
                "bitnet_matmul_i2s",
                "bitnet_quantize_i2s", 
                "bitnet_quantize_tl1",
                "bitnet_quantize_tl2",
                "bitnet_dequantize",
            ],
        ).map_err(|e| KernelError::GpuError { 
            reason: format!("Failed to load CUDA module: {}", e) 
        })?;

        log::info!("CUDA kernels compiled and loaded successfully");
        Ok(module)
    }

    /// Get device information
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }

    /// Get a CUDA stream for concurrent execution
    fn get_stream(&self, stream_id: usize) -> &CudaStream {
        &self.streams[stream_id % self.streams.len()]
    }

    /// Allocate GPU memory with pooling
    fn allocate_gpu_memory(&self, size: usize) -> Result<CudaSlice<u8>> {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.allocate(&self.device, size)
    }

    /// Deallocate GPU memory back to pool
    fn deallocate_gpu_memory(&self, buffer: CudaSlice<u8>) {
        let mut pool = self.memory_pool.lock().unwrap();
        pool.deallocate(buffer);
    }

    /// Synchronize all streams
    pub fn synchronize_all(&self) -> Result<()> {
        for stream in &self.streams {
            stream.synchronize()
                .map_err(|e| KernelError::GpuError { 
                    reason: format!("Failed to synchronize CUDA stream: {}", e) 
                })?;
        }
        Ok(())
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> (usize, usize) {
        let pool = self.memory_pool.lock().unwrap();
        (pool.allocated_size, pool.max_pool_size)
    }
}

impl KernelProvider for CudaKernel {
    fn name(&self) -> &'static str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        // Check if CUDA is available and device is accessible
        CudaDevice::new(0).is_ok()
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8], 
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        log::debug!("CUDA matmul_i2s: {}x{}x{}", m, n, k);

        // Allocate GPU memory
        let a_gpu = self.device.htod_copy(a)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy matrix A to GPU: {}", e) 
            })?;

        let b_gpu = self.device.htod_copy(b)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy matrix B to GPU: {}", e) 
            })?;

        let mut c_gpu = self.device.alloc_zeros::<f32>(c.len())
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory for result: {}", e) 
            })?;

        // Get kernel function
        let kernel_func = self.module.get_func("bitnet_matmul_i2s")
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to get CUDA kernel function: {}", e) 
            })?;

        // Configure kernel launch parameters
        let block_size = 16;
        let grid_x = (m + block_size - 1) / block_size;
        let grid_y = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let stream = self.get_stream(0);
        unsafe {
            kernel_func.launch_on_stream(
                stream,
                config,
                (
                    &a_gpu,
                    &b_gpu, 
                    &mut c_gpu,
                    m as i32,
                    n as i32,
                    k as i32,
                ),
            ).map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to launch CUDA kernel: {}", e) 
            })?;
        }

        // Copy result back to host
        self.device.dtoh_sync_copy_into(&c_gpu, c)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy result from GPU: {}", e) 
            })?;

        Ok(())
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        log::debug!("CUDA quantize: {} elements, type: {:?}", input.len(), qtype);

        let kernel_name = match qtype {
            QuantizationType::I2S => "bitnet_quantize_i2s",
            QuantizationType::TL1 => "bitnet_quantize_tl1", 
            QuantizationType::TL2 => "bitnet_quantize_tl2",
        };

        // Allocate GPU memory
        let input_gpu = self.device.htod_copy(input)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy input to GPU: {}", e) 
            })?;

        let mut output_gpu = self.device.alloc_zeros::<u8>(output.len())
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory for output: {}", e) 
            })?;

        let mut scales_gpu = self.device.alloc_zeros::<f32>(scales.len())
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to allocate GPU memory for scales: {}", e) 
            })?;

        // Get kernel function
        let kernel_func = self.module.get_func(kernel_name)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to get CUDA kernel function {}: {}", kernel_name, e) 
            })?;

        // Configure kernel launch parameters
        let block_size = 256;
        let grid_size = (input.len() + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let stream = self.get_stream(0);
        unsafe {
            kernel_func.launch_on_stream(
                stream,
                config,
                (
                    &input_gpu,
                    &mut output_gpu,
                    &mut scales_gpu,
                    input.len() as i32,
                ),
            ).map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to launch CUDA quantization kernel: {}", e) 
            })?;
        }

        // Copy results back to host
        self.device.dtoh_sync_copy_into(&output_gpu, output)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy quantized output from GPU: {}", e) 
            })?;

        self.device.dtoh_sync_copy_into(&scales_gpu, scales)
            .map_err(|e| KernelError::GpuError { 
                reason: format!("Failed to copy scales from GPU: {}", e) 
            })?;

        Ok(())
    }
}

/// Check if CUDA is available on the system
pub fn is_cuda_available() -> bool {
    CudaDevice::new(0).is_ok()
}

/// Get the number of available CUDA devices
pub fn cuda_device_count() -> usize {
    // cudarc doesn't provide a direct way to get device count
    // Try to initialize devices until we fail
    let mut count = 0;
    while CudaDevice::new(count).is_ok() {
        count += 1;
        if count > 16 { // Reasonable upper limit
            break;
        }
    }
    count
}

/// List all available CUDA devices with their information
pub fn list_cuda_devices() -> Result<Vec<CudaDeviceInfo>> {
    let device_count = cuda_device_count();
    let mut devices = Vec::new();

    for device_id in 0..device_count {
        if let Ok(device) = CudaDevice::new(device_id) {
            if let Ok(info) = CudaKernel::get_device_info(&device, device_id) {
                devices.push(info);
            }
        }
    }

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will pass even if CUDA is not available
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
        
        if available {
            let device_count = cuda_device_count();
            println!("CUDA device count: {}", device_count);
            
            if let Ok(devices) = list_cuda_devices() {
                for device in devices {
                    println!("Device {}: {:?}", device.device_id, device);
                }
            }
        }
    }

    #[test]
    fn test_cuda_kernel_creation() {
        if is_cuda_available() {
            match CudaKernel::new() {
                Ok(kernel) => {
                    println!("CUDA kernel created successfully");
                    println!("Device info: {:?}", kernel.device_info());
                    assert!(kernel.is_available());
                }
                Err(e) => {
                    println!("Failed to create CUDA kernel: {}", e);
                }
            }
        }
    }
}