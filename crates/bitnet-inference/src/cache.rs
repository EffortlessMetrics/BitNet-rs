//! KV cache implementation for efficient inference

use crate::Backend;
use bitnet_common::{BitNetConfig, BitNetError, BitNetTensor, Result};
use candle_core::{Device, DType, Tensor as CandleTensor};
use std::collections::HashMap;

/// KV cache for storing key-value pairs during inference
pub struct KVCache {
    /// Cache for each layer
    layer_caches: Vec<LayerCache>,
    /// Maximum sequence length
    max_length: usize,
    /// Current sequence length
    current_length: usize,
    /// Device for tensor operations
    device: Device,
    /// Memory pool for efficient allocation
    memory_pool: Option<MemoryPool>,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(config: &BitNetConfig, max_length: usize) -> Result<Self> {
        let device = Device::Cpu; // Default to CPU, can be overridden
        let num_layers = config.model.num_layers;
        let num_heads = config.model.num_heads;
        let head_dim = config.model.hidden_size / config.model.num_heads;
        
        let layer_caches = (0..num_layers)
            .map(|_| LayerCache::new(max_length, num_heads, head_dim, &device))
            .collect::<Result<Vec<_>>>()?;
        
        Ok(Self {
            layer_caches,
            max_length,
            current_length: 0,
            device,
            memory_pool: None,
        })
    }
    
    /// Create KV cache with memory pooling
    pub fn with_memory_pool(
        config: &BitNetConfig, 
        max_length: usize,
        pool_size_mb: usize,
    ) -> Result<Self> {
        let mut cache = Self::new(config, max_length)?;
        cache.memory_pool = Some(MemoryPool::new(pool_size_mb)?);
        Ok(cache)
    }
    
    /// Get cache for a specific layer
    pub fn get_layer_cache(&mut self, layer_idx: usize) -> Result<&mut LayerCache> {
        self.layer_caches
            .get_mut(layer_idx)
            .ok_or_else(|| BitNetError::Validation(
                format!("Layer index {} out of bounds", layer_idx)
            ))
    }
    
    /// Update cache for a layer
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        key: BitNetTensor,
        value: BitNetTensor,
        position: usize,
    ) -> Result<()> {
        let layer_cache = self.get_layer_cache(layer_idx)?;
        layer_cache.update(key, value, position)?;
        
        // Update current length if this is the last layer
        if layer_idx == self.layer_caches.len() - 1 {
            self.current_length = self.current_length.max(position + 1);
        }
        
        Ok(())
    }
    
    /// Get cached key-value pairs for a layer
    pub fn get_layer_kv(&self, layer_idx: usize) -> Result<(&BitNetTensor, &BitNetTensor)> {
        let layer_cache = self.layer_caches
            .get(layer_idx)
            .ok_or_else(|| BitNetError::Validation(
                format!("Layer index {} out of bounds", layer_idx)
            ))?;
        
        Ok((&layer_cache.key_cache, &layer_cache.value_cache))
    }
    
    /// Reset the cache
    pub fn reset(&mut self) {
        for layer_cache in &mut self.layer_caches {
            layer_cache.reset();
        }
        self.current_length = 0;
    }
    
    /// Get current sequence length
    pub fn current_length(&self) -> usize {
        self.current_length
    }
    
    /// Get maximum sequence length
    pub fn max_length(&self) -> usize {
        self.max_length
    }
    
    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.current_length >= self.max_length
    }
    
    /// Resize cache to new maximum length
    pub fn resize(&mut self, new_max_length: usize) -> Result<()> {
        if new_max_length < self.current_length {
            return Err(BitNetError::Validation(
                "Cannot resize cache to smaller than current length".to_string()
            ));
        }
        
        for layer_cache in &mut self.layer_caches {
            layer_cache.resize(new_max_length, &self.device)?;
        }
        
        self.max_length = new_max_length;
        Ok(())
    }
    
    /// Migrate cache to a different backend
    pub fn migrate_to_backend(&mut self, backend: &dyn Backend) -> Result<()> {
        let device_info = backend.device_info();
        let new_device = match device_info.device_type {
            crate::backend::DeviceType::Cpu => Device::Cpu,
            crate::backend::DeviceType::Cuda(id) => Device::new_cuda(id)
                .map_err(|e| BitNetError::Validation(e.to_string()))?,
            crate::backend::DeviceType::Metal => Device::Metal(candle_core::MetalDevice::new(0)
                .map_err(|e| BitNetError::Validation(e.to_string()))?),
        };
        
        // Migrate all layer caches to new device
        for layer_cache in &mut self.layer_caches {
            layer_cache.migrate_to_device(&new_device)?;
        }
        
        self.device = new_device;
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn memory_usage(&self) -> CacheMemoryStats {
        let mut total_bytes = 0;
        let mut allocated_bytes = 0;
        
        for layer_cache in &self.layer_caches {
            let stats = layer_cache.memory_usage();
            total_bytes += stats.total_bytes;
            allocated_bytes += stats.allocated_bytes;
        }
        
        CacheMemoryStats {
            total_bytes,
            allocated_bytes,
            utilization: if total_bytes > 0 {
                allocated_bytes as f64 / total_bytes as f64
            } else {
                0.0
            },
            num_layers: self.layer_caches.len(),
        }
    }
}

/// Cache for a single transformer layer
pub struct LayerCache {
    /// Cached keys
    pub key_cache: BitNetTensor,
    /// Cached values
    pub value_cache: BitNetTensor,
    /// Current position in cache
    position: usize,
    /// Maximum length
    max_length: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
}

impl LayerCache {
    /// Create a new layer cache
    pub fn new(
        max_length: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let key_shape = [max_length, num_heads, head_dim];
        let value_shape = [max_length, num_heads, head_dim];
        
        let key_cache = BitNetTensor::zeros(&key_shape, DType::F32, device)?;
        let value_cache = BitNetTensor::zeros(&value_shape, DType::F32, device)?;
        
        Ok(Self {
            key_cache,
            value_cache,
            position: 0,
            max_length,
            num_heads,
            head_dim,
        })
    }
    
    /// Update cache with new key-value pair
    pub fn update(
        &mut self,
        key: BitNetTensor,
        value: BitNetTensor,
        position: usize,
    ) -> Result<()> {
        if position >= self.max_length {
            return Err(BitNetError::Validation(
                format!("Position {} exceeds cache length {}", position, self.max_length)
            ));
        }
        
        // In a real implementation, we would copy the key/value tensors
        // into the appropriate positions in the cache tensors
        // For now, this is a placeholder
        self.position = position;
        
        Ok(())
    }
    
    /// Reset the cache
    pub fn reset(&mut self) {
        self.position = 0;
        // In a real implementation, we would zero out the cache tensors
    }
    
    /// Resize the cache
    pub fn resize(&mut self, new_max_length: usize, device: &Device) -> Result<()> {
        if new_max_length == self.max_length {
            return Ok(());
        }
        
        let key_shape = [new_max_length, self.num_heads, self.head_dim];
        let value_shape = [new_max_length, self.num_heads, self.head_dim];
        
        let new_key_cache = BitNetTensor::zeros(&key_shape, DType::F32, device)?;
        let new_value_cache = BitNetTensor::zeros(&value_shape, DType::F32, device)?;
        
        // Copy existing data if shrinking
        if new_max_length < self.max_length {
            self.position = self.position.min(new_max_length);
        }
        
        self.key_cache = new_key_cache;
        self.value_cache = new_value_cache;
        self.max_length = new_max_length;
        
        Ok(())
    }
    
    /// Migrate cache to different device
    pub fn migrate_to_device(&mut self, device: &Device) -> Result<()> {
        // In a real implementation, we would transfer the tensors to the new device
        // For now, recreate the cache on the new device
        let key_shape = [self.max_length, self.num_heads, self.head_dim];
        let value_shape = [self.max_length, self.num_heads, self.head_dim];
        
        self.key_cache = BitNetTensor::zeros(&key_shape, DType::F32, device)?;
        self.value_cache = BitNetTensor::zeros(&value_shape, DType::F32, device)?;
        
        Ok(())
    }
    
    /// Get memory usage for this layer
    pub fn memory_usage(&self) -> LayerMemoryStats {
        let key_bytes = self.max_length * self.num_heads * self.head_dim * 4; // F32 = 4 bytes
        let value_bytes = self.max_length * self.num_heads * self.head_dim * 4;
        let total_bytes = key_bytes + value_bytes;
        let allocated_bytes = self.position * self.num_heads * self.head_dim * 4 * 2; // key + value
        
        LayerMemoryStats {
            total_bytes,
            allocated_bytes,
        }
    }
}

/// Memory pool for efficient tensor allocation
pub struct MemoryPool {
    /// Available memory blocks
    available_blocks: Vec<MemoryBlock>,
    /// Allocated memory blocks
    allocated_blocks: HashMap<usize, MemoryBlock>,
    /// Total pool size in bytes
    total_size: usize,
    /// Next block ID
    next_id: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(size_mb: usize) -> Result<Self> {
        Ok(Self {
            available_blocks: Vec::new(),
            allocated_blocks: HashMap::new(),
            total_size: size_mb * 1024 * 1024,
            next_id: 0,
        })
    }
    
    /// Allocate a memory block
    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        // Find suitable block or create new one
        let block_id = self.next_id;
        self.next_id += 1;
        
        let block = MemoryBlock {
            id: block_id,
            size,
            offset: 0, // Simplified
        };
        
        self.allocated_blocks.insert(block_id, block);
        Ok(block_id)
    }
    
    /// Deallocate a memory block
    pub fn deallocate(&mut self, block_id: usize) -> Result<()> {
        if let Some(block) = self.allocated_blocks.remove(&block_id) {
            self.available_blocks.push(block);
        }
        Ok(())
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let allocated_size: usize = self.allocated_blocks.values().map(|b| b.size).sum();
        let available_size = self.total_size - allocated_size;
        
        PoolStats {
            total_size: self.total_size,
            allocated_size,
            available_size,
            num_allocated_blocks: self.allocated_blocks.len(),
            num_available_blocks: self.available_blocks.len(),
        }
    }
}

/// Memory block in the pool
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub id: usize,
    pub size: usize,
    pub offset: usize,
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_size: usize,
    pub allocated_size: usize,
    pub available_size: usize,
    pub num_allocated_blocks: usize,
    pub num_available_blocks: usize,
}

/// Cache memory statistics
#[derive(Debug, Clone)]
pub struct CacheMemoryStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub utilization: f64,
    pub num_layers: usize,
}

/// Layer memory statistics
#[derive(Debug, Clone)]
pub struct LayerMemoryStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kv_cache_creation() {
        let config = BitNetConfig::default();
        let cache = KVCache::new(&config, 2048);
        assert!(cache.is_ok());
        
        let cache = cache.unwrap();
        assert_eq!(cache.max_length(), 2048);
        assert_eq!(cache.current_length(), 0);
        assert!(!cache.is_full());
    }
    
    #[test]
    fn test_layer_cache() {
        let device = Device::Cpu;
        let cache = LayerCache::new(1024, 32, 128, &device);
        assert!(cache.is_ok());
        
        let cache = cache.unwrap();
        let stats = cache.memory_usage();
        assert!(stats.total_bytes > 0);
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(100).unwrap(); // 100MB
        
        let block_id = pool.allocate(1024).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.allocated_size, 1024);
        assert_eq!(stats.num_allocated_blocks, 1);
        
        pool.deallocate(block_id).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.allocated_size, 0);
        assert_eq!(stats.num_available_blocks, 1);
    }
}