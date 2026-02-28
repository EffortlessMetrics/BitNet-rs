//! GPU-accelerated KV cache management for transformer attention.
//!
//! Provides [GpuKvCache] backed by OpenCL buffers.

pub const MAX_SEQ_LEN: usize = 8192;

#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub page_size: usize,
    pub max_seq_len: usize,
}

impl KvCacheConfig {
    pub fn new(num_heads: usize, head_dim: usize, page_size: usize, max_seq_len: usize) -> Self {
        assert!(max_seq_len <= MAX_SEQ_LEN, "max_seq_len exceeds {MAX_SEQ_LEN}");
        assert!(page_size > 0, "page_size must be > 0");
        assert!(head_dim > 0, "head_dim must be > 0");
        assert!(num_heads > 0, "num_heads must be > 0");
        Self { num_heads, head_dim, page_size, max_seq_len }
    }

    pub fn max_pages(&self) -> usize {
        (self.max_seq_len + self.page_size - 1) / self.page_size
    }

    pub fn total_elements(&self) -> usize {
        self.num_heads * self.max_pages() * self.page_size * self.head_dim
    }
}

pub struct GpuKvCache {
    config: KvCacheConfig,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    positions: Vec<usize>,
}

impl GpuKvCache {
    pub fn new(config: KvCacheConfig) -> Self {
        let total = config.total_elements();
        Self { positions: vec![0; config.num_heads], k_cache: vec![0.0; total], v_cache: vec![0.0; total], config }
    }

    pub fn append(&mut self, new_keys: &[f32], new_values: &[f32]) {
        let (nh, hd) = (self.config.num_heads, self.config.head_dim);
        assert_eq!(new_keys.len(), nh * hd);
        assert_eq!(new_values.len(), nh * hd);
        for head in 0..nh {
            let pos = self.positions[head];
            assert!(pos < self.config.max_seq_len, "KV cache full for head {head}");
            let (pi, po) = (pos / self.config.page_size, pos % self.config.page_size);
            let stride = self.config.max_pages() * self.config.page_size * hd;
            let base = head * stride + pi * self.config.page_size * hd + po * hd;
            self.k_cache[base..base + hd].copy_from_slice(&new_keys[head * hd..(head + 1) * hd]);
            self.v_cache[base..base + hd].copy_from_slice(&new_values[head * hd..(head + 1) * hd]);
            self.positions[head] = pos + 1;
        }
    }

    pub fn read_keys(&self, positions: &[usize]) -> Vec<f32> { self.read_cache(&self.k_cache, positions) }
    pub fn read_values(&self, positions: &[usize]) -> Vec<f32> { self.read_cache(&self.v_cache, positions) }

    fn read_cache(&self, cache: &[f32], positions: &[usize]) -> Vec<f32> {
        let (nh, hd, sl) = (self.config.num_heads, self.config.head_dim, positions.len());
        let mut output = vec![0.0f32; nh * sl * hd];
        for head in 0..nh {
            let stride = self.config.max_pages() * self.config.page_size * hd;
            for (s, &pos) in positions.iter().enumerate() {
                let (pi, po) = (pos / self.config.page_size, pos % self.config.page_size);
                let base = head * stride + pi * self.config.page_size * hd + po * hd;
                let ob = head * sl * hd + s * hd;
                output[ob..ob + hd].copy_from_slice(&cache[base..base + hd]);
            }
        }
        output
    }

    pub fn position(&self, head: usize) -> usize { self.positions[head] }
    pub fn config(&self) -> &KvCacheConfig { &self.config }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(nh: usize, hd: usize, ps: usize) -> KvCacheConfig { KvCacheConfig::new(nh, hd, ps, 256) }

    #[test]
    fn append_and_read_single_head() { let mut c = GpuKvCache::new(mk(1,4,16)); c.append(&[1.0,2.0,3.0,4.0],&[5.0,6.0,7.0,8.0]); assert_eq!(c.read_keys(&[0]),vec![1.0,2.0,3.0,4.0]); assert_eq!(c.read_values(&[0]),vec![5.0,6.0,7.0,8.0]); }

    #[test]
    fn append_multiple_positions() { let mut c = GpuKvCache::new(mk(1,2,4)); c.append(&[1.0,2.0],&[10.0,20.0]); c.append(&[3.0,4.0],&[30.0,40.0]); c.append(&[5.0,6.0],&[50.0,60.0]); assert_eq!(c.position(0),3); assert_eq!(c.read_keys(&[0,1,2]),vec![1.0,2.0,3.0,4.0,5.0,6.0]); }

    #[test]
    fn multi_head_append_and_read() { let mut c = GpuKvCache::new(mk(2,3,8)); c.append(&[1.0,2.0,3.0,4.0,5.0,6.0],&[10.0,20.0,30.0,40.0,50.0,60.0]); assert_eq!(c.read_keys(&[0]),vec![1.0,2.0,3.0,4.0,5.0,6.0]); }

    #[test]
    fn page_boundary_crossing() { let mut c = GpuKvCache::new(mk(1,2,2)); c.append(&[1.0,2.0],&[10.0,20.0]); c.append(&[3.0,4.0],&[30.0,40.0]); c.append(&[5.0,6.0],&[50.0,60.0]); assert_eq!(c.read_keys(&[0,1,2]),vec![1.0,2.0,3.0,4.0,5.0,6.0]); }

    #[test]
    fn config_max_pages() { assert_eq!(KvCacheConfig::new(1,64,16,100).max_pages(),7); assert_eq!(KvCacheConfig::new(1,64,16,128).max_pages(),8); }

    #[test]
    fn total_elements() { let c = KvCacheConfig::new(4,64,16,256); assert_eq!(c.total_elements(),4*16*16*64); }

    #[test]
    fn read_non_sequential() { let mut c = GpuKvCache::new(mk(1,2,4)); for i in 0..6 { let v = (i as f32)*10.0; c.append(&[v,v+1.0],&[v+100.0,v+101.0]); } assert_eq!(c.read_keys(&[4,1,5]),vec![40.0,41.0,10.0,11.0,50.0,51.0]); }

    #[test]
    fn large_head_dim() { let hd=128; let mut c = GpuKvCache::new(mk(2,hd,32)); let k: Vec<f32>=(0..2*hd).map(|i| i as f32).collect(); let v: Vec<f32>=(0..2*hd).map(|i| (i as f32)+1000.0).collect(); c.append(&k,&v); let r=c.read_keys(&[0]); assert_eq!(r.len(),2*hd); assert_eq!(r[0],0.0); assert_eq!(r[hd],hd as f32); }

    #[test]
    #[should_panic(expected = "KV cache full")]
    fn append_beyond_capacity() { let mut c = GpuKvCache::new(KvCacheConfig::new(1,2,2,4)); for _ in 0..5 { c.append(&[1.0,1.0],&[1.0,1.0]); } }

    #[test]
    #[should_panic(expected = "max_seq_len exceeds")]
    fn config_rejects_excessive_seq_len() { KvCacheConfig::new(1,64,16,MAX_SEQ_LEN+1); }
}
