//! Unified weight loader abstraction for loading model tensors from multiple formats.
//!
//! Provides a common [`WeightLoader`] trait with concrete implementations:
//! - [`InMemoryWeightLoader`] — eagerly loads all tensors into RAM
//! - [`LazyWeightLoader`] — loads tensors on demand with optional caching
//!
//! Use [`WeightLoaderBuilder`] to construct a loader from a [`WeightFormat`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Supported serialisation formats for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// GGUF container (`.gguf` files).
    Gguf,
    /// SafeTensors container (`.safetensors` files).
    SafeTensors,
    /// Raw / unframed byte blobs.
    Raw,
}

impl WeightFormat {
    /// Attempt to detect the format from a file extension.
    pub fn detect(path: &str) -> Option<Self> {
        if path.ends_with(".gguf") {
            Some(Self::Gguf)
        } else if path.ends_with(".safetensors") {
            Some(Self::SafeTensors)
        } else {
            None
        }
    }
}

/// Element data type for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32,
}

impl DType {
    /// Size of a single element in bytes.
    pub fn element_size(self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 => 8,
        }
    }
}

/// Lightweight metadata about a tensor (no data payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Shape dimensions (e.g. `[4096, 4096]`).
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
    /// Byte offset within the backing store.
    pub offset: u64,
    /// Total byte size of the tensor data.
    pub size: u64,
}

/// Tensor data together with its metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
    /// Raw bytes (length must equal `shape.iter().product::<usize>() * dtype.element_size()`).
    pub data: Vec<u8>,
}

impl TensorData {
    /// Number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Expected byte length derived from shape and dtype.
    pub fn expected_byte_len(&self) -> usize {
        self.numel() * self.dtype.element_size()
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for loading model weights regardless of the underlying
/// storage format.
pub trait WeightLoader: Send + Sync {
    /// Load tensor data by name. Returns an error if the tensor does not exist
    /// or cannot be read.
    fn load_tensor(&self, name: &str) -> Result<TensorData>;

    /// List all tensor names available in the backing store.
    fn tensor_names(&self) -> Vec<String>;

    /// Check whether a tensor with the given name exists.
    fn has_tensor(&self, name: &str) -> bool;

    /// Retrieve metadata for a tensor without loading its data.
    fn tensor_info(&self, name: &str) -> Option<TensorInfo>;
}

// ---------------------------------------------------------------------------
// InMemoryWeightLoader
// ---------------------------------------------------------------------------

/// Eagerly stores every tensor in RAM.
#[derive(Debug, Clone)]
pub struct InMemoryWeightLoader {
    tensors: HashMap<String, TensorData>,
    infos: HashMap<String, TensorInfo>,
    format: WeightFormat,
}

impl InMemoryWeightLoader {
    /// Create a new empty loader for the given format.
    pub fn new(format: WeightFormat) -> Self {
        Self { tensors: HashMap::new(), infos: HashMap::new(), format }
    }

    /// Insert a tensor.
    pub fn insert(&mut self, name: impl Into<String>, data: TensorData) {
        let name = name.into();
        let info = TensorInfo {
            shape: data.shape.clone(),
            dtype: data.dtype,
            offset: 0,
            size: data.data.len() as u64,
        };
        self.infos.insert(name.clone(), info);
        self.tensors.insert(name, data);
    }

    /// Number of tensors held.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the loader is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The format this loader was created for.
    pub fn format(&self) -> WeightFormat {
        self.format
    }
}

impl WeightLoader for InMemoryWeightLoader {
    fn load_tensor(&self, name: &str) -> Result<TensorData> {
        self.tensors.get(name).cloned().ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.infos.get(name).cloned()
    }
}

// ---------------------------------------------------------------------------
// LazyWeightLoader
// ---------------------------------------------------------------------------

/// Callback that produces tensor data on demand.
pub type TensorProducer = Box<dyn Fn(&str) -> Result<TensorData> + Send + Sync>;

/// Loads tensors on first access and optionally caches them.
pub struct LazyWeightLoader {
    infos: HashMap<String, TensorInfo>,
    producer: Arc<TensorProducer>,
    cache: Mutex<HashMap<String, TensorData>>,
    caching_enabled: bool,
    format: WeightFormat,
}

impl std::fmt::Debug for LazyWeightLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyWeightLoader")
            .field("tensor_count", &self.infos.len())
            .field("caching_enabled", &self.caching_enabled)
            .field("format", &self.format)
            .finish()
    }
}

impl LazyWeightLoader {
    /// Create a new lazy loader.
    ///
    /// * `infos` — metadata for all available tensors.
    /// * `producer` — callback invoked the first time a tensor is accessed.
    /// * `caching_enabled` — when `true`, loaded tensors are kept in memory.
    /// * `format` — the weight format of the backing store.
    pub fn new(
        infos: HashMap<String, TensorInfo>,
        producer: TensorProducer,
        caching_enabled: bool,
        format: WeightFormat,
    ) -> Self {
        Self {
            infos,
            producer: Arc::new(producer),
            cache: Mutex::new(HashMap::new()),
            caching_enabled,
            format,
        }
    }

    /// Number of tensors available (registered via metadata).
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    /// Whether no tensors are registered.
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    /// Number of tensors currently cached.
    pub fn cached_count(&self) -> usize {
        self.cache.lock().expect("cache lock poisoned").len()
    }

    /// The format this loader was created for.
    pub fn format(&self) -> WeightFormat {
        self.format
    }
}

impl WeightLoader for LazyWeightLoader {
    fn load_tensor(&self, name: &str) -> Result<TensorData> {
        if !self.infos.contains_key(name) {
            bail!("tensor not found: {name}");
        }

        // Fast path: check cache.
        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(td) = cache.get(name) {
                return Ok(td.clone());
            }
        }

        // Slow path: produce tensor.
        let td = (self.producer)(name)?;

        if self.caching_enabled {
            let mut cache = self.cache.lock().expect("cache lock poisoned");
            cache.insert(name.to_string(), td.clone());
        }

        Ok(td)
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.infos.keys().cloned().collect();
        names.sort();
        names
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.infos.contains_key(name)
    }

    fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.infos.get(name).cloned()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing [`InMemoryWeightLoader`] or [`LazyWeightLoader`]
/// instances.
pub struct WeightLoaderBuilder {
    format: WeightFormat,
    tensors: HashMap<String, TensorData>,
    infos: HashMap<String, TensorInfo>,
    producer: Option<TensorProducer>,
    lazy: bool,
    caching: bool,
}

impl WeightLoaderBuilder {
    /// Start building a loader for the given format.
    pub fn new(format: WeightFormat) -> Self {
        Self {
            format,
            tensors: HashMap::new(),
            infos: HashMap::new(),
            producer: None,
            lazy: false,
            caching: true,
        }
    }

    /// Detect format from a file path and start building.
    pub fn from_path(path: &str) -> Result<Self> {
        let format = WeightFormat::detect(path)
            .ok_or_else(|| anyhow::anyhow!("cannot detect format from path: {path}"))?;
        Ok(Self::new(format))
    }

    /// Add a tensor (for in-memory loaders).
    pub fn add_tensor(mut self, name: impl Into<String>, data: TensorData) -> Self {
        let name = name.into();
        let info = TensorInfo {
            shape: data.shape.clone(),
            dtype: data.dtype,
            offset: 0,
            size: data.data.len() as u64,
        };
        self.infos.insert(name.clone(), info);
        self.tensors.insert(name, data);
        self
    }

    /// Register tensor metadata (for lazy loaders).
    pub fn add_tensor_info(mut self, name: impl Into<String>, info: TensorInfo) -> Self {
        self.infos.insert(name.into(), info);
        self
    }

    /// Enable lazy loading with the supplied producer callback.
    pub fn lazy(mut self, producer: TensorProducer) -> Self {
        self.lazy = true;
        self.producer = Some(producer);
        self
    }

    /// Enable or disable caching for lazy loaders (default: enabled).
    pub fn caching(mut self, enabled: bool) -> Self {
        self.caching = enabled;
        self
    }

    /// Consume the builder and return a boxed [`WeightLoader`].
    pub fn build(self) -> Result<Box<dyn WeightLoader>> {
        if self.lazy {
            let producer =
                self.producer.ok_or_else(|| anyhow::anyhow!("lazy loader requires a producer"))?;
            Ok(Box::new(LazyWeightLoader::new(self.infos, producer, self.caching, self.format)))
        } else {
            let mut loader = InMemoryWeightLoader::new(self.format);
            for (name, data) in self.tensors {
                loader.insert(name, data);
            }
            Ok(Box::new(loader))
        }
    }

    /// Consume the builder and return a concrete [`InMemoryWeightLoader`].
    ///
    /// Returns an error if the builder was configured for lazy loading.
    pub fn build_in_memory(self) -> Result<InMemoryWeightLoader> {
        if self.lazy {
            bail!("builder configured for lazy loading; use build() instead");
        }
        let mut loader = InMemoryWeightLoader::new(self.format);
        for (name, data) in self.tensors {
            loader.insert(name, data);
        }
        Ok(loader)
    }

    /// Consume the builder and return a concrete [`LazyWeightLoader`].
    ///
    /// Returns an error if no producer was set.
    pub fn build_lazy(self) -> Result<LazyWeightLoader> {
        let producer =
            self.producer.ok_or_else(|| anyhow::anyhow!("lazy loader requires a producer"))?;
        Ok(LazyWeightLoader::new(self.infos, producer, self.caching, self.format))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn make_f32_tensor(shape: Vec<usize>, value: f32) -> TensorData {
        let numel: usize = shape.iter().product();
        let data: Vec<u8> = value.to_le_bytes().iter().cycle().take(numel * 4).copied().collect();
        TensorData { shape, dtype: DType::F32, data }
    }

    fn sample_loader() -> InMemoryWeightLoader {
        let mut loader = InMemoryWeightLoader::new(WeightFormat::Gguf);
        loader.insert("weight_a", make_f32_tensor(vec![4, 4], 1.0));
        loader.insert("weight_b", make_f32_tensor(vec![2, 3], 2.0));
        loader.insert("bias", make_f32_tensor(vec![4], 0.5));
        loader
    }

    // -- DType ------------------------------------------------------------

    #[test]
    fn dtype_element_sizes() {
        assert_eq!(DType::U8.element_size(), 1);
        assert_eq!(DType::I8.element_size(), 1);
        assert_eq!(DType::F16.element_size(), 2);
        assert_eq!(DType::BF16.element_size(), 2);
        assert_eq!(DType::I16.element_size(), 2);
        assert_eq!(DType::U16.element_size(), 2);
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::I32.element_size(), 4);
        assert_eq!(DType::U32.element_size(), 4);
        assert_eq!(DType::F64.element_size(), 8);
    }

    // -- TensorData -------------------------------------------------------

    #[test]
    fn tensor_data_numel() {
        let td = make_f32_tensor(vec![3, 4, 5], 0.0);
        assert_eq!(td.numel(), 60);
    }

    #[test]
    fn tensor_data_expected_byte_len() {
        let td = make_f32_tensor(vec![2, 3], 0.0);
        assert_eq!(td.expected_byte_len(), 24); // 6 * 4
        assert_eq!(td.data.len(), td.expected_byte_len());
    }

    #[test]
    fn tensor_data_scalar() {
        let td = make_f32_tensor(vec![1], 42.0);
        assert_eq!(td.numel(), 1);
        assert_eq!(td.expected_byte_len(), 4);
    }

    // -- WeightFormat detection -------------------------------------------

    #[test]
    fn format_detect_gguf() {
        assert_eq!(WeightFormat::detect("model.gguf"), Some(WeightFormat::Gguf));
        assert_eq!(WeightFormat::detect("/path/to/model.gguf"), Some(WeightFormat::Gguf));
    }

    #[test]
    fn format_detect_safetensors() {
        assert_eq!(WeightFormat::detect("weights.safetensors"), Some(WeightFormat::SafeTensors));
    }

    #[test]
    fn format_detect_unknown() {
        assert_eq!(WeightFormat::detect("weights.bin"), None);
        assert_eq!(WeightFormat::detect("no_extension"), None);
    }

    // -- InMemoryWeightLoader ---------------------------------------------

    #[test]
    fn inmemory_load_existing_tensor() {
        let loader = sample_loader();
        let td = loader.load_tensor("weight_a").unwrap();
        assert_eq!(td.shape, vec![4, 4]);
        assert_eq!(td.dtype, DType::F32);
    }

    #[test]
    fn inmemory_load_missing_tensor_errors() {
        let loader = sample_loader();
        assert!(loader.load_tensor("nonexistent").is_err());
    }

    #[test]
    fn inmemory_tensor_names_sorted() {
        let loader = sample_loader();
        let names = loader.tensor_names();
        assert_eq!(names, vec!["bias", "weight_a", "weight_b"]);
    }

    #[test]
    fn inmemory_has_tensor() {
        let loader = sample_loader();
        assert!(loader.has_tensor("weight_a"));
        assert!(loader.has_tensor("bias"));
        assert!(!loader.has_tensor("missing"));
    }

    #[test]
    fn inmemory_tensor_info() {
        let loader = sample_loader();
        let info = loader.tensor_info("weight_b").unwrap();
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.dtype, DType::F32);
        assert_eq!(info.size, 24); // 6 * 4 bytes
    }

    #[test]
    fn inmemory_tensor_info_missing_returns_none() {
        let loader = sample_loader();
        assert!(loader.tensor_info("ghost").is_none());
    }

    #[test]
    fn inmemory_len_and_empty() {
        let loader = sample_loader();
        assert_eq!(loader.len(), 3);
        assert!(!loader.is_empty());

        let empty = InMemoryWeightLoader::new(WeightFormat::Raw);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn inmemory_format() {
        let loader = InMemoryWeightLoader::new(WeightFormat::SafeTensors);
        assert_eq!(loader.format(), WeightFormat::SafeTensors);
    }

    // -- LazyWeightLoader -------------------------------------------------

    fn lazy_loader_with_cache(caching: bool) -> LazyWeightLoader {
        let mut infos = HashMap::new();
        infos.insert(
            "lazy_a".to_string(),
            TensorInfo { shape: vec![2, 2], dtype: DType::F32, offset: 0, size: 16 },
        );
        infos.insert(
            "lazy_b".to_string(),
            TensorInfo { shape: vec![3], dtype: DType::F32, offset: 16, size: 12 },
        );

        let producer: TensorProducer = Box::new(|name: &str| {
            let shape = match name {
                "lazy_a" => vec![2, 2],
                "lazy_b" => vec![3],
                _ => bail!("unknown tensor: {name}"),
            };
            Ok(make_f32_tensor(shape, 1.0))
        });

        LazyWeightLoader::new(infos, producer, caching, WeightFormat::Gguf)
    }

    #[test]
    fn lazy_load_tensor_on_demand() {
        let loader = lazy_loader_with_cache(false);
        let td = loader.load_tensor("lazy_a").unwrap();
        assert_eq!(td.shape, vec![2, 2]);
    }

    #[test]
    fn lazy_missing_tensor_errors() {
        let loader = lazy_loader_with_cache(false);
        assert!(loader.load_tensor("no_such").is_err());
    }

    #[test]
    fn lazy_caching_stores_result() {
        let loader = lazy_loader_with_cache(true);
        assert_eq!(loader.cached_count(), 0);
        let _ = loader.load_tensor("lazy_a").unwrap();
        assert_eq!(loader.cached_count(), 1);
        // Second access should hit cache (no panic from producer).
        let _ = loader.load_tensor("lazy_a").unwrap();
        assert_eq!(loader.cached_count(), 1);
    }

    #[test]
    fn lazy_no_caching_does_not_store() {
        let loader = lazy_loader_with_cache(false);
        let _ = loader.load_tensor("lazy_a").unwrap();
        assert_eq!(loader.cached_count(), 0);
    }

    #[test]
    fn lazy_tensor_names_sorted() {
        let loader = lazy_loader_with_cache(false);
        assert_eq!(loader.tensor_names(), vec!["lazy_a", "lazy_b"]);
    }

    #[test]
    fn lazy_has_tensor() {
        let loader = lazy_loader_with_cache(false);
        assert!(loader.has_tensor("lazy_a"));
        assert!(!loader.has_tensor("ghost"));
    }

    #[test]
    fn lazy_tensor_info() {
        let loader = lazy_loader_with_cache(false);
        let info = loader.tensor_info("lazy_b").unwrap();
        assert_eq!(info.shape, vec![3]);
        assert_eq!(info.offset, 16);
    }

    #[test]
    fn lazy_len_and_format() {
        let loader = lazy_loader_with_cache(false);
        assert_eq!(loader.len(), 2);
        assert!(!loader.is_empty());
        assert_eq!(loader.format(), WeightFormat::Gguf);
    }

    // -- WeightLoaderBuilder (in-memory) ----------------------------------

    #[test]
    fn builder_in_memory_basic() {
        let loader = WeightLoaderBuilder::new(WeightFormat::SafeTensors)
            .add_tensor("t1", make_f32_tensor(vec![8], 0.0))
            .build_in_memory()
            .unwrap();
        assert_eq!(loader.len(), 1);
        assert!(loader.has_tensor("t1"));
    }

    #[test]
    fn builder_in_memory_via_build_returns_dyn() {
        let loader = WeightLoaderBuilder::new(WeightFormat::Gguf)
            .add_tensor("x", make_f32_tensor(vec![2], 1.0))
            .build()
            .unwrap();
        assert!(loader.has_tensor("x"));
        assert!(!loader.has_tensor("y"));
    }

    #[test]
    fn builder_lazy_basic() {
        let producer: TensorProducer = Box::new(|_name| Ok(make_f32_tensor(vec![1], 0.0)));
        let info = TensorInfo { shape: vec![1], dtype: DType::F32, offset: 0, size: 4 };
        let loader = WeightLoaderBuilder::new(WeightFormat::Raw)
            .add_tensor_info("t", info)
            .lazy(producer)
            .caching(false)
            .build_lazy()
            .unwrap();
        assert_eq!(loader.len(), 1);
        let td = loader.load_tensor("t").unwrap();
        assert_eq!(td.shape, vec![1]);
    }

    #[test]
    fn builder_lazy_via_build_returns_dyn() {
        let producer: TensorProducer = Box::new(|_name| Ok(make_f32_tensor(vec![4], 0.0)));
        let info = TensorInfo { shape: vec![4], dtype: DType::F32, offset: 0, size: 16 };
        let loader = WeightLoaderBuilder::new(WeightFormat::Gguf)
            .add_tensor_info("q", info)
            .lazy(producer)
            .build()
            .unwrap();
        assert!(loader.has_tensor("q"));
    }

    #[test]
    fn builder_from_path_gguf() {
        let builder = WeightLoaderBuilder::from_path("model.gguf").unwrap();
        let loader = builder.build_in_memory().unwrap();
        assert_eq!(loader.format(), WeightFormat::Gguf);
    }

    #[test]
    fn builder_from_path_safetensors() {
        let builder = WeightLoaderBuilder::from_path("/dir/weights.safetensors").unwrap();
        let loader = builder.build_in_memory().unwrap();
        assert_eq!(loader.format(), WeightFormat::SafeTensors);
    }

    #[test]
    fn builder_from_path_unknown_errors() {
        assert!(WeightLoaderBuilder::from_path("model.bin").is_err());
    }

    #[test]
    fn builder_in_memory_rejects_lazy_build() {
        let builder = WeightLoaderBuilder::new(WeightFormat::Gguf)
            .add_tensor("x", make_f32_tensor(vec![1], 0.0))
            .lazy(Box::new(|_| Ok(make_f32_tensor(vec![1], 0.0))));
        assert!(builder.build_in_memory().is_err());
    }

    #[test]
    fn builder_lazy_without_producer_errors() {
        let builder = WeightLoaderBuilder::new(WeightFormat::Gguf);
        assert!(builder.build_lazy().is_err());
    }

    // -- trait object usage -----------------------------------------------

    #[test]
    fn trait_object_dispatch() {
        let loader: Box<dyn WeightLoader> = Box::new(sample_loader());
        assert_eq!(loader.tensor_names().len(), 3);
        let td = loader.load_tensor("bias").unwrap();
        assert_eq!(td.shape, vec![4]);
    }

    #[test]
    fn trait_object_lazy() {
        let loader: Box<dyn WeightLoader> = Box::new(lazy_loader_with_cache(true));
        assert!(loader.has_tensor("lazy_a"));
        let td = loader.load_tensor("lazy_b").unwrap();
        assert_eq!(td.shape, vec![3]);
    }
}
