#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]

//! Metal pipeline state management tests for Apple Silicon.
//!
//! Comprehensive test suite covering compute pipeline state creation, caching,
//! reflection, thread execution width, threadgroup validation, statistics,
//! async compilation, serialization, linked/visible functions, pool management,
//! and error handling.
//!
//! All tests use local mock/simulated Metal types so they run on any platform
//! without GPU hardware dependencies.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, RwLock};

    // ───────────────────────────────────────────────────────────────────
    // Simulated Metal types
    // ───────────────────────────────────────────────────────────────────

    /// Simulated Metal function type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MetalFunctionType {
        Vertex,
        Fragment,
        Kernel,
    }

    /// Simulated Metal GPU family for capability checking.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum MetalGPUFamily {
        Apple7,
        Apple8,
        Apple9,
    }

    /// Simulated Metal data type for function arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MetalDataType {
        Float,
        Half,
        Int,
        UInt,
        Bool,
        Pointer,
    }

    /// Simulated Metal argument access.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MetalAccess {
        ReadOnly,
        WriteOnly,
        ReadWrite,
    }

    /// Simulated Metal mutability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MetalMutability {
        Default,
        Mutable,
        Immutable,
    }

    /// Pipeline compilation status for async compilation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CompilationStatus {
        Pending,
        Compiling,
        Success,
        Error,
    }

    /// A simulated Metal function argument descriptor.
    #[derive(Debug, Clone)]
    struct MetalArgument {
        name: String,
        index: u32,
        data_type: MetalDataType,
        access: MetalAccess,
        is_active: bool,
    }

    /// A simulated Metal function.
    #[derive(Debug, Clone)]
    struct MetalFunction {
        name: String,
        function_type: MetalFunctionType,
        arguments: Vec<MetalArgument>,
        patch_type: Option<String>,
    }

    impl MetalFunction {
        fn new_kernel(name: &str) -> Self {
            Self {
                name: name.to_string(),
                function_type: MetalFunctionType::Kernel,
                arguments: Vec::new(),
                patch_type: None,
            }
        }

        fn with_argument(mut self, arg: MetalArgument) -> Self {
            self.arguments.push(arg);
            self
        }
    }

    /// A simulated Metal library containing functions.
    #[derive(Debug, Clone)]
    struct MetalLibrary {
        label: Option<String>,
        functions: HashMap<String, MetalFunction>,
    }

    impl MetalLibrary {
        fn new(label: Option<&str>) -> Self {
            Self { label: label.map(String::from), functions: HashMap::new() }
        }

        fn add_function(&mut self, func: MetalFunction) {
            self.functions.insert(func.name.clone(), func);
        }

        fn get_function(&self, name: &str) -> Option<&MetalFunction> {
            self.functions.get(name)
        }

        fn function_names(&self) -> Vec<String> {
            self.functions.keys().cloned().collect()
        }
    }

    /// Pipeline state statistics.
    #[derive(Debug, Clone, Default)]
    struct PipelineStatistics {
        compilation_time_ns: u64,
        instruction_count: u64,
        spill_count: u64,
        occupancy: f32,
        sgpr_count: u32,
        vgpr_count: u32,
    }

    /// Compute pipeline descriptor.
    #[derive(Debug, Clone)]
    struct ComputePipelineDescriptor {
        label: Option<String>,
        compute_function: Option<MetalFunction>,
        thread_group_size_is_multiple_of_execution_width: bool,
        max_total_threads_per_threadgroup: u32,
        stage_input_descriptor: Option<StageInputDescriptor>,
        linked_functions: Vec<MetalFunction>,
        max_call_stack_depth: u32,
        support_indirect_command_buffers: bool,
        support_adding_binary_functions: bool,
        mutability: MetalMutability,
    }

    impl Default for ComputePipelineDescriptor {
        fn default() -> Self {
            Self {
                label: None,
                compute_function: None,
                thread_group_size_is_multiple_of_execution_width: false,
                max_total_threads_per_threadgroup: 1024,
                stage_input_descriptor: None,
                linked_functions: Vec::new(),
                max_call_stack_depth: 1,
                support_indirect_command_buffers: false,
                support_adding_binary_functions: false,
                mutability: MetalMutability::Default,
            }
        }
    }

    /// Simulated stage input descriptor.
    #[derive(Debug, Clone)]
    struct StageInputDescriptor {
        attributes: Vec<AttributeDescriptor>,
        layouts: Vec<BufferLayoutDescriptor>,
    }

    #[derive(Debug, Clone)]
    struct AttributeDescriptor {
        format: MetalDataType,
        offset: u32,
        buffer_index: u32,
    }

    #[derive(Debug, Clone)]
    struct BufferLayoutDescriptor {
        stride: u32,
        step_function: StepFunction,
        step_rate: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum StepFunction {
        Constant,
        PerVertex,
        PerInstance,
        ThreadPositionInGridX,
    }

    /// Simulated compute pipeline state.
    #[derive(Debug, Clone)]
    struct ComputePipelineState {
        label: Option<String>,
        max_total_threads_per_threadgroup: u32,
        thread_execution_width: u32,
        static_threadgroup_memory_length: u32,
        gpu_family: MetalGPUFamily,
        function_name: String,
        statistics: PipelineStatistics,
        supports_indirect_command_buffers: bool,
    }

    impl ComputePipelineState {
        fn new(
            descriptor: &ComputePipelineDescriptor,
            gpu_family: MetalGPUFamily,
        ) -> Result<Self, PipelineError> {
            let func =
                descriptor.compute_function.as_ref().ok_or(PipelineError::MissingFunction)?;

            if func.function_type != MetalFunctionType::Kernel {
                return Err(PipelineError::InvalidFunctionType(func.function_type));
            }

            if descriptor.max_total_threads_per_threadgroup == 0 {
                return Err(PipelineError::InvalidThreadgroupSize(0));
            }

            let execution_width = match gpu_family {
                MetalGPUFamily::Apple7 => 32,
                MetalGPUFamily::Apple8 => 32,
                MetalGPUFamily::Apple9 => 32,
            };

            let max_threads = descriptor.max_total_threads_per_threadgroup.min(1024);

            if descriptor.thread_group_size_is_multiple_of_execution_width
                && max_threads % execution_width != 0
            {
                return Err(PipelineError::ThreadgroupNotMultipleOfWidth {
                    threadgroup_size: max_threads,
                    execution_width,
                });
            }

            if descriptor.max_call_stack_depth > 16 {
                return Err(PipelineError::ExcessiveCallStackDepth(
                    descriptor.max_call_stack_depth,
                ));
            }

            Ok(Self {
                label: descriptor.label.clone(),
                max_total_threads_per_threadgroup: max_threads,
                thread_execution_width: execution_width,
                static_threadgroup_memory_length: 0,
                gpu_family,
                function_name: func.name.clone(),
                statistics: PipelineStatistics::default(),
                supports_indirect_command_buffers: descriptor.support_indirect_command_buffers,
            })
        }
    }

    /// Errors during pipeline state creation.
    #[derive(Debug, Clone)]
    enum PipelineError {
        MissingFunction,
        InvalidFunctionType(MetalFunctionType),
        InvalidThreadgroupSize(u32),
        ThreadgroupNotMultipleOfWidth { threadgroup_size: u32, execution_width: u32 },
        FunctionNotFound(String),
        CompilationFailed(String),
        SerializationFailed(String),
        DeserializationFailed(String),
        PoolExhausted,
        ExcessiveCallStackDepth(u32),
        LibraryError(String),
    }

    impl std::fmt::Display for PipelineError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::MissingFunction => {
                    write!(f, "compute function is required")
                }
                Self::InvalidFunctionType(t) => {
                    write!(f, "invalid function type: {t:?}")
                }
                Self::InvalidThreadgroupSize(s) => {
                    write!(f, "invalid threadgroup size: {s}")
                }
                Self::ThreadgroupNotMultipleOfWidth { threadgroup_size, execution_width } => {
                    write!(
                        f,
                        "threadgroup size {threadgroup_size} is not a \
                     multiple of execution width {execution_width}"
                    )
                }
                Self::FunctionNotFound(n) => {
                    write!(f, "function not found: {n}")
                }
                Self::CompilationFailed(m) => {
                    write!(f, "compilation failed: {m}")
                }
                Self::SerializationFailed(m) => {
                    write!(f, "serialization failed: {m}")
                }
                Self::DeserializationFailed(m) => {
                    write!(f, "deserialization failed: {m}")
                }
                Self::PoolExhausted => write!(f, "pipeline pool exhausted"),
                Self::ExcessiveCallStackDepth(d) => {
                    write!(f, "call stack depth {d} exceeds maximum 16")
                }
                Self::LibraryError(m) => {
                    write!(f, "library error: {m}")
                }
            }
        }
    }

    /// Serialized pipeline state blob.
    #[derive(Debug, Clone)]
    struct SerializedPipeline {
        data: Vec<u8>,
        function_name: String,
        gpu_family: MetalGPUFamily,
        version: u32,
    }

    impl ComputePipelineState {
        fn serialize(&self) -> Result<SerializedPipeline, PipelineError> {
            let data = format!(
                "{}:{}:{:?}",
                self.function_name, self.max_total_threads_per_threadgroup, self.gpu_family
            );
            Ok(SerializedPipeline {
                data: data.into_bytes(),
                function_name: self.function_name.clone(),
                gpu_family: self.gpu_family,
                version: 1,
            })
        }
    }

    impl SerializedPipeline {
        fn deserialize(&self) -> Result<ComputePipelineState, PipelineError> {
            let text = String::from_utf8(self.data.clone())
                .map_err(|e| PipelineError::DeserializationFailed(e.to_string()))?;
            let parts: Vec<&str> = text.split(':').collect();
            if parts.len() < 2 {
                return Err(PipelineError::DeserializationFailed("invalid format".to_string()));
            }
            let max_threads: u32 = parts[1].parse().map_err(|e: std::num::ParseIntError| {
                PipelineError::DeserializationFailed(e.to_string())
            })?;
            Ok(ComputePipelineState {
                label: None,
                max_total_threads_per_threadgroup: max_threads,
                thread_execution_width: 32,
                static_threadgroup_memory_length: 0,
                gpu_family: self.gpu_family,
                function_name: self.function_name.clone(),
                statistics: PipelineStatistics::default(),
                supports_indirect_command_buffers: false,
            })
        }
    }

    /// Async compilation handle.
    struct AsyncCompilationHandle {
        status: Arc<Mutex<CompilationStatus>>,
        result: Arc<Mutex<Option<Result<ComputePipelineState, PipelineError>>>>,
    }

    impl AsyncCompilationHandle {
        fn new() -> Self {
            Self {
                status: Arc::new(Mutex::new(CompilationStatus::Pending)),
                result: Arc::new(Mutex::new(None)),
            }
        }

        fn status(&self) -> CompilationStatus {
            *self.status.lock().unwrap()
        }

        fn complete(&self, result: Result<ComputePipelineState, PipelineError>) {
            let is_ok = result.is_ok();
            *self.result.lock().unwrap() = Some(result);
            *self.status.lock().unwrap() =
                if is_ok { CompilationStatus::Success } else { CompilationStatus::Error };
        }

        fn take_result(&self) -> Option<Result<ComputePipelineState, PipelineError>> {
            self.result.lock().unwrap().take()
        }
    }

    /// Pipeline state cache with LRU-style eviction.
    struct PipelineCache {
        entries: RwLock<HashMap<String, ComputePipelineState>>,
        max_entries: usize,
        hits: AtomicU64,
        misses: AtomicU64,
    }

    impl PipelineCache {
        fn new(max_entries: usize) -> Self {
            Self {
                entries: RwLock::new(HashMap::new()),
                max_entries,
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
            }
        }

        fn get(&self, key: &str) -> Option<ComputePipelineState> {
            let entries = self.entries.read().unwrap();
            if let Some(state) = entries.get(key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(state.clone())
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }

        fn insert(&self, key: String, state: ComputePipelineState) -> Result<(), PipelineError> {
            let mut entries = self.entries.write().unwrap();
            if entries.len() >= self.max_entries && !entries.contains_key(&key) {
                if let Some(first_key) = entries.keys().next().cloned() {
                    entries.remove(&first_key);
                }
            }
            entries.insert(key, state);
            Ok(())
        }

        fn remove(&self, key: &str) -> bool {
            self.entries.write().unwrap().remove(key).is_some()
        }

        fn len(&self) -> usize {
            self.entries.read().unwrap().len()
        }

        fn is_empty(&self) -> bool {
            self.len() == 0
        }

        fn clear(&self) {
            self.entries.write().unwrap().clear();
        }

        fn hit_rate(&self) -> f64 {
            let h = self.hits.load(Ordering::Relaxed) as f64;
            let m = self.misses.load(Ordering::Relaxed) as f64;
            let total = h + m;
            if total == 0.0 { 0.0 } else { h / total }
        }
    }

    /// Pipeline state pool with fixed capacity.
    struct PipelinePool {
        states: Mutex<Vec<Option<ComputePipelineState>>>,
        capacity: usize,
    }

    impl PipelinePool {
        fn new(capacity: usize) -> Self {
            let states = (0..capacity).map(|_| None).collect();
            Self { states: Mutex::new(states), capacity }
        }

        fn acquire(&self) -> Result<(usize, ComputePipelineState), PipelineError> {
            let mut states = self.states.lock().unwrap();
            for (i, slot) in states.iter_mut().enumerate() {
                if let Some(state) = slot.take() {
                    return Ok((i, state));
                }
            }
            Err(PipelineError::PoolExhausted)
        }

        fn release(&self, index: usize, state: ComputePipelineState) -> Result<(), PipelineError> {
            let mut states = self.states.lock().unwrap();
            if index >= self.capacity {
                return Err(PipelineError::InvalidThreadgroupSize(index as u32));
            }
            states[index] = Some(state);
            Ok(())
        }

        fn available(&self) -> usize {
            self.states.lock().unwrap().iter().filter(|s| s.is_some()).count()
        }

        fn capacity(&self) -> usize {
            self.capacity
        }
    }

    /// Visible function table.
    #[derive(Debug, Clone)]
    struct VisibleFunctionTable {
        functions: Vec<MetalFunction>,
    }

    impl VisibleFunctionTable {
        fn new() -> Self {
            Self { functions: Vec::new() }
        }

        fn add(&mut self, func: MetalFunction) {
            self.functions.push(func);
        }

        fn len(&self) -> usize {
            self.functions.len()
        }

        fn is_empty(&self) -> bool {
            self.functions.is_empty()
        }

        fn get(&self, index: usize) -> Option<&MetalFunction> {
            self.functions.get(index)
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Helper constructors
    // ───────────────────────────────────────────────────────────────────

    fn make_kernel(name: &str) -> MetalFunction {
        MetalFunction::new_kernel(name)
    }

    fn make_descriptor(func: MetalFunction) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor { compute_function: Some(func), ..Default::default() }
    }

    fn make_pipeline(name: &str) -> Result<ComputePipelineState, PipelineError> {
        let desc = make_descriptor(make_kernel(name));
        ComputePipelineState::new(&desc, MetalGPUFamily::Apple8)
    }

    fn make_pipeline_ok(name: &str) -> ComputePipelineState {
        make_pipeline(name).expect("pipeline creation should succeed")
    }

    // ═══════════════════════════════════════════════════════════════════
    // 1. Compute pipeline state creation and caching
    // ═══════════════════════════════════════════════════════════════════

    mod creation_and_caching {
        use super::*;

        #[test]
        fn create_pipeline_state_basic() {
            let state = make_pipeline_ok("matmul_kernel");
            assert_eq!(state.function_name, "matmul_kernel");
        }

        #[test]
        fn create_pipeline_with_label() {
            let mut desc = make_descriptor(make_kernel("add_kernel"));
            desc.label = Some("add_pipeline".to_string());
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.label.as_deref(), Some("add_pipeline"));
        }

        #[test]
        fn create_pipeline_without_label() {
            let state = make_pipeline_ok("kern");
            assert!(state.label.is_none());
        }

        #[test]
        fn cache_stores_pipeline() {
            let cache = PipelineCache::new(16);
            let state = make_pipeline_ok("kern_a");
            cache.insert("kern_a".to_string(), state).unwrap();
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn cache_hit_returns_pipeline() {
            let cache = PipelineCache::new(16);
            let state = make_pipeline_ok("kern_b");
            cache.insert("kern_b".to_string(), state).unwrap();
            let hit = cache.get("kern_b");
            assert!(hit.is_some());
            assert_eq!(hit.unwrap().function_name, "kern_b");
        }

        #[test]
        fn cache_miss_returns_none() {
            let cache = PipelineCache::new(16);
            assert!(cache.get("nonexistent").is_none());
        }

        #[test]
        fn cache_tracks_hit_count() {
            let cache = PipelineCache::new(16);
            cache.insert("k".to_string(), make_pipeline_ok("k")).unwrap();
            cache.get("k");
            cache.get("k");
            assert_eq!(cache.hits.load(Ordering::Relaxed), 2);
        }

        #[test]
        fn cache_tracks_miss_count() {
            let cache = PipelineCache::new(16);
            cache.get("x");
            cache.get("y");
            assert_eq!(cache.misses.load(Ordering::Relaxed), 2);
        }

        #[test]
        fn cache_hit_rate_empty() {
            let cache = PipelineCache::new(16);
            assert_eq!(cache.hit_rate(), 0.0);
        }

        #[test]
        fn cache_hit_rate_all_hits() {
            let cache = PipelineCache::new(16);
            cache.insert("k".to_string(), make_pipeline_ok("k")).unwrap();
            cache.get("k");
            cache.get("k");
            assert!((cache.hit_rate() - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn cache_evicts_when_full() {
            let cache = PipelineCache::new(2);
            cache.insert("a".to_string(), make_pipeline_ok("a")).unwrap();
            cache.insert("b".to_string(), make_pipeline_ok("b")).unwrap();
            cache.insert("c".to_string(), make_pipeline_ok("c")).unwrap();
            assert_eq!(cache.len(), 2);
        }

        #[test]
        fn cache_remove_existing() {
            let cache = PipelineCache::new(16);
            cache.insert("r".to_string(), make_pipeline_ok("r")).unwrap();
            assert!(cache.remove("r"));
            assert!(cache.is_empty());
        }

        #[test]
        fn cache_remove_nonexistent() {
            let cache = PipelineCache::new(16);
            assert!(!cache.remove("nope"));
        }

        #[test]
        fn cache_clear() {
            let cache = PipelineCache::new(16);
            for i in 0..5 {
                let name = format!("k{i}");
                cache.insert(name.clone(), make_pipeline_ok(&name)).unwrap();
            }
            cache.clear();
            assert!(cache.is_empty());
        }

        #[test]
        fn cache_overwrite_same_key() {
            let cache = PipelineCache::new(16);
            cache.insert("k".to_string(), make_pipeline_ok("k")).unwrap();
            let mut state2 = make_pipeline_ok("k");
            state2.static_threadgroup_memory_length = 512;
            cache.insert("k".to_string(), state2).unwrap();
            assert_eq!(cache.len(), 1);
            let s = cache.get("k").unwrap();
            assert_eq!(s.static_threadgroup_memory_length, 512);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. Pipeline state reflection and function inspection
    // ═══════════════════════════════════════════════════════════════════

    mod reflection {
        use super::*;

        #[test]
        fn reflect_function_name() {
            let state = make_pipeline_ok("softmax_kernel");
            assert_eq!(state.function_name, "softmax_kernel");
        }

        #[test]
        fn reflect_gpu_family() {
            let desc = make_descriptor(make_kernel("k"));
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple9).unwrap();
            assert_eq!(state.gpu_family, MetalGPUFamily::Apple9);
        }

        #[test]
        fn library_enumerate_functions() {
            let mut lib = MetalLibrary::new(Some("test_lib"));
            lib.add_function(make_kernel("fn_a"));
            lib.add_function(make_kernel("fn_b"));
            let names = lib.function_names();
            assert_eq!(names.len(), 2);
            assert!(names.contains(&"fn_a".to_string()));
            assert!(names.contains(&"fn_b".to_string()));
        }

        #[test]
        fn library_get_function_found() {
            let mut lib = MetalLibrary::new(None);
            lib.add_function(make_kernel("my_fn"));
            assert!(lib.get_function("my_fn").is_some());
        }

        #[test]
        fn library_get_function_not_found() {
            let lib = MetalLibrary::new(None);
            assert!(lib.get_function("missing").is_none());
        }

        #[test]
        fn function_type_is_kernel() {
            let f = make_kernel("kern");
            assert_eq!(f.function_type, MetalFunctionType::Kernel);
        }

        #[test]
        fn function_arguments_empty_by_default() {
            let f = make_kernel("kern");
            assert!(f.arguments.is_empty());
        }

        #[test]
        fn function_with_arguments() {
            let f = make_kernel("kern").with_argument(MetalArgument {
                name: "input".to_string(),
                index: 0,
                data_type: MetalDataType::Pointer,
                access: MetalAccess::ReadOnly,
                is_active: true,
            });
            assert_eq!(f.arguments.len(), 1);
            assert_eq!(f.arguments[0].name, "input");
            assert_eq!(f.arguments[0].data_type, MetalDataType::Pointer);
        }

        #[test]
        fn function_argument_access_types() {
            let f = make_kernel("kern")
                .with_argument(MetalArgument {
                    name: "in".to_string(),
                    index: 0,
                    data_type: MetalDataType::Pointer,
                    access: MetalAccess::ReadOnly,
                    is_active: true,
                })
                .with_argument(MetalArgument {
                    name: "out".to_string(),
                    index: 1,
                    data_type: MetalDataType::Pointer,
                    access: MetalAccess::WriteOnly,
                    is_active: true,
                })
                .with_argument(MetalArgument {
                    name: "scratch".to_string(),
                    index: 2,
                    data_type: MetalDataType::Pointer,
                    access: MetalAccess::ReadWrite,
                    is_active: true,
                });
            assert_eq!(f.arguments[0].access, MetalAccess::ReadOnly);
            assert_eq!(f.arguments[1].access, MetalAccess::WriteOnly);
            assert_eq!(f.arguments[2].access, MetalAccess::ReadWrite);
        }

        #[test]
        fn function_argument_data_types() {
            let types = [
                MetalDataType::Float,
                MetalDataType::Half,
                MetalDataType::Int,
                MetalDataType::UInt,
                MetalDataType::Bool,
                MetalDataType::Pointer,
            ];
            for (i, dt) in types.iter().enumerate() {
                let f = make_kernel("k").with_argument(MetalArgument {
                    name: format!("arg{i}"),
                    index: i as u32,
                    data_type: *dt,
                    access: MetalAccess::ReadOnly,
                    is_active: true,
                });
                assert_eq!(f.arguments[0].data_type, *dt);
            }
        }

        #[test]
        fn library_label() {
            let lib = MetalLibrary::new(Some("compute_shaders"));
            assert_eq!(lib.label.as_deref(), Some("compute_shaders"));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. Thread execution width configuration
    // ═══════════════════════════════════════════════════════════════════

    mod thread_execution_width {
        use super::*;

        #[test]
        fn execution_width_apple7() {
            let desc = make_descriptor(make_kernel("k"));
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple7).unwrap();
            assert_eq!(state.thread_execution_width, 32);
        }

        #[test]
        fn execution_width_apple8() {
            let desc = make_descriptor(make_kernel("k"));
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.thread_execution_width, 32);
        }

        #[test]
        fn execution_width_apple9() {
            let desc = make_descriptor(make_kernel("k"));
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple9).unwrap();
            assert_eq!(state.thread_execution_width, 32);
        }

        #[test]
        fn execution_width_divides_max_threads() {
            let state = make_pipeline_ok("k");
            assert_eq!(state.max_total_threads_per_threadgroup % state.thread_execution_width, 0);
        }

        #[test]
        fn execution_width_is_power_of_two() {
            let state = make_pipeline_ok("k");
            assert!(state.thread_execution_width.is_power_of_two());
        }

        #[test]
        fn execution_width_positive() {
            let state = make_pipeline_ok("k");
            assert!(state.thread_execution_width > 0);
        }

        #[test]
        fn descriptor_multiple_of_execution_width_flag_default_false() {
            let desc = ComputePipelineDescriptor::default();
            assert!(!desc.thread_group_size_is_multiple_of_execution_width);
        }

        #[test]
        fn descriptor_multiple_of_execution_width_flag_set() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.thread_group_size_is_multiple_of_execution_width = true;
            desc.max_total_threads_per_threadgroup = 256;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup % state.thread_execution_width, 0);
        }

        #[test]
        fn threadgroup_not_multiple_of_width_error() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.thread_group_size_is_multiple_of_execution_width = true;
            desc.max_total_threads_per_threadgroup = 33;
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. Maximum total threads per threadgroup validation
    // ═══════════════════════════════════════════════════════════════════

    mod threadgroup_validation {
        use super::*;

        #[test]
        fn default_max_threads_is_1024() {
            let desc = ComputePipelineDescriptor::default();
            assert_eq!(desc.max_total_threads_per_threadgroup, 1024);
        }

        #[test]
        fn max_threads_clamped_to_1024() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 2048;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup, 1024);
        }

        #[test]
        fn max_threads_zero_is_error() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 0;
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
        }

        #[test]
        fn max_threads_one_valid() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 1;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup, 1);
        }

        #[test]
        fn max_threads_32_valid() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 32;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup, 32);
        }

        #[test]
        fn max_threads_64_valid() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 64;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup, 64);
        }

        #[test]
        fn max_threads_256_valid() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 256;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup, 256);
        }

        #[test]
        fn max_threads_512_valid() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 512;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup, 512);
        }

        #[test]
        fn max_threads_1024_valid() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 1024;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.max_total_threads_per_threadgroup, 1024);
        }

        #[test]
        fn max_threads_exact_at_limit() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 1024;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert!(state.max_total_threads_per_threadgroup <= 1024);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. Pipeline statistics collection
    // ═══════════════════════════════════════════════════════════════════

    mod statistics {
        use super::*;

        #[test]
        fn default_statistics_zeroed() {
            let stats = PipelineStatistics::default();
            assert_eq!(stats.compilation_time_ns, 0);
            assert_eq!(stats.instruction_count, 0);
            assert_eq!(stats.spill_count, 0);
            assert_eq!(stats.occupancy, 0.0);
            assert_eq!(stats.sgpr_count, 0);
            assert_eq!(stats.vgpr_count, 0);
        }

        #[test]
        fn pipeline_state_has_default_statistics() {
            let state = make_pipeline_ok("k");
            assert_eq!(state.statistics.compilation_time_ns, 0);
        }

        #[test]
        fn statistics_compilation_time() {
            let mut stats = PipelineStatistics::default();
            stats.compilation_time_ns = 1_500_000;
            assert_eq!(stats.compilation_time_ns, 1_500_000);
        }

        #[test]
        fn statistics_instruction_count() {
            let mut stats = PipelineStatistics::default();
            stats.instruction_count = 42;
            assert_eq!(stats.instruction_count, 42);
        }

        #[test]
        fn statistics_spill_count() {
            let mut stats = PipelineStatistics::default();
            stats.spill_count = 3;
            assert_eq!(stats.spill_count, 3);
        }

        #[test]
        fn statistics_occupancy() {
            let mut stats = PipelineStatistics::default();
            stats.occupancy = 0.75;
            assert!((stats.occupancy - 0.75).abs() < f32::EPSILON);
        }

        #[test]
        fn statistics_register_counts() {
            let mut stats = PipelineStatistics::default();
            stats.sgpr_count = 16;
            stats.vgpr_count = 32;
            assert_eq!(stats.sgpr_count, 16);
            assert_eq!(stats.vgpr_count, 32);
        }

        #[test]
        fn statistics_clone() {
            let mut stats = PipelineStatistics::default();
            stats.instruction_count = 100;
            stats.occupancy = 0.5;
            let cloned = stats.clone();
            assert_eq!(cloned.instruction_count, 100);
            assert!((cloned.occupancy - 0.5).abs() < f32::EPSILON);
        }

        #[test]
        fn statistics_debug_format() {
            let stats = PipelineStatistics::default();
            let dbg = format!("{stats:?}");
            assert!(dbg.contains("compilation_time_ns"));
            assert!(dbg.contains("instruction_count"));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. Async pipeline compilation
    // ═══════════════════════════════════════════════════════════════════

    mod async_compilation {
        use super::*;

        #[test]
        fn handle_initial_status_is_pending() {
            let handle = AsyncCompilationHandle::new();
            assert_eq!(handle.status(), CompilationStatus::Pending);
        }

        #[test]
        fn handle_complete_success() {
            let handle = AsyncCompilationHandle::new();
            let state = make_pipeline_ok("k");
            handle.complete(Ok(state));
            assert_eq!(handle.status(), CompilationStatus::Success);
        }

        #[test]
        fn handle_complete_error() {
            let handle = AsyncCompilationHandle::new();
            handle.complete(Err(PipelineError::CompilationFailed("shader error".to_string())));
            assert_eq!(handle.status(), CompilationStatus::Error);
        }

        #[test]
        fn handle_take_result_success() {
            let handle = AsyncCompilationHandle::new();
            handle.complete(Ok(make_pipeline_ok("k")));
            let result = handle.take_result();
            assert!(result.is_some());
            assert!(result.unwrap().is_ok());
        }

        #[test]
        fn handle_take_result_error() {
            let handle = AsyncCompilationHandle::new();
            handle.complete(Err(PipelineError::MissingFunction));
            let result = handle.take_result();
            assert!(result.is_some());
            assert!(result.unwrap().is_err());
        }

        #[test]
        fn handle_take_result_consumes() {
            let handle = AsyncCompilationHandle::new();
            handle.complete(Ok(make_pipeline_ok("k")));
            let _ = handle.take_result();
            assert!(handle.take_result().is_none());
        }

        #[test]
        fn handle_no_result_before_completion() {
            let handle = AsyncCompilationHandle::new();
            assert!(handle.take_result().is_none());
        }

        #[test]
        fn handle_status_thread_safe() {
            let handle = Arc::new(AsyncCompilationHandle::new());
            let h2 = Arc::clone(&handle);
            let t = std::thread::spawn(move || {
                h2.complete(Ok(make_pipeline_ok("threaded")));
            });
            t.join().unwrap();
            assert_eq!(handle.status(), CompilationStatus::Success);
        }

        #[test]
        fn multiple_handles_independent() {
            let h1 = AsyncCompilationHandle::new();
            let h2 = AsyncCompilationHandle::new();
            h1.complete(Ok(make_pipeline_ok("k1")));
            assert_eq!(h1.status(), CompilationStatus::Success);
            assert_eq!(h2.status(), CompilationStatus::Pending);
        }

        #[test]
        fn compilation_status_values() {
            assert_ne!(CompilationStatus::Pending, CompilationStatus::Compiling);
            assert_ne!(CompilationStatus::Success, CompilationStatus::Error);
            assert_ne!(CompilationStatus::Pending, CompilationStatus::Success);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 7. Pipeline state serialization/deserialization
    // ═══════════════════════════════════════════════════════════════════

    mod serialization {
        use super::*;

        #[test]
        fn serialize_produces_data() {
            let state = make_pipeline_ok("matmul");
            let blob = state.serialize().unwrap();
            assert!(!blob.data.is_empty());
        }

        #[test]
        fn serialize_preserves_function_name() {
            let state = make_pipeline_ok("softmax");
            let blob = state.serialize().unwrap();
            assert_eq!(blob.function_name, "softmax");
        }

        #[test]
        fn serialize_preserves_gpu_family() {
            let desc = make_descriptor(make_kernel("k"));
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple9).unwrap();
            let blob = state.serialize().unwrap();
            assert_eq!(blob.gpu_family, MetalGPUFamily::Apple9);
        }

        #[test]
        fn serialize_has_version() {
            let state = make_pipeline_ok("k");
            let blob = state.serialize().unwrap();
            assert_eq!(blob.version, 1);
        }

        #[test]
        fn deserialize_roundtrip_function_name() {
            let state = make_pipeline_ok("layernorm");
            let blob = state.serialize().unwrap();
            let restored = blob.deserialize().unwrap();
            assert_eq!(restored.function_name, "layernorm");
        }

        #[test]
        fn deserialize_roundtrip_max_threads() {
            let state = make_pipeline_ok("k");
            let blob = state.serialize().unwrap();
            let restored = blob.deserialize().unwrap();
            assert_eq!(
                restored.max_total_threads_per_threadgroup,
                state.max_total_threads_per_threadgroup
            );
        }

        #[test]
        fn deserialize_roundtrip_gpu_family() {
            let desc = make_descriptor(make_kernel("k"));
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple7).unwrap();
            let blob = state.serialize().unwrap();
            let restored = blob.deserialize().unwrap();
            assert_eq!(restored.gpu_family, MetalGPUFamily::Apple7);
        }

        #[test]
        fn deserialize_invalid_data() {
            let blob = SerializedPipeline {
                data: vec![0xFF, 0xFE],
                function_name: "k".to_string(),
                gpu_family: MetalGPUFamily::Apple8,
                version: 1,
            };
            assert!(blob.deserialize().is_err());
        }

        #[test]
        fn deserialize_truncated_data() {
            let blob = SerializedPipeline {
                data: b"only_name".to_vec(),
                function_name: "k".to_string(),
                gpu_family: MetalGPUFamily::Apple8,
                version: 1,
            };
            assert!(blob.deserialize().is_err());
        }

        #[test]
        fn serialize_multiple_pipelines() {
            let s1 = make_pipeline_ok("k1");
            let s2 = make_pipeline_ok("k2");
            let b1 = s1.serialize().unwrap();
            let b2 = s2.serialize().unwrap();
            assert_ne!(b1.function_name, b2.function_name);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 8. Linked functions and visible functions
    // ═══════════════════════════════════════════════════════════════════

    mod linked_and_visible_functions {
        use super::*;

        #[test]
        fn descriptor_linked_functions_default_empty() {
            let desc = ComputePipelineDescriptor::default();
            assert!(desc.linked_functions.is_empty());
        }

        #[test]
        fn descriptor_with_linked_functions() {
            let mut desc = make_descriptor(make_kernel("main"));
            desc.linked_functions.push(make_kernel("helper_a"));
            desc.linked_functions.push(make_kernel("helper_b"));
            assert_eq!(desc.linked_functions.len(), 2);
        }

        #[test]
        fn pipeline_creation_with_linked_functions() {
            let mut desc = make_descriptor(make_kernel("main"));
            desc.linked_functions.push(make_kernel("utility"));
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.function_name, "main");
        }

        #[test]
        fn visible_function_table_empty() {
            let table = VisibleFunctionTable::new();
            assert!(table.is_empty());
            assert_eq!(table.len(), 0);
        }

        #[test]
        fn visible_function_table_add() {
            let mut table = VisibleFunctionTable::new();
            table.add(make_kernel("visible_fn"));
            assert_eq!(table.len(), 1);
        }

        #[test]
        fn visible_function_table_get() {
            let mut table = VisibleFunctionTable::new();
            table.add(make_kernel("vf"));
            let f = table.get(0).unwrap();
            assert_eq!(f.name, "vf");
        }

        #[test]
        fn visible_function_table_get_out_of_bounds() {
            let table = VisibleFunctionTable::new();
            assert!(table.get(0).is_none());
        }

        #[test]
        fn visible_function_table_multiple() {
            let mut table = VisibleFunctionTable::new();
            table.add(make_kernel("a"));
            table.add(make_kernel("b"));
            table.add(make_kernel("c"));
            assert_eq!(table.len(), 3);
            assert_eq!(table.get(1).unwrap().name, "b");
        }

        #[test]
        fn linked_function_names_preserved() {
            let mut desc = make_descriptor(make_kernel("entry"));
            desc.linked_functions.push(make_kernel("linked_1"));
            desc.linked_functions.push(make_kernel("linked_2"));
            assert_eq!(desc.linked_functions[0].name, "linked_1");
            assert_eq!(desc.linked_functions[1].name, "linked_2");
        }

        #[test]
        fn descriptor_max_call_stack_depth_default() {
            let desc = ComputePipelineDescriptor::default();
            assert_eq!(desc.max_call_stack_depth, 1);
        }

        #[test]
        fn descriptor_max_call_stack_depth_custom() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_call_stack_depth = 8;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert_eq!(state.function_name, "k");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 9. Pipeline state pool management
    // ═══════════════════════════════════════════════════════════════════

    mod pool_management {
        use super::*;

        #[test]
        fn pool_initial_capacity() {
            let pool = PipelinePool::new(4);
            assert_eq!(pool.capacity(), 4);
        }

        #[test]
        fn pool_initial_available_zero() {
            let pool = PipelinePool::new(4);
            assert_eq!(pool.available(), 0);
        }

        #[test]
        fn pool_release_then_acquire() {
            let pool = PipelinePool::new(4);
            let state = make_pipeline_ok("pooled");
            pool.release(0, state).unwrap();
            assert_eq!(pool.available(), 1);
            let (idx, acquired) = pool.acquire().unwrap();
            assert_eq!(idx, 0);
            assert_eq!(acquired.function_name, "pooled");
        }

        #[test]
        fn pool_acquire_empty_errors() {
            let pool = PipelinePool::new(4);
            assert!(pool.acquire().is_err());
        }

        #[test]
        fn pool_release_multiple() {
            let pool = PipelinePool::new(4);
            for i in 0..4 {
                let name = format!("k{i}");
                pool.release(i, make_pipeline_ok(&name)).unwrap();
            }
            assert_eq!(pool.available(), 4);
        }

        #[test]
        fn pool_acquire_decreases_available() {
            let pool = PipelinePool::new(4);
            pool.release(0, make_pipeline_ok("k")).unwrap();
            pool.release(1, make_pipeline_ok("k2")).unwrap();
            assert_eq!(pool.available(), 2);
            let _ = pool.acquire().unwrap();
            assert_eq!(pool.available(), 1);
        }

        #[test]
        fn pool_full_cycle() {
            let pool = PipelinePool::new(2);
            pool.release(0, make_pipeline_ok("a")).unwrap();
            pool.release(1, make_pipeline_ok("b")).unwrap();
            let (i1, s1) = pool.acquire().unwrap();
            let (i2, s2) = pool.acquire().unwrap();
            assert!(pool.acquire().is_err());
            pool.release(i1, s1).unwrap();
            pool.release(i2, s2).unwrap();
            assert_eq!(pool.available(), 2);
        }

        #[test]
        fn pool_release_preserves_state_data() {
            let pool = PipelinePool::new(2);
            let mut state = make_pipeline_ok("special");
            state.static_threadgroup_memory_length = 4096;
            pool.release(0, state).unwrap();
            let (_, acquired) = pool.acquire().unwrap();
            assert_eq!(acquired.static_threadgroup_memory_length, 4096);
            assert_eq!(acquired.function_name, "special");
        }

        #[test]
        fn pool_capacity_is_fixed() {
            let pool = PipelinePool::new(3);
            assert_eq!(pool.capacity(), 3);
            pool.release(0, make_pipeline_ok("k")).unwrap();
            assert_eq!(pool.capacity(), 3);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 10. Error handling for invalid pipeline configs
    // ═══════════════════════════════════════════════════════════════════

    mod error_handling {
        use super::*;

        #[test]
        fn error_missing_function() {
            let desc = ComputePipelineDescriptor::default();
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
            let err = result.unwrap_err();
            let msg = format!("{err}");
            assert!(msg.contains("function"));
        }

        #[test]
        fn error_invalid_function_type_vertex() {
            let func = MetalFunction {
                name: "vert".to_string(),
                function_type: MetalFunctionType::Vertex,
                arguments: Vec::new(),
                patch_type: None,
            };
            let desc = make_descriptor(func);
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
        }

        #[test]
        fn error_invalid_function_type_fragment() {
            let func = MetalFunction {
                name: "frag".to_string(),
                function_type: MetalFunctionType::Fragment,
                arguments: Vec::new(),
                patch_type: None,
            };
            let desc = make_descriptor(func);
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
        }

        #[test]
        fn error_zero_threadgroup_size() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_total_threads_per_threadgroup = 0;
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
            let msg = format!("{}", result.unwrap_err());
            assert!(msg.contains("threadgroup size"));
        }

        #[test]
        fn error_threadgroup_not_multiple_of_width() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.thread_group_size_is_multiple_of_execution_width = true;
            desc.max_total_threads_per_threadgroup = 100;
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
            let msg = format!("{}", result.unwrap_err());
            assert!(msg.contains("multiple"));
        }

        #[test]
        fn error_excessive_call_stack_depth() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_call_stack_depth = 32;
            let result = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8);
            assert!(result.is_err());
            let msg = format!("{}", result.unwrap_err());
            assert!(msg.contains("call stack"));
        }

        #[test]
        fn error_display_missing_function() {
            let err = PipelineError::MissingFunction;
            let msg = format!("{err}");
            assert!(!msg.is_empty());
        }

        #[test]
        fn error_display_function_not_found() {
            let err = PipelineError::FunctionNotFound("foo".to_string());
            let msg = format!("{err}");
            assert!(msg.contains("foo"));
        }

        #[test]
        fn error_display_compilation_failed() {
            let err = PipelineError::CompilationFailed("syntax error".to_string());
            let msg = format!("{err}");
            assert!(msg.contains("syntax error"));
        }

        #[test]
        fn error_display_serialization_failed() {
            let err = PipelineError::SerializationFailed("io error".to_string());
            let msg = format!("{err}");
            assert!(msg.contains("io error"));
        }

        #[test]
        fn error_display_deserialization_failed() {
            let err = PipelineError::DeserializationFailed("corrupt".to_string());
            let msg = format!("{err}");
            assert!(msg.contains("corrupt"));
        }

        #[test]
        fn error_display_pool_exhausted() {
            let err = PipelineError::PoolExhausted;
            let msg = format!("{err}");
            assert!(msg.contains("pool"));
        }

        #[test]
        fn error_display_library_error() {
            let err = PipelineError::LibraryError("bad source".to_string());
            let msg = format!("{err}");
            assert!(msg.contains("bad source"));
        }

        #[test]
        fn error_debug_format_all_variants() {
            let errors: Vec<PipelineError> = vec![
                PipelineError::MissingFunction,
                PipelineError::InvalidFunctionType(MetalFunctionType::Vertex),
                PipelineError::InvalidThreadgroupSize(0),
                PipelineError::ThreadgroupNotMultipleOfWidth {
                    threadgroup_size: 33,
                    execution_width: 32,
                },
                PipelineError::FunctionNotFound("f".to_string()),
                PipelineError::CompilationFailed("c".to_string()),
                PipelineError::SerializationFailed("s".to_string()),
                PipelineError::DeserializationFailed("d".to_string()),
                PipelineError::PoolExhausted,
                PipelineError::ExcessiveCallStackDepth(99),
                PipelineError::LibraryError("l".to_string()),
            ];
            for err in &errors {
                let dbg = format!("{err:?}");
                assert!(!dbg.is_empty());
            }
        }

        #[test]
        fn valid_kernel_function_succeeds() {
            let desc = make_descriptor(make_kernel("good_kernel"));
            assert!(ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).is_ok());
        }

        #[test]
        fn call_stack_depth_boundary_16_ok() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_call_stack_depth = 16;
            assert!(ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).is_ok());
        }

        #[test]
        fn call_stack_depth_boundary_17_error() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.max_call_stack_depth = 17;
            assert!(ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).is_err());
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Additional coverage: descriptor defaults, mutability, stage input,
    // indirect command buffers, and cross-cutting concerns
    // ═══════════════════════════════════════════════════════════════════

    mod descriptor_defaults_and_extras {
        use super::*;

        #[test]
        fn descriptor_default_label_none() {
            let desc = ComputePipelineDescriptor::default();
            assert!(desc.label.is_none());
        }

        #[test]
        fn descriptor_default_compute_function_none() {
            let desc = ComputePipelineDescriptor::default();
            assert!(desc.compute_function.is_none());
        }

        #[test]
        fn descriptor_default_stage_input_none() {
            let desc = ComputePipelineDescriptor::default();
            assert!(desc.stage_input_descriptor.is_none());
        }

        #[test]
        fn descriptor_default_indirect_cmd_buf_false() {
            let desc = ComputePipelineDescriptor::default();
            assert!(!desc.support_indirect_command_buffers);
        }

        #[test]
        fn descriptor_default_binary_functions_false() {
            let desc = ComputePipelineDescriptor::default();
            assert!(!desc.support_adding_binary_functions);
        }

        #[test]
        fn descriptor_default_mutability() {
            let desc = ComputePipelineDescriptor::default();
            assert_eq!(desc.mutability, MetalMutability::Default);
        }

        #[test]
        fn pipeline_supports_indirect_command_buffers() {
            let mut desc = make_descriptor(make_kernel("k"));
            desc.support_indirect_command_buffers = true;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple8).unwrap();
            assert!(state.supports_indirect_command_buffers);
        }

        #[test]
        fn pipeline_no_indirect_command_buffers_by_default() {
            let state = make_pipeline_ok("k");
            assert!(!state.supports_indirect_command_buffers);
        }

        #[test]
        fn stage_input_descriptor_attributes() {
            let sid = StageInputDescriptor {
                attributes: vec![AttributeDescriptor {
                    format: MetalDataType::Float,
                    offset: 0,
                    buffer_index: 0,
                }],
                layouts: vec![BufferLayoutDescriptor {
                    stride: 16,
                    step_function: StepFunction::PerVertex,
                    step_rate: 1,
                }],
            };
            assert_eq!(sid.attributes.len(), 1);
            assert_eq!(sid.layouts.len(), 1);
            assert_eq!(sid.layouts[0].stride, 16);
        }

        #[test]
        fn step_function_variants() {
            assert_ne!(StepFunction::Constant, StepFunction::PerVertex);
            assert_ne!(StepFunction::PerInstance, StepFunction::Constant);
            assert_ne!(StepFunction::ThreadPositionInGridX, StepFunction::PerVertex);
        }

        #[test]
        fn gpu_family_ordering() {
            assert!(MetalGPUFamily::Apple7 < MetalGPUFamily::Apple8);
            assert!(MetalGPUFamily::Apple8 < MetalGPUFamily::Apple9);
        }

        #[test]
        fn pipeline_static_threadgroup_memory_default_zero() {
            let state = make_pipeline_ok("k");
            assert_eq!(state.static_threadgroup_memory_length, 0);
        }
    }

    mod cross_cutting {
        use super::*;

        #[test]
        fn pipeline_creation_across_all_gpu_families() {
            for family in [MetalGPUFamily::Apple7, MetalGPUFamily::Apple8, MetalGPUFamily::Apple9] {
                let desc = make_descriptor(make_kernel("k"));
                let state = ComputePipelineState::new(&desc, family).unwrap();
                assert_eq!(state.gpu_family, family);
            }
        }

        #[test]
        fn cache_concurrent_access() {
            let cache = Arc::new(PipelineCache::new(64));
            let mut handles = Vec::new();
            for i in 0..8 {
                let c = Arc::clone(&cache);
                handles.push(std::thread::spawn(move || {
                    let name = format!("thread_{i}");
                    let state = make_pipeline_ok(&name);
                    c.insert(name.clone(), state).unwrap();
                    c.get(&name);
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            assert!(cache.len() <= 64);
        }

        #[test]
        fn pipeline_clone_preserves_all_fields() {
            let mut desc = make_descriptor(make_kernel("cloneme"));
            desc.label = Some("my_label".to_string());
            desc.max_total_threads_per_threadgroup = 256;
            let state = ComputePipelineState::new(&desc, MetalGPUFamily::Apple9).unwrap();
            let cloned = state.clone();
            assert_eq!(cloned.label, state.label);
            assert_eq!(cloned.function_name, state.function_name);
            assert_eq!(
                cloned.max_total_threads_per_threadgroup,
                state.max_total_threads_per_threadgroup
            );
            assert_eq!(cloned.gpu_family, state.gpu_family);
            assert_eq!(cloned.thread_execution_width, state.thread_execution_width);
        }
    }
}
