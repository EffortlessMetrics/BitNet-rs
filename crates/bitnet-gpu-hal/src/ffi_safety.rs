//! FFI safety layer for GPU library bindings.
//!
//! Provides safe wrappers around raw FFI calls: library loading,
//! symbol checking, handle management, null-pointer guards, and
//! bounds-checked buffer access.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────────────────

/// Errors that can occur during FFI operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfiError {
    /// The requested shared library could not be found.
    LibraryNotFound(String),
    /// A required symbol was not found in the loaded library.
    SymbolNotFound(String),
    /// The handle has been invalidated or was never valid.
    InvalidHandle,
    /// A null pointer was passed where a valid pointer is required.
    NullPointer,
    /// A buffer access would exceed the allocated size.
    BufferOverflow { expected: usize, actual: usize },
    /// An argument failed validation.
    InvalidArgument(String),
    /// A runtime error returned by the foreign library.
    RuntimeError { code: i32, message: String },
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound(name) => {
                write!(f, "library not found: {name}")
            }
            Self::SymbolNotFound(name) => {
                write!(f, "symbol not found: {name}")
            }
            Self::InvalidHandle => write!(f, "invalid handle"),
            Self::NullPointer => write!(f, "null pointer"),
            Self::BufferOverflow { expected, actual } => {
                write!(
                    f,
                    "buffer overflow: expected at most {expected}, \
                     got {actual}"
                )
            }
            Self::InvalidArgument(msg) => {
                write!(f, "invalid argument: {msg}")
            }
            Self::RuntimeError { code, message } => {
                write!(f, "runtime error {code}: {message}")
            }
        }
    }
}

impl std::error::Error for FfiError {}

// ── SymbolCheck ──────────────────────────────────────────────────────────

/// Record of a single symbol lookup in a loaded library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolCheck {
    /// Symbol name.
    pub name: String,
    /// Whether the symbol is required for correct operation.
    pub required: bool,
    /// Whether the symbol was found.
    pub found: bool,
}

// ── FfiGuard ─────────────────────────────────────────────────────────────

/// Tracks library load state and validates that required symbols exist.
#[derive(Debug, Clone)]
pub struct FfiGuard {
    library_name: String,
    loaded: bool,
    symbol_checks: Vec<SymbolCheck>,
}

impl FfiGuard {
    /// Create a new guard for the given library name.
    pub fn new(library_name: &str) -> Self {
        Self {
            library_name: library_name.to_string(),
            loaded: false,
            symbol_checks: Vec::new(),
        }
    }

    /// Mark the library as loaded.
    pub const fn set_loaded(&mut self) {
        self.loaded = true;
    }

    /// Whether the library has been loaded.
    pub const fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Library name.
    pub fn library_name(&self) -> &str {
        &self.library_name
    }

    /// Record a symbol check.
    ///
    /// Returns `true` if the symbol was found, `false` otherwise.
    pub fn check_symbol(&mut self, name: &str, required: bool) -> bool {
        // Simulate: symbol is "found" if name is non-empty and doesn't
        // start with `_missing_`.
        let found = !name.is_empty() && !name.starts_with("_missing_");
        self.symbol_checks.push(SymbolCheck {
            name: name.to_string(),
            required,
            found,
        });
        found
    }

    /// Register a pre-built [`SymbolCheck`].
    pub fn add_symbol_check(&mut self, check: SymbolCheck) {
        self.symbol_checks.push(check);
    }

    /// `true` when every *required* symbol has been found.
    pub fn all_required_found(&self) -> bool {
        self.symbol_checks
            .iter()
            .filter(|s| s.required)
            .all(|s| s.found)
    }

    /// Iterate over all recorded symbol checks.
    pub fn symbol_checks(&self) -> &[SymbolCheck] {
        &self.symbol_checks
    }

    /// Return names of required symbols that were **not** found.
    pub fn missing_required(&self) -> Vec<&str> {
        self.symbol_checks
            .iter()
            .filter(|s| s.required && !s.found)
            .map(|s| s.name.as_str())
            .collect()
    }
}

// ── SafeHandle ───────────────────────────────────────────────────────────

/// RAII wrapper around an opaque resource obtained via FFI.
///
/// The handle can be invalidated to prevent use-after-free, and
/// ownership can be moved out exactly once with [`take`](Self::take).
#[derive(Debug)]
pub struct SafeHandle<T> {
    inner: Option<T>,
    valid: bool,
    created_at: u64,
}

impl<T> SafeHandle<T> {
    /// Wrap a value in a new, valid handle.
    pub const fn new(value: T) -> Self {
        Self { inner: Some(value), valid: true, created_at: 0 }
    }

    /// Wrap a value with an explicit creation timestamp.
    pub const fn with_timestamp(value: T, ts: u64) -> Self {
        Self { inner: Some(value), valid: true, created_at: ts }
    }

    /// Borrow the inner value if the handle is still valid.
    pub fn get(&self) -> Result<&T, FfiError> {
        if !self.valid {
            return Err(FfiError::InvalidHandle);
        }
        self.inner.as_ref().ok_or(FfiError::InvalidHandle)
    }

    /// Mutably borrow the inner value if the handle is still valid.
    pub fn get_mut(&mut self) -> Result<&mut T, FfiError> {
        if !self.valid {
            return Err(FfiError::InvalidHandle);
        }
        self.inner.as_mut().ok_or(FfiError::InvalidHandle)
    }

    /// Take ownership of the inner value, consuming the handle's
    /// contents. Subsequent calls will return [`FfiError::InvalidHandle`].
    pub fn take(&mut self) -> Result<T, FfiError> {
        if !self.valid {
            return Err(FfiError::InvalidHandle);
        }
        self.valid = false;
        self.inner.take().ok_or(FfiError::InvalidHandle)
    }

    /// Invalidate the handle without returning the inner value.
    pub const fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Whether the handle is still valid.
    pub const fn is_valid(&self) -> bool {
        self.valid
    }

    /// Creation timestamp.
    pub const fn created_at(&self) -> u64 {
        self.created_at
    }
}

// ── FfiResult ────────────────────────────────────────────────────────────

/// Wrapper for results coming back from FFI calls that communicate
/// success or failure via integer error codes.
#[derive(Debug, Clone)]
pub struct FfiResult<T> {
    value: Option<T>,
    error_code: i32,
    error_message: Option<String>,
}

impl<T> FfiResult<T> {
    /// Construct a successful result.
    pub const fn ok(value: T) -> Self {
        Self { value: Some(value), error_code: 0, error_message: None }
    }

    /// Construct an error result.
    pub fn err(code: i32, message: &str) -> Self {
        Self {
            value: None,
            error_code: code,
            error_message: Some(message.to_string()),
        }
    }

    /// The raw error code (`0` means success).
    pub const fn error_code(&self) -> i32 {
        self.error_code
    }

    /// Whether this result represents success.
    pub const fn is_ok(&self) -> bool {
        self.error_code == 0 && self.value.is_some()
    }

    /// Convert to a standard `Result`.
    pub fn into_result(self) -> Result<T, FfiError> {
        if self.error_code != 0 {
            return Err(FfiError::RuntimeError {
                code: self.error_code,
                message: self
                    .error_message
                    .unwrap_or_else(|| "unknown error".to_string()),
            });
        }
        self.value.ok_or(FfiError::NullPointer)
    }
}

// ── LoadedLibrary ────────────────────────────────────────────────────────

/// Metadata for a library that has been successfully loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedLibrary {
    /// Library display name.
    pub name: String,
    /// Filesystem path from which the library was loaded.
    pub path: String,
    /// Optional version string (e.g. `"12.3"`).
    pub version: Option<String>,
    /// Symbols that were discovered in this library.
    pub symbols: Vec<String>,
}

// ── LibraryLoader ────────────────────────────────────────────────────────

/// Searches a set of paths for shared libraries and tracks loaded ones.
#[derive(Debug, Clone)]
pub struct LibraryLoader {
    search_paths: Vec<String>,
    loaded_libraries: Vec<LoadedLibrary>,
}

impl LibraryLoader {
    /// Create an empty loader with no search paths.
    pub const fn new() -> Self {
        Self { search_paths: Vec::new(), loaded_libraries: Vec::new() }
    }

    /// Append a search path.
    pub fn add_search_path(&mut self, path: &str) {
        self.search_paths.push(path.to_string());
    }

    /// The current ordered list of search paths.
    pub fn search_paths(&self) -> &[String] {
        &self.search_paths
    }

    /// Simulate finding a library on disk.
    ///
    /// Returns the first search path that conceptually contains `name`.
    /// For real usage this would probe the filesystem; here we return
    /// the first path whose string contains the library name.
    pub fn find_library(&self, name: &str) -> Option<&str> {
        self.search_paths
            .iter()
            .find(|p| p.contains(name))
            .map(String::as_str)
    }

    /// Register a library as loaded.
    pub fn register(&mut self, lib: LoadedLibrary) {
        self.loaded_libraries.push(lib);
    }

    /// All currently loaded libraries.
    pub fn loaded_libraries(&self) -> &[LoadedLibrary] {
        &self.loaded_libraries
    }

    /// Look up a loaded library by name.
    pub fn get_loaded(&self, name: &str) -> Option<&LoadedLibrary> {
        self.loaded_libraries.iter().find(|l| l.name == name)
    }
}

impl Default for LibraryLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pointer / bounds guards ──────────────────────────────────────────────

/// Zero-size guard for null-pointer validation.
pub struct NullCheckGuard;

impl NullCheckGuard {
    /// Return `Ok(())` when `ptr` is non-null, or
    /// `Err(FfiError::NullPointer)` otherwise.
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer is valid for reads if it is
    /// non-null. This function only checks for null.
    pub const fn check_ptr<T>(ptr: *const T) -> Result<(), FfiError> {
        if ptr.is_null() {
            Err(FfiError::NullPointer)
        } else {
            Ok(())
        }
    }

    /// Convenience wrapper for mutable pointers.
    pub const fn check_mut_ptr<T>(ptr: *mut T) -> Result<(), FfiError> {
        if ptr.is_null() {
            Err(FfiError::NullPointer)
        } else {
            Ok(())
        }
    }
}

/// Guard that validates buffer accesses stay within a declared size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundsCheckGuard {
    max_size: usize,
}

impl BoundsCheckGuard {
    /// Create a guard for buffers of up to `max_size` elements.
    pub const fn new(max_size: usize) -> Self {
        Self { max_size }
    }

    /// Maximum size this guard was configured with.
    pub const fn max_size(&self) -> usize {
        self.max_size
    }

    /// Validate that `offset + size` does not exceed `max_size`.
    pub const fn check(
        &self,
        offset: usize,
        size: usize,
    ) -> Result<(), FfiError> {
        let end = offset.saturating_add(size);
        if end > self.max_size {
            Err(FfiError::BufferOverflow { expected: self.max_size, actual: end })
        } else {
            Ok(())
        }
    }

    /// Validate a single index.
    pub const fn check_index(&self, index: usize) -> Result<(), FfiError> {
        if index >= self.max_size {
            Err(FfiError::BufferOverflow {
                expected: self.max_size,
                actual: index,
            })
        } else {
            Ok(())
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- FfiError Display -------------------------------------------------

    #[test]
    fn display_library_not_found() {
        let e = FfiError::LibraryNotFound("libfoo.so".into());
        assert_eq!(e.to_string(), "library not found: libfoo.so");
    }

    #[test]
    fn display_symbol_not_found() {
        let e = FfiError::SymbolNotFound("cuInit".into());
        assert_eq!(e.to_string(), "symbol not found: cuInit");
    }

    #[test]
    fn display_invalid_handle() {
        assert_eq!(FfiError::InvalidHandle.to_string(), "invalid handle");
    }

    #[test]
    fn display_null_pointer() {
        assert_eq!(FfiError::NullPointer.to_string(), "null pointer");
    }

    #[test]
    fn display_buffer_overflow() {
        let e = FfiError::BufferOverflow { expected: 100, actual: 120 };
        assert_eq!(
            e.to_string(),
            "buffer overflow: expected at most 100, got 120"
        );
    }

    #[test]
    fn display_invalid_argument() {
        let e = FfiError::InvalidArgument("bad size".into());
        assert_eq!(e.to_string(), "invalid argument: bad size");
    }

    #[test]
    fn display_runtime_error() {
        let e = FfiError::RuntimeError {
            code: -1,
            message: "device lost".into(),
        };
        assert_eq!(e.to_string(), "runtime error -1: device lost");
    }

    #[test]
    fn ffi_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FfiError>();
    }

    // -- FfiGuard ---------------------------------------------------------

    #[test]
    fn guard_new_default_state() {
        let g = FfiGuard::new("libcuda.so");
        assert_eq!(g.library_name(), "libcuda.so");
        assert!(!g.is_loaded());
        assert!(g.symbol_checks().is_empty());
    }

    #[test]
    fn guard_set_loaded() {
        let mut g = FfiGuard::new("lib");
        g.set_loaded();
        assert!(g.is_loaded());
    }

    #[test]
    fn guard_check_symbol_required_found() {
        let mut g = FfiGuard::new("lib");
        assert!(g.check_symbol("cuInit", true));
        assert!(g.all_required_found());
    }

    #[test]
    fn guard_check_symbol_required_not_found() {
        let mut g = FfiGuard::new("lib");
        assert!(!g.check_symbol("_missing_cuInit", true));
        assert!(!g.all_required_found());
    }

    #[test]
    fn guard_check_symbol_optional_not_found() {
        let mut g = FfiGuard::new("lib");
        assert!(!g.check_symbol("_missing_opt", false));
        // all_required_found is still true (no required symbols)
        assert!(g.all_required_found());
    }

    #[test]
    fn guard_mixed_required_and_optional() {
        let mut g = FfiGuard::new("lib");
        g.check_symbol("cuInit", true);
        g.check_symbol("_missing_opt", false);
        g.check_symbol("cuDeviceGet", true);
        assert!(g.all_required_found());
        assert_eq!(g.symbol_checks().len(), 3);
    }

    #[test]
    fn guard_missing_required_reported() {
        let mut g = FfiGuard::new("lib");
        g.check_symbol("cuInit", true);
        g.check_symbol("_missing_cuLaunch", true);
        assert!(!g.all_required_found());
        assert_eq!(g.missing_required(), vec!["_missing_cuLaunch"]);
    }

    #[test]
    fn guard_add_symbol_check_manually() {
        let mut g = FfiGuard::new("lib");
        g.add_symbol_check(SymbolCheck {
            name: "myFunc".into(),
            required: true,
            found: true,
        });
        assert!(g.all_required_found());
    }

    #[test]
    fn guard_all_required_found_empty() {
        let g = FfiGuard::new("lib");
        assert!(g.all_required_found());
    }

    #[test]
    fn guard_empty_name_not_found() {
        let mut g = FfiGuard::new("lib");
        assert!(!g.check_symbol("", true));
    }

    // -- SafeHandle -------------------------------------------------------

    #[test]
    fn handle_get_valid() {
        let h = SafeHandle::new(42u64);
        assert_eq!(*h.get().unwrap(), 42);
    }

    #[test]
    fn handle_get_mut_valid() {
        let mut h = SafeHandle::new(1u32);
        *h.get_mut().unwrap() = 99;
        assert_eq!(*h.get().unwrap(), 99);
    }

    #[test]
    fn handle_get_invalidated() {
        let mut h = SafeHandle::new(10);
        h.invalidate();
        assert!(!h.is_valid());
        assert_eq!(h.get().unwrap_err(), FfiError::InvalidHandle);
    }

    #[test]
    fn handle_take_moves_ownership() {
        let mut h = SafeHandle::new(String::from("hello"));
        let val = h.take().unwrap();
        assert_eq!(val, "hello");
        assert!(!h.is_valid());
    }

    #[test]
    fn handle_double_take_fails() {
        let mut h = SafeHandle::new(7);
        assert!(h.take().is_ok());
        assert_eq!(h.take().unwrap_err(), FfiError::InvalidHandle);
    }

    #[test]
    fn handle_with_timestamp() {
        let h = SafeHandle::with_timestamp(0u8, 1234);
        assert_eq!(h.created_at(), 1234);
        assert!(h.is_valid());
    }

    #[test]
    fn handle_invalidate_then_get_mut() {
        let mut h = SafeHandle::new(5);
        h.invalidate();
        assert_eq!(h.get_mut().unwrap_err(), FfiError::InvalidHandle);
    }

    #[test]
    fn handle_new_is_valid() {
        let h = SafeHandle::new(());
        assert!(h.is_valid());
    }

    #[test]
    fn handle_take_after_invalidate() {
        let mut h = SafeHandle::new(99);
        h.invalidate();
        assert_eq!(h.take().unwrap_err(), FfiError::InvalidHandle);
    }

    #[test]
    fn handle_default_timestamp_is_zero() {
        let h = SafeHandle::new(0);
        assert_eq!(h.created_at(), 0);
    }

    // -- FfiResult --------------------------------------------------------

    #[test]
    fn result_ok_round_trip() {
        let r = FfiResult::ok(42);
        assert!(r.is_ok());
        assert_eq!(r.error_code(), 0);
        assert_eq!(r.into_result().unwrap(), 42);
    }

    #[test]
    fn result_err_with_code_and_message() {
        let r: FfiResult<i32> = FfiResult::err(-5, "out of memory");
        assert!(!r.is_ok());
        assert_eq!(r.error_code(), -5);
        let e = r.into_result().unwrap_err();
        assert_eq!(
            e,
            FfiError::RuntimeError {
                code: -5,
                message: "out of memory".into(),
            }
        );
    }

    #[test]
    fn result_err_default_message() {
        let r: FfiResult<()> = FfiResult {
            value: None,
            error_code: 1,
            error_message: None,
        };
        match r.into_result().unwrap_err() {
            FfiError::RuntimeError { message, .. } => {
                assert_eq!(message, "unknown error");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn result_ok_no_value_gives_null_pointer() {
        let r: FfiResult<i32> = FfiResult {
            value: None,
            error_code: 0,
            error_message: None,
        };
        assert_eq!(r.into_result().unwrap_err(), FfiError::NullPointer);
    }

    #[test]
    fn result_ok_string_value() {
        let r = FfiResult::ok(String::from("gpu-0"));
        assert_eq!(r.into_result().unwrap(), "gpu-0");
    }

    #[test]
    fn result_err_zero_code_but_no_value() {
        let r: FfiResult<u32> = FfiResult {
            value: None,
            error_code: 0,
            error_message: None,
        };
        assert!(!r.is_ok());
    }

    // -- LibraryLoader ----------------------------------------------------

    #[test]
    fn loader_new_empty() {
        let loader = LibraryLoader::new();
        assert!(loader.search_paths().is_empty());
        assert!(loader.loaded_libraries().is_empty());
    }

    #[test]
    fn loader_default_is_empty() {
        let loader = LibraryLoader::default();
        assert!(loader.search_paths().is_empty());
    }

    #[test]
    fn loader_add_search_paths() {
        let mut loader = LibraryLoader::new();
        loader.add_search_path("/usr/lib");
        loader.add_search_path("/opt/cuda/lib64");
        assert_eq!(loader.search_paths().len(), 2);
        assert_eq!(loader.search_paths()[0], "/usr/lib");
    }

    #[test]
    fn loader_search_path_ordering() {
        let mut loader = LibraryLoader::new();
        loader.add_search_path("/first");
        loader.add_search_path("/second");
        loader.add_search_path("/third");
        assert_eq!(loader.search_paths()[0], "/first");
        assert_eq!(loader.search_paths()[2], "/third");
    }

    #[test]
    fn loader_find_library_present() {
        let mut loader = LibraryLoader::new();
        loader.add_search_path("/usr/lib/libcuda.so");
        assert_eq!(loader.find_library("libcuda"), Some("/usr/lib/libcuda.so"));
    }

    #[test]
    fn loader_find_library_not_found() {
        let mut loader = LibraryLoader::new();
        loader.add_search_path("/usr/lib/libfoo.so");
        assert!(loader.find_library("libbar").is_none());
    }

    #[test]
    fn loader_find_library_first_match_wins() {
        let mut loader = LibraryLoader::new();
        loader.add_search_path("/first/libcuda.so");
        loader.add_search_path("/second/libcuda.so");
        assert_eq!(
            loader.find_library("libcuda"),
            Some("/first/libcuda.so")
        );
    }

    #[test]
    fn loader_register_library() {
        let mut loader = LibraryLoader::new();
        loader.register(LoadedLibrary {
            name: "cuda".into(),
            path: "/usr/lib/libcuda.so".into(),
            version: Some("12.3".into()),
            symbols: vec!["cuInit".into()],
        });
        assert_eq!(loader.loaded_libraries().len(), 1);
        assert_eq!(loader.loaded_libraries()[0].name, "cuda");
    }

    #[test]
    fn loader_get_loaded_found() {
        let mut loader = LibraryLoader::new();
        loader.register(LoadedLibrary {
            name: "cuda".into(),
            path: "/lib/libcuda.so".into(),
            version: None,
            symbols: vec![],
        });
        let lib = loader.get_loaded("cuda").unwrap();
        assert_eq!(lib.path, "/lib/libcuda.so");
    }

    #[test]
    fn loader_get_loaded_not_found() {
        let loader = LibraryLoader::new();
        assert!(loader.get_loaded("nope").is_none());
    }

    #[test]
    fn loaded_library_version_none() {
        let lib = LoadedLibrary {
            name: "test".into(),
            path: "/tmp/libtest.so".into(),
            version: None,
            symbols: vec![],
        };
        assert!(lib.version.is_none());
    }

    #[test]
    fn loaded_library_symbols_list() {
        let lib = LoadedLibrary {
            name: "test".into(),
            path: "/tmp/libtest.so".into(),
            version: Some("1.0".into()),
            symbols: vec!["a".into(), "b".into(), "c".into()],
        };
        assert_eq!(lib.symbols.len(), 3);
    }

    // -- NullCheckGuard ---------------------------------------------------

    #[test]
    fn null_check_non_null() {
        let val = 42u32;
        let ptr: *const u32 = &raw const val;
        assert!(NullCheckGuard::check_ptr(ptr).is_ok());
    }

    #[test]
    fn null_check_null() {
        let ptr: *const u8 = std::ptr::null();
        assert_eq!(
            NullCheckGuard::check_ptr(ptr).unwrap_err(),
            FfiError::NullPointer,
        );
    }

    #[test]
    fn null_check_mut_non_null() {
        let mut val = 1u64;
        let ptr: *mut u64 = &raw mut val;
        assert!(NullCheckGuard::check_mut_ptr(ptr).is_ok());
    }

    #[test]
    fn null_check_mut_null() {
        let ptr: *mut f32 = std::ptr::null_mut();
        assert_eq!(
            NullCheckGuard::check_mut_ptr(ptr).unwrap_err(),
            FfiError::NullPointer,
        );
    }

    // -- BoundsCheckGuard -------------------------------------------------

    #[test]
    fn bounds_within() {
        let g = BoundsCheckGuard::new(100);
        assert!(g.check(0, 50).is_ok());
    }

    #[test]
    fn bounds_exact_fit() {
        let g = BoundsCheckGuard::new(100);
        assert!(g.check(0, 100).is_ok());
    }

    #[test]
    fn bounds_overflow() {
        let g = BoundsCheckGuard::new(100);
        let err = g.check(90, 20).unwrap_err();
        assert_eq!(
            err,
            FfiError::BufferOverflow { expected: 100, actual: 110 },
        );
    }

    #[test]
    fn bounds_offset_at_end() {
        let g = BoundsCheckGuard::new(100);
        assert!(g.check(100, 0).is_ok());
    }

    #[test]
    fn bounds_offset_past_end() {
        let g = BoundsCheckGuard::new(100);
        assert!(g.check(101, 0).is_err());
    }

    #[test]
    fn bounds_check_index_valid() {
        let g = BoundsCheckGuard::new(10);
        assert!(g.check_index(9).is_ok());
    }

    #[test]
    fn bounds_check_index_invalid() {
        let g = BoundsCheckGuard::new(10);
        assert_eq!(
            g.check_index(10).unwrap_err(),
            FfiError::BufferOverflow { expected: 10, actual: 10 },
        );
    }

    #[test]
    fn bounds_max_size_accessor() {
        let g = BoundsCheckGuard::new(256);
        assert_eq!(g.max_size(), 256);
    }

    #[test]
    fn bounds_zero_size_buffer() {
        let g = BoundsCheckGuard::new(0);
        assert!(g.check(0, 0).is_ok());
        assert!(g.check(0, 1).is_err());
    }

    #[test]
    fn bounds_saturating_add_no_panic() {
        let g = BoundsCheckGuard::new(100);
        // usize::MAX + 1 would overflow, saturating_add clamps.
        assert!(g.check(usize::MAX, 1).is_err());
    }

    // -- Integration / mixed scenarios ------------------------------------

    #[test]
    fn full_workflow_guard_then_handle() {
        let mut guard = FfiGuard::new("libcuda.so");
        guard.set_loaded();
        guard.check_symbol("cuInit", true);
        guard.check_symbol("cuDeviceGet", true);
        assert!(guard.all_required_found());

        let mut handle = SafeHandle::new(0xDEAD_BEEFu64);
        assert_eq!(*handle.get().unwrap(), 0xDEAD_BEEF);
        let val = handle.take().unwrap();
        assert_eq!(val, 0xDEAD_BEEF);
        assert!(handle.take().is_err());
    }

    #[test]
    fn full_workflow_ffi_result_to_handle() {
        let result = FfiResult::ok(42u32);
        let val = result.into_result().unwrap();
        let handle = SafeHandle::new(val);
        assert_eq!(*handle.get().unwrap(), 42);
    }

    #[test]
    fn full_workflow_loader_then_guard() {
        let mut loader = LibraryLoader::new();
        loader.add_search_path("/usr/lib/libgpu.so");
        let path = loader.find_library("libgpu").unwrap();
        assert!(path.contains("libgpu"));

        let mut guard = FfiGuard::new("libgpu.so");
        guard.set_loaded();
        guard.check_symbol("gpuInit", true);
        assert!(guard.all_required_found());
    }

    #[test]
    fn error_equality() {
        assert_eq!(FfiError::NullPointer, FfiError::NullPointer);
        assert_ne!(FfiError::NullPointer, FfiError::InvalidHandle);
    }

    #[test]
    fn error_clone() {
        let e = FfiError::RuntimeError {
            code: 7,
            message: "fail".into(),
        };
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    #[test]
    fn error_debug_formatting() {
        let e = FfiError::InvalidHandle;
        let dbg = format!("{e:?}");
        assert!(dbg.contains("InvalidHandle"));
    }
}
