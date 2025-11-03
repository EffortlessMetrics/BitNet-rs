# Stub code: `FfiKernel::new` in `ffi.rs` checks for `crate::ffi::bridge::cpp::is_available()`

The `FfiKernel::new` function in `crates/bitnet-kernels/src/ffi.rs` checks for `crate::ffi::bridge::cpp::is_available()` to determine if the FFI bridge is available. This is a form of stubbing, as the actual availability of the C++ kernel is determined by a function call.

**File:** `crates/bitnet-kernels/src/ffi.rs`

**Function:** `FfiKernel::new`

**Code:**
```rust
pub struct FfiKernel;

impl FfiKernel {
    pub fn new() -> Result<Self, &'static str> {
        if crate::ffi::bridge::cpp::is_available() {
            crate::ffi::bridge::cpp::init();
            Ok(Self)
        } else {
            Err("ffi bridge unavailable")
        }
    }
```

## Proposed Fix

The `FfiKernel::new` function should be implemented to directly check for the availability of the C++ kernel without relying on a function call. This would involve using conditional compilation or a feature flag to enable or disable the FFI bridge.

### Example Implementation

```rust
pub struct FfiKernel;

impl FfiKernel {
    pub fn new() -> Result<Self, &'static str> {
        #[cfg(all(feature = "ffi", have_cpp))]
        {
            crate::ffi::bridge::cpp::init();
            Ok(Self)
        }
        #[cfg(not(all(feature = "ffi", have_cpp)))]
        {
            Err("ffi bridge unavailable")
        }
    }
```
