#![cfg(not(feature = "ffi"))]

use bitnet_sys::*;

#[test]
fn stubs_return_disabled_errors() {
    assert!(!is_available());
    assert!(version().is_err());
    assert!(initialize().is_err());
    assert!(load_model("/dev/null").is_err());
    let mut handle = ModelHandle;
    assert!(generate(&mut handle, "hi", 1).is_err());
    assert!(cleanup().is_err());
}
