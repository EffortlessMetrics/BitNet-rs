#![cfg(feature = "integration-tests")]
// Include the tolerance validation code
include!("test_1e6_tolerance_validation_final.rs");

fn main() {
    demonstrate_1e6_tolerance_validation();
}
