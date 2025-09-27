//! Quantization Test Fixtures Module
//!
//! Provides test data for I2S, TL1, and TL2 quantization algorithms
//! used in Issue #260 mock elimination validation.

pub mod i2s_test_data;
pub mod tl_lookup_table_data;

// Re-export key types for convenience
pub use i2s_test_data::{
    DeviceType, I2SAccuracyFixture, I2SCrossValidationFixture, I2STestFixture,
};

pub use tl_lookup_table_data::{
    LookupTableParams, MemoryLayoutFixture, SimdArchitecture, TL1TestFixture, TL2TestFixture,
    TLPerformanceFixture,
};
