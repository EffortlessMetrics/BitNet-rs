#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct DeviceInput {
    device_index: usize,
    serialize: bool,
}

fuzz_target!(|input: DeviceInput| {
    use bitnet_common::Device;

    // Constructing any OpenCL device index should not panic
    let device = Device::OpenCL(input.device_index);
    assert!(device.is_opencl());
    assert!(!device.is_cpu());
    assert!(!device.is_cuda());

    // Candle conversion should not panic
    let _ = device.to_candle();

    // Serialization round-trip should not panic
    if input.serialize {
        if let Ok(json) = serde_json::to_string(&device) {
            let _ = serde_json::from_str::<Device>(&json);
        }
    }

    // Debug formatting should not panic
    let _ = format!("{:?}", device);
});
