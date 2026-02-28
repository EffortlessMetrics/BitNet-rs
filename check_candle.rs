fn main() {
    let d = candle_core::Device::Cpu;
    match d {
        candle_core::Device::Cpu => {},
        _ => {},
    }
}
