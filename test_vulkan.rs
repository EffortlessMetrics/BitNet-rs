use candle_core::Device;

fn main() {
    let d = Device::Vulkan(0);
    println!("{:?}", d);
}
