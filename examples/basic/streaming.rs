#[path = "../streaming_generation.rs"]
mod shared;

#[cfg(feature = "examples")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::main()
}

#[cfg(not(feature = "examples"))]
fn main() {
    shared::main();
}
