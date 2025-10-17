// Quick utility to list all tensors in a GGUF file
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf_file>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    let data = fs::read(path)?;

    // Use bitnet_models GgufReader
    use bitnet_models::formats::gguf::GgufReader;
    let reader = GgufReader::new(&data)?;

    println!("Total tensors: {}", reader.tensor_count());
    println!("\nTensors matching 'blk.0.ffn':");
    for i in 0..reader.tensor_count() {
        if let Ok(info) = reader.get_tensor_info(i as usize) {
            if info.name.contains("blk.0.ffn") {
                println!("  {}: shape={:?}, type={:?}, size={}",
                         info.name, info.shape, info.tensor_type, info.size);
            }
        }
    }

    println!("\nAll unique suffixes:");
    let mut suffixes = std::collections::HashSet::new();
    for i in 0..reader.tensor_count() {
        if let Ok(info) = reader.get_tensor_info(i as usize) {
            if let Some(suffix) = info.name.split('.').last() {
                suffixes.insert(suffix.to_string());
            }
        }
    }
    let mut suffixes: Vec<_> = suffixes.into_iter().collect();
    suffixes.sort();
    for s in suffixes {
        println!("  .{}", s);
    }

    Ok(())
}
