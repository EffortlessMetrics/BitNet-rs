// TL LUT Generator for Test Fixtures
// Run with: rustc generate_luts.rs && ./generate_luts

use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    // Generate TL1 LUT (256 entries, 1KB)
    // TL1: Single byte lookup, symmetric around 0
    let tl1_lut: Vec<f32> = (0..256)
        .map(|i| {
            let signed = i as i8; // -128..127
            signed as f32 / 127.0 // Normalize to [-1.008, 1.0]
        })
        .collect();

    let tl1_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tl1_lut.as_ptr() as *const u8,
            tl1_lut.len() * std::mem::size_of::<f32>()
        )
    };

    let mut tl1_file = File::create("tl1_lut.bin")?;
    tl1_file.write_all(tl1_bytes)?;
    println!("✓ Generated tl1_lut.bin ({} bytes)", tl1_bytes.len());

    // Generate TL2 LUT (65536 entries, 256KB)
    // TL2: Two-byte lookup, combines high and low bytes
    let tl2_lut: Vec<f32> = (0..65536)
        .map(|i| {
            let high = ((i >> 8) & 0xFF) as i8; // High byte as signed
            let low = (i & 0xFF) as i8;         // Low byte as signed

            // Combine with asymmetric weighting
            let high_contrib = high as f32 / 127.0;
            let low_contrib = low as f32 / 127.0 * 0.1; // Low byte has 10% weight

            high_contrib + low_contrib
        })
        .collect();

    let tl2_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            tl2_lut.as_ptr() as *const u8,
            tl2_lut.len() * std::mem::size_of::<f32>()
        )
    };

    let mut tl2_file = File::create("tl2_lut.bin")?;
    tl2_file.write_all(tl2_bytes)?;
    println!("✓ Generated tl2_lut.bin ({} bytes)", tl2_bytes.len());

    Ok(())
}
