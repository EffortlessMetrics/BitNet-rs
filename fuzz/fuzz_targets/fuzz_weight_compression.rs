#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::opencl_program_cache::Compression;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct CompressionInput {
    algo: u8,
    data: Vec<u8>,
    run_multi: bool,
}

fuzz_target!(|input: CompressionInput| {
    if input.data.len() > 64 * 1024 {
        return;
    }

    let algo = match input.algo % 2 {
        0 => Compression::None,
        _ => Compression::Zstd,
    };

    // Invariant 1: compress must not panic
    let compressed = algo.compress(&input.data);

    // Invariant 2: decompress must not panic and must succeed
    let decompressed = algo.decompress(&compressed);
    assert!(decompressed.is_ok(), "decompression failed: {:?}", decompressed.err());
    let decompressed = decompressed.unwrap();

    // Invariant 3: Roundtrip must be lossless
    assert_eq!(
        decompressed.len(),
        input.data.len(),
        "roundtrip length mismatch: compressed with {:?}",
        algo
    );
    assert_eq!(decompressed, input.data, "roundtrip data mismatch: compressed with {:?}", algo);

    // Invariant 4: Empty input produces empty output
    let empty_compressed = algo.compress(&[]);
    let empty_decompressed = algo.decompress(&empty_compressed).unwrap();
    assert!(empty_decompressed.is_empty(), "empty input should produce empty output");

    // Invariant 5: Decompress of raw garbage should not panic (may return error or data)
    let _ = algo.decompress(&input.data);

    // Invariant 6: Double compression roundtrip
    if input.run_multi {
        let double_compressed = algo.compress(&compressed);
        let step1 = algo.decompress(&double_compressed).unwrap();
        let step2 = algo.decompress(&step1).unwrap();
        assert_eq!(step2, input.data, "double compression roundtrip mismatch");
    }

    // Invariant 7: Both algorithms produce correct roundtrips on same data
    for &a in &[Compression::None, Compression::Zstd] {
        let c = a.compress(&input.data);
        let d = a.decompress(&c).unwrap();
        assert_eq!(d, input.data, "cross-algorithm roundtrip failed for {:?}", a);
    }
});
