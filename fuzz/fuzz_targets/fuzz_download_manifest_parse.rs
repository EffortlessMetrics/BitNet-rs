#![no_main]

use arbitrary::Arbitrary;
use bitnet_models::download_manager::{
    DownloadManifest, DownloadProgress, DownloadSpec, known_models, resolve_model,
    validate_download,
};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
struct ManifestInput {
    model_id: String,
    filenames: Vec<String>,
    urls: Vec<String>,
    expected_bytes: Vec<u64>,
    sha256_values: Vec<String>,
    actual_sizes: Vec<u64>,
    /// Progress tracking fields.
    file_index: usize,
    total_files: usize,
    bytes_downloaded: u64,
    bytes_total: u64,
}

fuzz_target!(|input: ManifestInput| {
    // Invariant 1: resolve_model must never panic on arbitrary strings
    let _ = resolve_model(&input.model_id);

    // Invariant 2: known_models must always return consistent data
    let models = known_models();
    for (_, manifest) in &models {
        assert!(!manifest.model_id.is_empty());
        assert!(manifest.file_count() > 0);
    }

    // Invariant 3: Construct manifest from arbitrary data and validate
    let file_count = input.filenames.len().min(input.urls.len()).min(8);
    if file_count == 0 {
        return;
    }

    let files: Vec<DownloadSpec> = (0..file_count)
        .map(|i| DownloadSpec {
            url: input.urls.get(i).cloned().unwrap_or_default(),
            filename: input.filenames.get(i).cloned().unwrap_or_default(),
            expected_bytes: input.expected_bytes.get(i).copied(),
            sha256: input.sha256_values.get(i).cloned(),
        })
        .collect();

    let manifest = DownloadManifest {
        model_id: input.model_id.clone(),
        files,
        total_bytes: Some(input.bytes_total),
    };

    // Invariant 4: file_count matches constructed files
    assert_eq!(manifest.file_count(), file_count);

    // Invariant 5: total_expected_bytes must not panic
    let _ = manifest.total_expected_bytes();

    // Invariant 6: has_checksums must not panic
    let _ = manifest.has_checksums();

    // Invariant 7: validate_download with arbitrary sizes must not panic
    let mut actual_map = HashMap::new();
    for (i, filename) in input.filenames.iter().take(file_count).enumerate() {
        if let Some(&size) = input.actual_sizes.get(i) {
            actual_map.insert(filename.clone(), size);
        }
    }
    let _ = validate_download(&manifest, &actual_map);

    // Invariant 8: DownloadProgress percent/file_percent must not panic
    let progress = DownloadProgress {
        file_index: input.file_index,
        total_files: input.total_files,
        bytes_downloaded: input.bytes_downloaded,
        bytes_total: input.bytes_total,
        current_file: input.filenames.first().cloned().unwrap_or_default(),
    };
    let pct = progress.percent();
    let fpct = progress.file_percent();
    assert!(pct >= 0.0, "percent must be non-negative");
    assert!(fpct >= 0.0, "file_percent must be non-negative");
});
