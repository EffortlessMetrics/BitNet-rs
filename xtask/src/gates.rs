use anyhow::Result;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Serialize)]
struct GateResult<'a> {
    name: &'a str,
    ok: bool,
    unmapped_count: usize,
    sample_unmapped: Vec<String>,
    counts: Counts,
}

#[derive(Serialize)]
struct Counts {
    n_kv: usize,
    n_tensors: usize,
}

pub fn mapper_gate(model: PathBuf) -> Result<i32> {
    // Read GGUF header and tensor names
    let bytes = fs::read(&model)?;
    let reader = bitnet_models::GgufReader::new(&bytes)?;

    // Get tensor names
    let names: Vec<String> = reader.tensor_names().into_iter().map(|s| s.to_string()).collect();

    // Dry-run map (names only; no tensor loads)
    let unmapped = bitnet_models::weight_mapper::dry_run_remap_names(names.clone());

    let res = GateResult {
        name: "ms_bitnet_names_map_clean",
        ok: unmapped.is_empty(),
        unmapped_count: unmapped.len(),
        sample_unmapped: unmapped.into_iter().take(10).collect(),
        counts: Counts {
            n_kv: reader.metadata_keys().len(),
            n_tensors: reader.tensor_count() as usize,
        },
    };

    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(if res.ok { 0 } else { 1 })
}
