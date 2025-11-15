use anyhow::{Context, Result, anyhow};
use clap::Parser;
use safetensors::tensor::{TensorView, serialize_to_file};
use safetensors::{Dtype, SafeTensors};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[path = "../common.rs"]
mod common;
use common::{cast_ln_to_f16, is_ln_gamma, read_safetensors_bytes};

#[derive(Parser, Debug)]
#[command(name = "st-merge-ln-f16")]
struct Cli {
    /// Directory containing *.safetensors shards OR a single .safetensors file
    #[arg(long)]
    input: PathBuf,

    /// Output merged .safetensors path
    #[arg(long)]
    output: PathBuf,

    /// Optional path to copy as config.json next to output
    #[arg(long)]
    config: Option<PathBuf>,
}

fn collect_files(input: &Path) -> Result<Vec<PathBuf>> {
    let mut out = vec![];
    if input.is_file() {
        if input.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            out.push(input.to_path_buf());
        } else {
            return Err(anyhow!("input file is not .safetensors"));
        }
    } else {
        for e in WalkDir::new(input).min_depth(1).max_depth(1) {
            let e = e?;
            if e.file_type().is_file() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    out.push(p.to_path_buf());
                }
            }
        }
    }
    if out.is_empty() {
        return Err(anyhow!("no .safetensors files found in {}", input.display()));
    }
    out.sort();
    Ok(out)
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let files = collect_files(&args.input)?;

    println!("[merge] found {} shard(s)", files.len());

    // Merge tensors; LN gamma forced to f16, others copied as-is.
    // Build intermediate storage: name -> (dtype, shape, data)
    let mut tensor_storage: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
    let mut ln_count = 0usize;

    for fp in files {
        println!("[merge] reading {}", fp.display());
        let buf = read_safetensors_bytes(&fp)?;
        let st = SafeTensors::deserialize(&buf)?;
        for (name, t) in st.tensors() {
            let shape = t.shape().to_vec();
            let dtype = t.dtype();
            let data = t.data();

            // Decide payload
            let (final_dtype, bytes) = if is_ln_gamma(&name) {
                ln_count += 1;
                // cast to f16 payload
                let cast = cast_ln_to_f16(&t)?;
                (Dtype::F16, cast)
            } else {
                // Keep original dtype and bytes
                (dtype, data.to_vec())
            };

            // Insert (later shards override same name, matching HF convention)
            tensor_storage.insert(name.to_string(), (final_dtype, shape, bytes));
        }
    }

    // Now build TensorViews from the stable storage
    let map: HashMap<String, TensorView<'_>> = tensor_storage
        .iter()
        .map(|(name, (dtype, shape, data))| {
            let tv = TensorView::new(*dtype, shape.clone(), data.as_slice())
                .map_err(|e| anyhow!("TensorView {}: {}", name, e))?;
            Ok((name.clone(), tv))
        })
        .collect::<Result<_>>()?;

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    // Attach minimal metadata
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "safetensors".to_string());
    meta.insert("ln_gamma".to_string(), "float16".to_string());

    serialize_to_file(&map, Some(meta), &args.output)
        .with_context(|| format!("writing {}", args.output.display()))?;

    println!(
        "[merge] wrote {} with {} tensors (LN gamma cast to f16: {})",
        args.output.display(),
        tensor_storage.len(),
        ln_count
    );

    if let Some(cfg) = args.config.as_ref()
        && cfg.exists()
    {
        let dst = args.output.parent().unwrap_or_else(|| Path::new(".")).join("config.json");
        std::fs::copy(cfg, &dst)
            .with_context(|| format!("copying {} -> {}", cfg.display(), dst.display()))?;
        println!("[merge] copied config.json -> {}", dst.display());
    }

    Ok(())
}
