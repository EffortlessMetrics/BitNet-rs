use anyhow::Result;
use vergen_gix::{BuildBuilder, CargoBuilder, Emitter, GixBuilder, RustcBuilder};

fn main() -> Result<()> {
    // Emit build metadata as environment variables
    let build = BuildBuilder::default().build_timestamp(true).build_date(true).build()?;

    let cargo =
        CargoBuilder::default().target_triple(true).features(true).opt_level(true).build()?;

    // Git metadata (branch, sha, describe, timestamps, etc.)
    let git = GixBuilder::all_git()?;

    let rustc = RustcBuilder::default()
        .channel(true)
        .commit_date(true)
        .host_triple(true)
        .semver(true)
        .build()?;

    // Emit all metadata including Git info
    Emitter::default()
        .add_instructions(&build)?
        .add_instructions(&cargo)?
        .add_instructions(&git)?
        .add_instructions(&rustc)?
        .emit()?;

    Ok(())
}
