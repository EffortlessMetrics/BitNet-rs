use anyhow::Result;
use vergen::{BuildBuilder, CargoBuilder, Emitter, GitBuilder, RustcBuilder};

fn main() -> Result<()> {
    // Emit build metadata as environment variables
    let build = BuildBuilder::default().build_timestamp(true).build_date(true).build()?;

    let cargo = CargoBuilder::default().target_triple(true).profile(true).features(true).build()?;

    let git = GitBuilder::default()
        .branch(true)
        .commit_date(true)
        .commit_timestamp(true)
        .describe(true, true, None)
        .sha(true)
        .build()?;

    let rustc = RustcBuilder::default()
        .channel(true)
        .commit_date(true)
        .host_triple(true)
        .semver(true)
        .build()?;

    // Emit all metadata
    Emitter::default()
        .add_instructions(&build)?
        .add_instructions(&cargo)?
        .add_instructions(&git)?
        .add_instructions(&rustc)?
        .emit()?;

    Ok(())
}
