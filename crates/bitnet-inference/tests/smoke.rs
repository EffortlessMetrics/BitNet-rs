#[cfg(feature = "rt-tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn opens_model_when_env_is_set() -> anyhow::Result<()> {
    let Some(path) = std::env::var_os("BITNET_GGUF") else {
        eprintln!("skipped: set BITNET_GGUF=/path/to/model.gguf to run");
        return Ok(());
    };
    let path = std::path::PathBuf::from(path);
    let header = bitnet_inference::gguf::read_header(&path).await?;
    assert!(header.version >= 1);
    Ok(())
}
