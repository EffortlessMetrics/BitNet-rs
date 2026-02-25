//! Snapshot tests for stable bitnet-server configuration defaults.
//! These pin key server configuration constants to catch silent regressions.

use bitnet_server::{
    batch_engine::BatchEngineConfig,
    concurrency::ConcurrencyConfig,
    config::{DeviceConfig, ServerSettings},
};

#[test]
fn server_settings_default_host_and_port() {
    let cfg = ServerSettings::default();
    insta::assert_snapshot!(format!("host={} port={}", cfg.host, cfg.port));
}

#[test]
fn server_settings_default_timeouts() {
    let cfg = ServerSettings::default();
    insta::assert_snapshot!(format!(
        "keep_alive={}s request_timeout={}s shutdown={}s",
        cfg.keep_alive.as_secs(),
        cfg.request_timeout.as_secs(),
        cfg.graceful_shutdown_timeout.as_secs()
    ));
}

#[test]
fn server_settings_default_device_is_auto() {
    let cfg = ServerSettings::default();
    insta::assert_snapshot!(format!("{:?}", cfg.default_device));
}

#[test]
fn batch_engine_config_default_batch_size() {
    let cfg = BatchEngineConfig::default();
    insta::assert_snapshot!(format!(
        "max_batch_size={} max_concurrent_batches={}",
        cfg.max_batch_size, cfg.max_concurrent_batches
    ));
}

#[test]
fn concurrency_config_default_max_requests() {
    let cfg = ConcurrencyConfig::default();
    insta::assert_snapshot!(format!("max_concurrent={}", cfg.max_concurrent_requests));
}

#[test]
fn device_config_variants_debug() {
    let variants = [format!("{:?}", DeviceConfig::Cpu), format!("{:?}", DeviceConfig::Auto)];
    insta::assert_snapshot!(variants.join("\n"));
}
