#![no_main]

use arbitrary::Arbitrary;
use bitnet_bdd_grid_core::{BitnetFeature, ExecutionEnvironment, TestingScenario, FeatureSet};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ContextParseInput {
    /// Arbitrary bytes to try parsing as a `TestingScenario`.
    scenario_bytes: Vec<u8>,
    /// Arbitrary bytes to try parsing as an `ExecutionEnvironment`.
    environment_bytes: Vec<u8>,
    /// Arbitrary bytes to try parsing as a `BitnetFeature`.
    feature_bytes: Vec<u8>,
    /// Short byte chunks for `feature_set_from_names`.
    names_bytes: Vec<u8>,
}

fuzz_target!(|input: ContextParseInput| {
    let scenario_str = String::from_utf8_lossy(&input.scenario_bytes);
    let env_str = String::from_utf8_lossy(&input.environment_bytes);
    let feature_str = String::from_utf8_lossy(&input.feature_bytes);

    // FromStr impls must not panic on arbitrary input — they may return Err.
    let _: Result<TestingScenario, _> = scenario_str.parse();
    let _: Result<ExecutionEnvironment, _> = env_str.parse();
    let _: Result<BitnetFeature, _> = feature_str.parse();

    // feature_set_from_names must not panic for any collection of arbitrary strings.
    let names: Vec<&str> = input
        .names_bytes
        .chunks(16)
        .filter_map(|b| std::str::from_utf8(b).ok())
        .take(32)
        .collect();
    let _: FeatureSet = bitnet_bdd_grid_core::feature_set_from_names(&names);
});
