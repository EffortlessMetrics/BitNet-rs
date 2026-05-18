//! Canonical curated BDD grid for BitNet.
//!
//! This crate intentionally keeps curated policy data here, while low-level
//! primitives (scenarios, features, grid and helper types) live in
//! `bitnet-bdd-grid-core` so they can be reused independently.

use std::sync::LazyLock;

pub use bitnet_bdd_grid_core::{
    BddCell, BddGrid, BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
    feature_set_from_names, try_feature_set_from_names,
};

static CURATED_ROWS: LazyLock<Box<[BddCell]>> = LazyLock::new(build_curated_rows);

mod curated_features;
mod curated_rows;

use curated_rows::build_curated_rows;

/// Canonical curated profile rows used by runtime profile resolution and tooling.
pub fn curated() -> BddGrid {
    BddGrid::from_rows(LazyLock::force(&CURATED_ROWS).as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_lookup_and_validation() {
        let grid = curated();
        let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
        assert!(cell.is_some());

        let active =
            crate::curated_features::curated_features(&["inference", "kernels", "tokenizers"]);
        let cell = cell.unwrap_or_else(|| panic!("unit/local row exists in curated grid"));
        assert!(cell.supports(&active));
        assert!(cell.violations(&active).0.is_empty());
        assert!(cell.violations(&active).1.is_empty());
    }
}
