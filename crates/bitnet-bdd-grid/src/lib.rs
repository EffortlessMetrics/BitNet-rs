//! Canonical curated BDD grid for BitNet.
//!
//! This crate intentionally keeps curated policy data here, while low-level
//! primitives (scenarios, features, grid and helper types) live in
//! `bitnet-bdd-grid-core` so they can be reused independently.

use std::sync::LazyLock;

mod features;
mod rows;

pub use bitnet_bdd_grid_core::{
    BddCell, BddGrid, BitnetFeature, ExecutionEnvironment, FeatureSet, TestingScenario,
    feature_set_from_names, try_feature_set_from_names,
};

use rows::build_curated_rows;

static CURATED_ROWS: LazyLock<Box<[BddCell]>> = LazyLock::new(build_curated_rows);

/// Canonical curated profile rows used by runtime profile resolution and tooling.
pub fn curated() -> BddGrid {
    BddGrid::from_rows(LazyLock::force(&CURATED_ROWS).as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::curated_features;

    #[test]
    fn test_grid_lookup_and_validation() -> Result<(), Box<dyn std::error::Error>> {
        let grid = curated();
        let cell = grid.find(TestingScenario::Unit, ExecutionEnvironment::Local);
        assert!(cell.is_some());

        let active = curated_features(&["inference", "kernels", "tokenizers"]);
        let Some(cell) = cell else {
            return Err("unit/local row exists in curated grid".into());
        };
        assert!(cell.supports(&active));
        assert!(cell.violations(&active).0.is_empty());
        assert!(cell.violations(&active).1.is_empty());
        Ok(())
    }
}
