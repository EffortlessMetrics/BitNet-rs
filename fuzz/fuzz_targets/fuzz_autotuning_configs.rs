#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::opencl_autotuner::{
    Autotuner, BenchmarkFn, ParamSet, SearchStrategy, TuningParam, TuningSpace,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct AutotuneInput {
    params: Vec<ParamInput>,
    strategy_byte: u8,
    sample_count: u8,
    iterations: u8,
}

#[derive(Arbitrary, Debug)]
struct ParamInput {
    name_byte: u8,
    min: u8,
    max: u8,
    step: u8,
}

fuzz_target!(|input: AutotuneInput| {
    if input.params.is_empty() || input.params.len() > 4 {
        return;
    }

    let mut tuning_params = Vec::new();
    for (i, p) in input.params.iter().enumerate() {
        let min = (p.min as u32 % 16) + 1;
        let max_val = min + (p.max as u32 % 8) + 1;
        let step = (p.step as u32 % 4) + 1;
        let name = format!("param_{i}_{}", p.name_byte % 4);
        tuning_params.push(TuningParam::new(name, min, max_val, step, min));
    }

    // Invariant 1: TuningParam::values must produce at least one value
    for tp in &tuning_params {
        let vals = tp.values();
        assert!(!vals.is_empty(), "param {} must have at least one value", tp.name);
        // Invariant 2: All values must be within [min, max]
        for &v in &vals {
            assert!(v >= tp.min, "value {v} < min {}", tp.min);
            assert!(v <= tp.max, "value {v} > max {}", tp.max);
        }
        // Invariant 3: num_values matches values().len()
        assert_eq!(tp.num_values(), vals.len());
    }

    let space = TuningSpace::new(tuning_params);

    // Invariant 4: total_configurations is product of individual param value counts
    let expected_total: usize = space.params().iter().map(|p| p.num_values()).product();
    assert_eq!(space.total_configurations(), expected_total, "total_configurations mismatch");

    // Invariant 5: Space must not be empty
    assert!(!space.is_empty(), "space should not be empty");

    // Invariant 6: enumerate produces exactly total_configurations entries
    let enumerated = space.enumerate();
    assert_eq!(enumerated.len(), expected_total, "enumerate count != total_configurations");

    // Invariant 7: default_config returns a valid param set
    let default = space.default_config();
    for tp in space.params() {
        let v = default.get(&tp.name);
        assert!(v.is_some(), "default config missing param {}", tp.name);
        let v = v.unwrap();
        assert!(v >= tp.min && v <= tp.max, "default param {} value {v} out of range", tp.name);
    }

    // Invariant 8: Each enumerated config has all params in range
    for (ci, cfg) in enumerated.iter().enumerate() {
        for tp in space.params() {
            let v = cfg.get(&tp.name);
            assert!(v.is_some(), "config {ci} missing param {}", tp.name);
            let v = v.unwrap();
            assert!(
                v >= tp.min && v <= tp.max,
                "config {ci} param {} = {v} out of [{}, {}]",
                tp.name,
                tp.min,
                tp.max
            );
        }
    }

    // Limit iterations for fuzzing performance
    let sample_n = ((input.sample_count as usize) % 8) + 1;
    let sa_iters = ((input.iterations as usize) % 8) + 1;

    let strategy = match input.strategy_byte % 4 {
        0 => {
            if expected_total > 64 {
                SearchStrategy::RandomSample(sample_n)
            } else {
                SearchStrategy::Exhaustive
            }
        }
        1 => SearchStrategy::RandomSample(sample_n),
        2 => SearchStrategy::SimulatedAnnealing {
            initial_temp: 1.0,
            cooling_rate: 0.9,
            iterations: sa_iters,
        },
        _ => SearchStrategy::BayesianOpt { evaluations: sample_n },
    };

    // A deterministic benchmark: sum of param values as "elapsed time"
    let bench: BenchmarkFn = Box::new(|params: &ParamSet| {
        let sum: f64 = params.0.iter().map(|(_, v)| *v as f64).sum();
        sum + 1.0
    });

    let autotuner = Autotuner::new(space, strategy, 1e9, 1e6);

    // Invariant 9: tune must not panic
    let report = autotuner.tune(&bench);

    // Invariant 10: Report must have at least 1 iteration
    assert!(report.iterations > 0, "report should have at least 1 iteration");

    // Invariant 11: Best elapsed must be positive
    assert!(
        report.best_elapsed_us > 0.0,
        "best elapsed should be positive: {}",
        report.best_elapsed_us
    );

    // Invariant 12: Speedup must be positive
    assert!(
        report.speedup_vs_default > 0.0,
        "speedup should be positive: {}",
        report.speedup_vs_default
    );

    // Invariant 13: all_results should not be empty
    assert!(!report.all_results.is_empty(), "all_results should not be empty");

    // Invariant 14: Best result should be <= all other results
    for r in &report.all_results {
        assert!(
            report.best_elapsed_us <= r.elapsed_us + 1e-9,
            "best {} > result {}",
            report.best_elapsed_us,
            r.elapsed_us
        );
    }
});
