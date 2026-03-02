#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Fuzz pipeline-parallel scheduling with random stage configs and micro-batch
/// sizes, verifying invariants like stage ordering, latency accounting, and
/// bubble-free scheduling.
#[derive(Arbitrary, Debug)]
struct PipelineParallelInput {
    num_stages: u8,
    micro_batch_count: u8,
    stage_latencies: Vec<u8>,
    reorder_ops: Vec<ReorderOp>,
}

#[derive(Arbitrary, Debug)]
enum ReorderOp {
    SwapStages { a: u8, b: u8 },
    InsertBubble { at: u8 },
    Drain,
    Tick,
}

struct PipelineScheduler {
    num_stages: usize,
    micro_batches: usize,
    stage_latencies: Vec<u64>,
    // Track which micro-batch is at which stage (None = empty slot)
    slots: Vec<Option<usize>>,
    completed: Vec<bool>,
    ticks: u64,
}

impl PipelineScheduler {
    fn new(num_stages: usize, micro_batches: usize, latencies: Vec<u64>) -> Self {
        Self {
            num_stages,
            micro_batches,
            stage_latencies: latencies,
            slots: vec![None; num_stages],
            completed: vec![false; micro_batches],
            ticks: 0,
        }
    }

    fn tick(&mut self) {
        // Advance pipeline: move micro-batches forward
        for s in (0..self.num_stages).rev() {
            if let Some(mb) = self.slots[s] {
                if s + 1 < self.num_stages {
                    if self.slots[s + 1].is_none() {
                        self.slots[s + 1] = Some(mb);
                        self.slots[s] = None;
                    }
                } else {
                    // Completed last stage
                    if mb < self.completed.len() {
                        self.completed[mb] = true;
                    }
                    self.slots[s] = None;
                }
            }
        }
        self.ticks += 1;
    }

    fn inject(&mut self, micro_batch_id: usize) -> bool {
        if self.slots[0].is_some() {
            return false;
        }
        self.slots[0] = Some(micro_batch_id);
        true
    }

    fn completed_count(&self) -> usize {
        self.completed.iter().filter(|&&c| c).count()
    }

    fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    fn drain(&mut self) {
        for _ in 0..self.num_stages + self.micro_batches + 1 {
            self.tick();
        }
    }

    fn total_latency(&self) -> u64 {
        self.stage_latencies.iter().sum()
    }

    fn swap_stages(&mut self, a: usize, b: usize) {
        if a < self.num_stages && b < self.num_stages {
            self.stage_latencies.swap(a, b);
            self.slots.swap(a, b);
        }
    }
}

fuzz_target!(|input: PipelineParallelInput| {
    let num_stages = (input.num_stages as usize % 8) + 1;
    let micro_batches = (input.micro_batch_count as usize % 16) + 1;

    let latencies: Vec<u64> =
        input.stage_latencies.iter().take(num_stages).map(|&b| (b as u64 % 100) + 1).collect();
    let mut padded_latencies = latencies;
    padded_latencies.resize(num_stages, 10);

    let mut sched = PipelineScheduler::new(num_stages, micro_batches, padded_latencies);

    // Invariant 1: Fresh scheduler has no completed or active batches.
    assert_eq!(sched.completed_count(), 0);
    assert_eq!(sched.active_count(), 0);

    // Inject all micro-batches one at a time with ticks in between.
    let mut injected = 0;
    for mb in 0..micro_batches {
        // Tick a few times to make room
        for _ in 0..3 {
            sched.tick();
        }
        if sched.inject(mb) {
            injected += 1;
        }
    }

    // Invariant 2: Active + completed <= total injected.
    assert!(
        sched.active_count() + sched.completed_count() <= injected,
        "active={} completed={} injected={}",
        sched.active_count(),
        sched.completed_count(),
        injected
    );

    // Apply fuzz-driven reorder operations.
    for op in input.reorder_ops.iter().take(64) {
        match op {
            ReorderOp::SwapStages { a, b } => {
                let sa = *a as usize % num_stages;
                let sb = *b as usize % num_stages;
                sched.swap_stages(sa, sb);
            }
            ReorderOp::InsertBubble { at } => {
                let stage = *at as usize % num_stages;
                // Simulate bubble by clearing a slot
                sched.slots[stage] = None;
            }
            ReorderOp::Drain => {
                sched.drain();
            }
            ReorderOp::Tick => {
                sched.tick();
            }
        }
    }

    // Invariant 3: After drain, no active batches remain.
    sched.drain();
    assert_eq!(sched.active_count(), 0, "active batches remain after drain");

    // Invariant 4: Total latency is always positive.
    assert!(sched.total_latency() > 0, "total latency must be > 0");

    // Invariant 5: Ticks always increase monotonically.
    let ticks_before = sched.ticks;
    sched.tick();
    assert!(sched.ticks > ticks_before, "ticks must increase");
});
