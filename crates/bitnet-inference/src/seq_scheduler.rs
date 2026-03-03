//! Sequence scheduling for batched inference.
//!
//! Manages multiple in-flight generation requests, scheduling
//! them into batches based on available compute and memory.

use std::collections::VecDeque;

/// State of a generation sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    Waiting,
    Prefilling,
    Generating,
    Completed,
    Failed,
}

/// A generation sequence tracked by the scheduler.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: u64,
    pub state: SeqState,
    pub prompt_len: usize,
    pub generated_len: usize,
    pub max_tokens: usize,
    pub priority: u32,
}

impl Sequence {
    pub fn new(id: u64, prompt_len: usize, max_tokens: usize) -> Self {
        Self { id, state: SeqState::Waiting, prompt_len, generated_len: 0, max_tokens, priority: 0 }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn total_len(&self) -> usize {
        self.prompt_len + self.generated_len
    }

    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.generated_len)
    }

    pub fn is_done(&self) -> bool {
        self.state == SeqState::Completed || self.state == SeqState::Failed
    }
}

/// Scheduling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePolicy {
    /// First come, first served.
    Fcfs,
    /// Shortest job first (by remaining tokens).
    ShortestFirst,
    /// Priority-based (higher priority first).
    PriorityFirst,
}

/// Sequence scheduler.
#[derive(Debug)]
pub struct SeqScheduler {
    policy: SchedulePolicy,
    waiting: VecDeque<Sequence>,
    active: Vec<Sequence>,
    completed: Vec<Sequence>,
    max_batch_size: usize,
    max_total_tokens: usize,
}

impl SeqScheduler {
    pub fn new(policy: SchedulePolicy, max_batch_size: usize, max_total_tokens: usize) -> Self {
        Self {
            policy,
            waiting: VecDeque::new(),
            active: Vec::new(),
            completed: Vec::new(),
            max_batch_size,
            max_total_tokens,
        }
    }

    /// Submit a new sequence for scheduling.
    pub fn submit(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    /// Get number of waiting sequences.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Get number of active sequences.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Get number of completed sequences.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Current total tokens across active sequences.
    pub fn active_tokens(&self) -> usize {
        self.active.iter().map(|s| s.total_len()).sum()
    }

    /// Schedule: move waiting sequences into active batch.
    pub fn schedule(&mut self) -> Vec<u64> {
        // Sort waiting queue by policy
        let mut waiting_vec: Vec<Sequence> = self.waiting.drain(..).collect();
        match self.policy {
            SchedulePolicy::Fcfs => {} // already in order
            SchedulePolicy::ShortestFirst => {
                waiting_vec.sort_by_key(|s| s.remaining_tokens());
            }
            SchedulePolicy::PriorityFirst => {
                waiting_vec.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
        }

        let mut scheduled = Vec::new();
        let mut remaining = Vec::new();

        for mut seq in waiting_vec {
            let would_be_tokens = self.active_tokens() + seq.total_len();
            if self.active.len() < self.max_batch_size && would_be_tokens <= self.max_total_tokens {
                seq.state = SeqState::Prefilling;
                scheduled.push(seq.id);
                self.active.push(seq);
            } else {
                remaining.push(seq);
            }
        }

        self.waiting = remaining.into();
        scheduled
    }

    /// Mark a sequence as having completed a generation step.
    pub fn step(&mut self, seq_id: u64) {
        if let Some(seq) = self.active.iter_mut().find(|s| s.id == seq_id) {
            seq.generated_len += 1;
            seq.state = SeqState::Generating;
            if seq.generated_len >= seq.max_tokens {
                seq.state = SeqState::Completed;
            }
        }
    }

    /// Move completed sequences out of active.
    pub fn collect_completed(&mut self) -> Vec<Sequence> {
        let (done, still_active): (Vec<_>, Vec<_>) =
            self.active.drain(..).partition(|s| s.is_done());
        self.active = still_active;
        self.completed.extend(done.iter().cloned());
        done
    }

    /// Mark a sequence as failed.
    pub fn fail(&mut self, seq_id: u64) {
        if let Some(seq) = self.active.iter_mut().find(|s| s.id == seq_id) {
            seq.state = SeqState::Failed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_and_schedule() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 4, 1024);
        sched.submit(Sequence::new(1, 10, 20));
        sched.submit(Sequence::new(2, 15, 30));
        let ids = sched.schedule();
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(sched.active_count(), 2);
        assert_eq!(sched.waiting_count(), 0);
    }

    #[test]
    fn test_batch_size_limit() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 2, 10000);
        sched.submit(Sequence::new(1, 10, 20));
        sched.submit(Sequence::new(2, 10, 20));
        sched.submit(Sequence::new(3, 10, 20));
        let ids = sched.schedule();
        assert_eq!(ids.len(), 2);
        assert_eq!(sched.waiting_count(), 1);
    }

    #[test]
    fn test_token_limit() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 10, 25);
        sched.submit(Sequence::new(1, 10, 20));
        sched.submit(Sequence::new(2, 10, 20));
        sched.submit(Sequence::new(3, 10, 20)); // would exceed 25
        let ids = sched.schedule();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_shortest_first() {
        let mut sched = SeqScheduler::new(SchedulePolicy::ShortestFirst, 10, 10000);
        sched.submit(Sequence::new(1, 10, 100));
        sched.submit(Sequence::new(2, 10, 10));
        sched.submit(Sequence::new(3, 10, 50));
        let ids = sched.schedule();
        assert_eq!(ids[0], 2); // shortest first
    }

    #[test]
    fn test_priority_first() {
        let mut sched = SeqScheduler::new(SchedulePolicy::PriorityFirst, 10, 10000);
        sched.submit(Sequence::new(1, 10, 20).with_priority(1));
        sched.submit(Sequence::new(2, 10, 20).with_priority(5));
        sched.submit(Sequence::new(3, 10, 20).with_priority(3));
        let ids = sched.schedule();
        assert_eq!(ids[0], 2); // highest priority first
    }

    #[test]
    fn test_step_and_complete() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 4, 1024);
        sched.submit(Sequence::new(1, 5, 2));
        sched.schedule();
        sched.step(1);
        sched.step(1); // should complete
        let done = sched.collect_completed();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].generated_len, 2);
    }

    #[test]
    fn test_fail() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 4, 1024);
        sched.submit(Sequence::new(1, 5, 20));
        sched.schedule();
        sched.fail(1);
        let done = sched.collect_completed();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].state, SeqState::Failed);
    }

    #[test]
    fn test_active_tokens() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 4, 1024);
        sched.submit(Sequence::new(1, 10, 20));
        sched.submit(Sequence::new(2, 15, 20));
        sched.schedule();
        assert_eq!(sched.active_tokens(), 25);
    }

    #[test]
    fn test_sequence_helpers() {
        let s = Sequence::new(1, 10, 20);
        assert_eq!(s.total_len(), 10);
        assert_eq!(s.remaining_tokens(), 20);
        assert!(!s.is_done());
    }

    #[test]
    fn test_empty_scheduler() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 4, 1024);
        let ids = sched.schedule();
        assert!(ids.is_empty());
        assert_eq!(sched.active_count(), 0);
    }

    #[test]
    fn test_collect_partial() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 4, 1024);
        sched.submit(Sequence::new(1, 5, 1));
        sched.submit(Sequence::new(2, 5, 100));
        sched.schedule();
        sched.step(1); // completes seq 1
        sched.step(2); // seq 2 still going
        let done = sched.collect_completed();
        assert_eq!(done.len(), 1);
        assert_eq!(sched.active_count(), 1);
    }

    #[test]
    fn test_completed_count() {
        let mut sched = SeqScheduler::new(SchedulePolicy::Fcfs, 4, 1024);
        sched.submit(Sequence::new(1, 5, 1));
        sched.schedule();
        sched.step(1);
        sched.collect_completed();
        assert_eq!(sched.completed_count(), 1);
    }
}
