/// A fixed-capacity ring buffer for token IDs.
#[derive(Debug, Clone)]
pub struct TokenRing {
    pub(super) buffer: Vec<u32>,
    pub(super) capacity: usize,
    pub(super) head: usize, // next write position
    pub(super) len: usize,  // current number of items
}

impl TokenRing {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring buffer capacity must be > 0");
        Self { buffer: vec![0; capacity], capacity, head: 0, len: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    pub(super) fn start_index(&self) -> usize {
        if self.len < self.capacity { 0 } else { self.head }
    }

    pub(super) fn physical_index(&self, logical_index: usize) -> usize {
        (self.start_index() + logical_index) % self.capacity
    }
}
