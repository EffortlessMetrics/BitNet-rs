//! CUDA warp-level utility types for reduction, scan, and shuffle operations.
//!
//! This module provides higher-level abstractions over raw warp primitives
//! found in [`crate::cuda::warp_ops`]. While `warp_ops` exposes free functions
//! that mirror individual CUDA intrinsics, `cuda_warp_utils` groups them into
//! composable builder types:
//!
//! - [`WarpConfig`] — warp geometry: size, lane mask, active mask
//! - [`WarpReducer`] — typed reduction operations (sum, max, min, product)
//!   with butterfly pattern and warp-level voting/ballot
//! - [`WarpShuffle`] — lane-to-lane communication (xor, up, down, broadcast,
//!   all-to-all)
//!
//! All public items are gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use inner::*;

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod inner {
    use bitnet_common::{KernelError, Result};

    // ------------------------------------------------------------------
    // WarpConfig
    // ------------------------------------------------------------------

    /// Standard CUDA warp size.
    pub const WARP_SIZE: u32 = 32;

    /// Configuration describing the warp geometry.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct WarpConfig {
        /// Number of lanes in the warp (always 32 for CUDA).
        pub warp_size: u32,
        /// Bit `i` set ⇒ lane `i` is considered when computing the lane mask.
        pub lane_mask: u32,
        /// Bit `i` set ⇒ lane `i` participates in operations.
        pub active_mask: u32,
    }

    impl Default for WarpConfig {
        fn default() -> Self {
            Self { warp_size: WARP_SIZE, lane_mask: 0xFFFF_FFFF, active_mask: 0xFFFF_FFFF }
        }
    }

    impl WarpConfig {
        /// Full warp with all 32 lanes active.
        pub fn full() -> Self {
            Self::default()
        }

        /// Create a config with a custom active mask.
        ///
        /// # Errors
        ///
        /// Returns an error if `active_mask` is zero.
        pub fn with_active_mask(active_mask: u32) -> Result<Self> {
            if active_mask == 0 {
                return Err(KernelError::InvalidArguments {
                    reason: "active_mask must have at least one lane active".into(),
                }
                .into());
            }
            Ok(Self { warp_size: WARP_SIZE, lane_mask: active_mask, active_mask })
        }

        /// Number of active lanes.
        #[inline]
        pub fn active_count(&self) -> u32 {
            self.active_mask.count_ones()
        }

        /// Returns `true` if lane `id` participates.
        #[inline]
        pub fn is_active(&self, lane: u32) -> bool {
            lane < self.warp_size && (self.active_mask & (1 << lane)) != 0
        }
    }

    // ------------------------------------------------------------------
    // WarpReducer
    // ------------------------------------------------------------------

    /// Warp-level reduction engine.
    ///
    /// Provides sum, max, min, and product reductions using the butterfly
    /// pattern, plus voting/ballot helpers. All operations respect the
    /// active mask in the associated [`WarpConfig`].
    #[derive(Debug, Clone)]
    pub struct WarpReducer {
        config: WarpConfig,
    }

    impl WarpReducer {
        /// Create a reducer that operates over the given configuration.
        pub fn new(config: WarpConfig) -> Self {
            Self { config }
        }

        // ---- reductions ------------------------------------------------

        /// Butterfly-pattern sum reduction.
        ///
        /// Every active lane receives the total sum.
        pub fn reduce_sum(&self, data: &mut [f32]) -> Result<()> {
            self.validate(data)?;
            let sum: f32 = self.active_values(data).sum();
            self.broadcast_to_active(data, sum);
            Ok(())
        }

        /// Butterfly-pattern max reduction.
        pub fn reduce_max(&self, data: &mut [f32]) -> Result<()> {
            self.validate(data)?;
            let val = self.active_values(data).fold(f32::NEG_INFINITY, f32::max);
            self.broadcast_to_active(data, val);
            Ok(())
        }

        /// Butterfly-pattern min reduction.
        pub fn reduce_min(&self, data: &mut [f32]) -> Result<()> {
            self.validate(data)?;
            let val = self.active_values(data).fold(f32::INFINITY, f32::min);
            self.broadcast_to_active(data, val);
            Ok(())
        }

        /// Butterfly-pattern product reduction.
        pub fn reduce_product(&self, data: &mut [f32]) -> Result<()> {
            self.validate(data)?;
            let val = self.active_values(data).fold(1.0f32, |a, b| a * b);
            self.broadcast_to_active(data, val);
            Ok(())
        }

        /// Butterfly reduction with a caller-supplied binary operator.
        ///
        /// `identity` is used as the initial accumulator value.
        pub fn reduce_butterfly(
            &self,
            data: &mut [f32],
            identity: f32,
            op: fn(f32, f32) -> f32,
        ) -> Result<()> {
            self.validate(data)?;
            let val = self.active_values(data).fold(identity, op);
            self.broadcast_to_active(data, val);
            Ok(())
        }

        // ---- voting / ballot -------------------------------------------

        /// Ballot vote — returns a bitmask where bit `i` is set iff lane `i`
        /// is active **and** `predicates[i]` is `true`.
        pub fn ballot(&self, predicates: &[bool]) -> Result<u32> {
            self.validate_len(predicates.len())?;
            let mut mask = 0u32;
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) && predicates[i as usize] {
                    mask |= 1 << i;
                }
            }
            Ok(mask)
        }

        /// `true` if **all** active lanes satisfy the predicate.
        pub fn vote_all(&self, predicates: &[bool]) -> Result<bool> {
            self.validate_len(predicates.len())?;
            Ok((0..self.config.warp_size)
                .filter(|&i| self.config.is_active(i))
                .all(|i| predicates[i as usize]))
        }

        /// `true` if **any** active lane satisfies the predicate.
        pub fn vote_any(&self, predicates: &[bool]) -> Result<bool> {
            self.validate_len(predicates.len())?;
            Ok((0..self.config.warp_size)
                .filter(|&i| self.config.is_active(i))
                .any(|i| predicates[i as usize]))
        }

        // ---- prefix scan -----------------------------------------------

        /// Inclusive prefix sum across active lanes.
        ///
        /// After the call, `data[i]` holds the sum of all active values in
        /// lanes `0..=i`.
        pub fn inclusive_scan(&self, data: &mut [f32]) -> Result<()> {
            self.validate(data)?;
            let mut running = 0.0f32;
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) {
                    running += data[i as usize];
                    data[i as usize] = running;
                }
            }
            Ok(())
        }

        /// Exclusive prefix scan across active lanes.
        ///
        /// After the call, `data[i]` holds the sum of all active values in
        /// lanes **before** `i`. The first active lane gets `0.0`.
        pub fn exclusive_scan(&self, data: &mut [f32]) -> Result<()> {
            self.validate(data)?;
            let mut running = 0.0f32;
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) {
                    let v = data[i as usize];
                    data[i as usize] = running;
                    running += v;
                }
            }
            Ok(())
        }

        // ---- helpers ---------------------------------------------------

        /// Borrow the underlying config.
        pub fn config(&self) -> &WarpConfig {
            &self.config
        }

        fn validate(&self, data: &[f32]) -> Result<()> {
            if data.len() < self.config.warp_size as usize {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "data length {} < warp_size {}",
                        data.len(),
                        self.config.warp_size
                    ),
                }
                .into());
            }
            Ok(())
        }

        fn validate_len(&self, len: usize) -> Result<()> {
            if len < self.config.warp_size as usize {
                return Err(KernelError::InvalidArguments {
                    reason: format!("buffer length {} < warp_size {}", len, self.config.warp_size),
                }
                .into());
            }
            Ok(())
        }

        fn active_values<'a>(&self, data: &'a [f32]) -> impl Iterator<Item = f32> + 'a {
            let cfg = self.config;
            (0..cfg.warp_size).filter(move |&i| cfg.is_active(i)).map(move |i| data[i as usize])
        }

        fn broadcast_to_active(&self, data: &mut [f32], val: f32) {
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) {
                    data[i as usize] = val;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // WarpShuffle
    // ------------------------------------------------------------------

    /// Warp-level lane-to-lane communication primitives.
    ///
    /// Simulates `__shfl_xor_sync`, `__shfl_up_sync`, `__shfl_down_sync`,
    /// and `__shfl_sync` on the CPU for testing and fallback paths.
    #[derive(Debug, Clone)]
    pub struct WarpShuffle {
        config: WarpConfig,
    }

    impl WarpShuffle {
        /// Create a shuffle engine for the given warp configuration.
        pub fn new(config: WarpConfig) -> Self {
            Self { config }
        }

        /// XOR shuffle — each active lane `i` reads from lane `i ^ xor_mask`.
        ///
        /// Mirrors `__shfl_xor_sync(active_mask, val, xor_mask)`.
        pub fn shuffle_xor(&self, data: &mut [f32], xor_mask: u32) -> Result<()> {
            self.validate(data)?;
            let snap: Vec<f32> = data[..self.config.warp_size as usize].to_vec();
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) {
                    let src = i ^ xor_mask;
                    if src < self.config.warp_size && self.config.is_active(src) {
                        data[i as usize] = snap[src as usize];
                    }
                    // If source lane is inactive, value is unchanged (matches HW).
                }
            }
            Ok(())
        }

        /// Up shuffle — each active lane `i` reads from lane `i - delta`.
        ///
        /// Lanes where `i < delta` keep their original value.
        /// Mirrors `__shfl_up_sync(active_mask, val, delta)`.
        pub fn shuffle_up(&self, data: &mut [f32], delta: u32) -> Result<()> {
            self.validate(data)?;
            let snap: Vec<f32> = data[..self.config.warp_size as usize].to_vec();
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) && i >= delta {
                    let src = i - delta;
                    if self.config.is_active(src) {
                        data[i as usize] = snap[src as usize];
                    }
                }
            }
            Ok(())
        }

        /// Down shuffle — each active lane `i` reads from lane `i + delta`.
        ///
        /// Lanes where `i + delta >= warp_size` keep their original value.
        /// Mirrors `__shfl_down_sync(active_mask, val, delta)`.
        pub fn shuffle_down(&self, data: &mut [f32], delta: u32) -> Result<()> {
            self.validate(data)?;
            let snap: Vec<f32> = data[..self.config.warp_size as usize].to_vec();
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) {
                    let src = i + delta;
                    if src < self.config.warp_size && self.config.is_active(src) {
                        data[i as usize] = snap[src as usize];
                    }
                }
            }
            Ok(())
        }

        /// Broadcast from lane 0 to all active lanes.
        ///
        /// Equivalent to `shuffle_xor` with source fixed to lane 0.
        pub fn broadcast(&self, data: &mut [f32]) -> Result<()> {
            self.validate(data)?;
            if !self.config.is_active(0) {
                return Err(KernelError::InvalidArguments {
                    reason: "broadcast requires lane 0 to be active".into(),
                }
                .into());
            }
            let val = data[0];
            for i in 0..self.config.warp_size {
                if self.config.is_active(i) {
                    data[i as usize] = val;
                }
            }
            Ok(())
        }

        /// All-to-all communication — each active lane `i` writes its value
        /// into every other active lane's position in `out`.
        ///
        /// `out` is a `warp_size × warp_size` flat buffer. After the call,
        /// `out[i * warp_size + j]` contains lane `j`'s value as seen by lane
        /// `i` (for active pairs).
        pub fn all_to_all(&self, data: &[f32], out: &mut [f32]) -> Result<()> {
            let ws = self.config.warp_size as usize;
            if data.len() < ws {
                return Err(KernelError::InvalidArguments {
                    reason: format!("data length {} < warp_size {}", data.len(), ws),
                }
                .into());
            }
            if out.len() < ws * ws {
                return Err(KernelError::InvalidArguments {
                    reason: format!("out length {} < warp_size² {}", out.len(), ws * ws),
                }
                .into());
            }
            for i in 0..self.config.warp_size {
                for j in 0..self.config.warp_size {
                    if self.config.is_active(i) && self.config.is_active(j) {
                        out[i as usize * ws + j as usize] = data[j as usize];
                    }
                }
            }
            Ok(())
        }

        /// Borrow the underlying config.
        pub fn config(&self) -> &WarpConfig {
            &self.config
        }

        fn validate(&self, data: &[f32]) -> Result<()> {
            if data.len() < self.config.warp_size as usize {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "data length {} < warp_size {}",
                        data.len(),
                        self.config.warp_size
                    ),
                }
                .into());
            }
            Ok(())
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[cfg(any(feature = "gpu", feature = "cuda"))]
mod tests {
    use super::inner::*;

    // -----------------------------------------------------------------
    // WarpConfig
    // -----------------------------------------------------------------

    #[test]
    fn config_defaults() {
        let c = WarpConfig::full();
        assert_eq!(c.warp_size, 32);
        assert_eq!(c.lane_mask, 0xFFFF_FFFF);
        assert_eq!(c.active_mask, 0xFFFF_FFFF);
        assert_eq!(c.active_count(), 32);
    }

    #[test]
    fn config_custom_mask() {
        let c = WarpConfig::with_active_mask(0x0F).unwrap();
        assert_eq!(c.active_count(), 4);
        assert!(c.is_active(0));
        assert!(c.is_active(3));
        assert!(!c.is_active(4));
    }

    #[test]
    fn config_zero_mask_rejected() {
        assert!(WarpConfig::with_active_mask(0).is_err());
    }

    #[test]
    fn config_out_of_range_inactive() {
        let c = WarpConfig::full();
        assert!(!c.is_active(32));
        assert!(!c.is_active(999));
    }

    // -----------------------------------------------------------------
    // WarpReducer — reductions
    // -----------------------------------------------------------------

    fn full_reducer() -> WarpReducer {
        WarpReducer::new(WarpConfig::full())
    }

    fn iota32() -> Vec<f32> {
        (0..32).map(|i| (i + 1) as f32).collect()
    }

    #[test]
    fn reduce_sum_all_lanes() {
        let r = full_reducer();
        let mut d = iota32();
        r.reduce_sum(&mut d).unwrap();
        let expected: f32 = (1..=32).sum::<u32>() as f32;
        assert!((d[0] - expected).abs() < 1e-4);
        assert!((d[31] - expected).abs() < 1e-4);
    }

    #[test]
    fn reduce_sum_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let r = WarpReducer::new(cfg);
        let mut d: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        d[4] = 999.0;
        r.reduce_sum(&mut d).unwrap();
        assert!((d[0] - 10.0).abs() < 1e-4); // 1+2+3+4
        assert!((d[4] - 999.0).abs() < 1e-4); // untouched
    }

    #[test]
    fn reduce_max_all_lanes() {
        let r = full_reducer();
        let mut d = iota32();
        r.reduce_max(&mut d).unwrap();
        assert!((d[0] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn reduce_min_all_lanes() {
        let r = full_reducer();
        let mut d = iota32();
        r.reduce_min(&mut d).unwrap();
        assert!((d[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn reduce_product_simple() {
        let cfg = WarpConfig::with_active_mask(0x07).unwrap(); // lanes 0-2
        let r = WarpReducer::new(cfg);
        let mut d = [0.0f32; 32];
        d[0] = 2.0;
        d[1] = 3.0;
        d[2] = 5.0;
        r.reduce_product(&mut d).unwrap();
        assert!((d[0] - 30.0).abs() < 1e-4);
    }

    #[test]
    fn reduce_butterfly_custom_op() {
        let r = full_reducer();
        let mut d = [1.0f32; 32];
        d[0] = 5.0;
        r.reduce_butterfly(&mut d, f32::NEG_INFINITY, f32::max).unwrap();
        assert!((d[15] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn reduce_sum_too_short() {
        let r = full_reducer();
        let mut d = [0.0f32; 16];
        assert!(r.reduce_sum(&mut d).is_err());
    }

    // -----------------------------------------------------------------
    // WarpReducer — voting / ballot
    // -----------------------------------------------------------------

    #[test]
    fn ballot_all_true() {
        let r = full_reducer();
        let preds = [true; 32];
        assert_eq!(r.ballot(&preds).unwrap(), 0xFFFF_FFFF);
    }

    #[test]
    fn ballot_none_true() {
        let r = full_reducer();
        let preds = [false; 32];
        assert_eq!(r.ballot(&preds).unwrap(), 0);
    }

    #[test]
    fn ballot_even_lanes() {
        let r = full_reducer();
        let preds: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        let b = r.ballot(&preds).unwrap();
        assert_eq!(b, 0x5555_5555);
    }

    #[test]
    fn vote_all_true() {
        let r = full_reducer();
        let preds = [true; 32];
        assert!(r.vote_all(&preds).unwrap());
    }

    #[test]
    fn vote_all_with_one_false() {
        let r = full_reducer();
        let mut preds = [true; 32];
        preds[17] = false;
        assert!(!r.vote_all(&preds).unwrap());
    }

    #[test]
    fn vote_any_one_true() {
        let r = full_reducer();
        let mut preds = [false; 32];
        preds[31] = true;
        assert!(r.vote_any(&preds).unwrap());
    }

    // -----------------------------------------------------------------
    // WarpReducer — prefix scan
    // -----------------------------------------------------------------

    #[test]
    fn inclusive_scan_ones() {
        let r = full_reducer();
        let mut d = [1.0f32; 32];
        r.inclusive_scan(&mut d).unwrap();
        for i in 0..32 {
            assert!((d[i] - (i + 1) as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn exclusive_scan_ones() {
        let r = full_reducer();
        let mut d = [1.0f32; 32];
        r.exclusive_scan(&mut d).unwrap();
        for i in 0..32 {
            assert!((d[i] - i as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn inclusive_exclusive_relationship() {
        // inclusive[i] == exclusive[i] + original[i]
        let r = full_reducer();
        let orig = iota32();
        let mut inc = orig.clone();
        let mut exc = orig.clone();
        r.inclusive_scan(&mut inc).unwrap();
        r.exclusive_scan(&mut exc).unwrap();
        for i in 0..32 {
            assert!((inc[i] - (exc[i] + orig[i])).abs() < 1e-3);
        }
    }

    // -----------------------------------------------------------------
    // WarpShuffle
    // -----------------------------------------------------------------

    fn full_shuffle() -> WarpShuffle {
        WarpShuffle::new(WarpConfig::full())
    }

    #[test]
    fn shuffle_xor_swap_pairs() {
        let s = full_shuffle();
        let mut d = iota32();
        s.shuffle_xor(&mut d, 1).unwrap();
        // lane 0 reads from lane 1, lane 1 reads from lane 0, etc.
        assert!((d[0] - 2.0).abs() < 1e-4);
        assert!((d[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn shuffle_xor_identity() {
        let s = full_shuffle();
        let mut d = iota32();
        let orig = d.clone();
        s.shuffle_xor(&mut d, 0).unwrap();
        assert_eq!(d, orig);
    }

    #[test]
    fn shuffle_up_by_one() {
        let s = full_shuffle();
        let mut d = iota32();
        s.shuffle_up(&mut d, 1).unwrap();
        // lane 0 keeps original, lane 1 gets lane 0's value, etc.
        assert!((d[0] - 1.0).abs() < 1e-4); // unchanged
        assert!((d[1] - 1.0).abs() < 1e-4); // was lane 0
        assert!((d[2] - 2.0).abs() < 1e-4); // was lane 1
    }

    #[test]
    fn shuffle_down_by_one() {
        let s = full_shuffle();
        let mut d = iota32();
        s.shuffle_down(&mut d, 1).unwrap();
        assert!((d[0] - 2.0).abs() < 1e-4); // reads from lane 1
        assert!((d[30] - 32.0).abs() < 1e-4); // reads from lane 31
        assert!((d[31] - 32.0).abs() < 1e-4); // no source, unchanged
    }

    #[test]
    fn broadcast_from_lane0() {
        let s = full_shuffle();
        let mut d = iota32();
        s.broadcast(&mut d).unwrap();
        for v in &d[..32] {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn broadcast_requires_lane0_active() {
        let cfg = WarpConfig::with_active_mask(0xFFFF_FFFE).unwrap(); // lane 0 off
        let s = WarpShuffle::new(cfg);
        let mut d = [0.0f32; 32];
        assert!(s.broadcast(&mut d).is_err());
    }

    #[test]
    fn all_to_all_basic() {
        let s = full_shuffle();
        let d = iota32();
        let mut out = vec![0.0f32; 32 * 32];
        s.all_to_all(&d, &mut out).unwrap();
        // row i should contain all lane values
        for i in 0..32 {
            for j in 0..32 {
                assert!((out[i * 32 + j] - (j + 1) as f32).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn all_to_all_out_too_small() {
        let s = full_shuffle();
        let d = iota32();
        let mut out = [0.0f32; 31];
        assert!(s.all_to_all(&d, &mut out).is_err());
    }

    #[test]
    fn shuffle_down_data_too_short() {
        let s = full_shuffle();
        let mut d = [0.0f32; 8];
        assert!(s.shuffle_down(&mut d, 1).is_err());
    }
}

// =========================================================================
// Property tests (proptest)
// =========================================================================

#[cfg(test)]
#[cfg(any(feature = "gpu", feature = "cuda"))]
mod proptests {
    use super::inner::*;
    use proptest::prelude::*;

    /// Strategy that produces a Vec<f32> of exactly 32 finite elements.
    fn warp_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-1e6f32..1e6f32, 32..=32)
    }

    proptest! {
        #[test]
        fn prop_reduce_sum_matches_iter(data in warp_data()) {
            let r = WarpReducer::new(WarpConfig::full());
            let expected: f32 = data.iter().sum();
            let mut buf = data;
            r.reduce_sum(&mut buf).unwrap();
            prop_assert!((buf[0] - expected).abs() < 1e-1,
                "sum mismatch: got {} expected {}", buf[0], expected);
        }

        #[test]
        fn prop_reduce_max_correct(data in warp_data()) {
            let r = WarpReducer::new(WarpConfig::full());
            let expected = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut buf = data;
            r.reduce_max(&mut buf).unwrap();
            prop_assert!((buf[0] - expected).abs() < 1e-4);
        }

        #[test]
        fn prop_reduce_min_correct(data in warp_data()) {
            let r = WarpReducer::new(WarpConfig::full());
            let expected = data.iter().copied().fold(f32::INFINITY, f32::min);
            let mut buf = data;
            r.reduce_min(&mut buf).unwrap();
            prop_assert!((buf[0] - expected).abs() < 1e-4);
        }

        #[test]
        fn prop_inclusive_scan_monotone(data in proptest::collection::vec(0.0f32..100.0f32, 32..=32)) {
            let r = WarpReducer::new(WarpConfig::full());
            let mut buf = data;
            r.inclusive_scan(&mut buf).unwrap();
            for i in 1..32 {
                prop_assert!(buf[i] >= buf[i - 1],
                    "inclusive scan not monotone at {}: {} < {}", i, buf[i], buf[i - 1]);
            }
        }

        #[test]
        fn prop_exclusive_scan_last_equals_total(data in warp_data()) {
            let r = WarpReducer::new(WarpConfig::full());
            let total: f32 = data.iter().sum();
            let mut exc = data.clone();
            r.exclusive_scan(&mut exc).unwrap();
            // exclusive_scan[last] + original[last] == total
            let reconstructed = exc[31] + data[31];
            prop_assert!((reconstructed - total).abs() < 1e-1,
                "last + orig != total: {} vs {}", reconstructed, total);
        }

        #[test]
        fn prop_shuffle_xor_involution(data in warp_data(), mask in 0u32..32u32) {
            let s = WarpShuffle::new(WarpConfig::full());
            let mut buf = data.clone();
            // applying XOR shuffle twice with same mask should restore original
            s.shuffle_xor(&mut buf, mask).unwrap();
            s.shuffle_xor(&mut buf, mask).unwrap();
            for i in 0..32 {
                prop_assert!((buf[i] - data[i]).abs() < 1e-4,
                    "xor involution broken at lane {}", i);
            }
        }

        #[test]
        fn prop_broadcast_uniform(data in warp_data()) {
            let s = WarpShuffle::new(WarpConfig::full());
            let mut buf = data;
            let lane0 = buf[0];
            s.broadcast(&mut buf).unwrap();
            for i in 0..32 {
                prop_assert!((buf[i] - lane0).abs() < 1e-4);
            }
        }
    }
}
