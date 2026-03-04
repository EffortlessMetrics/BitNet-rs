sed -i 's/for i in 0..32 {/for (i, v_i) in data.iter_mut().enumerate().take(32) {/g' crates/bitnet-kernels/src/cuda/warp_ops.rs
sed -i 's/for i in 0..8 {/for v_i in data.iter_mut().take(8) {/g' crates/bitnet-kernels/src/cuda/warp_ops.rs
sed -i 's/for i in 8..16 {/for v_i in data.iter_mut().take(16).skip(8) {/g' crates/bitnet-kernels/src/cuda/warp_ops.rs
sed -i 's/for i in 16..32 {/for v_i in data.iter_mut().take(32).skip(16) {/g' crates/bitnet-kernels/src/cuda/warp_ops.rs
