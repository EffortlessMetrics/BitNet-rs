#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct BufferSizeInput {
    alloc_sizes: Vec<u64>,
    max_pool_bytes: u64,
    max_cached_buffers: u32,
    workgroup_x: u32,
    workgroup_y: u32,
    workgroup_z: u32,
    global_x: u64,
    global_y: u64,
    global_z: u64,
    memory_budget_mb: u32,
    tensor_bytes: u64,
    alignment: u32,
}

const MAX_ALLOC: u64 = 4 * 1024 * 1024 * 1024; // 4 GB (SecurityLimits)

fn validate_allocation(size: u64, budget: u64) -> Result<u64, &'static str> {
    if size == 0 {
        return Err("zero-size allocation");
    }
    if size > MAX_ALLOC {
        return Err("exceeds max memory allocation");
    }
    if size > budget {
        return Err("exceeds memory budget");
    }
    Ok(size)
}

fn validate_alignment(size: u64, alignment: u32) -> Result<u64, &'static str> {
    if alignment == 0 {
        return Err("zero alignment");
    }
    if !alignment.is_power_of_two() {
        return Err("alignment must be power of two");
    }
    let align = alignment as u64;
    // Round up to alignment boundary
    let aligned = size
        .checked_add(align - 1)
        .ok_or("overflow in alignment")?
        & !(align - 1);
    Ok(aligned)
}

fn validate_workgroup_size(x: u32, y: u32, z: u32) -> Result<u32, &'static str> {
    if x == 0 || y == 0 || z == 0 {
        return Err("zero workgroup dimension");
    }
    let total = (x as u64)
        .checked_mul(y as u64)
        .and_then(|v| v.checked_mul(z as u64))
        .ok_or("workgroup size overflow")?;
    // GPU workgroup limits: typically max 1024 threads
    if total > 1024 {
        return Err("workgroup exceeds max threads");
    }
    Ok(total as u32)
}

fn compute_dispatch_groups(
    global_x: u64,
    global_y: u64,
    global_z: u64,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
) -> Result<(u64, u64, u64), &'static str> {
    if wg_x == 0 || wg_y == 0 || wg_z == 0 {
        return Err("zero workgroup dimension");
    }
    let groups_x = global_x.checked_add(wg_x as u64 - 1).ok_or("overflow")? / wg_x as u64;
    let groups_y = global_y.checked_add(wg_y as u64 - 1).ok_or("overflow")? / wg_y as u64;
    let groups_z = global_z.checked_add(wg_z as u64 - 1).ok_or("overflow")? / wg_z as u64;
    // GPU dispatch limit: typically 65535 per dimension
    const MAX_DISPATCH: u64 = 65535;
    if groups_x > MAX_DISPATCH || groups_y > MAX_DISPATCH || groups_z > MAX_DISPATCH {
        return Err("dispatch groups exceed GPU limit");
    }
    Ok((groups_x, groups_y, groups_z))
}

fn compute_memory_budget(budget_mb: u32) -> Result<u64, &'static str> {
    let budget_bytes = (budget_mb as u64).checked_mul(1024 * 1024).ok_or("overflow")?;
    if budget_bytes > MAX_ALLOC {
        return Err("budget exceeds max allocation");
    }
    Ok(budget_bytes)
}

fuzz_target!(|input: BufferSizeInput| {
    let budget = match compute_memory_budget(input.memory_budget_mb) {
        Ok(b) => b,
        Err(_) => MAX_ALLOC,
    };

    // Fuzz allocation sizes — must return Err, never panic
    let mut total_allocated: u64 = 0;
    for &size in input.alloc_sizes.iter().take(256) {
        match validate_allocation(size, budget) {
            Ok(s) => {
                total_allocated = total_allocated.saturating_add(s);
            }
            Err(_) => {}
        }
    }

    // Fuzz aligned allocation
    let _ = validate_alignment(input.tensor_bytes, input.alignment);
    let _ = validate_alignment(0, input.alignment);
    let _ = validate_alignment(u64::MAX, input.alignment);

    // Fuzz workgroup size validation — must not panic
    let _ = validate_workgroup_size(input.workgroup_x, input.workgroup_y, input.workgroup_z);

    // Fuzz dispatch group calculation — must not panic
    let wg_x = input.workgroup_x.max(1);
    let wg_y = input.workgroup_y.max(1);
    let wg_z = input.workgroup_z.max(1);
    let _ = compute_dispatch_groups(
        input.global_x,
        input.global_y,
        input.global_z,
        wg_x,
        wg_y,
        wg_z,
    );

    // Fuzz pool config bounds
    let _ = input.max_pool_bytes.checked_mul(input.max_cached_buffers as u64);
});
