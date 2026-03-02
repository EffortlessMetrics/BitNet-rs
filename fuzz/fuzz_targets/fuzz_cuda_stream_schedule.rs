#![no_main]

use arbitrary::Arbitrary;
use bitnet_kernels::cuda::stream_mgmt::{
    DefaultStreamBehavior, StreamConfig, StreamPool, StreamPriority,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct StreamInput {
    num_streams: u8,
    priority_byte: u8,
    ops: Vec<(u8, u8)>,
}

fuzz_target!(|input: StreamInput| {
    let num_streams = (input.num_streams as usize % 8) + 1;
    let priority = match input.priority_byte % 3 {
        0 => StreamPriority::Low,
        1 => StreamPriority::Normal,
        _ => StreamPriority::High,
    };
    let config = StreamConfig {
        num_streams,
        priority,
        default_stream_behavior: DefaultStreamBehavior::PerThread,
        enable_profiling: false,
    };

    let mut pool = match StreamPool::new(config) {
        Ok(p) => p,
        Err(_) => return,
    };

    // Invariant: pool has the requested number of streams.
    assert_eq!(pool.num_streams(), num_streams);

    let mut next_event_id: u64 = 1;
    let mut created_events = Vec::new();

    for &(op, param) in input.ops.iter().take(128) {
        let stream_idx = param as usize % num_streams;
        match op % 7 {
            // acquire_next
            0 => {
                let idx = pool.acquire_next();
                assert!(idx < num_streams, "acquire_next returned out-of-bounds index");
            }
            // acquire_least_loaded
            1 => {
                let idx = pool.acquire_least_loaded();
                assert!(idx < num_streams, "acquire_least_loaded returned out-of-bounds index");
            }
            // create_event + record
            2 => {
                let event = pool.create_event();
                let eid = event.id;
                created_events.push(eid);
                let _ = pool.record_event(eid, stream_idx);
                next_event_id = next_event_id.wrapping_add(1);
            }
            // wait on existing event
            3 => {
                if let Some(&eid) = created_events.last() {
                    let _ = pool.wait_event(eid, stream_idx);
                }
            }
            // sync single stream
            4 => {
                let _ = pool.sync_stream(stream_idx);
            }
            // sync all
            5 => {
                let _ = pool.sync_all();
            }
            // destroy event
            _ => {
                if let Some(eid) = created_events.pop() {
                    let _ = pool.destroy_event(eid);
                }
            }
        }
    }

    // Invariant: stream access with valid index must not panic.
    for i in 0..num_streams {
        assert!(pool.stream(i).is_ok());
    }

    // Invariant: out-of-bounds access returns Err.
    assert!(pool.stream(num_streams).is_err());

    // Final sync must not panic.
    let _ = pool.sync_all();
});
