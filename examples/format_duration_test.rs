use std::time::Duration;

fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let nanos = duration.subsec_nanos();

    if total_secs >= 3600 {
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if total_secs >= 60 {
        let minutes = total_secs / 60;
        let seconds = total_secs % 60;
        format!("{}m {}s", minutes, seconds)
    } else if total_secs > 0 {
        format!("{}.{:03}s", total_secs, nanos / 1_000_000)
    } else if nanos >= 1_000_000 {
        format!("{:.1}ms", nanos as f64 / 1_000_000.0)
    } else if nanos >= 1_000 {
        format!("{:.1}μs", nanos as f64 / 1_000.0)
    } else {
        format!("{}ns", nanos)
    }
}

fn main() {
    let tests = vec![
        Duration::from_nanos(500),
        Duration::from_micros(1500),
        Duration::from_millis(1500),
        Duration::from_secs(1),
        Duration::from_secs(65),
        Duration::from_secs(3665),
    ];

    for duration in tests {
        println!("{} -> {}", duration.as_nanos(), format_duration(duration));
    }
}
