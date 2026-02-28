import re

with open("crates/bitnet-server/src/concurrency.rs", "r") as f:
    content = f.read()

content = content.replace(
"""        for (ip, bucket) in limiters.iter() {
            if let Ok(last_refill) = bucket.last_refill.try_lock() {
                if now.duration_since(*last_refill) > max_idle {
                    keys_to_remove.push(*ip);
                }
            }
        }""",
"""        for (ip, bucket) in limiters.iter() {
            if let Ok(last_refill) = bucket.last_refill.try_lock() {
                #[allow(clippy::collapsible_if)]
                if now.duration_since(*last_refill) > max_idle {
                    keys_to_remove.push(*ip);
                }
            }
        }""")

content = content.replace(
"""        let mut config = ConcurrencyConfig::default();
        config.per_ip_rate_limit = Some(10);""",
"""        let config = ConcurrencyConfig {
            per_ip_rate_limit: Some(10),
            ..Default::default()
        };""")

with open("crates/bitnet-server/src/concurrency.rs", "w") as f:
    f.write(content)
