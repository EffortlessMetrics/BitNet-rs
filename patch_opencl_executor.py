import re

with open("crates/bitnet-kernels/src/opencl_async_executor.rs", "r") as f:
    content = f.read()

# Fix the formatting by splitting the if let ... && ... since rustfmt might complain
# or just make it compliant
new_content = content.replace("""            if let Some(last) = merged.last_mut()
                && interval.0 <= last.1 {
                    last.1 = last.1.max(interval.1);
                    continue;
                }""", """            if let Some(last) = merged.last_mut() {
                if interval.0 <= last.1 {
                    last.1 = last.1.max(interval.1);
                    continue;
                }
            }""")

with open("crates/bitnet-kernels/src/opencl_async_executor.rs", "w") as f:
    f.write(new_content)
