import re

file_path = 'crates/bitnet-quantization/src/device_aware_quantizer.rs'

with open(file_path, 'r') as f:
    content = f.read()

# Fix unused Arc import
search = "use std::sync::Arc;"
replace = '#[cfg(any(feature = "gpu", feature = "cuda"))]\nuse std::sync::Arc;'

if search in content and replace not in content:
    content = content.replace(search, replace)
    with open(file_path, 'w') as f:
        f.write(content)
    print("Fixed unused import in device_aware_quantizer.rs")
else:
    print("Already fixed or import not found")
