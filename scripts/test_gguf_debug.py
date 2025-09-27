#!/usr/bin/env python3
import struct

def read_string(data, offset):
    """Read GGUF string (u64 length + utf8 bytes)"""
    str_len = struct.unpack('<Q', data[offset:offset+8])[0]
    offset += 8
    s = data[offset:offset+str_len]
    offset += str_len
    return s, offset

def align_up(offset, alignment=32):
    """Align offset to alignment boundary"""
    return ((offset + alignment - 1) // alignment) * alignment

# Read GGUF file
with open('models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf', 'rb') as f:
    data = f.read()

offset = 0

# Parse header
magic = data[offset:offset+4]
offset += 4
version = struct.unpack('<I', data[offset:offset+4])[0]
offset += 4
tensor_count = struct.unpack('<Q', data[offset:offset+8])[0]
offset += 8
metadata_count = struct.unpack('<Q', data[offset:offset+8])[0]
offset += 8

print(f"Header: magic={magic}, version={version}, tensors={tensor_count}, metadata={metadata_count}")
print(f"Offset after header: {offset}")

# Parse metadata entries
for i in range(min(5, metadata_count)):  # Just first 5 for debugging
    print(f"\n--- Metadata entry {i} ---")
    print(f"Offset before key: {offset}")
    
    # Read key
    key, offset = read_string(data, offset)
    print(f"Key: {key.decode('utf-8', errors='replace')}")
    
    # Read value type
    value_type = struct.unpack('B', data[offset:offset+1])[0]
    offset += 1
    print(f"Value type: {value_type}")
    
    # Skip value parsing for now, just advance offset based on type
    if value_type == 8:  # String
        val, offset = read_string(data, offset)
        print(f"String value: {val.decode('utf-8', errors='replace')}")
    elif value_type == 4:  # U32
        val = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        print(f"U32 value: {val}")
    elif value_type == 9:  # Array
        array_type = struct.unpack('B', data[offset:offset+1])[0]
        offset += 1
        array_len = struct.unpack('<Q', data[offset:offset+8])[0]
        offset += 8
        print(f"Array: type={array_type}, len={array_len}")
        if array_type == 8 and array_len > 0:  # String array
            # Just read first string to check
            s, new_off = read_string(data, offset)
            print(f"First array string: {repr(s[:50])}")
    else:
        print(f"Skipping value type {value_type}")
        break
    
    # Apply alignment after value
    old_offset = offset
    offset = align_up(offset, 32)
    print(f"Aligned offset: {old_offset} -> {offset} (delta={offset-old_offset})")