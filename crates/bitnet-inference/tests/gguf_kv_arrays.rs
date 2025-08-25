//! Tests for GGUF KV array parsing

#[test]
fn reads_numeric_array_kv() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arr.gguf");
    let mut f = std::fs::File::create(&path).unwrap();

    // Header: magic, v=2, n_tensors=0, n_kv=1
    f.write_all(b"GGUF").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap();

    // KV #1: key "arr.u32", ARRAY of UINT32 with 3 elements: [7,8,9]
    let key = b"arr.u32";
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key).unwrap();
    f.write_all(&9u32.to_le_bytes()).unwrap(); // ARRAY
    f.write_all(&4u32.to_le_bytes()).unwrap(); // elem type = UINT32
    f.write_all(&3u64.to_le_bytes()).unwrap(); // length
    for v in [7u32, 8, 9] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();

    let kvs = bitnet_inference::gguf::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs.len(), 1);
    assert_eq!(kvs[0].key, "arr.u32");
    match &kvs[0].value {
        bitnet_inference::gguf::GgufValue::Array(items) => {
            assert_eq!(items.len(), 3);
            assert!(matches!(items[0], bitnet_inference::gguf::GgufValue::U32(7)));
            assert!(matches!(items[1], bitnet_inference::gguf::GgufValue::U32(8)));
            assert!(matches!(items[2], bitnet_inference::gguf::GgufValue::U32(9)));
        }
        _ => panic!("expected array"),
    }
}

#[test]
fn reads_string_array_kv() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arr_str.gguf");
    let mut f = std::fs::File::create(&path).unwrap();

    // Header: v=2, 0 tensors, 1 kv
    f.write_all(b"GGUF").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap();

    // KV #1: key "arr.str", ARRAY(STRING) of 2: ["a", "bc"]
    let key = b"arr.str";
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key).unwrap();
    f.write_all(&9u32.to_le_bytes()).unwrap(); // ARRAY
    f.write_all(&8u32.to_le_bytes()).unwrap(); // elem type = STRING
    f.write_all(&2u64.to_le_bytes()).unwrap(); // length
    // "a"
    f.write_all(&1u64.to_le_bytes()).unwrap();
    f.write_all(b"a").unwrap();
    // "bc"
    f.write_all(&2u64.to_le_bytes()).unwrap();
    f.write_all(b"bc").unwrap();
    f.flush().unwrap();

    let kvs = bitnet_inference::gguf::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].key, "arr.str");
    match &kvs[0].value {
        bitnet_inference::gguf::GgufValue::Array(items) => {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0], bitnet_inference::gguf::GgufValue::String("a".into()));
            assert_eq!(items[1], bitnet_inference::gguf::GgufValue::String("bc".into()));
        }
        _ => panic!("expected array"),
    }
}

#[test]
fn handles_large_arrays_with_sampling() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_arr.gguf");
    let mut f = std::fs::File::create(&path).unwrap();

    // Header: v=2, 0 tensors, 1 kv
    f.write_all(b"GGUF").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap();

    // KV #1: key "big.arr", ARRAY of 1000 UINT32s
    let key = b"big.arr";
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key).unwrap();
    f.write_all(&9u32.to_le_bytes()).unwrap(); // ARRAY
    f.write_all(&4u32.to_le_bytes()).unwrap(); // elem type = UINT32
    f.write_all(&1000u64.to_le_bytes()).unwrap(); // length = 1000
    for i in 0u32..1000 {
        f.write_all(&i.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();

    let kvs = bitnet_inference::gguf::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].key, "big.arr");
    match &kvs[0].value {
        bitnet_inference::gguf::GgufValue::Array(items) => {
            // Should be capped at ARRAY_SAMPLE_LIMIT (256)
            assert_eq!(items.len(), 256);
            // Check first few values
            assert!(matches!(items[0], bitnet_inference::gguf::GgufValue::U32(0)));
            assert!(matches!(items[1], bitnet_inference::gguf::GgufValue::U32(1)));
            assert!(matches!(items[255], bitnet_inference::gguf::GgufValue::U32(255)));
        }
        _ => panic!("expected array"),
    }
}

#[test]
fn reads_multiple_kvs_with_arrays() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_kv.gguf");
    let mut f = std::fs::File::create(&path).unwrap();

    // Header: v=2, 0 tensors, 3 kvs
    f.write_all(b"GGUF").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&3u64.to_le_bytes()).unwrap();

    // KV #1: simple u32
    let key1 = b"simple";
    f.write_all(&(key1.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key1).unwrap();
    f.write_all(&4u32.to_le_bytes()).unwrap(); // UINT32
    f.write_all(&42u32.to_le_bytes()).unwrap();

    // KV #2: array of 2 bools
    let key2 = b"bools";
    f.write_all(&(key2.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key2).unwrap();
    f.write_all(&9u32.to_le_bytes()).unwrap(); // ARRAY
    f.write_all(&7u32.to_le_bytes()).unwrap(); // elem type = BOOL
    f.write_all(&2u64.to_le_bytes()).unwrap(); // length
    f.write_all(&[1u8, 0u8]).unwrap(); // true, false

    // KV #3: string
    let key3 = b"name";
    f.write_all(&(key3.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key3).unwrap();
    f.write_all(&8u32.to_le_bytes()).unwrap(); // STRING
    f.write_all(&5u64.to_le_bytes()).unwrap(); // length
    f.write_all(b"hello").unwrap();

    f.flush().unwrap();

    let kvs = bitnet_inference::gguf::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs.len(), 3);

    // Check first KV
    assert_eq!(kvs[0].key, "simple");
    assert_eq!(kvs[0].value, bitnet_inference::gguf::GgufValue::U32(42));

    // Check second KV (array)
    assert_eq!(kvs[1].key, "bools");
    match &kvs[1].value {
        bitnet_inference::gguf::GgufValue::Array(items) => {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0], bitnet_inference::gguf::GgufValue::Bool(true));
            assert_eq!(items[1], bitnet_inference::gguf::GgufValue::Bool(false));
        }
        _ => panic!("expected array"),
    }

    // Check third KV
    assert_eq!(kvs[2].key, "name");
    assert_eq!(kvs[2].value, bitnet_inference::gguf::GgufValue::String("hello".to_string()));
}

#[test]
fn handles_float_arrays() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("float_arr.gguf");
    let mut f = std::fs::File::create(&path).unwrap();

    // Header
    f.write_all(b"GGUF").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap();

    // KV: array of 3 f32 values
    let key = b"floats";
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key).unwrap();
    f.write_all(&9u32.to_le_bytes()).unwrap(); // ARRAY
    f.write_all(&6u32.to_le_bytes()).unwrap(); // elem type = FLOAT32
    f.write_all(&3u64.to_le_bytes()).unwrap(); // length
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&2.5f32.to_le_bytes()).unwrap();
    f.write_all(&(-3.14f32).to_le_bytes()).unwrap();
    f.flush().unwrap();

    let kvs = bitnet_inference::gguf::read_kv_pairs(&path, None).unwrap();
    assert_eq!(kvs[0].key, "floats");
    match &kvs[0].value {
        bitnet_inference::gguf::GgufValue::Array(items) => {
            assert_eq!(items.len(), 3);
            match &items[0] {
                bitnet_inference::gguf::GgufValue::F32(v) => assert!((v - 1.0).abs() < 0.001),
                _ => panic!("expected f32"),
            }
            match &items[1] {
                bitnet_inference::gguf::GgufValue::F32(v) => assert!((v - 2.5).abs() < 0.001),
                _ => panic!("expected f32"),
            }
            match &items[2] {
                bitnet_inference::gguf::GgufValue::F32(v) => assert!((v + 3.14).abs() < 0.001),
                _ => panic!("expected f32"),
            }
        }
        _ => panic!("expected array"),
    }
}
