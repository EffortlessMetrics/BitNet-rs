use std::fs::File;
use std::io::{Read, Seek};
use memmap2::Mmap;

fn main() -> anyhow::Result<()> {
    let path = "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    
    // Read magic
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    println!("Magic: {:?}", std::str::from_utf8(&magic)?);
    
    // Read version
    let version = read_u32(&mut cursor)?;
    println!("Version: {}", version);
    
    // Read counts
    let n_tensors = read_u64(&mut cursor)?;
    let n_kv = read_u64(&mut cursor)?;
    println!("Tensors: {}, KV pairs: {}", n_tensors, n_kv);
    
    // Skip KV pairs
    for i in 0..n_kv {
        let key = read_string(&mut cursor)?;
        let ty = read_u32(&mut cursor)?;
        if i < 5 {
            println!("KV[{}]: {} (type {})", i, key, ty);
        }
        skip_value(&mut cursor, ty)?;
    }
    
    // Read tensor infos
    println!("\nTensor names:");
    for i in 0..n_tensors.min(10) {
        let name = read_string(&mut cursor)?;
        let n_dims = read_u32(&mut cursor)?;
        let mut dims = Vec::new();
        for _ in 0..n_dims {
            dims.push(read_u64(&mut cursor)?);
        }
        let ty = read_u32(&mut cursor)?;
        let _offset = read_u64(&mut cursor)?;
        
        println!("  [{}] {} {:?} (type {})", i, name, dims, ty);
        
        if name.contains("emb") || name.contains("output") || name.contains("lm_head") {
            println!("    ^^ IMPORTANT TENSOR");
        }
    }
    
    Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> anyhow::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> anyhow::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn skip_value<R: Read>(r: &mut R, ty: u32) -> anyhow::Result<()> {
    match ty {
        0..=7 => {
            let sizes = [1, 1, 2, 2, 4, 4, 4, 1];
            let mut buf = vec![0u8; sizes[ty as usize]];
            r.read_exact(&mut buf)?;
        }
        8 => {
            let n = read_u64(r)? as usize;
            let mut buf = vec![0u8; n];
            r.read_exact(&mut buf)?;
        }
        9 => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem_ty)?;
            }
        }
        10..=12 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
        }
        _ => {}
    }
    Ok(())
}