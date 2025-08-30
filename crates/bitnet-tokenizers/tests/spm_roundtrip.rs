#![cfg(feature = "spm")]

use bitnet_tokenizers::sp_tokenizer::SpTokenizer;
use std::process::Command;

#[test]
fn sentencepiece_encode_decode_roundtrip() {
    let dir = tempfile::tempdir().expect("tmp dir");
    let input_path = dir.path().join("train.txt");
    std::fs::write(&input_path, "hello world").expect("write data");
    let model_prefix = dir.path().join("spm");
    let script = format!(
        "import sentencepiece as spm; spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size=11')",
        input_path.display(),
        model_prefix.display()
    );
    let status = Command::new("python").arg("-c").arg(script).status().expect("run python");
    assert!(status.success());
    let model_path = dir.path().join("spm.model");
    let tokenizer = SpTokenizer::from_file(&model_path).expect("load spm");
    let ids = tokenizer.encode("hello world", false, false).expect("encode");
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, "hello world");
}
