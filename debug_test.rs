use bitnet_tokenizers::BasicTokenizer;

fn main() {
    let tokenizer = BasicTokenizer::with_config(5000, Some(4999), Some(1));
    let text = "custom configuration";
    let tokens = tokenizer.encode(text, true).unwrap();
    let decoded_with_special = tokenizer.decode(&tokens, false).unwrap();
    let decoded_without_special = tokenizer.decode(&tokens, true).unwrap();

    println!("Text: '{}'", text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded with special: '{}'", decoded_with_special);
    println!("Decoded without special: '{}'", decoded_without_special);
}
