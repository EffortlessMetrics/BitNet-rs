fn main() {
    let mut logits = vec![1.0, 2.0];
    let mut acc = 1.0;
    let count_penalty: f32 = 2.0;
    let count: i32 = 3;
    let penalty1 = count_penalty.powi(count);

    let mut penalty2 = 1.0;
    for _ in 0..count {
        penalty2 *= count_penalty;
    }

    let mut logit_1 = 8.0;
    let mut logit_2 = 8.0;

    logit_1 /= penalty1;

    let inv_penalty = 1.0 / count_penalty;
    for _ in 0..count {
        logit_2 *= inv_penalty;
    }
    println!("logit_1: {}, logit_2: {}", logit_1, logit_2);
}
