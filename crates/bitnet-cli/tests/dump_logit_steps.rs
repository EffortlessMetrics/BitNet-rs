#[test]
fn greedy_check_skipped_without_dump_logit_steps() {
    let assert_greedy = true;
    let greedy = true;
    let dump_logit_steps: Option<usize> = None;
    let step_idx = 0;

    let mut executed = false;
    if assert_greedy && greedy && dump_logit_steps.is_some_and(|max_steps| step_idx < max_steps) {
        executed = true;
    }

    assert!(!executed, "greedy check should be skipped when dump_logit_steps is None");
}
