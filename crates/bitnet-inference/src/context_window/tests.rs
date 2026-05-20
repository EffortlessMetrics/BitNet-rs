use super::*;

#[test]
fn test_new_window() {
    let w = ContextWindow::new(4096);
    assert_eq!(w.max_length(), 4096);
    assert!(w.is_empty());
    assert_eq!(w.remaining(), 4096);
}

#[test]
fn test_append() {
    let mut w = ContextWindow::new(5);
    assert_eq!(w.append(&[1, 2, 3]), 3);
    assert_eq!(w.current_length(), 3);
    assert_eq!(w.remaining(), 2);
}

#[test]
fn test_append_overflow() {
    let mut w = ContextWindow::new(3);
    assert_eq!(w.append(&[1, 2, 3, 4, 5]), 3);
    assert!(w.is_full());
}

#[test]
fn test_last_n() {
    let mut w = ContextWindow::new(10);
    w.append(&[1, 2, 3, 4, 5]);
    assert_eq!(w.last_n(3), &[3, 4, 5]);
    assert_eq!(w.last_n(10), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_truncate() {
    let mut w = ContextWindow::new(10);
    w.append(&[1, 2, 3, 4, 5]);
    w.truncate_to_last(3);
    assert_eq!(w.tokens(), &[3, 4, 5]);
}

#[test]
fn test_clear() {
    let mut w = ContextWindow::new(10);
    w.append(&[1, 2]);
    w.clear();
    assert!(w.is_empty());
}

#[test]
fn test_can_fit() {
    let mut w = ContextWindow::new(5);
    w.append(&[1, 2, 3]);
    assert!(w.can_fit(2));
    assert!(!w.can_fit(3));
}

#[test]
fn test_utilization() {
    let mut w = ContextWindow::new(4);
    w.append(&[1, 2]);
    assert!((w.utilization() - 0.5).abs() < 0.01);
}

#[test]
fn test_fixed_budget() {
    let (p, g) = compute_budgets(4096, 1000, AllocationStrategy::Fixed { max_prompt: 2048 });
    assert_eq!(p, 1000);
    assert_eq!(g, 3096);
}

#[test]
fn test_dynamic_budget() {
    let (_p, g) = compute_budgets(4096, 3900, AllocationStrategy::Dynamic { min_generation: 256 });
    assert!(g >= 256);
}

#[test]
fn test_even_split() {
    let (p, g) = compute_budgets(4096, 3000, AllocationStrategy::EvenSplit);
    assert_eq!(p, 2048);
    assert_eq!(g, 2048);
}

#[test]
fn test_report() {
    let mut w = ContextWindow::new(100);
    w.append(&[1, 2, 3]);
    let r = w.report();
    assert_eq!(r.max_length, 100);
    assert_eq!(r.used, 3);
    assert_eq!(r.remaining, 97);
}
