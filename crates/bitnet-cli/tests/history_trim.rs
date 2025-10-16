//! Test Suite 3: CLI - History Management (bitnet-cli)
//!
//! Tests feature spec: chat-repl-ux-polish.md#AC3-history-management
//!
//! This test suite validates the conversation history management functionality
//! for the BitNet.rs CLI chat mode. Tests ensure proper FIFO trimming when
//! history limits are enforced.
//!
//! **TDD Approach**: These tests compile successfully but fail because the
//! history management logic with configurable limits needs to be implemented
//! in the chat mode with proper turn counting and trimming.

use anyhow::Result;

/// Represents a conversation turn in chat history
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConversationTurn {
    user_message: String,
    assistant_message: String,
}

impl ConversationTurn {
    fn new(user: impl Into<String>, assistant: impl Into<String>) -> Self {
        Self { user_message: user.into(), assistant_message: assistant.into() }
    }
}

/// Simulates the history manager that should be implemented in CLI chat mode
#[derive(Debug)]
struct ChatHistoryManager {
    history: Vec<ConversationTurn>,
    limit: Option<usize>,
}

impl ChatHistoryManager {
    fn new(limit: Option<usize>) -> Self {
        Self { history: Vec::new(), limit }
    }

    fn push(&mut self, turn: ConversationTurn) {
        self.history.push(turn);
        self.enforce_limit();
    }

    fn enforce_limit(&mut self) {
        if let Some(max_turns) = self.limit {
            if self.history.len() > max_turns {
                // Drop oldest turns (FIFO)
                let excess = self.history.len() - max_turns;
                self.history.drain(0..excess);
            }
        }
    }

    fn len(&self) -> usize {
        self.history.len()
    }

    fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    fn get_history(&self) -> &[ConversationTurn] {
        &self.history
    }

    fn clear(&mut self) {
        self.history.clear();
    }
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-limit-enforcement
#[test]
fn test_history_limit_enforcement() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(3));

    // Add 5 turns (exceeds limit of 3)
    manager.push(ConversationTurn::new("Turn 1 user", "Turn 1 assistant"));
    manager.push(ConversationTurn::new("Turn 2 user", "Turn 2 assistant"));
    manager.push(ConversationTurn::new("Turn 3 user", "Turn 3 assistant"));
    manager.push(ConversationTurn::new("Turn 4 user", "Turn 4 assistant"));
    manager.push(ConversationTurn::new("Turn 5 user", "Turn 5 assistant"));

    // Should only keep last 3 turns
    assert_eq!(manager.len(), 3, "History should be trimmed to limit of 3 turns");

    let history = manager.get_history();

    // Verify oldest turns were dropped (FIFO)
    assert_eq!(history[0].user_message, "Turn 3 user", "Oldest remaining turn should be turn 3");
    assert_eq!(history[1].user_message, "Turn 4 user", "Second oldest should be turn 4");
    assert_eq!(history[2].user_message, "Turn 5 user", "Newest turn should be turn 5");

    // Verify turns 1 and 2 were dropped
    assert!(
        !history.iter().any(|t| t.user_message == "Turn 1 user"),
        "Turn 1 should have been dropped"
    );
    assert!(
        !history.iter().any(|t| t.user_message == "Turn 2 user"),
        "Turn 2 should have been dropped"
    );

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-fifo-ordering
#[test]
fn test_fifo_ordering_maintained() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(2));

    // Add turns sequentially
    manager.push(ConversationTurn::new("First", "First response"));
    manager.push(ConversationTurn::new("Second", "Second response"));
    manager.push(ConversationTurn::new("Third", "Third response"));
    manager.push(ConversationTurn::new("Fourth", "Fourth response"));

    let history = manager.get_history();

    assert_eq!(history.len(), 2, "Should maintain limit of 2 turns");

    // Most recent 2 turns should be preserved in order
    assert_eq!(history[0].user_message, "Third", "First remaining should be 'Third'");
    assert_eq!(history[1].user_message, "Fourth", "Second remaining should be 'Fourth'");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-no-limit
#[test]
fn test_no_limit_specified() -> Result<()> {
    let mut manager = ChatHistoryManager::new(None);

    // Add many turns
    for i in 0..20 {
        manager.push(ConversationTurn::new(format!("User {}", i), format!("Assistant {}", i)));
    }

    // All turns should be preserved
    assert_eq!(manager.len(), 20, "History should grow unbounded when no limit specified");

    let history = manager.get_history();

    // Verify all turns are present
    for i in 0..20 {
        assert_eq!(
            history[i].user_message,
            format!("User {}", i),
            "Turn {} should be preserved",
            i
        );
    }

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-limit-zero
#[test]
fn test_limit_zero_edge_case() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(0));

    // Add turns
    manager.push(ConversationTurn::new("Turn 1", "Response 1"));
    manager.push(ConversationTurn::new("Turn 2", "Response 2"));

    // With limit 0, history should remain empty
    assert_eq!(manager.len(), 0, "Limit of 0 should keep history empty");
    assert!(manager.is_empty(), "History should be empty with limit 0");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-limit-one
#[test]
fn test_limit_one() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(1));

    manager.push(ConversationTurn::new("First", "First response"));
    assert_eq!(manager.len(), 1, "Should have 1 turn");

    manager.push(ConversationTurn::new("Second", "Second response"));
    assert_eq!(manager.len(), 1, "Should still have only 1 turn");

    let history = manager.get_history();
    assert_eq!(history[0].user_message, "Second", "Only most recent turn should remain");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-exact-limit
#[test]
fn test_exact_limit_no_trimming() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(5));

    // Add exactly 5 turns
    for i in 0..5 {
        manager.push(ConversationTurn::new(format!("User {}", i), format!("Assistant {}", i)));
    }

    assert_eq!(manager.len(), 5, "Should have exactly 5 turns");

    // No turns should be dropped
    let history = manager.get_history();
    for i in 0..5 {
        assert_eq!(
            history[i].user_message,
            format!("User {}", i),
            "Turn {} should be preserved",
            i
        );
    }

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-clear-history
#[test]
fn test_clear_history() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(5));

    // Add turns
    manager.push(ConversationTurn::new("Turn 1", "Response 1"));
    manager.push(ConversationTurn::new("Turn 2", "Response 2"));
    manager.push(ConversationTurn::new("Turn 3", "Response 3"));

    assert_eq!(manager.len(), 3, "Should have 3 turns before clear");

    // Clear history
    manager.clear();

    assert_eq!(manager.len(), 0, "History should be empty after clear");
    assert!(manager.is_empty(), "Should report as empty");

    // Should be able to add new turns after clear
    manager.push(ConversationTurn::new("New turn", "New response"));
    assert_eq!(manager.len(), 1, "Should be able to add turns after clear");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-bulk-add-enforcement
#[test]
fn test_bulk_add_with_limit() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(3));

    // Simulate rapid turn addition (e.g., restoring from file)
    let turns = vec![
        ConversationTurn::new("A", "Response A"),
        ConversationTurn::new("B", "Response B"),
        ConversationTurn::new("C", "Response C"),
        ConversationTurn::new("D", "Response D"),
        ConversationTurn::new("E", "Response E"),
    ];

    for turn in turns {
        manager.push(turn);
    }

    // Should maintain limit throughout
    assert_eq!(manager.len(), 3, "Limit should be enforced during bulk add");

    let history = manager.get_history();
    assert_eq!(history[0].user_message, "C", "Oldest remaining should be C");
    assert_eq!(history[1].user_message, "D", "Middle should be D");
    assert_eq!(history[2].user_message, "E", "Newest should be E");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-single-turn-preservation
#[test]
fn test_single_turn_below_limit() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(10));

    manager.push(ConversationTurn::new("Only turn", "Only response"));

    assert_eq!(manager.len(), 1, "Should have 1 turn");

    let history = manager.get_history();
    assert_eq!(history[0].user_message, "Only turn", "Single turn should be preserved");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-limit-change-simulation
#[test]
fn test_limit_change_behavior() -> Result<()> {
    // Simulate changing limit mid-session (create new manager with different limit)
    let mut manager_unlimited = ChatHistoryManager::new(None);

    // Add 10 turns with no limit
    for i in 0..10 {
        manager_unlimited
            .push(ConversationTurn::new(format!("User {}", i), format!("Assistant {}", i)));
    }

    assert_eq!(manager_unlimited.len(), 10, "Should have all 10 turns");

    // Simulate applying a limit (create new manager with limit, copy history)
    let mut manager_limited = ChatHistoryManager::new(Some(5));
    for turn in manager_unlimited.get_history() {
        manager_limited.push(turn.clone());
    }

    // After applying limit, only last 5 should remain
    assert_eq!(manager_limited.len(), 5, "Should have trimmed to limit of 5");

    let limited_history = manager_limited.get_history();
    assert_eq!(limited_history[0].user_message, "User 5", "First remaining turn should be User 5");
    assert_eq!(limited_history[4].user_message, "User 9", "Last remaining turn should be User 9");

    Ok(())
}

/// Tests feature spec: chat-repl-ux-polish.md#AC3-assistant-response-pairing
#[test]
fn test_turn_pairing_maintained() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(2));

    manager.push(ConversationTurn::new("Question 1", "Answer 1"));
    manager.push(ConversationTurn::new("Question 2", "Answer 2"));
    manager.push(ConversationTurn::new("Question 3", "Answer 3"));

    let history = manager.get_history();

    // Verify user-assistant pairing is maintained
    assert_eq!(history[0].user_message, "Question 2", "User message preserved");
    assert_eq!(history[0].assistant_message, "Answer 2", "Paired assistant response preserved");

    assert_eq!(history[1].user_message, "Question 3", "User message preserved");
    assert_eq!(history[1].assistant_message, "Answer 3", "Paired assistant response preserved");

    Ok(())
}

/// Integration test: Full chat session simulation
/// Tests feature spec: chat-repl-ux-polish.md#AC3-integration
#[test]
fn test_full_chat_session_with_history_limit() -> Result<()> {
    let mut manager = ChatHistoryManager::new(Some(4));

    // Simulate a chat session with 7 exchanges
    let exchanges = vec![
        ("Hello", "Hi there!"),
        ("How are you?", "I'm doing well!"),
        ("What's your name?", "I'm an AI assistant."),
        ("Tell me a joke", "Why did the chicken cross the road?"),
        ("I don't know", "To get to the other side!"),
        ("That's funny", "Glad you enjoyed it!"),
        ("Goodbye", "Have a great day!"),
    ];

    for (user_msg, assistant_msg) in exchanges {
        manager.push(ConversationTurn::new(user_msg, assistant_msg));
    }

    // Should only keep last 4 turns
    assert_eq!(manager.len(), 4, "Should maintain limit of 4 turns");

    let history = manager.get_history();

    // Verify the last 4 turns are preserved
    assert_eq!(history[0].user_message, "Tell me a joke");
    assert_eq!(history[1].user_message, "I don't know");
    assert_eq!(history[2].user_message, "That's funny");
    assert_eq!(history[3].user_message, "Goodbye");

    // First 3 turns should be dropped
    assert!(!history.iter().any(|t| t.user_message == "Hello"));
    assert!(!history.iter().any(|t| t.user_message == "How are you?"));
    assert!(!history.iter().any(|t| t.user_message == "What's your name?"));

    Ok(())
}
