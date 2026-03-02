//! Inference session manager.
//!
//! Tracks multi-turn conversations and inference state.

use std::collections::HashMap;

/// Unique session identifier.
pub type SessionId = u64;

/// A single turn in a conversation.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
    pub token_count: usize,
}

/// Conversation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// Session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Active,
    Completed,
    Expired,
}

/// Inference session tracking a conversation.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub state: SessionState,
    pub turns: Vec<Turn>,
    pub total_tokens: usize,
    pub max_context: usize,
    pub model_id: String,
}

impl Session {
    pub fn new(id: SessionId, model_id: String, max_context: usize) -> Self {
        Self {
            id,
            state: SessionState::Active,
            turns: Vec::new(),
            total_tokens: 0,
            max_context,
            model_id,
        }
    }

    pub fn add_turn(&mut self, role: Role, content: String, token_count: usize) -> bool {
        if self.state != SessionState::Active {
            return false;
        }
        if self.total_tokens + token_count > self.max_context {
            return false;
        }
        self.total_tokens += token_count;
        self.turns.push(Turn { role, content, token_count });
        true
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    pub fn remaining_tokens(&self) -> usize {
        self.max_context.saturating_sub(self.total_tokens)
    }

    pub fn is_active(&self) -> bool {
        self.state == SessionState::Active
    }

    pub fn complete(&mut self) {
        self.state = SessionState::Completed;
    }

    pub fn expire(&mut self) {
        self.state = SessionState::Expired;
    }

    /// Trim oldest non-system turns to make room.
    pub fn trim_to_fit(&mut self, needed: usize) -> usize {
        let mut freed = 0;
        while self.total_tokens + needed > self.max_context && !self.turns.is_empty() {
            // Keep system prompts
            if let Some(pos) = self.turns.iter().position(|t| t.role != Role::System) {
                freed += self.turns[pos].token_count;
                self.total_tokens -= self.turns[pos].token_count;
                self.turns.remove(pos);
            } else {
                break;
            }
        }
        freed
    }
}

/// Session manager.
#[derive(Debug)]
pub struct SessionManager {
    sessions: HashMap<SessionId, Session>,
    next_id: SessionId,
    max_sessions: usize,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl SessionManager {
    pub fn new(max_sessions: usize) -> Self {
        Self { sessions: HashMap::new(), next_id: 1, max_sessions }
    }

    pub fn create(&mut self, model_id: String, max_context: usize) -> Option<SessionId> {
        if self.sessions.len() >= self.max_sessions {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(id, Session::new(id, model_id, max_context));
        Some(id)
    }

    pub fn get(&self, id: SessionId) -> Option<&Session> {
        self.sessions.get(&id)
    }

    pub fn get_mut(&mut self, id: SessionId) -> Option<&mut Session> {
        self.sessions.get_mut(&id)
    }

    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    pub fn active_count(&self) -> usize {
        self.sessions.values().filter(|s| s.is_active()).count()
    }

    pub fn remove(&mut self, id: SessionId) -> bool {
        self.sessions.remove(&id).is_some()
    }

    pub fn expire_all(&mut self) {
        for session in self.sessions.values_mut() {
            if session.is_active() {
                session.expire();
            }
        }
    }

    pub fn cleanup_expired(&mut self) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(|_, s| s.state != SessionState::Expired);
        before - self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_session() {
        let mut mgr = SessionManager::new(10);
        let id = mgr.create("phi-4".into(), 4096).unwrap();
        assert_eq!(id, 1);
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_add_turn() {
        let mut mgr = SessionManager::new(10);
        let id = mgr.create("phi-4".into(), 100).unwrap();
        let s = mgr.get_mut(id).unwrap();
        assert!(s.add_turn(Role::User, "Hello".into(), 5));
        assert_eq!(s.turn_count(), 1);
        assert_eq!(s.total_tokens, 5);
    }

    #[test]
    fn test_context_limit() {
        let mut s = Session::new(1, "test".into(), 10);
        assert!(s.add_turn(Role::User, "a".into(), 8));
        assert!(!s.add_turn(Role::User, "b".into(), 5)); // over limit
        assert_eq!(s.turn_count(), 1);
    }

    #[test]
    fn test_remaining_tokens() {
        let mut s = Session::new(1, "test".into(), 100);
        s.add_turn(Role::User, "test".into(), 30);
        assert_eq!(s.remaining_tokens(), 70);
    }

    #[test]
    fn test_complete_session() {
        let mut s = Session::new(1, "test".into(), 100);
        s.complete();
        assert!(!s.is_active());
        assert!(!s.add_turn(Role::User, "hi".into(), 2));
    }

    #[test]
    fn test_trim_to_fit() {
        let mut s = Session::new(1, "test".into(), 20);
        s.add_turn(Role::System, "sys".into(), 5);
        s.add_turn(Role::User, "u1".into(), 8);
        s.add_turn(Role::Assistant, "a1".into(), 5);
        let freed = s.trim_to_fit(10);
        assert!(freed >= 5);
        // System prompt should be preserved
        assert!(s.turns.iter().any(|t| t.role == Role::System));
    }

    #[test]
    fn test_max_sessions() {
        let mut mgr = SessionManager::new(2);
        mgr.create("m1".into(), 100).unwrap();
        mgr.create("m2".into(), 100).unwrap();
        assert!(mgr.create("m3".into(), 100).is_none());
    }

    #[test]
    fn test_remove_session() {
        let mut mgr = SessionManager::new(10);
        let id = mgr.create("test".into(), 100).unwrap();
        assert!(mgr.remove(id));
        assert!(mgr.get(id).is_none());
    }

    #[test]
    fn test_expire_all() {
        let mut mgr = SessionManager::new(10);
        mgr.create("m1".into(), 100);
        mgr.create("m2".into(), 100);
        mgr.expire_all();
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_cleanup_expired() {
        let mut mgr = SessionManager::new(10);
        let id = mgr.create("test".into(), 100).unwrap();
        mgr.get_mut(id).unwrap().expire();
        let cleaned = mgr.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_role_str() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    #[test]
    fn test_default_manager() {
        let mgr = SessionManager::default();
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_multiple_turns() {
        let mut s = Session::new(1, "phi-4".into(), 1000);
        s.add_turn(Role::System, "You are helpful".into(), 10);
        s.add_turn(Role::User, "Hi".into(), 5);
        s.add_turn(Role::Assistant, "Hello!".into(), 8);
        s.add_turn(Role::User, "How are you?".into(), 12);
        assert_eq!(s.turn_count(), 4);
        assert_eq!(s.total_tokens, 35);
    }
}
