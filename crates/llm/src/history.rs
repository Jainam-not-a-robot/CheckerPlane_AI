//! # Conversation History Formatting
//!
//! **Responsibility:** Formats and compresses prior conversation turns into concise string summaries
//! for premise grounding and LLM prompt context injection.
//! **Pipeline Position:** Prepared before LLM generation and Tier 2 grounding.
//! **Latency Budget:** <50 µs.
//! **Failure Mode:** Infallible string formatter.

use serde::{Deserialize, Serialize};

/// Individual turn in a multi-turn conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Role identifier ("user" or "assistant").
    pub role: String,
    /// Content of the message turn.
    pub content: String,
}

/// Helper container for managing conversation history summaries.
#[derive(Debug, Clone, Default)]
pub struct ConversationHistory {
    turns: Vec<ConversationTurn>,
}

impl ConversationHistory {
    /// Constructs a new empty conversation history container.
    #[must_use]
    pub const fn new() -> Self {
        Self { turns: Vec::new() }
    }

    /// Adds a conversation turn to the history.
    pub fn push(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.turns.push(ConversationTurn {
            role: role.into(),
            content: content.into(),
        });
    }

    /// Formats the conversation history into a structured premise string.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let mut formatted = String::new();
        for turn in &self.turns {
            formatted.push_str(&format!("{}: {}\n", turn.role, turn.content));
        }
        formatted
    }
}
