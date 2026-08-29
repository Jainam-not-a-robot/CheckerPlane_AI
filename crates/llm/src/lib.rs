//! # LLM Generation Backends and Client Orchestration
//!
//! **Responsibility:** Provides the abstraction and implementations for text generation using
//! external Large Language Models (e.g. Google Gemini 2.0 Flash) or an offline reproducible mock backend.
//! **Pipeline Position:** Middle stage executed only when user query survives all Tier 1 input guardrails.
//! **Latency Budget:** Gemini: 500–1500 ms; Mock: 700 ± 150 ms.
//! **Failure Mode:** Propagates `LlmError`.

#![deny(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]

pub mod gemini;
pub mod history;
pub mod mock;

use controlplane_core::error::LlmError;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub use gemini::GeminiClient;
pub use history::ConversationHistory;
pub use mock::MockLlm;

/// Model generation response payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Generated output text.
    pub text: String,
    /// Estimated or reported prompt tokens consumed.
    pub prompt_tokens: usize,
    /// Completion tokens generated.
    pub completion_tokens: usize,
    /// Generation wall-clock duration.
    pub latency: Duration,
}

/// Uniform asynchronous interface for Large Language Model generation.
#[async_trait::async_trait]
pub trait LlmBackend: Send + Sync {
    /// Generates a response to the given user query with optional compressed conversation history.
    ///
    /// # Errors
    /// Returns `LlmError` if generation fails, times out, or provider returns an error.
    async fn generate(&self, query: &str, history: Option<&str>) -> Result<LlmResponse, LlmError>;
}
