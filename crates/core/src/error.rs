//! # Error Taxonomy
//!
//! **Responsibility:** Enumerates structured, strongly typed error variants across inference,
//! gate evaluation, LLM calls, pipeline orchestration, and configuration loading.
//! **Pipeline Position:** Propagated across all layers to ensure graceful degradation and error visibility.
//! **Latency Budget:** Zero overhead.
//! **Failure Mode:** Infallible type definitions using `thiserror`.

use crate::verdict::GateId;
use thiserror::Error;

/// Errors arising during inference execution or model management.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// Model with the given identifier was not found in the registry.
    #[error("model '{0}' not found in registry")]
    ModelNotFound(String),

    /// Model files (weights or tokenizer) are missing from disk.
    #[error("model files missing for '{model}': {reason}")]
    FilesMissing { model: String, reason: String },

    /// Tokenizer failed to encode input text.
    #[error("tokenization error for model '{model}': {message}")]
    TokenizerError { model: String, message: String },

    /// ONNX Runtime engine execution error.
    #[error("ONNX Runtime error for model '{model}': {message}")]
    OnnxError { model: String, message: String },

    /// Inference session pool was exhausted or failed to provide a session.
    #[error("session pool error for model '{model}': {message}")]
    PoolExhausted { model: String, message: String },

    /// Model output tensor had an unexpected shape or missing dimension.
    #[error("shape mismatch for model '{model}': expected {expected}, found {actual}")]
    ShapeMismatch {
        model: String,
        expected: String,
        actual: String,
    },

    /// Background task failed during blocking inference execution.
    #[error("blocking task join error: {0}")]
    JoinError(String),
}

/// Errors arising during gate evaluation.
#[derive(Debug, Error)]
pub enum GateError {
    /// Gate timed out before completing evaluation.
    #[error("gate '{0}' timed out")]
    Timeout(GateId),

    /// Underlying inference backend error during gate evaluation.
    #[error("inference failure in gate '{gate}': {source}")]
    Inference {
        gate: GateId,
        #[source]
        source: InferenceError,
    },

    /// Gate encountered an internal evaluation failure.
    #[error("internal evaluation failure in gate '{gate}': {message}")]
    Internal { gate: GateId, message: String },

    /// Gate was invoked while disabled in configuration.
    #[error("gate '{0}' is disabled")]
    Disabled(GateId),
}

/// Errors arising during LLM generation.
#[derive(Debug, Error)]
pub enum LlmError {
    /// HTTP communication failure with external LLM provider.
    #[error("HTTP error communicating with LLM provider: {0}")]
    Http(String),

    /// LLM provider returned an API error status.
    #[error("LLM provider returned status {status}: {message}")]
    ProviderError { status: u16, message: String },

    /// LLM generation call exceeded configured timeout.
    #[error("LLM generation timed out after {0}ms")]
    Timeout(u64),

    /// Required API authentication key is missing.
    #[error("LLM API key not configured for provider '{0}'")]
    MissingApiKey(String),

    /// LLM provider response payload was invalid or unparseable.
    #[error("failed to parse LLM response: {0}")]
    InvalidResponse(String),
}

/// Errors arising during configuration loading and validation.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Configuration syntax error or schema mismatch.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Required configuration file could not be read.
    #[error("failed to read configuration file '{path}': {message}")]
    FileReadError { path: String, message: String },
}

/// Errors arising during overall pipeline orchestration.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Request processing exceeded the overall input or output gate wall-clock budget.
    #[error("stage '{stage}' wall-clock budget exceeded ({budget_ms}ms)")]
    BudgetExceeded { stage: &'static str, budget_ms: u64 },

    /// Unrecoverable internal error in the pipeline.
    #[error("pipeline internal error: {0}")]
    Internal(String),

    /// LLM generation failure in the pipeline.
    #[error("pipeline LLM failure: {0}")]
    Llm(#[from] LlmError),
}
