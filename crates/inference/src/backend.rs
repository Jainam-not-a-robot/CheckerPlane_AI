//! # Model Backend Trait
//!
//! **Responsibility:** Defines the uniform asynchronous interface implemented by both live ONNX
//! sessions and weightless synthetic stub backends.
//! **Pipeline Position:** Core inference abstraction consumed by all gate implementations.
//! **Latency Budget:** Forward-pass dependent (5–30 ms ONNX, <50 µs Stub).
//! **Failure Mode:** Returns `Result<_, InferenceError>`.

use controlplane_core::error::InferenceError;



/// Uniform interface over a real ONNX session and the weightless stub.
/// Every gate depends on this trait, never on `ort` directly.
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync {
    /// Unique identifier for this model.
    fn id(&self) -> &str;

    /// True when backed by real weights; false for the stub.
    fn is_live(&self) -> bool;

    /// Returns the class names in the order they correspond to logits.
    fn class_names(&self) -> Vec<String>;

    /// Truncates a text to the last `max_tokens` tokens (sliding window).
    ///
    /// # Errors
    /// Returns `InferenceError` if tokenization or decoding fails.
    fn sliding_window_truncate(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<String, InferenceError>;

    /// Returns per-class logits for sequence classification.
    ///
    /// # Errors
    /// Returns `InferenceError` if tokenization or model forward pass fails.
    async fn classify(&self, text: &str) -> Result<Vec<f32>, InferenceError>;

    /// Returns per-class logits for a (premise, hypothesis) cross-encoder pair.
    ///
    /// # Errors
    /// Returns `InferenceError` if tokenization or model forward pass fails.
    async fn classify_pair(&self, a: &str, b: &str) -> Result<Vec<f32>, InferenceError>;
}
