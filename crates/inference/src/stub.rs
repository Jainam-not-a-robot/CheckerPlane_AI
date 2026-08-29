//! # Deterministic Weightless Stub Backend
//!
//! **Responsibility:** Implements a synthetic, deterministic model backend that enables the pipeline
//! to execute full end-to-end tests and serve traffic without requiring ONNX model weights on disk.
//! **Pipeline Position:** Drop-in fallback when model files are missing or `force_stub = true`.
//! **Latency Budget:** <50 µs (except for intentional `__STUB_SLOW__` testing triggers).
//! **Failure Mode:** Infallible synthetic generator.

use crate::backend::ModelBackend;
use controlplane_core::error::InferenceError;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

/// Deterministic mock inference backend.
#[derive(Debug, Clone)]
pub struct StubBackend {
    id: String,
    num_classes: usize,
}

impl StubBackend {
    /// Constructs a new deterministic stub backend for a given model ID and expected class count.
    #[must_use]
    pub fn new(id: impl Into<String>, num_classes: usize) -> Self {
        Self {
            id: id.into(),
            num_classes,
        }
    }

    /// Computes a stable 64-bit hash from input strings to drive reproducible logits.
    fn compute_hash(&self, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Checks if a slow trigger was requested to test timeouts.
    async fn check_slow_trigger(&self, text: &str) {
        if text.contains("__STUB_SLOW__") {
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for StubBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_live(&self) -> bool {
        false
    }

    fn class_names(&self) -> Vec<String> {
        match self.id.as_str() {
            "cross_encoder" => vec![
                "contradiction".to_string(),
                "entailment".to_string(),
                "neutral".to_string(),
            ],
            _ => vec![],
        }
    }

    fn sliding_window_truncate(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<String, InferenceError> {
        // Simple approximate word-based truncation for the stub
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() <= max_tokens {
            Ok(text.to_string())
        } else {
            Ok(words[words.len() - max_tokens..].join(" "))
        }
    }

    async fn classify(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        self.check_slow_trigger(text).await;

        // Check for model-specific or global blocking triggers
        let is_block_trigger = text.contains("__STUB_BLOCK__")
            || (self.id == "coherence" && text.contains("__STUB_BLOCK_COHERENCE__"))
            || (self.id == "toxicity" && text.contains("__STUB_BLOCK_TOXICITY__"))
            || (self.id == "intent" && text.contains("__STUB_BLOCK_INTENT__"));

        if is_block_trigger {
            match self.id.as_str() {
                "coherence" => {
                    // Classes: [clean, mild_gibberish, noise, word_salad]
                    // Return high probability for word salad
                    return Ok(vec![0.02, 0.05, 0.03, 0.90]);
                }
                "toxicity" => {
                    // Classes: [clean, toxic]
                    return Ok(vec![0.05, 0.95]);
                }
                "intent" => {
                    // Classes: [benign, attack]
                    return Ok(vec![0.02, 0.98]);
                }
                _ => {
                    // Generic multi-class block: put dominant mass on highest risk class
                    let mut logits = vec![0.05; self.num_classes.max(2)];
                    if let Some(last) = logits.last_mut() {
                        *last = 0.95;
                    }
                    return Ok(logits);
                }
            }
        }

        // Deterministic pseudo-random generation based on input hash
        let hash = self.compute_hash(text);
        let n = self.num_classes.max(2);
        let mut raw = Vec::with_capacity(n);

        for i in 0..n {
            let val = ((hash.wrapping_add(i as u64 * 31)) % 100) as f32 / 100.0;
            raw.push(val);
        }

        // Bias class 0 (benign / clean) so default text passes
        raw[0] += 5.0;

        // Apply softmax normalization
        let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = raw.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();

        Ok(probs)
    }

    async fn classify_pair(&self, a: &str, b: &str) -> Result<Vec<f32>, InferenceError> {
        let combined = format!("{a} [SEP] {b}");
        self.check_slow_trigger(&combined).await;

        let is_block_trigger =
            combined.contains("__STUB_BLOCK__") || combined.contains("__STUB_BLOCK_GROUNDING__");

        if is_block_trigger {
            match self.id.as_str() {
                "cross_encoder" => {
                    // Classes: [contradiction, entailment, neutral]
                    // Return high contradiction for block
                    return Ok(vec![0.92, 0.03, 0.05]);
                }
                _ => return Ok(vec![0.90, 0.10]),
            }
        }

        // Default: high entailment / grounded
        match self.id.as_str() {
            "cross_encoder" => {
                // Classes: [contradiction, entailment, neutral] -> high entailment
                Ok(vec![0.05, 0.90, 0.05])
            }
            _ => {
                let hash = self.compute_hash(&combined);
                let n = self.num_classes.max(2);
                let mut raw = vec![0.1; n];
                raw[1] = 0.85 + ((hash % 10) as f32 / 100.0);
                Ok(raw)
            }
        }
    }
}
