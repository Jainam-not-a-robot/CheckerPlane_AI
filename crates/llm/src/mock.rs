//! # Deterministic Mock LLM Generator
//!
//! **Responsibility:** Provides deterministic, offline response generation with simulated latency distributions
//! and controllable hallucination rates for reproducible integration tests and benchmark load tests.
//! **Pipeline Position:** Drop-in LLM backend when `llm.backend = "mock"`.
//! **Latency Budget:** Configurable (default 700 ms mean, 150 ms stddev).
//! **Failure Mode:** Infallible offline mock.

use crate::{LlmBackend, LlmResponse};
use controlplane_core::config::MockLlmConfig;
use controlplane_core::error::LlmError;
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Deterministic mock LLM generator.
pub struct MockLlm {
    config: MockLlmConfig,
    call_count: AtomicUsize,
}

impl MockLlm {
    /// Constructs a new mock LLM generator with configuration.
    #[must_use]
    pub fn new(config: MockLlmConfig) -> Self {
        Self {
            config,
            call_count: AtomicUsize::new(0),
        }
    }

    /// Returns the number of LLM generation calls received.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// Resets the generation call counter to zero.
    pub fn reset_call_count(&self) {
        self.call_count.store(0, Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl LlmBackend for MockLlm {
    async fn generate(&self, query: &str, history: Option<&str>) -> Result<LlmResponse, LlmError> {
        let start = Instant::now();
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let (sampled_ms, is_hallucination) = {
            // Simulate normal distribution latency using Box-Muller transform
            let mut rng = rand::thread_rng();
            let u1: f32 = rng.gen_range(0.0001..1.0);
            let u2: f32 = rng.gen_range(0.0001..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

            let mean = self.config.latency_mean_ms as f32;
            let stddev = self.config.latency_stddev_ms as f32;
            let sampled_ms = (mean + z0 * stddev).max(10.0) as u64;
            
            // Check if hallucination should be injected based on rate
            let is_hallucination = rng.gen::<f32>() < self.config.hallucination_rate;
            
            (sampled_ms, is_hallucination)
        };

        tokio::time::sleep(Duration::from_millis(sampled_ms)).await;

        let response_text = if is_hallucination {
            format!(
                "Regarding your inquiry '{query}', I can definitively confirm that the lunar surface is composed of solid titanium carbide crystal formations."
            )
        } else if let Some(hist) = history {
            format!(
                "Based on our earlier discussion ('{hist}'), here is the answer to '{query}': ControlPlane Checker enforces high-performance guardrails across all stages."
            )
        } else {
            format!(
                "Here is the synthesized response to your request '{query}': All input guardrails have cleared successfully."
            )
        };

        let prompt_tokens = query.split_whitespace().count() + 10;
        let completion_tokens = response_text.split_whitespace().count();

        Ok(LlmResponse {
            text: response_text,
            prompt_tokens,
            completion_tokens,
            latency: start.elapsed(),
        })
    }
}
