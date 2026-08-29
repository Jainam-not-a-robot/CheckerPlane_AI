//! # Coherence and Word-Salad Classifier Gate
//!
//! **Responsibility:** Classifies user queries against a 4-class gibberish detector (`clean`, `mild gibberish`,
//! `noise`, `word salad`) to discard unanswerable queries before invoking expensive LLM calls.
//! **Pipeline Position:** Tier 1 input gate running concurrently with PII, Toxicity, and Intent gates.
//! **Latency Budget:** 40 ms timeout.
//! **Failure Mode:** Fail-open (`FailurePolicy::Open`).
//!
//! ### Known False-Positive Risk on Terse Keyword Queries
//! Terse queries such as `"best rust orm postgres"`, `"jodhpur weather tomorrow"`, or
//! `"docker compose healthcheck syntax"` structurally resemble word salad to general-purpose NLP
//! models. To mitigate false rejections on valid technical searches:
//! 1. The gate is tuned via a single `strictness` parameter (`threshold = 1.0 - strictness`).
//! 2. It is configured to fail open on degradation or ambiguity.
//! 3. Long queries (> `max_tokens_for_model`) bypass model evaluation entirely.

use crate::Gate;
use controlplane_core::config::CoherenceConfig;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use controlplane_inference::ModelBackend;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Coherence guardrail gate.
pub struct CoherenceGate {
    config: CoherenceConfig,
    backend: Arc<dyn ModelBackend>,
}

impl CoherenceGate {
    /// Constructs a new coherence gate from configuration and model backend.
    #[must_use]
    pub fn new(config: CoherenceConfig, backend: Arc<dyn ModelBackend>) -> Self {
        Self { config, backend }
    }

    /// Fast heuristic estimation of token count using whitespace word splitting.
    fn estimate_token_count(text: &str) -> usize {
        text.split_whitespace().count()
    }
}

#[async_trait::async_trait]
impl Gate for CoherenceGate {
    fn id(&self) -> GateId {
        GateId::Coherence
    }

    fn stage(&self) -> Stage {
        Stage::Input
    }

    fn failure_policy(&self) -> FailurePolicy {
        self.config.failure_policy
    }

    fn timeout(&self) -> Duration {
        Duration::from_millis(self.config.timeout_ms)
    }

    async fn evaluate(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let query = ctx.query;
        let token_count = Self::estimate_token_count(query);

        // WHY: The gibberish classifier is configured for short sequences (64 tokens).
        // On a long query it would judge only the opening fragment, and long queries are almost never
        // word salad. Above max_tokens_for_model, pass automatically without a forward pass.
        if token_count > self.config.max_tokens_for_model {
            return Ok(GateOutcome {
                gate: GateId::Coherence,
                verdict: Verdict::Pass,
                score: 0.0,
                threshold: 1.0 - self.config.strictness,
                detail: serde_json::json!({
                    "bypassed": true,
                    "reason": "query_exceeds_max_tokens_for_model",
                    "estimated_tokens": token_count
                }),
                latency: start.elapsed(),
                degraded: false,
            });
        }

        // Bypassed if query has fewer than minimum tokens
        if token_count < self.config.min_tokens_for_model {
            return Ok(GateOutcome {
                gate: GateId::Coherence,
                verdict: Verdict::Pass,
                score: 0.0,
                threshold: 1.0 - self.config.strictness,
                detail: serde_json::json!({
                    "bypassed": true,
                    "reason": "query_below_min_tokens_for_model",
                    "estimated_tokens": token_count
                }),
                latency: start.elapsed(),
                degraded: false,
            });
        }

        let probs = self
            .backend
            .classify(query)
            .await
            .map_err(|err| GateError::Inference {
                gate: GateId::Coherence,
                source: err,
            })?;

        // 4 classes: [clean, mild_gibberish, noise, word_salad]
        let p_clean = probs.first().copied().unwrap_or(1.0);
        let p_mild = probs.get(1).copied().unwrap_or(0.0);
        let p_noise = probs.get(2).copied().unwrap_or(0.0);
        let p_salad = probs.get(3).copied().unwrap_or(0.0);

        // Aggregate risk score
        let risk_score = (self.config.weight_noise * p_noise)
            + (self.config.weight_word_salad * p_salad)
            + (self.config.weight_mild_gibberish * p_mild);

        // Effective threshold mapped from single strictness dial: 0.0 -> 1.0 threshold, 1.0 -> 0.0 threshold
        let effective_threshold = (1.0 - self.config.strictness).clamp(0.01, 0.99);

        let verdict = if risk_score >= effective_threshold {
            Verdict::Block {
                reason: BlockReason::Incoherent {
                    score: risk_score,
                    threshold: effective_threshold,
                },
            }
        } else {
            Verdict::Pass
        };

        Ok(GateOutcome {
            gate: GateId::Coherence,
            verdict,
            score: risk_score,
            threshold: effective_threshold,
            detail: serde_json::json!({
                "clean_prob": p_clean,
                "mild_gibberish_prob": p_mild,
                "noise_prob": p_noise,
                "word_salad_prob": p_salad,
                "strictness": self.config.strictness
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
