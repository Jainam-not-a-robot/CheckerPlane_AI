//! # Output Relevance Gate (Tier 2)
//!
//! **Responsibility:** Evaluates if the LLM-generated response actually answers the user's query.
//! **Pipeline Position:** Tier 2 output gate executed after LLM completion.
//! **Latency Budget:** 80 ms timeout.
//! **Failure Mode:** Fail-open (`FailurePolicy::Open`).

use crate::Gate;
use controlplane_core::config::RelevanceConfig;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use controlplane_inference::ModelBackend;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Tier 2 Relevance guardrail gate.
pub struct RelevanceGate {
    config: RelevanceConfig,
    backend: Arc<dyn ModelBackend>,
    contradiction_idx: usize,
    entailment_idx: usize,
    neutral_idx: usize,
}

impl RelevanceGate {
    /// Constructs a new Relevance gate.
    /// Returns an error if the model does not have the expected NLI classes.
    /// # Errors
    /// Returns a `String` if the backend is incompatible.
    pub fn new(config: RelevanceConfig, backend: Arc<dyn ModelBackend>) -> Result<Self, String> {
        let classes = backend.class_names();
        let mut contradiction_idx = 0;
        let mut entailment_idx = 1;
        let mut neutral_idx = 2;

        if !classes.is_empty() {
            let get_idx = |name: &str| -> Result<usize, String> {
                classes.iter().position(|c| c == name).ok_or_else(|| {
                    format!("Model {} missing required class '{}'", backend.id(), name)
                })
            };

            contradiction_idx = get_idx("contradiction")?;
            entailment_idx = get_idx("entailment")?;
            neutral_idx = get_idx("neutral")?;
        }

        Ok(Self {
            config,
            backend,
            contradiction_idx,
            entailment_idx,
            neutral_idx,
        })
    }
}

#[async_trait::async_trait]
impl Gate for RelevanceGate {
    fn id(&self) -> GateId {
        GateId::Relevance
    }

    fn stage(&self) -> Stage {
        Stage::Output
    }

    fn failure_policy(&self) -> FailurePolicy {
        self.config.failure_policy
    }

    fn timeout(&self) -> Duration {
        Duration::from_millis(self.config.timeout_ms)
    }

    async fn evaluate(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let response = ctx.response.unwrap_or("");

        let premise_raw = ctx.query;
        // WHY: For relevance, the premise is the raw user query.
        // We use sliding window truncation to cap it, ensuring the response (hypothesis) isn't dropped.
        let premise_trunc = self
            .backend
            .sliding_window_truncate(premise_raw, self.config.max_premise_tokens)
            .unwrap_or_else(|_| premise_raw.to_string());

        let premise = &premise_trunc;

        let probs = self
            .backend
            .classify_pair(premise, response)
            .await
            .map_err(|err| GateError::Inference {
                gate: GateId::Relevance,
                source: err,
            })?;

        let p_contra = probs.get(self.contradiction_idx).copied().unwrap_or(0.0);
        let p_entail = probs.get(self.entailment_idx).copied().unwrap_or(1.0);
        let p_neutral = probs.get(self.neutral_idx).copied().unwrap_or(0.0);

        let score = p_entail;

        let verdict = if score < self.config.threshold {
            Verdict::Block {
                reason: BlockReason::IrrelevantResponse {
                    score,
                    threshold: self.config.threshold,
                },
            }
        } else {
            Verdict::Pass
        };

        Ok(GateOutcome {
            gate: GateId::Relevance,
            verdict,
            score,
            threshold: self.config.threshold,
            detail: serde_json::json!({
                "entailment_prob": p_entail,
                "contradiction_prob": p_contra,
                "neutral_prob": p_neutral,
                "premise_length": premise.len(),
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
