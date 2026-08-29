//! # Output Grounding and Hallucination Gate (Tier 2)
//!
//! **Responsibility:** Evaluates LLM-generated candidate responses against conversation history
//! and source context to detect hallucinations and ungrounded statements before returning to the user.
//! **Pipeline Position:** Tier 2 output gate executed after LLM completion.
//! **Latency Budget:** 150 ms timeout.
//! **Failure Mode:** Fail-closed (`FailurePolicy::Closed`).

use crate::Gate;
use controlplane_core::config::GroundingConfig;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use controlplane_inference::ModelBackend;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Tier 2 Grounding guardrail gate.
pub struct GroundingGate {
    config: GroundingConfig,
    backend: Arc<dyn ModelBackend>,
    contradiction_idx: usize,
    entailment_idx: usize,
    neutral_idx: usize,
}

impl GroundingGate {
    /// Constructs a new grounding gate from configuration and a `cross_encoder` backend.
    ///
    /// # Errors
    /// Returns a `String` if the backend is incompatible.
    pub fn new(config: GroundingConfig, backend: Arc<dyn ModelBackend>) -> Result<Self, String> {
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
impl Gate for GroundingGate {
    fn id(&self) -> GateId {
        GateId::Grounding
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
        let query = ctx.query;

        // If history summary is available, use it as premise; otherwise fall back to user query
        let premise_raw = ctx.request.history_summary.as_deref().unwrap_or(query);

        // WHY: We need sliding-window truncation on the premise to avoid eating the response (hypothesis)
        // when concatenating. By capping the premise on the way in, we keep the most recent context
        // and ensure the response stays intact.
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
                gate: GateId::Grounding,
                source: err,
            })?;

        let p_contra = probs.get(self.contradiction_idx).copied().unwrap_or(0.0);
        let p_entail = probs.get(self.entailment_idx).copied().unwrap_or(1.0);
        let p_neutral = probs.get(self.neutral_idx).copied().unwrap_or(0.0);

        let contradiction_risk = p_contra;
        let neutral_risk = p_neutral;

        let score = (contradiction_risk * self.config.weight_contradiction)
            + (neutral_risk * self.config.weight_neutral);

        let verdict = if score >= self.config.threshold {
            Verdict::Block {
                reason: BlockReason::UngroundedResponse {
                    score,
                    threshold: self.config.threshold,
                },
            }
        } else {
            Verdict::Pass
        };

        Ok(GateOutcome {
            gate: GateId::Grounding,
            verdict,
            score,
            threshold: self.config.threshold,
            detail: serde_json::json!({
                "contradiction_prob": p_contra,
                "entailment_prob": p_entail,
                "neutral_prob": p_neutral,
                "premise_length": premise.len(),
                "response_length": response.len(),
                "nli_folded_into_grounding": true
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
