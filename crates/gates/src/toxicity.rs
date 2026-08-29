//! # Toxicity Classifier Gate
//!
//! **Responsibility:** Classifies user query text against abusive and toxic speech classifiers (`martin-ha/toxic-comment-model`).
//! **Pipeline Position:** Tier 1 input gate evaluated concurrently with Coherence, PII, and Intent gates.
//! **Latency Budget:** 40 ms timeout.
//! **Failure Mode:** Fail-closed (`FailurePolicy::Closed`).

use crate::Gate;
use controlplane_core::config::ToxicityConfig;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use controlplane_inference::ModelBackend;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Toxicity guardrail gate.
pub struct ToxicityGate {
    config: ToxicityConfig,
    backend: Arc<dyn ModelBackend>,
}

impl ToxicityGate {
    /// Constructs a new toxicity gate from configuration and model backend.
    #[must_use]
    pub fn new(config: ToxicityConfig, backend: Arc<dyn ModelBackend>) -> Self {
        Self { config, backend }
    }
}

#[async_trait::async_trait]
impl Gate for ToxicityGate {
    fn id(&self) -> GateId {
        GateId::Toxicity
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

        let probs = self
            .backend
            .classify(query)
            .await
            .map_err(|err| GateError::Inference {
                gate: GateId::Toxicity,
                source: err,
            })?;

        // 2 classes: [clean, toxic]
        let p_clean = probs.first().copied().unwrap_or(1.0);
        let p_toxic = probs.get(1).copied().unwrap_or(0.0);

        let verdict = if p_toxic >= self.config.threshold {
            Verdict::Block {
                reason: BlockReason::ToxicContent {
                    score: p_toxic,
                    threshold: self.config.threshold,
                },
            }
        } else {
            Verdict::Pass
        };

        Ok(GateOutcome {
            gate: GateId::Toxicity,
            verdict,
            score: Some(p_toxic),
            threshold: self.config.threshold,
            detail: serde_json::json!({
                "clean_prob": p_clean,
                "toxic_prob": p_toxic
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
