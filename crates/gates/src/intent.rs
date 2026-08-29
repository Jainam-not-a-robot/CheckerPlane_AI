//! # Intent and Prompt Injection Classifier Gate
//!
//! **Responsibility:** Classifies user query text against prompt injection and jailbreak attacks
//! using `meta-llama/Llama-Prompt-Guard-2-22M`.
//! **Pipeline Position:** Tier 1 input gate evaluated concurrently with Coherence, PII, and Toxicity.
//! **Latency Budget:** 40 ms timeout.
//! **Failure Mode:** Fail-closed (`FailurePolicy::Closed`).

use crate::Gate;
use controlplane_core::config::IntentConfig;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use controlplane_inference::ModelBackend;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Intent and jailbreak detector gate.
pub struct IntentGate {
    config: IntentConfig,
    backend: Arc<dyn ModelBackend>,
}

impl IntentGate {
    /// Constructs a new intent gate from configuration and model backend.
    #[must_use]
    pub fn new(config: IntentConfig, backend: Arc<dyn ModelBackend>) -> Self {
        Self { config, backend }
    }
}

#[async_trait::async_trait]
impl Gate for IntentGate {
    fn id(&self) -> GateId {
        GateId::Intent
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
                gate: GateId::Intent,
                source: err,
            })?;

        // 2 classes: [LABEL_0 (benign), LABEL_1 (attack)]
        let p_benign = probs.first().copied().unwrap_or(1.0);
        let p_attack = probs.get(1).copied().unwrap_or(0.0);

        let verdict = if p_attack >= self.config.threshold {
            Verdict::Block {
                reason: BlockReason::PromptAttack {
                    score: p_attack,
                    threshold: self.config.threshold,
                },
            }
        } else {
            Verdict::Pass
        };

        Ok(GateOutcome {
            gate: GateId::Intent,
            verdict,
            score: p_attack,
            threshold: self.config.threshold,
            detail: serde_json::json!({
                "benign_prob": p_benign,
                "attack_prob": p_attack,
                // Deliberately high default threshold (0.90) to prevent visible false-positive rejections on normal requests
                "threshold_mode": "conservative"
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
