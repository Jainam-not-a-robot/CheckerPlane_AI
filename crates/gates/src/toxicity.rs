//! # Toxicity Classifier Gate
//!
//! **Responsibility:** Classifies user query text against abusive and toxic speech classifiers
//! (`minuva/MiniLMv2-toxic-jigsaw-onnx`).
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

/// Label names that denote the *absence* of toxicity. Everything else in a toxicity
/// head is a category of harm we want to block on.
const BENIGN_LABELS: &[&str] = &[
    "clean",
    "neutral",
    "safe",
    "benign",
    "non-toxic",
    "non_toxic",
    "nontoxic",
    "not_toxic",
    "no_toxic",
    "nothate",
];

/// Toxicity guardrail gate.
pub struct ToxicityGate {
    config: ToxicityConfig,
    backend: Arc<dyn ModelBackend>,
    /// Class names in logit order; empty when the backend exposes no `id2label`.
    class_names: Vec<String>,
    /// Indices of the classes that count as toxic.
    toxic_indices: Vec<usize>,
}

impl ToxicityGate {
    /// Constructs a new toxicity gate from configuration and model backend.
    ///
    /// WHY the index resolution: the toxicity slot has held both a binary `clean`/`toxic`
    /// head and the six-way Jigsaw head (`toxic`, `severe_toxic`, `obscene`, `threat`,
    /// `insult`, `identity_hate`). Hard-coding "index 1 is toxic" silently reads
    /// `severe_toxic` off the Jigsaw model — a label that is near-zero even for plainly
    /// abusive text — so the gate stops discriminating without ever erroring. Resolving
    /// against `id2label` makes a model swap a configuration change, not a silent outage.
    #[must_use]
    pub fn new(config: ToxicityConfig, backend: Arc<dyn ModelBackend>) -> Self {
        let class_names = backend.class_names();
        let toxic_indices = Self::resolve_toxic_indices(&class_names);

        tracing::info!(
            gate = "toxicity",
            model = %backend.id(),
            multi_label = backend.is_multi_label(),
            classes = ?class_names,
            toxic_indices = ?toxic_indices,
            "resolved toxicity label mapping"
        );

        Self {
            config,
            backend,
            class_names,
            toxic_indices,
        }
    }

    /// Picks the logit indices that represent toxicity.
    ///
    /// Falls back to index 1 when labels are absent or uninformative (`LABEL_0`/`LABEL_1`),
    /// which preserves the conventional binary `[negative, positive]` ordering.
    fn resolve_toxic_indices(class_names: &[String]) -> Vec<usize> {
        let uninformative = class_names.is_empty()
            || class_names
                .iter()
                .all(|c| c.to_ascii_lowercase().starts_with("label_"));

        if uninformative {
            return vec![1];
        }

        let indices: Vec<usize> = class_names
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                let lowered = name.to_ascii_lowercase();
                !BENIGN_LABELS.contains(&lowered.as_str())
            })
            .map(|(idx, _)| idx)
            .collect();

        if indices.is_empty() {
            vec![1]
        } else {
            indices
        }
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

        // WHY max and not a sum: on a multi-label head the categories overlap heavily
        // (an insult is usually also `toxic`), so summing double-counts one utterance.
        // The strongest single category is the honest severity signal.
        let (top_idx, p_toxic) = self
            .toxic_indices
            .iter()
            .filter_map(|&idx| probs.get(idx).map(|&p| (idx, p)))
            .fold(
                (0_usize, 0.0_f32),
                |acc, item| {
                    if item.1 > acc.1 {
                        item
                    } else {
                        acc
                    }
                },
            );

        let top_label = self
            .class_names
            .get(top_idx)
            .cloned()
            .unwrap_or_else(|| format!("class_{top_idx}"));

        // On a single-label head the complement is meaningful; on a multi-label head the
        // probabilities are independent, so "clean" is simply the absence of any category.
        let p_clean = 1.0 - p_toxic;

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

        let per_label: serde_json::Map<String, serde_json::Value> = self
            .class_names
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                probs.get(idx).and_then(|&p| {
                    serde_json::Number::from_f64(f64::from(p))
                        .map(|n| (name.clone(), serde_json::Value::Number(n)))
                })
            })
            .collect();

        Ok(GateOutcome {
            gate: GateId::Toxicity,
            verdict,
            score: Some(p_toxic),
            threshold: self.config.threshold,
            detail: serde_json::json!({
                "clean_prob": p_clean,
                "toxic_prob": p_toxic,
                "top_label": top_label,
                "multi_label": self.backend.is_multi_label(),
                "per_label": per_label
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
