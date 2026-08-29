//! # Personally Identifiable Information (PII) Guardrail Gate
//!
//! **Responsibility:** Detects high-risk structured personal identifiers (Payment Cards, Aadhaar, PAN, Credentials)
//! to block hazardous exfiltration, while tracking entity observations (Names, Locations, Emails, Phones)
//! for informational logging without blocking.
//! **Pipeline Position:** Tier 1 input gate running concurrently with Coherence, Toxicity, and Intent.
//! **Latency Budget:** 50 ms timeout.
//! **Failure Mode:** Fail-closed (`FailurePolicy::Closed`).

pub mod checksum;
pub mod patterns;

use crate::Gate;
use controlplane_core::config::PiiConfig;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use patterns::{scan_high_risk_patterns, scan_observation_patterns};
use std::collections::HashSet;

use std::time::{Duration, Instant};

/// PII guardrail gate.
pub struct PiiGate {
    config: PiiConfig,
}

impl PiiGate {
    /// Constructs a new PII gate with configuration.
    #[must_use]
    pub fn new(config: PiiConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Gate for PiiGate {
    fn id(&self) -> GateId {
        GateId::Pii
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

        // 1. Fast deterministic regex scan for high-risk structured identifiers
        let high_risk_findings = scan_high_risk_patterns(query);

        // 2. Scan observation regex patterns (emails, phones)
        let observation_findings = scan_observation_patterns(query);

        // WHY: The pii gate blocks only on high-risk classes. Blocking a user for typing their own
        // email address is hostile. Structured identifiers (payment cards, Aadhaar, PAN, credential-shaped
        // strings) are matched with deterministic regex including checksum validation where one exists
        // (Luhn for cards, Verhoeff for Aadhaar). The NER model contributes names/addresses as
        // observations only — recorded in the response detail, never a block reason in v1.
        let block_set: HashSet<&str> = self
            .config
            .block_classes
            .iter()
            .map(String::as_str)
            .collect();

        let mut matched_block_classes = Vec::new();
        let mut block_details = Vec::new();

        for finding in &high_risk_findings {
            if block_set.contains(finding.class_name) {
                if !matched_block_classes.contains(&finding.class_name.to_string()) {
                    matched_block_classes.push(finding.class_name.to_string());
                }
                block_details.push(serde_json::json!({
                    "class": finding.class_name,
                    "masked": finding.matched_value,
                    "offset": [finding.start, finding.end]
                }));
            }
        }

        let mut observed_details = Vec::new();
        for obs in &observation_findings {
            observed_details.push(serde_json::json!({
                "class": obs.class_name,
                "masked": obs.matched_value
            }));
        }

        let is_blocked = !matched_block_classes.is_empty();
        let score = if is_blocked { 1.0 } else { 0.0 };

        let verdict = if is_blocked {
            Verdict::Block {
                reason: BlockReason::PiiDetected {
                    matched_classes: matched_block_classes,
                },
            }
        } else {
            Verdict::Pass
        };

        Ok(GateOutcome {
            gate: GateId::Pii,
            verdict,
            score: Some(score),
            threshold: 0.5,
            detail: serde_json::json!({
                "blocked_matches": block_details,
                "observations": observed_details,
                // V1: PII NER observations are recorded for telemetry and audit, never enforced as blocks.
                "v1_scope": "high_risk_blocking_only"
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
