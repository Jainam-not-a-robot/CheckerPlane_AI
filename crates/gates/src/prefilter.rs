//! # Tier 0 Heuristic Prefilter Gate
//!
//! **Responsibility:** Performs instant synchronous heuristic validation on raw user queries,
//! catching empty inputs, oversized payloads, wrong-script binary garbage, and keyboard mashing noise.
//! **Pipeline Position:** Tier 0 gate executed before parallel Tier 1 fan-out.
//! **Latency Budget:** <100 µs (pure synchronous string analysis).
//! **Failure Mode:** Fail-open on unexpected internal errors.

use crate::Gate;
use controlplane_core::config::PrefilterConfig;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Tier 0 heuristic input prefilter.
pub struct PrefilterGate {
    config: PrefilterConfig,
}

impl PrefilterGate {
    /// Constructs a new prefilter gate from configuration.
    #[must_use]
    pub const fn new(config: PrefilterConfig) -> Self {
        Self { config }
    }

    /// Computes the Shannon entropy in bits per character of the input string.
    #[must_use]
    pub fn calculate_entropy(text: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        let mut counts = HashMap::new();
        let mut total_chars = 0usize;

        for ch in text.chars() {
            *counts.entry(ch).or_insert(0usize) += 1;
            total_chars += 1;
        }

        let total = total_chars as f32;
        let mut entropy = 0.0f32;

        for &count in counts.values() {
            let p = count as f32 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Computes the ratio of expected ASCII/Latin characters to total characters.
    #[must_use]
    pub fn calculate_script_ratio(text: &str) -> f32 {
        if text.is_empty() {
            return 1.0;
        }

        let mut expected = 0usize;
        let mut total = 0usize;

        for ch in text.chars() {
            total += 1;
            if ch.is_ascii()
                || ch.is_alphanumeric()
                || ch.is_whitespace()
                || ch.is_ascii_punctuation()
            {
                expected += 1;
            }
        }

        expected as f32 / total as f32
    }
}

#[async_trait::async_trait]
impl Gate for PrefilterGate {
    fn id(&self) -> GateId {
        GateId::Prefilter
    }

    fn stage(&self) -> Stage {
        Stage::Input
    }

    fn failure_policy(&self) -> FailurePolicy {
        FailurePolicy::Open
    }

    fn timeout(&self) -> Duration {
        Duration::from_millis(10)
    }

    async fn evaluate(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let query = ctx.query;
        let trimmed = query.trim();

        // 1. Length validation (min_chars)
        if trimmed.len() < self.config.min_chars {
            return Ok(GateOutcome {
                gate: GateId::Prefilter,
                verdict: Verdict::Block {
                    reason: BlockReason::PrefilterNoise {
                        message: format!(
                            "query too short (length={}, min_chars={})",
                            trimmed.len(),
                            self.config.min_chars
                        ),
                    },
                },
                score: 1.0,
                threshold: 0.0,
                detail: serde_json::json!({ "reason": "too_short", "len": trimmed.len() }),
                latency: start.elapsed(),
                degraded: false,
            });
        }

        // 2. Length validation (max_chars)
        if query.len() > self.config.max_chars {
            return Ok(GateOutcome {
                gate: GateId::Prefilter,
                verdict: Verdict::Block {
                    reason: BlockReason::PrefilterNoise {
                        message: format!(
                            "query exceeds maximum size (length={}, max_chars={})",
                            query.len(),
                            self.config.max_chars
                        ),
                    },
                },
                score: 1.0,
                threshold: 0.0,
                detail: serde_json::json!({ "reason": "too_long", "len": query.len() }),
                latency: start.elapsed(),
                degraded: false,
            });
        }

        // 3. Script validation
        let script_ratio = Self::calculate_script_ratio(query);
        if script_ratio < self.config.min_script_ratio {
            return Ok(GateOutcome {
                gate: GateId::Prefilter,
                verdict: Verdict::Block {
                    reason: BlockReason::PrefilterNoise {
                        message: format!(
                            "unexpected character script (ratio={script_ratio:.2}, min={})",
                            self.config.min_script_ratio
                        ),
                    },
                },
                score: 1.0 - script_ratio,
                threshold: 1.0 - self.config.min_script_ratio,
                detail: serde_json::json!({ "reason": "invalid_script", "script_ratio": script_ratio }),
                latency: start.elapsed(),
                degraded: false,
            });
        }

        // 4. Shannon character entropy (keyboard mashing / random noise)
        let entropy = Self::calculate_entropy(query);
        // Only evaluate entropy if query is sufficiently long to avoid false positives on short words
        if trimmed.len() >= 16 && entropy > self.config.max_char_entropy {
            return Ok(GateOutcome {
                gate: GateId::Prefilter,
                verdict: Verdict::Block {
                    reason: BlockReason::PrefilterNoise {
                        message: format!(
                            "high character entropy / keyboard mashing (entropy={entropy:.2}, max={})",
                            self.config.max_char_entropy
                        ),
                    },
                },
                score: entropy / 8.0,
                threshold: self.config.max_char_entropy / 8.0,
                detail: serde_json::json!({ "reason": "high_entropy", "entropy": entropy }),
                latency: start.elapsed(),
                degraded: false,
            });
        }

        Ok(GateOutcome {
            gate: GateId::Prefilter,
            verdict: Verdict::Pass,
            score: 0.0,
            threshold: 0.0,
            detail: serde_json::json!({
                "len": query.len(),
                "script_ratio": script_ratio,
                "entropy": entropy
            }),
            latency: start.elapsed(),
            degraded: false,
        })
    }
}
