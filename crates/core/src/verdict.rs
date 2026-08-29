//! # Verdicts and Evaluation Outcomes
//!
//! **Responsibility:** Defines evaluation outcomes, verdict types, block reasons, and gate identifiers.
//! **Pipeline Position:** Core contract consumed by gate evaluators, the fan-out executor, and the router.
//! **Latency Budget:** Zero overhead.
//! **Failure Mode:** Infallible type definitions.

use serde::{Deserialize, Serialize, Serializer};
use std::fmt;
use std::time::Duration;

/// Identifier for individual gates in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateId {
    /// Tier 0 heuristic prefilter.
    Prefilter,
    /// Tier 1 coherence sequence classifier.
    Coherence,
    /// Tier 1 PII detector (regex + NER).
    Pii,
    /// Tier 1 toxicity sequence classifier.
    Toxicity,
    /// Tier 1 prompt injection and jailbreak detector.
    Intent,
    /// Tier 2 grounding and hallucination evaluator.
    Grounding,
    /// Tier 2 output relevance evaluator.
    Relevance,
}

impl GateId {
    /// Returns the static string representation of this gate ID.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefilter => "prefilter",
            Self::Coherence => "coherence",
            Self::Pii => "pii",
            Self::Toxicity => "toxicity",
            Self::Intent => "intent",
            Self::Grounding => "grounding",
            Self::Relevance => "relevance",
        }
    }
}

impl fmt::Display for GateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Pipeline processing stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Executed on user query before routing to the LLM.
    Input,
    /// Executed on model response before returning to user.
    Output,
}

/// Failure policy for gate timeout or execution error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Allow request to proceed on gate error or timeout.
    Open,
    /// Block request on gate error or timeout.
    Closed,
}

/// Operator-facing reason for blocking a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockReason {
    /// Input failed heuristic prefilter.
    PrefilterNoise { message: String },
    /// Input classified as incoherent / word salad.
    Incoherent { score: f32, threshold: f32 },
    /// High-risk personal identifiable information detected.
    PiiDetected { matched_classes: Vec<String> },
    /// Toxic or abusive language detected.
    ToxicContent { score: f32, threshold: f32 },
    /// Prompt injection or jailbreak attempt detected.
    PromptAttack { score: f32, threshold: f32 },
    /// Output hallucination or ungrounded response detected.
    UngroundedResponse { score: f32, threshold: f32 },
    /// Output irrelevant to the user query.
    IrrelevantResponse { score: f32, threshold: f32 },
    /// Gate timed out under fail-closed policy.
    Timeout { gate: GateId, timeout_ms: u64 },
    /// Gate encountered an unrecoverable error under fail-closed policy.
    GateError { gate: GateId, message: String },
    /// Forward-compatible Clarify verdict mapped to Block in v1.
    ClarificationRequired { hint: String },
}

impl fmt::Display for BlockReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrefilterNoise { message } => write!(f, "prefilter noise: {message}"),
            Self::Incoherent { score, threshold } => {
                write!(
                    f,
                    "incoherent query (score={score:.3}, threshold={threshold:.3})"
                )
            }
            Self::PiiDetected { matched_classes } => {
                write!(f, "high-risk PII detected: {}", matched_classes.join(", "))
            }
            Self::ToxicContent { score, threshold } => {
                write!(
                    f,
                    "toxic content (score={score:.3}, threshold={threshold:.3})"
                )
            }
            Self::PromptAttack { score, threshold } => {
                write!(
                    f,
                    "prompt injection detected (score={score:.3}, threshold={threshold:.3})"
                )
            }
            Self::UngroundedResponse { score, threshold } => {
                write!(
                    f,
                    "ungrounded response (score={score:.3}, threshold={threshold:.3})"
                )
            }
            Self::IrrelevantResponse { score, threshold } => {
                write!(
                    f,
                    "irrelevant response (score={score:.3}, threshold={threshold:.3})"
                )
            }
            Self::Timeout { gate, timeout_ms } => {
                write!(
                    f,
                    "gate {gate} timed out after {timeout_ms}ms (fail-closed)"
                )
            }
            Self::GateError { gate, message } => {
                write!(f, "gate {gate} error: {message} (fail-closed)")
            }
            Self::ClarificationRequired { hint } => {
                write!(f, "clarification required: {hint}")
            }
        }
    }
}

/// Outcome of a single gate evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// Gate found nothing objectionable.
    Pass,
    /// Gate wants the request stopped. `reason` is operator-facing, not user-facing.
    Block { reason: BlockReason },
    /// Gate believes the request is answerable only after clarification.
    /// Defined for forward compatibility; mapped to `Block` in v1 unless
    /// `PipelineConfig::clarify_enabled` is true.
    Clarify { hint: String },
}

impl Verdict {
    /// Returns true if the verdict is `Pass`.
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    /// Returns true if the verdict is `Block`.
    #[must_use]
    pub const fn is_block(&self) -> bool {
        matches!(self, Self::Block { .. })
    }

    /// Returns true if the verdict is `Clarify`.
    #[must_use]
    pub const fn is_clarify(&self) -> bool {
        matches!(self, Self::Clarify { .. })
    }
}

/// Evaluated outcome of an individual gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateOutcome {
    /// Identifier of the evaluated gate.
    pub gate: GateId,
    /// Evaluation verdict.
    pub verdict: Verdict,
    /// Raw model score in [0.0, 1.0]. For multi-class gates this is the
    /// aggregated risk score, not a single class probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// The threshold this score was compared against, echoed for explainability.
    pub threshold: f32,
    /// Gate-specific structured detail (matched PII classes, class probabilities, ...).
    pub detail: serde_json::Value,
    /// Execution latency.
    #[serde(
        serialize_with = "serialize_duration_as_ms",
        deserialize_with = "deserialize_duration_from_ms",
        rename = "latency_ms"
    )]
    pub latency: Duration,
    /// True if this outcome came from the fail-open/fail-closed policy
    /// rather than from a successful model evaluation.
    pub degraded: bool,
}

fn serialize_duration_as_ms<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let ms = duration.as_secs_f64() * 1000.0;
    // Round to 3 decimal places for clean reporting
    let rounded = (ms * 1000.0).round() / 1000.0;
    serializer.serialize_f64(rounded)
}

fn deserialize_duration_from_ms<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let ms = f64::deserialize(deserializer)?;
    Ok(Duration::from_secs_f64(ms / 1000.0))
}

/// Overall pipeline decision for the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Request cleared all gates and was fulfilled.
    Allow,
    /// Request was blocked by a gate.
    Block,
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => f.write_str("allow"),
            Self::Block => f.write_str("block"),
        }
    }
}
