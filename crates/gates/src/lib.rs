//! # Guardrail Gate Implementations and Concurrent Executor
//!
//! **Responsibility:** Houses all input and output guardrail gates (Prefilter, Coherence, PII, Toxicity,
//! Intent, Grounding, NLI) and executes them concurrently using the fan-out executor.
//! **Pipeline Position:** Core classification layer of the pipeline (Tier 0, Tier 1, Tier 2).
//! **Latency Budget:** Tier 1 Input budget: 120 ms max; Tier 2 Output budget: 200 ms max.
//! **Failure Mode:** Handled via per-gate `FailurePolicy` (Open/Closed) and stage timeouts.

#![deny(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]

pub mod coherence;
pub mod executor;
pub mod grounding;
pub mod intent;
pub mod pii;
pub mod prefilter;
pub mod relevance;
pub mod toxicity;

use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{FailurePolicy, GateId, GateOutcome, Stage};
use std::sync::Arc;
use std::time::Duration;

pub use coherence::CoherenceGate;
pub use executor::{FanOutResult, GateExecutor};
pub use grounding::GroundingGate;
pub use intent::IntentGate;
pub use pii::PiiGate;
pub use prefilter::PrefilterGate;
pub use relevance::RelevanceGate;
pub use toxicity::ToxicityGate;

/// Core trait representing an individual guardrail classifier gate.
#[async_trait::async_trait]
pub trait Gate: Send + Sync + 'static {
    /// Unique identifier for this gate.
    fn id(&self) -> GateId;

    /// Pipeline stage in which this gate executes (Input or Output).
    fn stage(&self) -> Stage;

    /// Failure policy when gate encounters an error or timeout.
    fn failure_policy(&self) -> FailurePolicy;

    /// Execution timeout for a single evaluation.
    fn timeout(&self) -> Duration;

    /// Evaluates the input or output content against this gate's policy.
    ///
    /// # Errors
    /// Returns `GateError` on evaluation failure or timeout.
    async fn evaluate(&self, ctx: &GateContext) -> Result<GateOutcome, GateError>;
}

/// Type alias for thread-safe shared gate handle.
pub type DynGate = Arc<dyn Gate>;
