//! # Core Domain Models and Configuration Types
//!
//! **Responsibility:** Defines foundational data types, configuration structures, error taxonomies,
//! request contexts, and gate outcomes used across the entire ControlPlane Checker pipeline.
//! **Pipeline Position:** Cross-cutting substrate used in all pipeline tiers (Tier 0, Tier 1, LLM, Tier 2).
//! **Latency Budget:** Zero computational overhead (<1 µs); purely in-memory data structures.
//! **Failure Mode:** Infallible data definitions.

#![deny(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]

pub mod config;
pub mod context;
pub mod error;
pub mod verdict;

pub use config::AppConfig;
pub use context::{GateContext, RequestContext, RequestOptions};
pub use error::{ConfigError, GateError, InferenceError, LlmError, PipelineError};
pub use verdict::{BlockReason, Decision, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
