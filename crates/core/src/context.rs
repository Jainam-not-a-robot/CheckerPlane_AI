//! # Request and Gate Execution Contexts
//!
//! **Responsibility:** Carries request-level metadata, telemetry tracing identifiers, execution options,
//! input query data, and optional conversation history summaries through the gate pipeline.
//! **Pipeline Position:** Passed into every gate evaluation and downstream LLM invocation.
//! **Latency Budget:** Zero overhead.
//! **Failure Mode:** Infallible type definitions.

use crate::verdict::GateId;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;
use uuid::Uuid;

/// Request execution options supplied by the caller.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestOptions {
    /// Gates to bypass during this evaluation.
    #[serde(default)]
    pub skip_gates: HashSet<GateId>,
    /// When true, evaluate all gates and log findings without blocking requests.
    #[serde(default)]
    pub dry_run: bool,
}

/// Request-level context containing tracing metadata and execution options.
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Unique trace identifier assigned to the request.
    pub trace_id: Uuid,
    /// Optional conversation or session identifier.
    pub session_id: Option<String>,
    /// Optional compressed summary of prior conversation turns.
    pub history_summary: Option<String>,
    /// Caller-specified execution options (dry run, gate bypass).
    pub options: RequestOptions,
    /// Monotonic start timestamp for wall-clock latency measurement.
    pub started_at: Instant,
}

impl RequestContext {
    /// Constructs a new request context with a freshly generated UUID v4 trace ID.
    #[must_use]
    pub fn new(
        session_id: Option<String>,
        history_summary: Option<String>,
        options: RequestOptions,
    ) -> Self {
        Self {
            trace_id: Uuid::new_v4(),
            session_id,
            history_summary,
            options,
            started_at: Instant::now(),
        }
    }

    /// Constructs a request context with an explicit trace ID (e.g. propagated from upstream headers).
    #[must_use]
    pub fn with_trace_id(
        trace_id: Uuid,
        session_id: Option<String>,
        history_summary: Option<String>,
        options: RequestOptions,
    ) -> Self {
        Self {
            trace_id,
            session_id,
            history_summary,
            options,
            started_at: Instant::now(),
        }
    }

    /// Returns true if the given gate ID should be skipped.
    #[must_use]
    pub fn should_skip(&self, gate_id: GateId) -> bool {
        self.options.skip_gates.contains(&gate_id)
    }

    /// Returns true if running in dry-run observability mode.
    #[must_use]
    pub fn is_dry_run(&self) -> bool {
        self.options.dry_run
    }
}

/// Context provided to individual gate evaluators.
#[derive(Debug, Clone)]
pub struct GateContext<'a> {
    /// Reference to the parent request context.
    pub request: &'a RequestContext,
    /// Raw user query string.
    pub query: &'a str,
    /// LLM-generated response candidate (populated only for Tier 2 output gates).
    pub response: Option<&'a str>,
}

impl<'a> GateContext<'a> {
    /// Constructs a gate context for Tier 0 / Tier 1 input gates.
    #[must_use]
    pub const fn for_input(request: &'a RequestContext, query: &'a str) -> Self {
        Self {
            request,
            query,
            response: None,
        }
    }

    /// Constructs a gate context for Tier 2 output gates.
    #[must_use]
    pub const fn for_output(
        request: &'a RequestContext,
        query: &'a str,
        response: &'a str,
    ) -> Self {
        Self {
            request,
            query,
            response: Some(response),
        }
    }
}
