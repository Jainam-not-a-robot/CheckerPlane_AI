//! # Concurrent Fan-Out Gate Executor
//!
//! **Responsibility:** Executes a set of gates in parallel, manages per-gate and stage timeouts,
//! enforces early cancellation on the first blocking verdict, and handles fail-open/fail-closed policies.
//! **Pipeline Position:** Invoked during Tier 1 input evaluation and Tier 2 output evaluation.
//! **Latency Budget:** Bounded by `input_gate_budget_ms` (120 ms) and `output_gate_budget_ms` (200 ms).
//! **Failure Mode:** Applies per-gate `FailurePolicy` on timeout or error; degraded mode tracked.

use crate::DynGate;
use controlplane_core::context::GateContext;
use controlplane_core::error::GateError;
use controlplane_core::verdict::{BlockReason, FailurePolicy, GateId, GateOutcome, Stage, Verdict};
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use metrics::{counter, histogram};
use std::time::{Duration, Instant};
use tracing::{debug, error, info_span, warn, Instrument};

/// Aggregated result of a fan-out gate stage execution.
#[derive(Debug, Clone)]
pub struct FanOutResult {
    /// Outcomes recorded for all evaluated gates.
    pub outcomes: Vec<GateOutcome>,
    /// Gate that triggered a block, if any.
    pub blocked_by: Option<GateId>,
    /// Whether any gate evaluated in degraded mode.
    pub degraded_mode: bool,
}

/// Executor for concurrent gate evaluation.
pub struct GateExecutor;

impl GateExecutor {
    /// Executes a slice of gates concurrently against the given context.
    ///
    /// # Features:
    /// 1. True parallel fan-out via `FuturesUnordered`.
    /// 2. Instant cancellation of outstanding gates upon first `Block` verdict.
    /// 3. Per-gate timeout and error handling with fail-open/fail-closed policies.
    /// 4. Hard overall stage wall-clock budget.
    pub async fn execute(
        stage: Stage,
        gates: &[DynGate],
        ctx: &GateContext<'_>,
        stage_budget: Duration,
        clarify_enabled: bool,
    ) -> FanOutResult {
        let stage_name = match stage {
            Stage::Input => "input",
            Stage::Output => "output",
        };

        let mut active_gates: Vec<DynGate> = Vec::new();
        let mut outcomes: Vec<GateOutcome> = Vec::new();
        let mut blocked_by: Option<GateId> = None;
        let mut degraded_mode = false;

        for gate in gates {
            if ctx.request.should_skip(gate.id()) {
                debug!(gate = %gate.id(), "Skipping gate per request options");
                continue;
            }
            active_gates.push(gate.clone());
        }

        if active_gates.is_empty() {
            return FanOutResult {
                outcomes,
                blocked_by: None,
                degraded_mode: false,
            };
        }

        let stage_start = Instant::now();
        let mut futures = FuturesUnordered::new();

        for gate in &active_gates {
            let gate_clone = gate.clone();
            let gate_id = gate.id();
            let timeout_dur = gate.timeout();
            let ctx_owned = ctx.clone();

            let span = info_span!(
                "gate_evaluate",
                gate = %gate_id,
                stage = %stage_name,
                trace_id = %ctx.request.trace_id
            );

            futures.push(
                async move {
                    let start = Instant::now();
                    let eval_res =
                        tokio::time::timeout(timeout_dur, gate_clone.evaluate(&ctx_owned)).await;

                    (gate_clone, eval_res, start.elapsed())
                }
                .instrument(span),
            );
        }

        let outer_deadline = tokio::time::sleep(stage_budget);
        tokio::pin!(outer_deadline);

        while !futures.is_empty() {
            tokio::select! {
                biased;

                Some((gate, eval_result, elapsed)) = futures.next() => {
                    let gate_id = gate.id();
                    let policy = gate.failure_policy();

                    let mut outcome = match eval_result {
                        // Successful evaluation within gate timeout
                        Ok(Ok(mut gate_outcome)) => {
                            // Forward-compatibility handling for Clarify
                            if let Verdict::Clarify { hint } = &gate_outcome.verdict {
                                if !clarify_enabled {
                                    // WHY: Forward-compatible Clarify is mapped to Block in v1 unless clarify_enabled is explicitly set.
                                    // V1: Clarify prompts are defined in the schema but not user-visible in v1.
                                    gate_outcome.verdict = Verdict::Block {
                                        reason: BlockReason::ClarificationRequired { hint: hint.clone() },
                                    };
                                }
                            }
                            gate_outcome
                        }
                        // Gate timed out
                        Err(_elapsed_err) => {
                            warn!(
                                gate = %gate_id,
                                elapsed_ms = elapsed.as_millis(),
                                policy = ?policy,
                                "Gate execution timed out"
                            );
                            degraded_mode = true;
                            counter!("cp_gate_degraded_total", "gate" => gate_id.as_str(), "reason" => "timeout").increment(1);

                            match policy {
                                FailurePolicy::Open => GateOutcome {
                                    gate: gate_id,
                                    verdict: Verdict::Pass,
                                    score: 0.0,
                                    threshold: 0.0,
                                    detail: serde_json::json!({ "degraded_reason": "timeout_fail_open" }),
                                    latency: elapsed,
                                    degraded: true,
                                },
                                FailurePolicy::Closed => GateOutcome {
                                    gate: gate_id,
                                    verdict: Verdict::Block {
                                        reason: BlockReason::Timeout {
                                            gate: gate_id,
                                            timeout_ms: gate.timeout().as_millis() as u64,
                                        },
                                    },
                                    score: 1.0,
                                    threshold: 0.0,
                                    detail: serde_json::json!({ "degraded_reason": "timeout_fail_closed" }),
                                    latency: elapsed,
                                    degraded: true,
                                },
                            }
                        }
                        // Gate returned an internal error
                        Ok(Err(gate_err)) => {
                            error!(
                                gate = %gate_id,
                                error = %gate_err,
                                policy = ?policy,
                                "Gate evaluation error"
                            );
                            degraded_mode = true;
                            counter!("cp_gate_degraded_total", "gate" => gate_id.as_str(), "reason" => "error").increment(1);

                            match policy {
                                FailurePolicy::Open => GateOutcome {
                                    gate: gate_id,
                                    verdict: Verdict::Pass,
                                    score: 0.0,
                                    threshold: 0.0,
                                    detail: serde_json::json!({
                                        "degraded_reason": "error_fail_open",
                                        "error": gate_err.to_string()
                                    }),
                                    latency: elapsed,
                                    degraded: true,
                                },
                                FailurePolicy::Closed => GateOutcome {
                                    gate: gate_id,
                                    verdict: Verdict::Block {
                                        reason: BlockReason::GateError {
                                            gate: gate_id,
                                            message: gate_err.to_string(),
                                        },
                                    },
                                    score: 1.0,
                                    threshold: 0.0,
                                    detail: serde_json::json!({
                                        "degraded_reason": "error_fail_closed",
                                        "error": gate_err.to_string()
                                    }),
                                    latency: elapsed,
                                    degraded: true,
                                },
                            }
                        }
                    };

                    // Record metrics for this gate outcome
                    let verdict_str = match &outcome.verdict {
                        Verdict::Pass => "pass",
                        Verdict::Block { .. } => "block",
                        Verdict::Clarify { .. } => "clarify",
                    };
                    histogram!("cp_gate_latency_seconds", "gate" => gate_id.as_str())
                        .record(outcome.latency.as_secs_f64());
                    counter!("cp_gate_verdict_total", "gate" => gate_id.as_str(), "verdict" => verdict_str)
                        .increment(1);

                    let is_block = outcome.verdict.is_block();
                    outcomes.push(outcome);

                    // Check if block occurred and dry_run is not active
                    if is_block && !ctx.request.is_dry_run() {
                        blocked_by = Some(gate_id);
                        // WHY: On the first Verdict::Block, record the outcome, then drop the
                        // FuturesUnordered to cancel every outstanding gate. Do not wait for the rest.
                        drop(futures);
                        break;
                    }
                }

                () = &mut outer_deadline => {
                    warn!(
                        stage = %stage_name,
                        elapsed_ms = stage_start.elapsed().as_millis(),
                        budget_ms = stage_budget.as_millis(),
                        "Stage wall-clock budget exceeded; cancelling remaining futures"
                    );
                    degraded_mode = true;
                    // Any outstanding gates in futures get dropped
                    drop(futures);
                    break;
                }
            }
        }

        FanOutResult {
            outcomes,
            blocked_by,
            degraded_mode,
        }
    }
}
