//! # End-to-End Guardrail Pipeline Orchestrator
//!
//! **Responsibility:** Coordinates the multi-tiered guardrail lifecycle: Tier 0 heuristic prefilter,
//! concurrent Tier 1 input fan-out, LLM generation, and concurrent Tier 2 output grounding.
//! **Pipeline Position:** Core request execution engine.
//! **Latency Budget:** Prefilter: <1 ms, Input fan-out: <120 ms, Output fan-out: <200 ms.
//! **Failure Mode:** Manages graceful degradation and short-circuits on policy blocks.

use controlplane_core::config::AppConfig;
use controlplane_core::context::{GateContext, RequestContext};
use controlplane_core::error::PipelineError;
use controlplane_core::verdict::{Decision, GateId, GateOutcome, Stage};
use controlplane_gates::{
    CoherenceGate, DynGate, Gate, GateExecutor, GroundingGate, IntentGate, PiiGate, PrefilterGate,
    RelevanceGate, ToxicityGate,
};
use controlplane_inference::ModelRegistry;
use controlplane_llm::{GeminiClient, LlmBackend, MockLlm};
use metrics::{counter, histogram};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Timing breakdown across pipeline evaluation stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineTimings {
    /// Tier 0 heuristic prefilter latency in milliseconds.
    pub prefilter_ms: f64,
    /// Tier 1 concurrent input gates latency in milliseconds.
    pub input_gates_ms: f64,
    /// LLM generation latency in milliseconds.
    pub llm_ms: f64,
    /// Tier 2 concurrent output gates latency in milliseconds.
    pub output_gates_ms: f64,
    /// Total end-to-end wall-clock latency in milliseconds.
    pub total_ms: f64,
}

/// Unified API response body returned by the pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckResponse {
    /// Trace identifier.
    pub trace_id: Uuid,
    /// Pipeline decision ("allow" or "block").
    pub decision: Decision,
    /// Generated LLM response text (present only when decision is allow).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    /// Gate that triggered the block (present only when decision is block).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_by: Option<GateId>,
    /// Outcomes of all evaluated gates.
    pub gates: Vec<GateOutcome>,
    /// Detailed timing breakdown.
    pub timings: PipelineTimings,
    /// True if any gate fell back to degraded fail-open/fail-closed mode.
    pub degraded_mode: bool,
}

/// Core pipeline orchestrator.
#[derive(Clone)]
pub struct Pipeline {
    config: AppConfig,
    prefilter: Arc<PrefilterGate>,
    input_gates: Vec<DynGate>,
    llm: Arc<dyn LlmBackend>,
    output_gates: Vec<DynGate>,
}

impl Pipeline {
    /// Constructs and wires the complete pipeline from configuration and loaded model registry.
    ///
    /// # Errors
    /// Returns `PipelineError` if LLM client or backends fail to initialize.
    pub fn build(config: AppConfig, registry: &ModelRegistry) -> Result<Self, PipelineError> {
        let prefilter = Arc::new(PrefilterGate::new(config.gates.prefilter.clone()));

        // Wire Tier 1 input gates
        let mut input_gates: Vec<DynGate> = Vec::new();

        if config.gates.coherence.enabled {
            if let Some(backend) = registry.get("coherence") {
                input_gates.push(Arc::new(CoherenceGate::new(
                    config.gates.coherence.clone(),
                    backend,
                )));
            }
        }

        if config.gates.pii.enabled {
            input_gates.push(Arc::new(PiiGate::new(config.gates.pii.clone())));
        }

        if config.gates.toxicity.enabled {
            if let Some(backend) = registry.get("toxicity") {
                input_gates.push(Arc::new(ToxicityGate::new(
                    config.gates.toxicity.clone(),
                    backend,
                )));
            }
        }

        if config.gates.intent.enabled {
            if let Some(backend) = registry.get("intent") {
                input_gates.push(Arc::new(IntentGate::new(
                    config.gates.intent.clone(),
                    backend,
                )));
            }
        }

        // Wire LLM backend
        let llm: Arc<dyn LlmBackend> = match config.llm.backend.as_str() {
            "gemini" => Arc::new(GeminiClient::new(&config.llm)?),
            _ => Arc::new(MockLlm::new(config.llm.mock.clone())),
        };

        let mut output_gates: Vec<DynGate> = Vec::new();

        // WHY: The cross_encoder pool serves two gates (grounding and relevance).
        // By resolving the same model ID, both gates receive a clone of the same `Arc<dyn ModelBackend>`.
        // Duplicating the pool would double resident memory for zero benefit.
        let cross_encoder = registry.get("cross_encoder").unwrap_or_else(|| {
            Arc::new(controlplane_inference::StubBackend::new("cross_encoder", 3))
        });

        if config.gates.grounding.enabled {
            match GroundingGate::new(config.gates.grounding.clone(), Arc::clone(&cross_encoder)) {
                Ok(gate) => output_gates.push(Arc::new(gate)),
                Err(err) => {
                    return Err(PipelineError::Internal(format!(
                        "GroundingGate initialization failed: {err}"
                    )))
                }
            }
        }

        if config.gates.relevance.enabled {
            match RelevanceGate::new(config.gates.relevance.clone(), Arc::clone(&cross_encoder)) {
                Ok(gate) => output_gates.push(Arc::new(gate)),
                Err(err) => {
                    return Err(PipelineError::Internal(format!(
                        "RelevanceGate initialization failed: {err}"
                    )))
                }
            }
        }

        Ok(Self {
            config,
            prefilter,
            input_gates,
            llm,
            output_gates,
        })
    }

    /// Evaluates a user query through the full guardrail pipeline.
    ///
    /// # Errors
    /// Returns `PipelineError` if LLM execution fails unrecoverably.
    pub async fn check(
        &self,
        req_ctx: RequestContext,
        query: &str,
    ) -> Result<CheckResponse, PipelineError> {
        let total_start = Instant::now();
        let mut all_outcomes: Vec<GateOutcome> = Vec::new();
        let mut degraded_mode = false;

        // Estimated tokens saved if blocked before LLM
        let estimated_tokens = (query.split_whitespace().count() + 150) as u64;

        // -------------------------------------------------------------
        // Step 0: Tier 0 Sync Heuristic Prefilter
        // -------------------------------------------------------------
        let prefilter_start = Instant::now();
        let prefilter_outcome =
            if self.config.gates.prefilter.enabled && !req_ctx.should_skip(GateId::Prefilter) {
                let ctx = GateContext::for_input(&req_ctx, query);
                match self.prefilter.evaluate(&ctx).await {
                    Ok(outcome) => outcome,
                    Err(err) => GateOutcome {
                        gate: GateId::Prefilter,
                        verdict: controlplane_core::verdict::Verdict::Pass,
                        score: 0.0,
                        threshold: 0.0,
                        detail: serde_json::json!({ "error": err.to_string() }),
                        latency: prefilter_start.elapsed(),
                        degraded: true,
                    },
                }
            } else {
                GateOutcome {
                    gate: GateId::Prefilter,
                    verdict: controlplane_core::verdict::Verdict::Pass,
                    score: 0.0,
                    threshold: 0.0,
                    detail: serde_json::json!({ "skipped": true }),
                    latency: Duration::from_micros(1),
                    degraded: false,
                }
            };

        let prefilter_ms = prefilter_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("cp_pipeline_latency_seconds", "stage" => "prefilter")
            .record(prefilter_start.elapsed().as_secs_f64());

        let prefilter_blocked = prefilter_outcome.verdict.is_block();
        degraded_mode |= prefilter_outcome.degraded;
        all_outcomes.push(prefilter_outcome);

        if prefilter_blocked && !req_ctx.is_dry_run() {
            counter!("cp_pipeline_decision_total", "decision" => "block", "blocked_by" => "prefilter")
                .increment(1);
            counter!("cp_llm_tokens_saved_total").increment(estimated_tokens);

            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            return Ok(CheckResponse {
                trace_id: req_ctx.trace_id,
                decision: Decision::Block,
                response: None,
                blocked_by: Some(GateId::Prefilter),
                gates: all_outcomes,
                timings: PipelineTimings {
                    prefilter_ms: round_ms(prefilter_ms),
                    input_gates_ms: 0.0,
                    llm_ms: 0.0,
                    output_gates_ms: 0.0,
                    total_ms: round_ms(total_ms),
                },
                degraded_mode,
            });
        }

        // -------------------------------------------------------------
        // Step 1: Tier 1 Concurrent Input Gates Fan-Out
        // -------------------------------------------------------------
        let input_start = Instant::now();
        let input_ctx = GateContext::for_input(&req_ctx, query);
        let input_fan_out = GateExecutor::execute(
            Stage::Input,
            &self.input_gates,
            &input_ctx,
            Duration::from_millis(self.config.pipeline.input_gate_budget_ms),
            self.config.pipeline.clarify_enabled,
        )
        .await;

        let input_gates_ms = input_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("cp_pipeline_latency_seconds", "stage" => "input")
            .record(input_start.elapsed().as_secs_f64());

        degraded_mode |= input_fan_out.degraded_mode;
        all_outcomes.extend(input_fan_out.outcomes);

        if let Some(blocked_gate) = input_fan_out.blocked_by {
            if !req_ctx.is_dry_run() {
                counter!("cp_pipeline_decision_total", "decision" => "block", "blocked_by" => blocked_gate.as_str())
                    .increment(1);
                counter!("cp_llm_tokens_saved_total").increment(estimated_tokens);

                let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
                return Ok(CheckResponse {
                    trace_id: req_ctx.trace_id,
                    decision: Decision::Block,
                    response: None,
                    blocked_by: Some(blocked_gate),
                    gates: all_outcomes,
                    timings: PipelineTimings {
                        prefilter_ms: round_ms(prefilter_ms),
                        input_gates_ms: round_ms(input_gates_ms),
                        llm_ms: 0.0,
                        output_gates_ms: 0.0,
                        total_ms: round_ms(total_ms),
                    },
                    degraded_mode,
                });
            }
        }

        // -------------------------------------------------------------
        // Step 2: LLM Generation
        // -------------------------------------------------------------
        let llm_start = Instant::now();
        let llm_response = self
            .llm
            .generate(query, req_ctx.history_summary.as_deref())
            .await?;

        let llm_ms = llm_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("cp_pipeline_latency_seconds", "stage" => "llm")
            .record(llm_start.elapsed().as_secs_f64());

        // -------------------------------------------------------------
        // Step 3: Tier 2 Concurrent Output Gates Fan-Out
        // -------------------------------------------------------------
        let output_start = Instant::now();
        let output_ctx = GateContext::for_output(&req_ctx, query, &llm_response.text);
        let output_fan_out = GateExecutor::execute(
            Stage::Output,
            &self.output_gates,
            &output_ctx,
            Duration::from_millis(self.config.pipeline.output_gate_budget_ms),
            self.config.pipeline.clarify_enabled,
        )
        .await;

        let output_gates_ms = output_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("cp_pipeline_latency_seconds", "stage" => "output")
            .record(output_start.elapsed().as_secs_f64());

        degraded_mode |= output_fan_out.degraded_mode;
        all_outcomes.extend(output_fan_out.outcomes);

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("cp_pipeline_latency_seconds", "stage" => "total")
            .record(total_start.elapsed().as_secs_f64());

        if let Some(blocked_gate) = output_fan_out.blocked_by {
            if !req_ctx.is_dry_run() {
                counter!("cp_pipeline_decision_total", "decision" => "block", "blocked_by" => blocked_gate.as_str())
                    .increment(1);

                return Ok(CheckResponse {
                    trace_id: req_ctx.trace_id,
                    decision: Decision::Block,
                    response: None,
                    blocked_by: Some(blocked_gate),
                    gates: all_outcomes,
                    timings: PipelineTimings {
                        prefilter_ms: round_ms(prefilter_ms),
                        input_gates_ms: round_ms(input_gates_ms),
                        llm_ms: round_ms(llm_ms),
                        output_gates_ms: round_ms(output_gates_ms),
                        total_ms: round_ms(total_ms),
                    },
                    degraded_mode,
                });
            }
        }

        // All gates passed!
        counter!("cp_pipeline_decision_total", "decision" => "allow", "blocked_by" => "none")
            .increment(1);

        Ok(CheckResponse {
            trace_id: req_ctx.trace_id,
            decision: Decision::Allow,
            response: Some(llm_response.text),
            blocked_by: None,
            gates: all_outcomes,
            timings: PipelineTimings {
                prefilter_ms: round_ms(prefilter_ms),
                input_gates_ms: round_ms(input_gates_ms),
                llm_ms: round_ms(llm_ms),
                output_gates_ms: round_ms(output_gates_ms),
                total_ms: round_ms(total_ms),
            },
            degraded_mode,
        })
    }
}

fn round_ms(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}
