//! # Microbenchmarks: Gate Latency and Fan-Out Concurrency
//!
//! **Responsibility:** Measures single-gate p50/p99 latency distributions and quantitatively compares
//! parallel fan-out latency against sequential chaining.

use controlplane_core::config::AppConfig;
use controlplane_core::context::{GateContext, RequestContext, RequestOptions};
use controlplane_core::verdict::Stage;
use controlplane_gates::{CoherenceGate, DynGate, GateExecutor, IntentGate, PiiGate, ToxicityGate};
use controlplane_inference::StubBackend;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use std::time::Duration;

fn create_stub_gates() -> Vec<DynGate> {
    let config = AppConfig::default();

    let coh_backend = Arc::new(StubBackend::new("coherence", 4));
    let tox_backend = Arc::new(StubBackend::new("toxicity", 2));
    let int_backend = Arc::new(StubBackend::new("intent", 2));


    vec![
        Arc::new(CoherenceGate::new(config.gates.coherence, coh_backend)),
        Arc::new(PiiGate::new(config.gates.pii)),
        Arc::new(ToxicityGate::new(config.gates.toxicity, tox_backend)),
        Arc::new(IntentGate::new(config.gates.intent, int_backend)),
    ]
}

fn bench_gate_latencies(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gates = create_stub_gates();

    let req_ctx = RequestContext::new(None, None, RequestOptions::default());
    let query = "Explain the mechanics of concurrent memory synchronization";
    let gate_ctx = GateContext::for_input(&req_ctx, query);

    let mut group = c.benchmark_group("guardrail_gates");

    // Single gate latency benchmarks
    for gate in &gates {
        let gate_clone = gate.clone();
        let gate_id = gate.id();
        let ctx = gate_ctx.clone();

        group.bench_with_input(
            BenchmarkId::new("single_gate_evaluate", gate_id.as_str()),
            &gate_id,
            |b, _| {
                b.to_async(&rt).iter(|| {
                    let g = gate_clone.clone();
                    let c = ctx.clone();
                    async move {
                        let _ = g.evaluate(&c).await;
                    }
                });
            },
        );
    }

    // Parallel fan-out vs sequential chaining comparison
    group.bench_function("input_tier_parallel_fan_out", |b| {
        let gates_ref = gates.clone();
        let ctx = gate_ctx.clone();

        b.to_async(&rt).iter(|| {
            let g = gates_ref.clone();
            let c = ctx.clone();
            async move {
                GateExecutor::execute(Stage::Input, &g, &c, Duration::from_millis(120), false).await
            }
        });
    });

    group.bench_function("input_tier_sequential_chain", |b| {
        let gates_ref = gates.clone();
        let ctx = gate_ctx.clone();

        b.to_async(&rt).iter(|| {
            let g = gates_ref.clone();
            let c = ctx.clone();
            async move {
                let mut outcomes = Vec::new();
                for gate in &g {
                    if let Ok(outcome) = gate.evaluate(&c).await {
                        let is_block = outcome.verdict.is_block();
                        outcomes.push(outcome);
                        if is_block {
                            break;
                        }
                    }
                }
                outcomes
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_gate_latencies);
criterion_main!(benches);
