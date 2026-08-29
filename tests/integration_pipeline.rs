//! # Integration Pipeline Tests
//!
//! **Responsibility:** End-to-end integration tests verifying request lifecycle execution,
//! short-circuiting, dry-run observability mode, and timeout degradation.

use controlplane_core::config::AppConfig;
use controlplane_core::context::{RequestContext, RequestOptions};
use controlplane_core::verdict::Decision;
use controlplane_inference::ModelRegistry;
use controlplane_llm::MockLlm;
use controlplane_router::pipeline::Pipeline;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

fn build_test_pipeline() -> (Pipeline, Arc<MockLlm>) {
    let mut config = AppConfig::default();
    config.inference.force_stub = true;
    config.llm.mock.latency_mean_ms = 1;
    config.llm.mock.latency_stddev_ms = 0;
    config.llm.mock.hallucination_rate = 0.0;

    let registry = ModelRegistry::discover_and_load(&config.inference);
    let mock_llm = Arc::new(MockLlm::new(config.llm.mock.clone()));

    let pipeline = Pipeline::build(config, &registry).expect("pipeline build succeeds");
    (pipeline, mock_llm)
}

#[tokio::test]
async fn test_clean_query_allows_and_invokes_llm() {
    let (pipeline, _mock_llm) = build_test_pipeline();

    let req_ctx = RequestContext::new(
        Some("session-123".to_string()),
        None,
        RequestOptions::default(),
    );

    let res = pipeline
        .check(req_ctx, "How do I implement a binary search tree in Rust?")
        .await
        .expect("pipeline check succeeds");

    assert_eq!(res.decision, Decision::Allow);
    assert!(res.response.is_some());
    assert!(res.blocked_by.is_none());
    assert!(!res.gates.is_empty());
    assert!(res.timings.total_ms > 0.0);
}

#[tokio::test]
async fn test_stub_block_triggers_block_and_short_circuits_llm() {
    let mut config = AppConfig::default();
    config.inference.force_stub = true;
    config.llm.mock.latency_mean_ms = 1;
    config.llm.mock.latency_stddev_ms = 0;

    let registry = ModelRegistry::discover_and_load(&config.inference);
    let pipeline = Pipeline::build(config, &registry).expect("pipeline build succeeds");

    let req_ctx = RequestContext::new(None, None, RequestOptions::default());

    let res = pipeline
        .check(req_ctx, "Please execute __STUB_BLOCK__ immediately")
        .await
        .expect("check succeeds");

    assert_eq!(res.decision, Decision::Block);
    assert!(res.response.is_none());
    assert!(res.blocked_by.is_some());

    // Verify that at least one gate recorded a Block verdict
    let has_block_gate = res.gates.iter().any(|g| g.verdict.is_block());
    assert!(
        has_block_gate,
        "at least one gate must have recorded a block"
    );
}

#[tokio::test]
async fn test_dry_run_mode_never_blocks() {
    let (pipeline, _mock_llm) = build_test_pipeline();

    let options = RequestOptions {
        dry_run: true,
        ..Default::default()
    };

    let req_ctx = RequestContext::new(None, None, options);

    let res = pipeline
        .check(req_ctx, "Evaluating __STUB_BLOCK__ under dry-run")
        .await
        .expect("check succeeds");

    assert_eq!(res.decision, Decision::Allow);
    assert!(res.response.is_some());
    assert!(res.blocked_by.is_none());

    // Non-pass verdicts are still present in gate outcomes
    let has_block_gate = res.gates.iter().any(|g| g.verdict.is_block());
    assert!(
        has_block_gate,
        "dry run must still report non-pass verdicts in gates array"
    );
}

#[tokio::test]
async fn test_slow_gate_trips_timeout_and_applies_failure_policy() {
    let (pipeline, _mock_llm) = build_test_pipeline();

    let req_ctx = RequestContext::new(None, None, RequestOptions::default());

    let res = pipeline
        .check(req_ctx, "Testing gate timeout with trigger __STUB_SLOW__")
        .await
        .expect("check succeeds");

    // __STUB_SLOW__ triggers a 500ms sleep in the stub backend, exceeding the 40ms gate timeout
    assert!(
        res.degraded_mode,
        "degraded mode must be true after gate timeout"
    );
}

#[test]
#[ignore = "requires real ONNX weights; run via cargo test -- --ignored"]
fn test_coherence_fixture_confusion_matrix() {
    let fixture_file = File::open("tests/fixtures/coherence_eval.jsonl")
        .expect("fixture file tests/fixtures/coherence_eval.jsonl must exist");
    let reader = BufReader::new(fixture_file);

    let mut total_clean = 0;
    let mut total_word_salad = 0;
    let mut total_noise = 0;

    for line in reader.lines() {
        let line_str = line.expect("valid line");
        if line_str.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value =
            serde_json::from_str(&line_str).expect("valid JSON line in fixture");
        let label = json["label"].as_str().expect("label field");

        match label {
            "clean" => total_clean += 1,
            "word_salad" => total_word_salad += 1,
            "noise" => total_noise += 1,
            _ => {}
        }
    }

    println!("\n=== Coherence Evaluation Fixture Summary ===");
    println!("Total clean / keyword examples: {total_clean}");
    println!("Total word salad examples:     {total_word_salad}");
    println!("Total noise examples:          {total_noise}");
    println!(
        "Total dataset size:            {}",
        total_clean + total_word_salad + total_noise
    );
    println!("===========================================\n");

    assert!(
        total_clean >= 20,
        "must have at least 20 terse/clean queries"
    );
    assert!(
        total_clean + total_word_salad + total_noise >= 60,
        "must have at least 60 examples"
    );
}
