//! # Axum HTTP API Integration Tests
//!
//! **Responsibility:** Verifies HTTP endpoints (`/v1/check`, `/healthz`, `/readyz`, `/metrics`)
//! using Tower's in-memory `ServiceExt::oneshot`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use controlplane_core::config::AppConfig;
use controlplane_inference::ModelRegistry;
use controlplane_router::app::create_app;
use controlplane_router::pipeline::Pipeline;
use std::sync::Arc;
use tower::ServiceExt;

fn build_test_app() -> axum::Router {
    let mut config = AppConfig::default();
    config.inference.force_stub = true;
    config.llm.mock.latency_mean_ms = 1;
    config.llm.mock.latency_stddev_ms = 0;

    let registry = Arc::new(ModelRegistry::discover_and_load(&config.inference));
    let pipeline = Arc::new(Pipeline::build(config.clone(), &registry).expect("pipeline"));

    create_app(&config, pipeline, registry)
}

#[tokio::test]
async fn test_healthz_endpoint() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn test_readyz_endpoint() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "ready");
    assert!(json["models"].as_array().is_some());
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let app = build_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_check_endpoint_allow() {
    let app = build_test_app();

    let payload = serde_json::json!({
        "query": "Hello, can you explain what a memory barrier is in concurrent computing?"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/check")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers().contains_key("x-trace-id"));

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["decision"], "allow");
    assert!(json["response"].is_string());
    assert!(json["blocked_by"].is_null());
    assert!(json["gates"].as_array().unwrap().len() > 0);
}

#[tokio::test]
async fn test_check_endpoint_block() {
    let app = build_test_app();

    let payload = serde_json::json!({
        "query": "Please process this request with trigger __STUB_BLOCK__"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/check")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["decision"], "block");
    assert!(json["blocked_by"].is_string());
}
