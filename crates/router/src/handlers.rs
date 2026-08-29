//! # HTTP API Request Handlers
//!
//! **Responsibility:** Implements Axum request handlers for `/v1/check`, `/healthz`, `/readyz`, and `/metrics`.
//! **Pipeline Position:** Public HTTP interface layer.
//! **Latency Budget:** Handler overhead: <100 µs (excluding pipeline evaluation).
//! **Failure Mode:** Maps internal errors to appropriate 4xx/5xx responses; guardrail blocks return 200 OK.

use crate::pipeline::Pipeline;
use crate::telemetry::render_metrics;
use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use controlplane_core::context::{RequestContext, RequestOptions};
use controlplane_inference::ModelRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info_span, Instrument};
use uuid::Uuid;

/// Application state shared across Axum routes.
#[derive(Clone)]
pub struct AppState {
    /// Pipeline orchestrator instance.
    pub pipeline: Arc<Pipeline>,
    /// Model registry containing loaded model info.
    pub registry: Arc<ModelRegistry>,
}

/// Request body for `POST /v1/check`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckRequest {
    /// Required user query string.
    pub query: String,
    /// Optional session/conversation identifier.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Optional compressed chat history summary.
    #[serde(default)]
    pub history_summary: Option<String>,
    /// Optional execution options.
    #[serde(default)]
    pub options: RequestOptions,
}

/// Handler for `POST /v1/check`.
/// # Errors
/// Returns an error if the request cannot be handled by the pipeline.
pub async fn check_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<CheckRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    // Propagate existing trace_id if supplied in x-trace-id header, otherwise generate a new UUID
    let trace_id = headers
        .get("x-trace-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| Uuid::parse_str(s).ok())
        .unwrap_or_else(Uuid::new_v4);

    let span = info_span!(
        "http_check",
        trace_id = %trace_id,
        session_id = ?payload.session_id,
        dry_run = payload.options.dry_run
    );

    async move {
        let req_ctx = RequestContext::with_trace_id(
            trace_id,
            payload.session_id,
            payload.history_summary,
            payload.options,
        );

        match state.pipeline.check(req_ctx, &payload.query).await {
            Ok(check_response) => {
                let mut resp_headers = HeaderMap::new();
                if let Ok(val) = HeaderValue::from_str(&check_response.trace_id.to_string()) {
                    resp_headers.insert("x-trace-id", val);
                }
                Ok((StatusCode::OK, resp_headers, Json(check_response)).into_response())
            }
            Err(err) => {
                error!(error = %err, "Pipeline execution failed");
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": "internal_error",
                        "message": err.to_string(),
                        "trace_id": trace_id
                    })),
                ))
            }
        }
    }
    .instrument(span)
    .await
}

/// Handler for `GET /healthz` liveness probe.
pub async fn healthz_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}

/// Handler for `GET /readyz` readiness probe.
pub async fn readyz_handler(State(state): State<AppState>) -> impl IntoResponse {
    let inventory = state.registry.inventory();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ready",
            "models": inventory
        })),
    )
}

/// Handler for `GET /metrics` Prometheus scraping.
pub async fn metrics_handler() -> impl IntoResponse {
    let metrics_text = render_metrics();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        metrics_text,
    )
}
