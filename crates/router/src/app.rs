//! # Axum Router Construction
//!
//! **Responsibility:** Assembles HTTP route endpoints, attaches middleware layers (CORS, tracing, body limit),
//! and injects shared application state.
//! **Pipeline Position:** HTTP service layer.
//! **Latency Budget:** <20 µs router matching.
//! **Failure Mode:** Infallible router builder.

use crate::handlers::{check_handler, healthz_handler, metrics_handler, readyz_handler, AppState};
use crate::pipeline::Pipeline;
use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use controlplane_core::config::AppConfig;
use controlplane_inference::ModelRegistry;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Builds the Axum application router with all routes and middleware layers.
pub fn create_app(
    config: &AppConfig,
    pipeline: Arc<Pipeline>,
    registry: Arc<ModelRegistry>,
) -> Router {
    let state = AppState { pipeline, registry };

    Router::new()
        .route("/v1/check", post(check_handler))
        .route("/healthz", get(healthz_handler))
        .route("/readyz", get(readyz_handler))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(config.server.max_body_bytes))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
