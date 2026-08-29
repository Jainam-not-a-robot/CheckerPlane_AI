//! # ControlPlane Checker Service Entrypoint
//!
//! **Responsibility:** Bootstraps runtime configuration, telemetry, model discovery,
//! Tokio thread pools, pipeline assembly, and the Axum HTTP listener with graceful shutdown.
//! **Pipeline Position:** Process entrypoint.
//! **Latency Budget:** Startup: <50 ms.
//! **Failure Mode:** Aborts on fatal initialization errors with actionable operator messages.

use controlplane_core::config::AppConfig;
use controlplane_inference::{ModelRegistry, MODEL_REGISTRY_TABLE};
use controlplane_router::app::create_app;
use controlplane_router::pipeline::Pipeline;
use controlplane_router::shutdown::shutdown_signal;
use controlplane_router::telemetry::init_telemetry;
use std::sync::Arc;
use tracing::info;

fn main() {
    // 1. Load layered configuration
    let config = AppConfig::load(None).expect(
        "Failed to load application configuration. Ensure 'config/default.toml' exists and environment variables are valid.",
    );

    // 2. Initialize tracing subscriber and Prometheus metrics
    init_telemetry(&config.telemetry).expect(
        "Failed to initialize telemetry logging or Prometheus metrics recorder. Check log level configuration.",
    );

    info!(
        version = env!("CARGO_PKG_VERSION"),
        bind = %config.server.bind,
        llm_backend = %config.llm.backend,
        "Starting ControlPlane Checker Guardrail Service"
    );

    // 3. Size blocking thread pool to accommodate all ONNX model session pools plus spare workers
    // Total sessions across all registered models
    let total_sessions = MODEL_REGISTRY_TABLE.len() * config.inference.pool_size_per_model;
    let max_blocking_threads = total_sessions + 8;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .max_blocking_threads(max_blocking_threads)
        .enable_all()
        .build()
        .expect("Failed to build Tokio multi-threaded runtime. Check system resource limits.");

    runtime.block_on(async move {
        // 4. Discover and instantiate model inference backends (live ONNX or deterministic stubs)
        let registry = Arc::new(ModelRegistry::discover_and_load(&config.inference));

        // 5. Construct end-to-end pipeline
        let pipeline = Arc::new(Pipeline::build(config.clone(), &registry).expect(
            "Failed to assemble guardrail pipeline. Verify gate configurations and model registry.",
        ));

        // 6. Assemble Axum HTTP router
        let app = create_app(&config, pipeline, registry);

        // 7. Bind TCP listener socket
        let listener = tokio::net::TcpListener::bind(&config.server.bind)
            .await
            .expect("Failed to bind TCP listener. Verify that port is not already in use.");

        info!(address = %config.server.bind, "HTTP server listening and ready to accept traffic");

        // 8. Serve HTTP traffic with graceful shutdown handling
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .expect("Server encountered a fatal error while running HTTP listener.");

        info!("ControlPlane Checker shutdown complete. Exiting gracefully.");
    });
}
