//! # Telemetry, Tracing, and Metrics Instrumentation
//!
//! **Responsibility:** Configures structured JSON/pretty log formatting via `tracing-subscriber` and
//! installs Prometheus metric recorders for latency histograms, saturation gauges, and decision counters.
//! **Pipeline Position:** Cross-cutting observability layer initialized at application startup.
//! **Latency Budget:** In-memory atomic metric increments (<50 ns).
//! **Failure Mode:** Infallible logger with graceful fallback.

use controlplane_core::config::TelemetryConfig;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

static PROMETHEUS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Initializes structured logging and Prometheus metrics exporter based on configuration.
///
/// # Errors
/// Returns an error if tracing subscriber fails to initialize.
pub fn init_telemetry(
    config: &TelemetryConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    if config.log_format == "json" {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().json().flatten_event(true))
            .try_init()?;
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().compact())
            .try_init()?;
    }

    if config.metrics_enabled {
        let builder = PrometheusBuilder::new();
        let handle = builder
            .install_recorder()
            .map_err(|err| format!("failed to install Prometheus recorder: {err}"))?;
        let _ = PROMETHEUS_HANDLE.set(handle);
    }

    Ok(())
}

/// Renders the Prometheus exposition format text for scraping.
#[must_use]
pub fn render_metrics() -> String {
    PROMETHEUS_HANDLE.get().map_or_else(
        || "# Metrics disabled\n".to_string(),
        PrometheusHandle::render,
    )
}
