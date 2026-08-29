//! # ControlPlane Checker HTTP Router Crate
//!
//! **Responsibility:** Houses HTTP route handlers, telemetry configuration, and request orchestration.
//! **Pipeline Position:** Top-level application server.
//! **Latency Budget:** Bounded by pipeline execution.
//! **Failure Mode:** Maps internal outcomes to HTTP status codes.

#![deny(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]

pub mod app;
pub mod handlers;
pub mod pipeline;
pub mod shutdown;
pub mod telemetry;

pub use app::create_app;
pub use handlers::{AppState, CheckRequest};
pub use pipeline::{CheckResponse, Pipeline, PipelineTimings};
