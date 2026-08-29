//! # Inference Engine and Session Pool
//!
//! **Responsibility:** Manages ONNX Runtime sessions, worker thread pools, tokenization,
//! deterministic stub fallbacks, and model discovery.
//! **Pipeline Position:** Invoked by all machine-learning gate evaluators in Tier 1 and Tier 2.
//! **Latency Budget:** Single forward pass: 5–30 ms (ONNX); <50 µs (Stub).
//! **Failure Mode:** Propagates `InferenceError`; handled per gate via `FailurePolicy`.

#![deny(clippy::pedantic)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]

pub mod backend;
pub mod onnx;
pub mod pool;
pub mod registry;
pub mod stub;
pub mod tokenizer;

pub use backend::{ModelBackend, TokenTag};
pub use onnx::OnnxBackend;
pub use pool::{SessionGuard, SessionPool};
pub use registry::{ModelInfo, ModelRegistry, RegisteredModel};
pub use stub::StubBackend;
pub use tokenizer::SharedTokenizer;
