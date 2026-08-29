//! # Model Registry and Discovery
//!
//! **Responsibility:** Discovers available model weights on disk at startup, instantiates either
//! `OnnxBackend` or fallback `StubBackend`, and serves the live model inventory for readiness probes.
//! **Pipeline Position:** Executed once at application startup.
//! **Latency Budget:** Startup scan: <10 ms.
//! **Failure Mode:** Infallible; missing models cleanly fall back to `StubBackend` with a `WARN` log.

use crate::backend::ModelBackend;
use crate::onnx::OnnxBackend;
use crate::stub::StubBackend;
use controlplane_core::config::InferenceConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

/// Static metadata definition for a known model.
#[derive(Debug, Clone, Copy)]
pub struct RegisteredModel {
    /// Model identifier key.
    pub model_id: &'static str,
    /// Upstream HuggingFace repository identifier.
    pub hf_repo: &'static str,
    /// Task description.
    pub task: &'static str,
    /// Expected class output count.
    pub num_classes: usize,
}

/// Hardcoded model registry table.
pub const MODEL_REGISTRY_TABLE: &[RegisteredModel] = &[
    RegisteredModel {
        model_id: "coherence",
        hf_repo: "madhurjindal/autonlp-Gibberish-Detector-492513457",
        task: "text-classification",
        num_classes: 4,
    },
    RegisteredModel {
        model_id: "toxicity",
        hf_repo: "martin-ha/toxic-comment-model",
        task: "text-classification",
        num_classes: 2,
    },
    RegisteredModel {
        model_id: "intent",
        hf_repo: "meta-llama/Llama-Prompt-Guard-2-22M",
        task: "text-classification",
        num_classes: 2,
    },
    RegisteredModel {
        model_id: "cross_encoder",
        hf_repo: "cross-encoder/nli-distilroberta-base",
        task: "text-classification",
        num_classes: 3,
    },
];

/// Status report for an individual model in `/readyz`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub model_id: String,
    /// Backend kind ("onnx" or "stub").
    pub backend: String,
    /// True when backed by live weights.
    pub is_live: bool,
    /// Upstream HuggingFace repository.
    pub hf_repo: String,
    /// Task category.
    pub task: String,
}

/// Container for all instantiated model backends.
#[derive(Clone)]
pub struct ModelRegistry {
    backends: HashMap<String, Arc<dyn ModelBackend>>,
    infos: Vec<ModelInfo>,
}

impl ModelRegistry {
    /// Discovers and initializes all registered models based on disk presence and configuration.
    #[must_use]
    pub fn discover_and_load(config: &InferenceConfig) -> Self {
        let mut backends: HashMap<String, Arc<dyn ModelBackend>> = HashMap::new();
        let mut infos = Vec::new();

        let base_dir = Path::new(&config.model_dir);

        for registered in MODEL_REGISTRY_TABLE {
            let model_id = registered.model_id;
            let model_dir = base_dir.join(model_id);

            let pool_size = config.pool_size_per_model;
            let weights_file = "model.onnx".to_string();

            let model_file = model_dir.join(&weights_file);
            let tokenizer_file = model_dir.join("tokenizer.json");

            let has_files = model_file.exists() && tokenizer_file.exists();

            let (backend, info): (Arc<dyn ModelBackend>, ModelInfo) =
                if has_files && !config.force_stub {
                    match OnnxBackend::from_dir(
                        model_id,
                        &model_dir,
                        pool_size,
                        &weights_file,
                        &config.graph_optimization_level,
                    ) {
                        Ok(onnx) => {
                            info!(
                                model = %model_id,
                                backend = "onnx",
                                status = "live",
                                "Model loaded with live ONNX weights"
                            );
                            (
                                Arc::new(onnx),
                                ModelInfo {
                                    model_id: model_id.to_string(),
                                    backend: "onnx".to_string(),
                                    is_live: true,
                                    hf_repo: registered.hf_repo.to_string(),
                                    task: registered.task.to_string(),
                                },
                            )
                        }
                        Err(err) => {
                            warn!(
                                model = %model_id,
                                backend = "stub",
                                status = "placeholder",
                                error = %err,
                                "Failed to load ONNX session, falling back to stub backend"
                            );
                            (
                                Arc::new(StubBackend::new(model_id, registered.num_classes)),
                                ModelInfo {
                                    model_id: model_id.to_string(),
                                    backend: "stub".to_string(),
                                    is_live: false,
                                    hf_repo: registered.hf_repo.to_string(),
                                    task: registered.task.to_string(),
                                },
                            )
                        }
                    }
                } else {
                    warn!(
                        model = %model_id,
                        backend = "stub",
                        status = "placeholder",
                        reason = "weights_not_found",
                        "Pipeline will run but classifications are synthetic"
                    );
                    (
                        Arc::new(StubBackend::new(model_id, registered.num_classes)),
                        ModelInfo {
                            model_id: model_id.to_string(),
                            backend: "stub".to_string(),
                            is_live: false,
                            hf_repo: registered.hf_repo.to_string(),
                            task: registered.task.to_string(),
                        },
                    )
                };

            backends.insert(model_id.to_string(), backend);
            infos.push(info);
        }

        Self { backends, infos }
    }

    /// Retrieves an instantiated model backend by model ID.
    #[must_use]
    pub fn get(&self, model_id: &str) -> Option<Arc<dyn ModelBackend>> {
        self.backends.get(model_id).cloned()
    }

    /// Returns the inventory of loaded models for readiness reporting.
    #[must_use]
    pub fn inventory(&self) -> &[ModelInfo] {
        &self.infos
    }
}
