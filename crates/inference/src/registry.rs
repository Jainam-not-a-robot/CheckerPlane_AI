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
    /// Upstream `HuggingFace` repository identifier.
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
        hf_repo: "minuva/MiniLMv2-toxic-jigsaw-onnx",
        task: "text-classification",
        num_classes: 2,
    },
    RegisteredModel {
        model_id: "intent",
        hf_repo: "testsavantai/prompt-injection-defender-base-v0-onnx",
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
    /// Upstream `HuggingFace` repository.
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
    #[allow(clippy::too_many_lines)]
    pub fn discover_and_load(
        config: &InferenceConfig,
        overrides: &std::collections::HashMap<
            String,
            controlplane_core::config::ModelOverrideConfig,
        >,
    ) -> Self {
        let mut backends: HashMap<String, Arc<dyn ModelBackend>> = HashMap::new();
        let mut infos = Vec::new();

        let valid_model_ids: std::collections::HashSet<&str> =
            MODEL_REGISTRY_TABLE.iter().map(|r| r.model_id).collect();
        for override_id in overrides.keys() {
            if !valid_model_ids.contains(override_id.as_str()) {
                warn!(
                    model = %override_id,
                    "Model override provided in config but model ID is not registered"
                );
            }
        }

        let base_dir = Path::new(&config.model_dir);

        for registered in MODEL_REGISTRY_TABLE {
            let model_id = registered.model_id;
            let model_dir = base_dir.join(model_id);

            let model_override = overrides.get(model_id);
            let pool_size = model_override
                .and_then(|o| o.pool_size)
                .unwrap_or(config.pool_size_per_model);
            let weights_file = model_override
                .and_then(|o| o.weights_file.clone())
                .unwrap_or_else(|| "model.onnx".to_string());

            let model_file = model_dir.join(&weights_file);
            let tokenizer_file = model_dir.join("tokenizer.json");
            let config_file = model_dir.join("config.json");

            let mut missing_files = Vec::new();
            if !model_file.exists() {
                missing_files.push("weights");
            }
            if !tokenizer_file.exists() {
                missing_files.push("tokenizer.json");
            }
            if !config_file.exists() {
                missing_files.push("config.json");
            }

            let has_files = missing_files.is_empty();

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
                    let missing_str = missing_files.join(", ");
                    let checked_path = model_file.display().to_string();
                    warn!(
                        model = %model_id,
                        backend = "stub",
                        status = "placeholder",
                        reason = %missing_str,
                        path = %checked_path,
                        "Pipeline will run but classifications are synthetic due to missing files"
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use controlplane_core::config::{InferenceConfig, ModelOverrideConfig};
    use std::collections::HashMap;
    use std::fs;

    #[test]
    fn test_model_override_resolution() {
        let temp_dir = std::env::temp_dir().join("cp_test_overrides");
        let model_id = "toxicity";
        let model_dir = temp_dir.join(model_id);
        fs::create_dir_all(&model_dir).unwrap();

        let custom_weights = "my_custom_weights.onnx";
        fs::write(model_dir.join(custom_weights), b"fake_weights").unwrap();
        fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();
        fs::write(model_dir.join("config.json"), b"{}").unwrap();

        let config = InferenceConfig {
            model_dir: temp_dir.to_string_lossy().to_string(),
            ..Default::default()
        };

        let mut overrides = HashMap::new();
        overrides.insert(
            model_id.to_string(),
            ModelOverrideConfig {
                pool_size: Some(42),
                weights_file: Some(custom_weights.to_string()),
            },
        );

        let registry = ModelRegistry::discover_and_load(&config, &overrides);

        let info = registry
            .inventory()
            .iter()
            .find(|i| i.model_id == model_id)
            .unwrap();
        assert_eq!(info.backend, "stub"); // falls back to stub

        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
