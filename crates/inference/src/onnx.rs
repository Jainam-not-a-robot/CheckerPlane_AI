//! # Live ONNX Model Backend
//!
//! **Responsibility:** Executes tensor inference over live ONNX models using the session pool,
//! tokenizers, and dynamic class label mappings loaded from HuggingFace `config.json`.
//! **Pipeline Position:** Invoked by gates during live model evaluation.
//! **Latency Budget:** Forward pass: 5–30 ms.
//! **Failure Mode:** Returns `Result<_, InferenceError>`.

use crate::backend::ModelBackend;
use crate::pool::SessionPool;
use crate::tokenizer::{EncodedInput, SharedTokenizer};
use controlplane_core::error::InferenceError;
use ort::value::Tensor;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Live ONNX model inference backend.
pub struct OnnxBackend {
    id: String,
    pool: SessionPool,
    tokenizer: SharedTokenizer,
    id2label: Vec<String>,
}

impl OnnxBackend {
    /// Loads an ONNX backend from a directory containing `model.onnx`, `tokenizer.json`, and optional `config.json`.
    ///
    /// # Errors
    /// Returns `InferenceError` if model or tokenizer cannot be loaded.
    pub fn from_dir(
        model_id: impl Into<String>,
        dir: impl AsRef<Path>,
        pool_size: usize,
        weights_file: &str,
        optimization_level: &str,
    ) -> Result<Self, InferenceError> {
        let model_id = model_id.into();
        let dir_path = dir.as_ref();

        let model_path = dir_path.join(weights_file);
        let tokenizer_path = dir_path.join("tokenizer.json");
        let config_path = dir_path.join("config.json");

        if !model_path.exists() || !tokenizer_path.exists() {
            return Err(InferenceError::FilesMissing {
                model: model_id.clone(),
                reason: format!(
                    "directory {} missing {} or tokenizer.json",
                    dir_path.display(),
                    weights_file
                ),
            });
        }

        let pool = SessionPool::from_file(&model_id, &model_path, pool_size, optimization_level)?;
        let tokenizer = SharedTokenizer::from_file(&model_id, &tokenizer_path)?;
        let id2label = Self::load_id2label(&model_id, &config_path);

        Ok(Self {
            id: model_id,
            pool,
            tokenizer,
            id2label,
        })
    }

    /// Loads the `id2label` mapping from `config.json` if available.
    ///
    /// WHY: HuggingFace model exports can order classes arbitrarily in id2label (e.g. clean, noise, etc.).
    /// Reading config.json dynamically prevents catastrophic classification index misalignment.
    fn load_id2label(model_id: &str, config_path: &Path) -> Vec<String> {
        if let Ok(content) = fs::read_to_string(config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(map) = json.get("id2label").and_then(|v| v.as_object()) {
                    let mut btree: BTreeMap<usize, String> = BTreeMap::new();
                    for (k, v) in map {
                        if let (Ok(idx), Some(label)) = (k.parse::<usize>(), v.as_str()) {
                            btree.insert(idx, label.to_string());
                        }
                    }
                    if !btree.is_empty() {
                        return btree.into_values().collect();
                    }
                }
            }
        }

        tracing::debug!(model = %model_id, "no id2label found in config.json; using default indices");
        Vec::new()
    }

    /// Runs a sequence classification forward pass on encoded input.
    fn run_sequence_classification(
        model_id: &str,
        session: &mut ort::session::Session,
        encoded: &EncodedInput,
    ) -> Result<Vec<f32>, InferenceError> {
        let seq_len = encoded.length;
        let shape = [1, seq_len];

        let input_ids_tensor =
            Tensor::from_array((shape, encoded.input_ids.clone())).map_err(|err| {
                InferenceError::OnnxError {
                    model: model_id.to_string(),
                    message: format!("failed to create input_ids tensor: {err}"),
                }
            })?;

        let attention_mask_tensor = Tensor::from_array((shape, encoded.attention_mask.clone()))
            .map_err(|err| InferenceError::OnnxError {
                model: model_id.to_string(),
                message: format!("failed to create attention_mask tensor: {err}"),
            })?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            ])
            .map_err(|err| InferenceError::OnnxError {
                model: model_id.to_string(),
                message: format!("session run failed: {err}"),
            })?;

        // Extract logits tensor: shape should be [1, num_classes]
        let logits_val = outputs
            .get("logits")
            .or_else(|| outputs.get("output_0"))
            .ok_or_else(|| InferenceError::ShapeMismatch {
                model: model_id.to_string(),
                expected: "logits or output_0".to_string(),
                actual: "none".to_string(),
            })?;

        let (out_shape, raw_data) =
            logits_val
                .try_extract_tensor::<f32>()
                .map_err(|err| InferenceError::OnnxError {
                    model: model_id.to_string(),
                    message: format!("failed to extract logits: {err}"),
                })?;

        let num_classes = *out_shape.last().unwrap_or(&0) as usize;
        if num_classes == 0 || raw_data.is_empty() {
            return Err(InferenceError::ShapeMismatch {
                model: model_id.to_string(),
                expected: "[1, num_classes]".to_string(),
                actual: format!("{out_shape:?}"),
            });
        }

        let logits: Vec<f32> = raw_data.iter().take(num_classes).copied().collect();

        // Softmax normalization
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&x| x / sum).collect();

        Ok(probs)
    }
}

#[async_trait::async_trait]
impl ModelBackend for OnnxBackend {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_live(&self) -> bool {
        true
    }

    fn class_names(&self) -> Vec<String> {
        self.id2label.clone()
    }

    fn sliding_window_truncate(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<String, InferenceError> {
        self.tokenizer.sliding_window_truncate(text, max_tokens)
    }

    async fn classify(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let encoded = self.tokenizer.encode(text)?;
        let model_id = self.id.clone();

        self.pool
            .run_blocking(move |session| {
                Self::run_sequence_classification(&model_id, session, &encoded)
            })
            .await
    }

    async fn classify_pair(&self, a: &str, b: &str) -> Result<Vec<f32>, InferenceError> {
        let encoded = self.tokenizer.encode_pair(a, b)?;
        let model_id = self.id.clone();

        self.pool
            .run_blocking(move |session| {
                Self::run_sequence_classification(&model_id, session, &encoded)
            })
            .await
    }
}
