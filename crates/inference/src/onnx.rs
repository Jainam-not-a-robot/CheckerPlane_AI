//! # Live ONNX Model Backend
//!
//! **Responsibility:** Executes tensor inference over live ONNX models using the session pool,
//! tokenizers, and dynamic class label mappings loaded from `HuggingFace` `config.json`.
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

/// How raw logits are turned into probabilities.
///
/// WHY: this is not cosmetic. A `multi_label_classification` head (e.g. the Jigsaw
/// toxicity models) is trained with BCE, so each logit is an independent probability and
/// must go through a sigmoid. Running softmax over those logits normalises independent
/// scores against each other and destroys the signal — an unmistakably toxic sentence and
/// a benign one both land in the same narrow band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Single-label head: classes are mutually exclusive.
    Softmax,
    /// Multi-label head: each class is an independent binary decision.
    Sigmoid,
}

/// Live ONNX model inference backend.
pub struct OnnxBackend {
    id: String,
    pool: SessionPool,
    tokenizer: SharedTokenizer,
    id2label: Vec<String>,
    activation: Activation,
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
        let max_sequence_length = Self::load_max_sequence_length(&model_id, &config_path);
        let tokenizer =
            SharedTokenizer::from_file(&model_id, &tokenizer_path, max_sequence_length)?;
        let id2label = Self::load_id2label(&model_id, &config_path);
        let activation = Self::load_activation(&model_id, &config_path);

        tracing::info!(
            model = %model_id,
            max_sequence_length,
            activation = ?activation,
            classes = id2label.len(),
            "loaded ONNX backend"
        );

        Ok(Self {
            id: model_id,
            pool,
            tokenizer,
            id2label,
            activation,
        })
    }

    /// Reads the model's maximum input length from `config.json`.
    ///
    /// WHY: exceeding the positional-embedding table is not a soft failure — the forward
    /// pass dies inside the embedding Gather, which surfaces as a degraded gate (a *block*
    /// under a fail-closed policy). `RoBERTa` reserves the first `padding_idx` + 1 position
    /// slots, so its usable length is `max_position_embeddings - 2`, not the raw value.
    /// (The reserved slots are the ones below `padding_idx`.)
    fn load_max_sequence_length(model_id: &str, config_path: &Path) -> usize {
        const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 512;

        let Ok(content) = fs::read_to_string(config_path) else {
            return DEFAULT_MAX_SEQUENCE_LENGTH;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
            return DEFAULT_MAX_SEQUENCE_LENGTH;
        };

        let Some(raw) = json
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
        else {
            return DEFAULT_MAX_SEQUENCE_LENGTH;
        };

        #[allow(clippy::cast_possible_truncation)]
        let raw = raw as usize;

        let model_type = json
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();

        // `RoBERTa`-family position ids are offset past the padding index.
        let usable = if matches!(model_type, "roberta" | "xlm-roberta" | "camembert") {
            raw.saturating_sub(2)
        } else {
            raw
        };

        if usable == 0 {
            tracing::warn!(
                model = %model_id,
                "config.json reported an unusable max_position_embeddings; falling back to default"
            );
            return DEFAULT_MAX_SEQUENCE_LENGTH;
        }

        usable
    }

    /// Determines the output activation from the `HuggingFace` `problem_type` field.
    fn load_activation(model_id: &str, config_path: &Path) -> Activation {
        if let Ok(content) = fs::read_to_string(config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                match json.get("problem_type").and_then(serde_json::Value::as_str) {
                    Some("multi_label_classification") => return Activation::Sigmoid,
                    Some("single_label_classification" | "regression") => {
                        return Activation::Softmax
                    }
                    _ => {}
                }
            }
        }

        tracing::debug!(
            model = %model_id,
            "no problem_type in config.json; assuming single-label (softmax)"
        );
        Activation::Softmax
    }

    /// Loads the `id2label` mapping from `config.json` if available.
    ///
    /// WHY: `HuggingFace` model exports can order classes arbitrarily in id2label (e.g. clean, noise, etc.).
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
        activation: Activation,
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

        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let outputs = if has_token_type_ids {
            let token_type_ids_tensor = Tensor::from_array((shape, encoded.token_type_ids.clone()))
                .map_err(|err| InferenceError::OnnxError {
                    model: model_id.to_string(),
                    message: format!("failed to create token_type_ids tensor: {err}"),
                })?;

            session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            ])
        }
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

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let num_classes = *out_shape.last().unwrap_or(&0) as usize;
        if num_classes == 0 || raw_data.is_empty() {
            return Err(InferenceError::ShapeMismatch {
                model: model_id.to_string(),
                expected: "[1, num_classes]".to_string(),
                actual: format!("{out_shape:?}"),
            });
        }

        let logits: Vec<f32> = raw_data.iter().take(num_classes).copied().collect();

        let probs = match activation {
            Activation::Softmax => {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = exp.iter().sum();
                exp.iter().map(|&x| x / sum).collect()
            }
            Activation::Sigmoid => logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        };

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

    fn is_multi_label(&self) -> bool {
        self.activation == Activation::Sigmoid
    }

    async fn classify(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let encoded = self.tokenizer.encode(text)?;
        let model_id = self.id.clone();
        let activation = self.activation;

        self.pool
            .run_blocking(move |session| {
                Self::run_sequence_classification(&model_id, session, &encoded, activation)
            })
            .await
    }

    async fn classify_pair(&self, a: &str, b: &str) -> Result<Vec<f32>, InferenceError> {
        let encoded = self.tokenizer.encode_pair(a, b)?;
        let model_id = self.id.clone();
        let activation = self.activation;

        self.pool
            .run_blocking(move |session| {
                Self::run_sequence_classification(&model_id, session, &encoded, activation)
            })
            .await
    }
}
