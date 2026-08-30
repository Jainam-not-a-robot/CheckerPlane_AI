//! # Tokenizer Wrapper
//!
//! **Responsibility:** Provides shared, thread-safe tokenization for Transformer architectures
//! using the native Rust `tokenizers` crate.
//! **Pipeline Position:** Invoked immediately prior to ONNX session forward passes.
//! **Latency Budget:** <1 ms per text sequence.
//! **Failure Mode:** Returns `InferenceError::TokenizerError`.

use controlplane_core::error::InferenceError;
use std::path::Path;
use std::sync::Arc;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// Container for Transformer model input tensors.
#[derive(Debug, Clone)]
pub struct EncodedInput {
    /// Token IDs.
    pub input_ids: Vec<i64>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<i64>,
    /// Token type IDs (segment IDs for pairs, 0 for single sequences).
    pub token_type_ids: Vec<i64>,
    /// Sequence length.
    pub length: usize,
}

/// Thread-safe shared tokenizer instance.
#[derive(Clone)]
pub struct SharedTokenizer {
    model_id: String,
    tokenizer: Arc<Tokenizer>,
    /// Same vocabulary, truncation disabled. Used only for *measuring* a text's true
    /// token length in `sliding_window_truncate`; the truncating `tokenizer` would cap
    /// the count at `max_sequence_length` and silently turn "keep the last N tokens"
    /// into "keep tokens N..2N from the start".
    raw_tokenizer: Arc<Tokenizer>,
    max_sequence_length: usize,
}

impl SharedTokenizer {
    /// Loads a tokenizer from a local JSON file path.
    ///
    /// # Errors
    /// Returns `InferenceError::TokenizerError` if file cannot be read or parsed.
    pub fn from_file(
        model_id: impl Into<String>,
        path: impl AsRef<Path>,
        max_sequence_length: usize,
    ) -> Result<Self, InferenceError> {
        let model_id = model_id.into();
        let path_ref = path.as_ref();
        let raw_tokenizer =
            Tokenizer::from_file(path_ref).map_err(|err| InferenceError::TokenizerError {
                model: model_id.clone(),
                message: format!(
                    "failed to load tokenizer from {}: {err}",
                    path_ref.display()
                ),
            })?;
        let mut tokenizer =
            Tokenizer::from_file(path_ref).map_err(|err| InferenceError::TokenizerError {
                model: model_id.clone(),
                message: format!(
                    "failed to load tokenizer from {}: {err}",
                    path_ref.display()
                ),
            })?;

        // WHY: this is the hard safety net for the model's positional-embedding limit.
        // Gates cap the premise themselves (see `sliding_window_truncate`), but nothing
        // capped the *hypothesis*, so a long LLM response could push the encoded pair past
        // the graph's max positions and fail the forward pass with an out-of-bounds Gather.
        // `LongestFirst` trims whichever side is longer, so a short premise is preserved
        // intact and only the oversized side gives up tokens.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_sequence_length,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|err| InferenceError::TokenizerError {
                model: model_id.clone(),
                message: format!("failed to configure truncation: {err}"),
            })?;

        Ok(Self {
            model_id,
            tokenizer: Arc::new(tokenizer),
            raw_tokenizer: Arc::new(raw_tokenizer),
            max_sequence_length,
        })
    }

    /// Returns the maximum encoded sequence length this tokenizer will emit.
    #[must_use]
    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    /// Encodes a single text sequence into model input arrays.
    ///
    /// # Errors
    /// Returns `InferenceError::TokenizerError` if encoding fails.
    pub fn encode(&self, text: &str) -> Result<EncodedInput, InferenceError> {
        let encoding =
            self.tokenizer
                .encode(text, true)
                .map_err(|err| InferenceError::TokenizerError {
                    model: self.model_id.clone(),
                    message: format!("tokenization failed: {err}"),
                })?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();
        let token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let length = input_ids.len();

        Ok(EncodedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            length,
        })
    }

    /// Encodes a (premise, hypothesis) text pair into model input arrays.
    ///
    /// # Errors
    /// Returns `InferenceError::TokenizerError` if encoding fails.
    pub fn encode_pair(&self, a: &str, b: &str) -> Result<EncodedInput, InferenceError> {
        let encoding =
            self.tokenizer
                .encode((a, b), true)
                .map_err(|err| InferenceError::TokenizerError {
                    model: self.model_id.clone(),
                    message: format!("pair tokenization failed: {err}"),
                })?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();
        let token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&t| i64::from(t))
            .collect();
        let length = input_ids.len();

        Ok(EncodedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            length,
        })
    }

    /// Returns the underlying tokenizer model ID.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Truncates a text to the last `max_tokens` tokens (sliding window).
    ///
    /// WHY: Default `HuggingFace` truncation trims from the end. For cross-encoders,
    /// this eats the hypothesis (response) first, which can invert the verdict.
    /// We cap the premise (history) on the way in, keeping the most recent context.
    ///
    /// # Errors
    /// Returns `InferenceError::TokenizerError` if encoding or decoding fails.
    pub fn sliding_window_truncate(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<String, InferenceError> {
        let encoding = self.raw_tokenizer.encode(text, false).map_err(|err| {
            InferenceError::TokenizerError {
                model: self.model_id.clone(),
                message: format!("truncation tokenization failed: {err}"),
            }
        })?;

        let ids = encoding.get_ids();
        if ids.len() <= max_tokens {
            return Ok(text.to_string());
        }

        let truncated_ids = &ids[ids.len() - max_tokens..];
        self.raw_tokenizer
            .decode(truncated_ids, true)
            .map_err(|err| InferenceError::TokenizerError {
                model: self.model_id.clone(),
                message: format!("truncation decoding failed: {err}"),
            })
    }
}
