//! # Google Gemini API Client
//!
//! **Responsibility:** Implements generation using Google's Generative Language API (e.g. `gemini-2.0-flash`).
//! **Pipeline Position:** Live LLM generation backend when `llm.backend = "gemini"`.
//! **Latency Budget:** Bounded by `llm.timeout_ms` (15,000 ms).
//! **Failure Mode:** Returns `LlmError::Http`, `LlmError::ProviderError`, or `LlmError::Timeout`.

use crate::{LlmBackend, LlmResponse};
use controlplane_core::config::LlmConfig;
use controlplane_core::error::LlmError;
use reqwest::Client;
use serde_json::json;
use std::time::{Duration, Instant};

/// Google Gemini API generation client.
pub struct GeminiClient {
    client: Client,
    model: String,
    api_key: String,
    max_output_tokens: usize,
    timeout: Duration,
}

impl GeminiClient {
    /// Constructs a new Gemini client from configuration.
    ///
    /// # Errors
    /// Returns `LlmError::MissingApiKey` if no API key is provided.
    pub fn new(config: &LlmConfig) -> Result<Self, LlmError> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("CP_LLM__API_KEY").ok())
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| LlmError::MissingApiKey("gemini".to_string()))?;

        let timeout = Duration::from_millis(config.timeout_ms);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|err| LlmError::Http(err.to_string()))?;

        Ok(Self {
            client,
            model: config.model.clone(),
            api_key,
            max_output_tokens: config.max_output_tokens,
            timeout,
        })
    }
}

#[async_trait::async_trait]
impl LlmBackend for GeminiClient {
    async fn generate(&self, query: &str, history: Option<&str>) -> Result<LlmResponse, LlmError> {
        let start = Instant::now();

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let mut contents = Vec::new();

        if let Some(hist) = history {
            contents.push(json!({
                "role": "user",
                "parts": [{ "text": format!("Prior conversation context:\n{hist}") }]
            }));
            contents.push(json!({
                "role": "model",
                "parts": [{ "text": "Acknowledged. I will consider the prior context in my response." }]
            }));
        }

        contents.push(json!({
            "role": "user",
            "parts": [{ "text": query }]
        }));

        let payload = json!({
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_output_tokens,
                "temperature": 0.2
            }
        });

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|err| {
                if err.is_timeout() {
                    LlmError::Timeout(u64::try_from(self.timeout.as_millis()).unwrap_or(u64::MAX))
                } else {
                    LlmError::Http(err.to_string())
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            return Err(LlmError::ProviderError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|err| LlmError::InvalidResponse(err.to_string()))?;

        let text = body
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                LlmError::InvalidResponse("missing candidate text in Gemini response".to_string())
            })?
            .to_string();

        let prompt_tokens = usize::try_from(
            body.pointer("/usageMetadata/promptTokenCount")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(0);

        let completion_tokens = usize::try_from(
            body.pointer("/usageMetadata/candidatesTokenCount")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .unwrap_or(0);

        Ok(LlmResponse {
            text,
            prompt_tokens,
            completion_tokens,
            latency: start.elapsed(),
        })
    }
}
