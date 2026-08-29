//! # Configuration Management
//!
//! **Responsibility:** Parses, validates, and provides layered access to all server, pipeline,
//! inference, gate, LLM, and telemetry configuration parameters.
//! **Pipeline Position:** Loaded at process startup in `main.rs` and passed by reference to all components.
//! **Latency Budget:** One-time startup cost (<5 ms).
//! **Failure Mode:** Aborts startup if invalid config is supplied; all fields provide sensible defaults.

use crate::error::ConfigError;
use crate::verdict::FailurePolicy;
use figment::{
    providers::{Env, Format, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Root configuration container for ControlPlane Checker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    /// HTTP server configuration.
    #[serde(default)]
    pub server: ServerConfig,
    /// Pipeline coordination and latency budget configuration.
    #[serde(default)]
    pub pipeline: PipelineConfig,
    /// Inference session pool and ONNX runtime configuration.
    #[serde(default)]
    pub inference: InferenceConfig,
    /// Individual gate tuning configurations.
    #[serde(default)]
    pub gates: GatesConfig,
    /// LLM client and mock generator configuration.
    #[serde(default)]
    pub llm: LlmConfig,
    /// Tracing and telemetry metrics configuration.
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// Model-specific overrides (pool_size, weights_file).
    #[serde(default)]
    pub models: std::collections::HashMap<String, ModelOverrideConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            pipeline: PipelineConfig::default(),
            inference: InferenceConfig::default(),
            gates: GatesConfig::default(),
            llm: LlmConfig::default(),
            telemetry: TelemetryConfig::default(),
            models: std::collections::HashMap::new(),
        }
    }
}

/// Model-specific configuration overrides (e.g., [models.toxicity]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelOverrideConfig {
    /// Override for the number of pooled sessions.
    pub pool_size: Option<usize>,
    /// Override for the weights filename (e.g. "model_optimized_quantized.onnx").
    pub weights_file: Option<String>,
}

impl AppConfig {
    /// Loads configuration hierarchically from default TOML, optional local override TOML,
    /// and `CP_` prefixed environment variables.
    ///
    /// # Errors
    /// Returns `ConfigError` if configuration syntax or data types are invalid.
    pub fn load(custom_path: Option<&str>) -> Result<Self, ConfigError> {
        let mut figment = Figment::new();

        // 1. Load built-in default TOML if present
        let default_toml = Path::new("config/default.toml");
        if default_toml.exists() {
            figment = figment.merge(Toml::file(default_toml));
        }

        // 2. Load optional custom path or local.toml
        if let Some(path) = custom_path {
            figment = figment.merge(Toml::file(path));
        } else {
            let local_toml = Path::new("config/local.toml");
            if local_toml.exists() {
                figment = figment.merge(Toml::file(local_toml));
            }
        }

        // 3. Merge environment variables with CP_ prefix and double-underscore nesting (e.g. CP_LLM__API_KEY)
        figment = figment.merge(Env::prefixed("CP_").split("__"));

        figment
            .extract::<Self>()
            .map_err(|err| ConfigError::InvalidConfig(err.to_string()))
    }
}

/// HTTP server settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Socket binding address (e.g. "0.0.0.0:8080").
    #[serde(default = "default_bind")]
    pub bind: String,
    /// HTTP request timeout in milliseconds.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    /// Maximum allowed request body size in bytes.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Maximum grace period in milliseconds for server shutdown.
    #[serde(default = "default_shutdown_grace_ms")]
    pub shutdown_grace_ms: u64,
}

fn default_bind() -> String {
    "0.0.0.0:8080".to_string()
}
const fn default_request_timeout_ms() -> u64 {
    5000
}
const fn default_max_body_bytes() -> usize {
    65536
}
const fn default_shutdown_grace_ms() -> u64 {
    10000
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            request_timeout_ms: default_request_timeout_ms(),
            max_body_bytes: default_max_body_bytes(),
            shutdown_grace_ms: default_shutdown_grace_ms(),
        }
    }
}

/// Pipeline coordination and overall budget settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Enables user-facing `Clarify` verdicts. In v1, clarify is mapped to Block unless true.
    #[serde(default)]
    pub clarify_enabled: bool,
    /// Wall-clock timeout budget for the entire Tier 1 input gate fan-out in milliseconds.
    #[serde(default = "default_input_budget_ms")]
    pub input_gate_budget_ms: u64,
    /// Wall-clock timeout budget for the Tier 2 output gate fan-out in milliseconds.
    #[serde(default = "default_output_budget_ms")]
    pub output_gate_budget_ms: u64,
}

const fn default_input_budget_ms() -> u64 {
    120
}
const fn default_output_budget_ms() -> u64 {
    200
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            clarify_enabled: false,
            input_gate_budget_ms: default_input_budget_ms(),
            output_gate_budget_ms: default_output_budget_ms(),
        }
    }
}

/// Inference session pool and ONNX engine settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Number of independent ONNX session instances pooled per model.
    #[serde(default = "default_pool_size")]
    pub pool_size_per_model: usize,
    /// Internal intra-op threads per session (must be 1 for multi-session pool).
    #[serde(default = "default_threads")]
    pub intra_op_threads: usize,
    /// Internal inter-op threads per session (must be 1 for multi-session pool).
    #[serde(default = "default_threads")]
    pub inter_op_threads: usize,
    /// Graph optimization level string ("all", "basic", "extended", "disable").
    #[serde(default = "default_optimization_level")]
    pub graph_optimization_level: String,
    /// When true, forces stub backend even if ONNX weights exist on disk.
    #[serde(default)]
    pub force_stub: bool,
    /// Directory where exported model folders are located.
    #[serde(default = "default_model_dir")]
    pub model_dir: String,
}

const fn default_pool_size() -> usize {
    4
}
const fn default_threads() -> usize {
    1
}
fn default_optimization_level() -> String {
    "all".to_string()
}
fn default_model_dir() -> String {
    "models".to_string()
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            pool_size_per_model: default_pool_size(),
            intra_op_threads: default_threads(),
            inter_op_threads: default_threads(),
            graph_optimization_level: default_optimization_level(),
            force_stub: false,
            model_dir: default_model_dir(),
        }
    }
}

/// Container for all gate configurations.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GatesConfig {
    /// Heuristic prefilter configuration.
    #[serde(default)]
    pub prefilter: PrefilterConfig,
    /// Coherence classifier configuration.
    #[serde(default)]
    pub coherence: CoherenceConfig,
    /// PII detector configuration.
    #[serde(default)]
    pub pii: PiiConfig,
    /// Toxicity classifier configuration.
    #[serde(default)]
    pub toxicity: ToxicityConfig,
    /// Intent and prompt injection classifier configuration.
    #[serde(default)]
    pub intent: IntentConfig,
    /// Grounding and hallucination evaluator configuration.
    #[serde(default)]
    pub grounding: GroundingConfig,
    /// Output relevance evaluator configuration.
    #[serde(default)]
    pub relevance: RelevanceConfig,
}

/// Heuristic prefilter settings (Tier 0).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefilterConfig {
    /// Whether prefilter is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Minimum input character length.
    #[serde(default = "default_min_chars")]
    pub min_chars: usize,
    /// Maximum input character length.
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    /// Minimum expected Latin/ASCII script character ratio.
    #[serde(default = "default_min_script_ratio")]
    pub min_script_ratio: f32,
    /// Maximum Shannon character entropy threshold.
    #[serde(default = "default_max_entropy")]
    pub max_char_entropy: f32,
}

const fn default_true() -> bool {
    true
}
const fn default_min_chars() -> usize {
    2
}
const fn default_max_chars() -> usize {
    8000
}
const fn default_min_script_ratio() -> f32 {
    0.5
}
const fn default_max_entropy() -> f32 {
    4.2
}

impl Default for PrefilterConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            min_chars: default_min_chars(),
            max_chars: default_max_chars(),
            min_script_ratio: default_min_script_ratio(),
            max_char_entropy: default_max_entropy(),
        }
    }
}

/// Coherence / gibberish gate settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Whether coherence gate is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Single tuning dial: 0.0 (permissive) to 1.0 (paranoid).
    /// Effective threshold is `1.0 - strictness`.
    #[serde(default = "default_strictness")]
    pub strictness: f32,
    /// Weight assigned to the noise class probability.
    #[serde(default = "default_weight_1_0")]
    pub weight_noise: f32,
    /// Weight assigned to the word salad class probability.
    #[serde(default = "default_weight_1_0")]
    pub weight_word_salad: f32,
    /// Weight assigned to the mild gibberish class probability.
    #[serde(default = "default_weight_mild")]
    pub weight_mild_gibberish: f32,
    /// Minimum token count to run the classifier.
    #[serde(default = "default_min_tokens")]
    pub min_tokens_for_model: usize,
    /// Maximum token count above which model inference is bypassed.
    #[serde(default = "default_max_tokens")]
    pub max_tokens_for_model: usize,
    /// Execution timeout in milliseconds.
    #[serde(default = "default_coherence_timeout")]
    pub timeout_ms: u64,
    /// Fallback policy on error or timeout ("open" or "closed").
    #[serde(default = "default_open_policy")]
    pub failure_policy: FailurePolicy,
}

const fn default_strictness() -> f32 {
    0.55
}
const fn default_weight_1_0() -> f32 {
    1.0
}
const fn default_weight_mild() -> f32 {
    0.35
}
const fn default_min_tokens() -> usize {
    3
}
const fn default_max_tokens() -> usize {
    64
}
const fn default_coherence_timeout() -> u64 {
    40
}
const fn default_open_policy() -> FailurePolicy {
    FailurePolicy::Open
}
const fn default_closed_policy() -> FailurePolicy {
    FailurePolicy::Closed
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            strictness: default_strictness(),
            weight_noise: default_weight_1_0(),
            weight_word_salad: default_weight_1_0(),
            weight_mild_gibberish: default_weight_mild(),
            min_tokens_for_model: default_min_tokens(),
            max_tokens_for_model: default_max_tokens(),
            timeout_ms: default_coherence_timeout(),
            failure_policy: default_open_policy(),
        }
    }
}

/// PII detection settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiiConfig {
    /// Whether PII detection is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Identifier classes that trigger a request block.
    #[serde(default = "default_block_classes")]
    pub block_classes: Vec<String>,
    /// Execution timeout in milliseconds.
    #[serde(default = "default_pii_timeout")]
    pub timeout_ms: u64,
    /// Fallback policy on error or timeout.
    #[serde(default = "default_closed_policy")]
    pub failure_policy: FailurePolicy,
}

fn default_block_classes() -> Vec<String> {
    vec![
        "payment_card".to_string(),
        "aadhaar".to_string(),
        "pan".to_string(),
        "credential".to_string(),
    ]
}

const fn default_pii_timeout() -> u64 {
    50
}

impl Default for PiiConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            block_classes: default_block_classes(),
            timeout_ms: default_pii_timeout(),
            failure_policy: default_closed_policy(),
        }
    }
}

/// Toxicity gate settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToxicityConfig {
    /// Whether toxicity gate is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Threshold probability for blocking toxic queries.
    #[serde(default = "default_toxicity_threshold")]
    pub threshold: f32,
    /// Execution timeout in milliseconds.
    #[serde(default = "default_toxicity_timeout")]
    pub timeout_ms: u64,
    /// Fallback policy on error or timeout.
    #[serde(default = "default_closed_policy")]
    pub failure_policy: FailurePolicy,
}

const fn default_toxicity_threshold() -> f32 {
    0.80
}
const fn default_toxicity_timeout() -> u64 {
    40
}

impl Default for ToxicityConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            threshold: default_toxicity_threshold(),
            timeout_ms: default_toxicity_timeout(),
            failure_policy: default_closed_policy(),
        }
    }
}

/// Intent and prompt injection gate settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentConfig {
    /// Whether intent gate is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Threshold probability for blocking prompt injection attacks.
    #[serde(default = "default_intent_threshold")]
    pub threshold: f32,
    /// Execution timeout in milliseconds.
    #[serde(default = "default_intent_timeout")]
    pub timeout_ms: u64,
    /// Fallback policy on error or timeout.
    #[serde(default = "default_closed_policy")]
    pub failure_policy: FailurePolicy,
}

const fn default_intent_threshold() -> f32 {
    0.90
}
const fn default_intent_timeout() -> u64 {
    40
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            threshold: default_intent_threshold(),
            timeout_ms: default_intent_timeout(),
            failure_policy: default_closed_policy(),
        }
    }
}

/// Grounding and hallucination gate settings (Tier 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundingConfig {
    /// Whether grounding evaluation is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Threshold for blocking ungrounded responses.
    #[serde(default = "default_grounding_threshold")]
    pub threshold: f32,
    /// Weight of the contradiction logit for the grounding score.
    #[serde(default = "default_weight_contradiction")]
    pub weight_contradiction: f32,
    /// Weight of the neutral logit for the grounding score.
    #[serde(default = "default_weight_neutral")]
    pub weight_neutral: f32,
    /// Max tokens for the premise before sliding-window truncation.
    #[serde(default = "default_grounding_max_premise")]
    pub max_premise_tokens: usize,
    /// Execution timeout in milliseconds.
    #[serde(default = "default_grounding_timeout")]
    pub timeout_ms: u64,
    /// Fallback policy on error or timeout.
    #[serde(default = "default_closed_policy")]
    pub failure_policy: FailurePolicy,
}

const fn default_grounding_threshold() -> f32 {
    0.50
}
const fn default_weight_contradiction() -> f32 {
    1.0
}
const fn default_weight_neutral() -> f32 {
    0.45
}
const fn default_grounding_max_premise() -> usize {
    256
}
const fn default_grounding_timeout() -> u64 {
    150
}

impl Default for GroundingConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            threshold: default_grounding_threshold(),
            weight_contradiction: default_weight_contradiction(),
            weight_neutral: default_weight_neutral(),
            max_premise_tokens: default_grounding_max_premise(),
            timeout_ms: default_grounding_timeout(),
            failure_policy: default_closed_policy(),
        }
    }
}

/// Output relevance evaluator settings (Tier 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelevanceConfig {
    /// Whether relevance evaluation is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Threshold for allowing relevant responses (fail if below).
    #[serde(default = "default_relevance_threshold")]
    pub threshold: f32,
    /// Max tokens for the premise before sliding-window truncation.
    #[serde(default = "default_relevance_max_premise")]
    pub max_premise_tokens: usize,
    /// Execution timeout in milliseconds.
    #[serde(default = "default_relevance_timeout")]
    pub timeout_ms: u64,
    /// Fallback policy on error or timeout.
    #[serde(default = "default_open_policy")]
    pub failure_policy: FailurePolicy,
}

const fn default_relevance_threshold() -> f32 {
    0.40
}
const fn default_relevance_max_premise() -> usize {
    64
}
const fn default_relevance_timeout() -> u64 {
    80
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            threshold: default_relevance_threshold(),
            max_premise_tokens: default_relevance_max_premise(),
            timeout_ms: default_relevance_timeout(),
            failure_policy: default_open_policy(),
        }
    }
}

/// LLM client and mock configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmConfig {
    /// LLM backend implementation ("mock" or "gemini").
    #[serde(default = "default_llm_backend")]
    pub backend: String,
    /// LLM model name (e.g. "gemini-2.0-flash").
    #[serde(default = "default_llm_model")]
    pub model: String,
    /// Timeout for LLM generation call in milliseconds.
    #[serde(default = "default_llm_timeout")]
    pub timeout_ms: u64,
    /// Maximum tokens generated in LLM response.
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: usize,
    /// API key for external LLM provider.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Settings for the deterministic mock LLM backend.
    #[serde(default)]
    pub mock: MockLlmConfig,
}

fn default_llm_backend() -> String {
    "mock".to_string()
}
fn default_llm_model() -> String {
    "gemini-2.0-flash".to_string()
}
const fn default_llm_timeout() -> u64 {
    15000
}
const fn default_max_output_tokens() -> usize {
    1024
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            backend: default_llm_backend(),
            model: default_llm_model(),
            timeout_ms: default_llm_timeout(),
            max_output_tokens: default_max_output_tokens(),
            api_key: None,
            mock: MockLlmConfig::default(),
        }
    }
}

/// Deterministic mock LLM parameters for reproducible load testing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MockLlmConfig {
    /// Mean latency in milliseconds for simulated LLM generation.
    #[serde(default = "default_mock_latency_mean")]
    pub latency_mean_ms: u64,
    /// Standard deviation in milliseconds for simulated latency.
    #[serde(default = "default_mock_latency_stddev")]
    pub latency_stddev_ms: u64,
    /// Rate in [0.0, 1.0] of synthetic hallucinations injected.
    #[serde(default = "default_mock_hallucination_rate")]
    pub hallucination_rate: f32,
}

const fn default_mock_latency_mean() -> u64 {
    700
}
const fn default_mock_latency_stddev() -> u64 {
    150
}
const fn default_mock_hallucination_rate() -> f32 {
    0.15
}

impl Default for MockLlmConfig {
    fn default() -> Self {
        Self {
            latency_mean_ms: default_mock_latency_mean(),
            latency_stddev_ms: default_mock_latency_stddev(),
            hallucination_rate: default_mock_hallucination_rate(),
        }
    }
}

/// Telemetry, logging, and metrics configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Log output format ("json" or "pretty").
    #[serde(default = "default_log_format")]
    pub log_format: String,
    /// Log filtering level ("trace", "debug", "info", "warn", "error").
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Whether Prometheus metrics scraping endpoint is enabled.
    #[serde(default = "default_true")]
    pub metrics_enabled: bool,
}

fn default_log_format() -> String {
    "json".to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_format: default_log_format(),
            log_level: default_log_level(),
            metrics_enabled: default_true(),
        }
    }
}
