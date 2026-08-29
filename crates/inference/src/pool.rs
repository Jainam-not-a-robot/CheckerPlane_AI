//! # ONNX Session Pool and Concurrency Manager
//!
//! **Responsibility:** Manages a bounded pool of independent ONNX Runtime `Session` instances per model,
//! enforcing thread isolation and strict `spawn_blocking` execution.
//! **Pipeline Position:** Core inference driver inside `OnnxBackend`.
//! **Latency Budget:** Acquisition wait: <1 ms under typical load; tracks `cp_inference_pool_wait_seconds`.
//! **Failure Mode:** Returns `InferenceError::PoolExhausted` if pool acquisition fails.

use controlplane_core::error::InferenceError;
use metrics::histogram;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// RAII Guard for an ONNX Session acquired from the pool.
///
/// Ensures that sessions are returned to the pool even if a panic or early return occurs.
pub struct SessionGuard {
    session: Option<Session>,
    pool: Arc<Mutex<Vec<Session>>>,
    _permit: OwnedSemaphorePermit,
}

impl SessionGuard {
    /// Borrows a mutable reference to the underlying ONNX session.
    #[must_use]
    pub fn session_mut(&mut self) -> &mut Session {
        self.session.as_mut().expect("session must be present")
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let mut lock = self.pool.lock();
            lock.push(session);
        }
    }
}

/// Thread-safe bounded pool of ONNX Runtime sessions.
#[derive(Clone)]
pub struct SessionPool {
    model_id: String,
    sessions: Arc<Mutex<Vec<Session>>>,
    semaphore: Arc<Semaphore>,
    pool_size: usize,
}

impl SessionPool {
    /// Creates and initializes a session pool by loading `pool_size` independent instances of the ONNX model.
    ///
    /// # Errors
    /// Returns `InferenceError::OnnxError` if session creation fails.
    pub fn from_file(
        model_id: impl Into<String>,
        model_path: impl AsRef<Path>,
        pool_size: usize,
        optimization_level: &str,
    ) -> Result<Self, InferenceError> {
        let model_id = model_id.into();
        let path = model_path.as_ref().to_path_buf();
        let pool_size = pool_size.max(1);

        let opt_level = match optimization_level {
            "disable" => GraphOptimizationLevel::Disable,
            "basic" => GraphOptimizationLevel::Level1,
            "extended" => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::Level3,
        };

        let mut session_vec = Vec::with_capacity(pool_size);

        for i in 0..pool_size {
            let session = Self::create_session(&model_id, &path, opt_level, i)?;
            session_vec.push(session);
        }

        Ok(Self {
            model_id,
            sessions: Arc::new(Mutex::new(session_vec)),
            semaphore: Arc::new(Semaphore::new(pool_size)),
            pool_size,
        })
    }

    fn create_session(
        model_id: &str,
        path: &PathBuf,
        opt_level: GraphOptimizationLevel,
        index: usize,
    ) -> Result<Session, InferenceError> {
        let builder = Session::builder().map_err(|err| InferenceError::OnnxError {
            model: model_id.to_string(),
            message: format!("failed to create session builder for instance {index}: {err}"),
        })?;

        // WHY: Parallelism comes from having N sessions, not from ONNX's internal thread pools.
        // Letting ONNX spawn its own threads per session oversubscribes the CPU and destroys tail latency.
        let builder = builder
            .with_intra_threads(1)
            .map_err(|err| InferenceError::OnnxError {
                model: model_id.to_string(),
                message: format!("failed to set intra_op_num_threads: {err}"),
            })?
            .with_inter_threads(1)
            .map_err(|err| InferenceError::OnnxError {
                model: model_id.to_string(),
                message: format!("failed to set inter_op_num_threads: {err}"),
            })?
            .with_optimization_level(opt_level)
            .map_err(|err| InferenceError::OnnxError {
                model: model_id.to_string(),
                message: format!("failed to set optimization level: {err}"),
            })?;

        let session = builder
            .commit_from_file(path)
            .map_err(|err| InferenceError::OnnxError {
                model: model_id.to_string(),
                message: format!("failed to load model from {}: {err}", path.display()),
            })?;

        Ok(session)
    }

    /// Acquires a session permit and returns an RAII guard containing an ONNX session.
    ///
    /// # Errors
    /// Returns `InferenceError::PoolExhausted` if acquisition fails.
    pub async fn acquire(&self) -> Result<SessionGuard, InferenceError> {
        let wait_start = Instant::now();

        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|err| InferenceError::PoolExhausted {
                model: self.model_id.clone(),
                message: format!("semaphore acquisition failed: {err}"),
            })?;

        let wait_duration = wait_start.elapsed();
        histogram!("cp_inference_pool_wait_seconds", "model" => self.model_id.clone())
            .record(wait_duration.as_secs_f64());

        let session = {
            let mut lock = self.sessions.lock();
            lock.pop().ok_or_else(|| InferenceError::PoolExhausted {
                model: self.model_id.clone(),
                message: "session pool was empty despite semaphore permit".to_string(),
            })?
        };

        Ok(SessionGuard {
            session: Some(session),
            pool: self.sessions.clone(),
            _permit: permit,
        })
    }

    /// Executes an inference closure strictly on a blocking background worker thread.
    ///
    /// # Errors
    /// Returns `InferenceError` if acquisition, execution, or join fails.
    pub async fn run_blocking<F, R>(&self, f: F) -> Result<R, InferenceError>
    where
        F: FnOnce(&mut Session) -> Result<R, InferenceError> + Send + 'static,
        R: Send + 'static,
    {
        let mut guard = self.acquire().await?;

        // WHY: `ort::Session::run` is a blocking call. Calling it directly inside an async fn body
        // would park Tokio worker threads inside ONNX, starving the async reactor and serializing parallel gates.
        tokio::task::spawn_blocking(move || {
            let session = guard.session_mut();
            f(session)
        })
        .await
        .map_err(|err| InferenceError::JoinError(err.to_string()))?
    }

    /// Returns the total configured pool size.
    #[must_use]
    pub const fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Returns the model identifier.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}
