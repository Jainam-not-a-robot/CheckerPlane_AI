//! # Graceful Process Shutdown Listener
//!
//! **Responsibility:** Listens for operating system termination signals (SIGINT, SIGTERM, Ctrl+C)
//! and initiates graceful drainage of active HTTP connections within the configured grace period.
//! **Pipeline Position:** Process lifecycle management.
//! **Latency Budget:** Bounded by `server.shutdown_grace_ms`.
//! **Failure Mode:** Infallible signal listener.

use tokio::signal;
use tracing::info;

/// Awaits OS shutdown signals across Unix and Windows platforms.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C interrupt; initiating graceful shutdown");
        }
        () = terminate => {
            info!("Received SIGTERM terminate signal; initiating graceful shutdown");
        }
    }
}
