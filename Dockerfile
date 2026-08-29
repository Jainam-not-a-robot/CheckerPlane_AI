# ==============================================================================
# Stage 1: Build & Compile
# ==============================================================================
FROM rust:1.83-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace configuration and manifests for dependency caching
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/core/Cargo.toml crates/core/Cargo.toml
COPY crates/inference/Cargo.toml crates/inference/Cargo.toml
COPY crates/gates/Cargo.toml crates/gates/Cargo.toml
COPY crates/llm/Cargo.toml crates/llm/Cargo.toml
COPY crates/router/Cargo.toml crates/router/Cargo.toml

# Create dummy source files for layer caching
RUN mkdir -p crates/core/src && echo "pub fn dummy() {}" > crates/core/src/lib.rs && \
    mkdir -p crates/inference/src && echo "pub fn dummy() {}" > crates/inference/src/lib.rs && \
    mkdir -p crates/gates/src && echo "pub fn dummy() {}" > crates/gates/src/lib.rs && \
    mkdir -p crates/llm/src && echo "pub fn dummy() {}" > crates/llm/src/lib.rs && \
    mkdir -p crates/router/src && echo "fn main() {}" > crates/router/src/main.rs && \
    touch crates/router/src/lib.rs

# Fetch and build dependencies
RUN cargo build --release --locked --bin controlplane-router || true

# Copy real source files and configuration
COPY config/ config/
COPY crates/ crates/
COPY tests/ tests/
COPY benches/ benches/

# Touch source files to invalidate dummy build timestamps
RUN touch crates/core/src/lib.rs \
    crates/inference/src/lib.rs \
    crates/gates/src/lib.rs \
    crates/llm/src/lib.rs \
    crates/router/src/main.rs \
    crates/router/src/lib.rs

# Build final release binary
RUN cargo build --release --locked --bin controlplane-router

# ==============================================================================
# Stage 2: Runtime Container
# ==============================================================================
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime certificates and networking utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root service user
RUN groupadd -g 10001 controlplane && \
    useradd -u 10001 -g controlplane -s /bin/bash -m controlplane

# Copy binary and configuration templates
COPY --from=builder /build/target/release/controlplane-router /usr/local/bin/controlplane-router
COPY config/default.toml config/default.toml

# Prepare models mount directory with proper permissions
# NOTE: Model weights are NEVER copied into the image; they are volume-mounted at runtime.
RUN mkdir -p models && chown -R controlplane:controlplane /app

USER controlplane:controlplane

EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

ENTRYPOINT ["controlplane-router"]
