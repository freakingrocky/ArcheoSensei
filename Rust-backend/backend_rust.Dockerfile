# syntax=docker/dockerfile:1.7
FROM rust:1.79 AS builder

WORKDIR /app

# Only copy the backend folder into the container
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Switch to nightly
RUN rustup toolchain install nightly
RUN rustup default nightly
ENV CARGO_UNSTABLE_EDITION2024=1

# Build
RUN cargo build --release

# ---- Runtime image ----
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rust-backend /usr/local/bin/rust-backend

ENV BACKEND_PORT=8000
EXPOSE 8000
CMD ["/usr/local/bin/rust-backend"]
