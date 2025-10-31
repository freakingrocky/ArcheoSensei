# syntax=docker/dockerfile:1.7
FROM rust:1.79 AS builder
WORKDIR /app
COPY ./ ./Rust-backend
WORKDIR /app/Rust-backend
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/Rust-backend/target/release/rust-backend /usr/local/bin/rust-backend
ENV BACKEND_PORT=8000
EXPOSE 8000
CMD ["/usr/local/bin/rust-backend"]
