# Agent Guidelines

- Prefer the Rust backend under `Rust-backend/` when implementing or debugging server functionality.
- Keep logging structured via `tracing` and prefer middleware-based request logging instead of ad-hoc prints.
- Run `cargo fmt` and `cargo test` inside `Rust-backend/` for changes to the Rust services when possible.
- Update this file in future PRs if you add new contributor workflows that others should know about.
