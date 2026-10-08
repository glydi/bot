//! GLYDI: the wiring. Config, the [`App`] that instantiates every crate
//! and connects them, the environment check, and the observation tee that
//! `--record` uses. The binary in `main.rs` is a thin clap front over
//! these; tests build the [`App`] with mock parts (`--features mock`).

#![forbid(unsafe_code)]

pub mod app;
pub mod check;
pub mod config;
pub mod health;
pub mod school;
pub mod tee;
pub mod text;

pub use app::{App, Parts, UiParts};
pub use config::{Config, Tts};
