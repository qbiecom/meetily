#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use log;
use env_logger;

fn main() {
    // SAFETY: This process-wide logging default is set at startup before
    // other application threads are spawned. The key/value are static UTF-8
    // strings without interior NULs.
    unsafe {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // Async logger will be initialized lazily when first needed (after Tauri runtime starts)
    log::info!("Starting application...");
    app_lib::run();
}
