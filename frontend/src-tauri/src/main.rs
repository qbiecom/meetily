#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use log::{LevelFilter, Log, Metadata, Record};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

struct StartupFileLogger {
    file: Mutex<File>,
    level: LevelFilter,
}

impl Log for StartupFileLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let line = format!(
            "{} [{}] {} - {}\n",
            timestamp,
            record.level(),
            record.target(),
            record.args()
        );

        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }

        #[cfg(debug_assertions)]
        eprint!("{}", line);
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

fn log_file_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("meetily")
        .join("logs")
        .join("meetily.log")
}

fn init_logging() -> Option<PathBuf> {
    let path = log_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok()?;

    let logger = StartupFileLogger {
        file: Mutex::new(file),
        level: LevelFilter::Error,
    };

    if log::set_boxed_logger(Box::new(logger)).is_ok() {
        log::set_max_level(LevelFilter::Error);
        Some(path)
    } else {
        None
    }
}

fn install_panic_logger() {
    std::panic::set_hook(Box::new(|panic_info| {
        log::error!("Application panic: {}", panic_info);
        log::error!("Backtrace: {}", std::backtrace::Backtrace::force_capture());
    }));
}

fn main() {
    // SAFETY: This process-wide logging default is set at startup before
    // other application threads are spawned. The key/value are static UTF-8
    // strings without interior NULs.
    unsafe {
        std::env::set_var("RUST_LOG", "error");
    }
    let log_path = init_logging();
    install_panic_logger();

    // Async logger will be initialized lazily when first needed (after Tauri runtime starts)
    log::info!("Starting application...");
    if let Some(path) = log_path {
        log::info!("Writing application log to {}", path.display());
    } else {
        log::warn!("File logging could not be initialized");
    }
    app_lib::run();
}
