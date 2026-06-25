// Audio capture implementations module

pub mod backend_config;
pub mod microphone;
pub mod system;

#[cfg(target_os = "macos")]
pub mod core_audio;

// Re-export capture functionality
pub use system::{
    SystemAudioCapture, SystemAudioStream, check_system_audio_permissions,
    list_system_audio_devices, start_system_audio_capture,
};

#[cfg(target_os = "macos")]
pub use core_audio::{CoreAudioCapture, CoreAudioStream};

// Re-export backend configuration
pub use backend_config::{
    AudioCaptureBackend, BACKEND_CONFIG, BackendConfig, get_available_backends,
    get_current_backend, set_current_backend,
};
