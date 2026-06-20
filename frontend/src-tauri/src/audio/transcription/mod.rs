// audio/transcription/mod.rs
//
// Transcription module: Provider abstraction, engine management, and worker pool.

pub mod engine;
pub mod parakeet_provider;
pub mod provider;
pub mod sherpa_provider;
pub mod whisper_provider;
pub mod worker;

// Re-export commonly used types
pub use engine::{
    TranscriptionEngine, get_or_init_transcription_engine, get_or_init_whisper,
    validate_transcription_model_ready,
};
pub use parakeet_provider::ParakeetProvider;
pub use provider::{TranscriptResult, TranscriptionError, TranscriptionProvider};
pub use sherpa_provider::SherpaProvider;
pub use whisper_provider::WhisperProvider;
pub use worker::{TranscriptUpdate, reset_speech_detected_flag, start_transcription_task};
