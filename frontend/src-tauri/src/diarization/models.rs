// diarization/models.rs
//
// Model location and download for speaker identification.
// Mirrors the parakeet_engine download pattern: stream from a stable URL
// into <app_data>/models/diarization/, .tmp + rename for atomicity,
// progress emitted as Tauri events.

use futures_util::StreamExt;
use serde::Serialize;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, Manager, Runtime};

use super::clustering::{DEFAULT_MAX_ANONYMOUS_SPEAKERS, SpeakerClusteringConfig};
use super::session::{DEFAULT_MIN_RELIABLE_SEGMENT_MS, DiarizationSessionConfig};

/// NOTE: "recongition" is the canonical (misspelled) sherpa-onnx release tag.
pub const DEFAULT_EMBEDDING_MODEL_ID: &str = "3dspeaker-eres2net-en";

/// Legacy WeSpeaker CAM++ speaker-embedding model retained for existing users.
pub const LEGACY_EMBEDDING_MODEL_ID: &str = "wespeaker-campp";

/// Imports can legitimately contain larger groups. Keep live conservative, but
/// do not force batch audio with many speakers into two anonymous clusters.
pub const DEFAULT_IMPORT_MAX_ANONYMOUS_SPEAKERS: usize = 8;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EmbeddingModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub size_mb: u32,
    pub recommended: bool,
    pub legacy: bool,
    pub embedding_dimension: usize,
    pub cluster_similarity_threshold: f32,
    pub profile_match_threshold: f32,
    pub live_max_anonymous_speakers: usize,
    pub import_max_anonymous_speakers: usize,
    pub min_reliable_segment_ms: u32,
    pub default_import_vad_redemption_ms: u32,
}

pub const EMBEDDING_MODELS: &[EmbeddingModelInfo] = &[
    EmbeddingModelInfo {
        id: "3dspeaker-eres2net-en",
        name: "3D-Speaker ERes2Net English",
        description: "Recommended newer English speaker embedding model from 3D-Speaker.",
        filename: "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        size_mb: 26,
        recommended: true,
        legacy: false,
        embedding_dimension: 192,
        cluster_similarity_threshold: 0.55,
        profile_match_threshold: 0.60,
        live_max_anonymous_speakers: DEFAULT_MAX_ANONYMOUS_SPEAKERS,
        import_max_anonymous_speakers: DEFAULT_IMPORT_MAX_ANONYMOUS_SPEAKERS,
        min_reliable_segment_ms: DEFAULT_MIN_RELIABLE_SEGMENT_MS,
        default_import_vad_redemption_ms: 400,
    },
    EmbeddingModelInfo {
        id: "3dspeaker-campp-en",
        name: "3D-Speaker CAM++ English",
        description: "Newer CAM++ English speaker embedding model from 3D-Speaker.",
        filename: "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        size_mb: 30,
        recommended: false,
        legacy: false,
        embedding_dimension: 192,
        cluster_similarity_threshold: 0.55,
        profile_match_threshold: 0.60,
        live_max_anonymous_speakers: DEFAULT_MAX_ANONYMOUS_SPEAKERS,
        import_max_anonymous_speakers: DEFAULT_IMPORT_MAX_ANONYMOUS_SPEAKERS,
        min_reliable_segment_ms: DEFAULT_MIN_RELIABLE_SEGMENT_MS,
        default_import_vad_redemption_ms: 400,
    },
    EmbeddingModelInfo {
        id: LEGACY_EMBEDDING_MODEL_ID,
        name: "WeSpeaker CAM++ English (legacy)",
        description: "Original Meetily speaker embedding model.",
        filename: "wespeaker_en_voxceleb_CAM++.onnx",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx",
        size_mb: 28,
        recommended: false,
        legacy: true,
        embedding_dimension: 192,
        cluster_similarity_threshold: 0.55,
        profile_match_threshold: 0.60,
        live_max_anonymous_speakers: DEFAULT_MAX_ANONYMOUS_SPEAKERS,
        import_max_anonymous_speakers: DEFAULT_IMPORT_MAX_ANONYMOUS_SPEAKERS,
        min_reliable_segment_ms: DEFAULT_MIN_RELIABLE_SEGMENT_MS,
        default_import_vad_redemption_ms: 400,
    },
];

pub fn embedding_models() -> &'static [EmbeddingModelInfo] {
    EMBEDDING_MODELS
}

pub fn embedding_model_by_id(model_id: &str) -> Option<&'static EmbeddingModelInfo> {
    EMBEDDING_MODELS.iter().find(|model| model.id == model_id)
}

pub fn valid_or_default_model_id(model_id: Option<&str>) -> &'static str {
    model_id
        .and_then(embedding_model_by_id)
        .map(|model| model.id)
        .unwrap_or(DEFAULT_EMBEDDING_MODEL_ID)
}

pub fn available_embedding_model_id<R: Runtime>(
    app: &AppHandle<R>,
    preferred_model_id: Option<&str>,
) -> Option<&'static str> {
    select_available_model_id(preferred_model_id, |model| {
        is_embedding_model_present_for_id(app, model.id)
    })
}

fn select_available_model_id<F>(
    preferred_model_id: Option<&str>,
    is_present: F,
) -> Option<&'static str>
where
    F: Fn(&EmbeddingModelInfo) -> bool,
{
    let preferred_model_id = valid_or_default_model_id(preferred_model_id);

    if let Some(model) = embedding_model_by_id(preferred_model_id).filter(|model| is_present(model))
    {
        return Some(model.id);
    }

    EMBEDDING_MODELS
        .iter()
        .find(|model| is_present(model))
        .map(|model| model.id)
}

pub fn session_config_for_model_id(model_id: &str) -> DiarizationSessionConfig {
    let model = embedding_model_by_id(model_id)
        .or_else(|| embedding_model_by_id(DEFAULT_EMBEDDING_MODEL_ID))
        .expect("default diarization model must be registered");

    session_config_for_model(model, model.live_max_anonymous_speakers)
}

pub fn import_session_config_for_model_id(model_id: &str) -> DiarizationSessionConfig {
    let model = embedding_model_by_id(model_id)
        .or_else(|| embedding_model_by_id(DEFAULT_EMBEDDING_MODEL_ID))
        .expect("default diarization model must be registered");

    session_config_for_model(model, model.import_max_anonymous_speakers)
}

fn session_config_for_model(
    model: &EmbeddingModelInfo,
    max_anonymous_speakers: usize,
) -> DiarizationSessionConfig {
    DiarizationSessionConfig {
        model_id: model.id,
        clustering: SpeakerClusteringConfig {
            cluster_similarity_threshold: model.cluster_similarity_threshold,
            profile_match_threshold: model.profile_match_threshold,
            max_anonymous_speakers,
        },
        min_reliable_segment_ms: model.min_reliable_segment_ms,
    }
}

pub fn default_import_vad_redemption_ms_for_model_id(model_id: &str) -> u32 {
    embedding_model_by_id(model_id)
        .or_else(|| embedding_model_by_id(DEFAULT_EMBEDDING_MODEL_ID))
        .map(|model| model.default_import_vad_redemption_ms)
        .unwrap_or(400)
}

pub fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {}", e))?;
    Ok(app_data_dir.join("models").join("diarization"))
}

pub fn embedding_model_path<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    embedding_model_path_for_id(app, DEFAULT_EMBEDDING_MODEL_ID)
}

pub fn embedding_model_path_for_id<R: Runtime>(
    app: &AppHandle<R>,
    model_id: &str,
) -> Result<PathBuf, String> {
    let model = embedding_model_by_id(model_id)
        .ok_or_else(|| format!("Unknown diarization model: {}", model_id))?;
    Ok(models_dir(app)?.join(model.filename))
}

pub fn is_embedding_model_present<R: Runtime>(app: &AppHandle<R>) -> bool {
    is_embedding_model_present_for_id(app, DEFAULT_EMBEDDING_MODEL_ID)
}

pub fn is_embedding_model_present_for_id<R: Runtime>(app: &AppHandle<R>, model_id: &str) -> bool {
    embedding_model_path_for_id(app, model_id)
        .map(|p| {
            p.exists()
                && std::fs::metadata(&p)
                    .map(|m| m.len() > 1_000_000)
                    .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// Download the embedding model, emitting `diarization-model-download-progress`
/// events with { downloaded_bytes, total_bytes, percent }.
pub async fn download_embedding_model<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    download_embedding_model_for_id(app, DEFAULT_EMBEDDING_MODEL_ID).await
}

pub async fn download_embedding_model_for_id<R: Runtime>(
    app: &AppHandle<R>,
    model_id: &str,
) -> Result<(), String> {
    let model = embedding_model_by_id(model_id)
        .ok_or_else(|| format!("Unknown diarization model: {}", model_id))?;
    let dir = models_dir(app)?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create models dir: {}", e))?;

    let final_path = dir.join(model.filename);
    if is_embedding_model_present_for_id(app, model.id) {
        log::info!(
            "Diarization embedding model already present at {}",
            final_path.display()
        );
        return Ok(());
    }
    let tmp_path = dir.join(format!("{}.tmp", model.filename));

    log::info!("Downloading diarization embedding model from {}", model.url);
    let client = reqwest::Client::new();
    let response = client
        .get(model.url)
        .send()
        .await
        .map_err(|e| format!("Download request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed with HTTP {}", response.status()));
    }

    let total_bytes = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut last_emitted_percent: i64 = -1;

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download stream error: {}", e))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|e| format!("Failed to write model file: {}", e))?;
        downloaded += chunk.len() as u64;

        let percent = if total_bytes > 0 {
            (downloaded * 100 / total_bytes) as i64
        } else {
            0
        };
        if percent != last_emitted_percent {
            last_emitted_percent = percent;
            let _ = app.emit(
                "diarization-model-download-progress",
                serde_json::json!({
                    "downloaded_bytes": downloaded,
                    "total_bytes": total_bytes,
                    "percent": percent,
                }),
            );
        }
    }
    tokio::io::AsyncWriteExt::flush(&mut file)
        .await
        .map_err(|e| format!("Failed to flush model file: {}", e))?;
    drop(file);

    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| format!("Failed to finalize model file: {}", e))?;

    log::info!(
        "Diarization embedding model downloaded to {} ({} bytes)",
        final_path.display(),
        downloaded
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_model_prefers_selected_when_present() {
        let selected = select_available_model_id(Some(LEGACY_EMBEDDING_MODEL_ID), |model| {
            model.id == LEGACY_EMBEDDING_MODEL_ID || model.id == DEFAULT_EMBEDDING_MODEL_ID
        });

        assert_eq!(selected, Some(LEGACY_EMBEDDING_MODEL_ID));
    }

    #[test]
    fn available_model_falls_back_to_downloaded_model() {
        let selected = select_available_model_id(Some(DEFAULT_EMBEDDING_MODEL_ID), |model| {
            model.id == LEGACY_EMBEDDING_MODEL_ID
        });

        assert_eq!(selected, Some(LEGACY_EMBEDDING_MODEL_ID));
    }

    #[test]
    fn available_model_returns_none_when_no_models_are_present() {
        let selected = select_available_model_id(Some(DEFAULT_EMBEDDING_MODEL_ID), |_| false);

        assert_eq!(selected, None);
    }
}
