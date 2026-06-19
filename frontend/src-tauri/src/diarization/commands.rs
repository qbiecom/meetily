// diarization/commands.rs
//
// Tauri command surface for speaker identification: feature toggle
// (persisted in the diarization_settings table), model status, and
// model download.

use crate::database::repositories::speaker_profile::SpeakerProfilesRepository;
use crate::state::AppState;
use sqlx::SqlitePool;
use tauri::{AppHandle, Runtime, command};

pub async fn is_enabled(pool: &SqlitePool) -> bool {
    sqlx::query_scalar::<_, i64>("SELECT enabled FROM diarization_settings WHERE id = '1'")
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(|v| v != 0)
        .unwrap_or(false)
}

pub async fn selected_model_id(pool: &SqlitePool) -> String {
    let selected = sqlx::query_scalar::<_, String>(
        "SELECT selected_model_id FROM diarization_settings WHERE id = '1'",
    )
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();

    super::models::valid_or_default_model_id(selected.as_deref()).to_string()
}

#[command]
pub async fn diarization_get_status<R: Runtime>(
    app: AppHandle<R>,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let enabled = is_enabled(state.db_manager.pool()).await;
    let selected_model_id = selected_model_id(state.db_manager.pool()).await;
    let model_present = super::models::is_embedding_model_present_for_id(&app, &selected_model_id);
    let models = super::models::embedding_models()
        .iter()
        .map(|model| {
            serde_json::json!({
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "filename": model.filename,
                "size_mb": model.size_mb,
                "recommended": model.recommended,
                "legacy": model.legacy,
                "embedding_dimension": model.embedding_dimension,
                "cluster_similarity_threshold": model.cluster_similarity_threshold,
                "profile_match_threshold": model.profile_match_threshold,
                "max_anonymous_speakers": model.live_max_anonymous_speakers,
                "live_max_anonymous_speakers": model.live_max_anonymous_speakers,
                "import_max_anonymous_speakers": model.import_max_anonymous_speakers,
                "min_reliable_segment_ms": model.min_reliable_segment_ms,
                "default_import_vad_redemption_ms": model.default_import_vad_redemption_ms,
                "present": super::models::is_embedding_model_present_for_id(&app, model.id),
            })
        })
        .collect::<Vec<_>>();
    let selected_model = super::models::embedding_model_by_id(&selected_model_id)
        .or_else(|| super::models::embedding_model_by_id(super::models::DEFAULT_EMBEDDING_MODEL_ID))
        .ok_or_else(|| "No diarization models are registered".to_string())?;
    Ok(serde_json::json!({
        "enabled": enabled,
        "model_present": model_present,
        "model_filename": selected_model.filename,
        "selected_model_id": selected_model.id,
        "models": models,
    }))
}

#[command]
pub async fn diarization_set_enabled(
    state: tauri::State<'_, AppState>,
    enabled: bool,
) -> Result<(), String> {
    sqlx::query(
        r#"
        INSERT INTO diarization_settings (id, enabled) VALUES ('1', $1)
        ON CONFLICT(id) DO UPDATE SET enabled = excluded.enabled
        "#,
    )
    .bind(enabled as i64)
    .execute(state.db_manager.pool())
    .await
    .map_err(|e| format!("Failed to save diarization setting: {}", e))?;
    log::info!(
        "Speaker identification {}",
        if enabled { "enabled" } else { "disabled" }
    );
    Ok(())
}

#[command]
pub async fn diarization_set_model(
    state: tauri::State<'_, AppState>,
    model_id: String,
) -> Result<(), String> {
    let model_id = super::models::embedding_model_by_id(&model_id)
        .map(|model| model.id)
        .ok_or_else(|| format!("Unknown diarization model: {}", model_id))?;

    sqlx::query(
        r#"
        INSERT INTO diarization_settings (id, enabled, selected_model_id) VALUES ('1', 0, $1)
        ON CONFLICT(id) DO UPDATE SET selected_model_id = excluded.selected_model_id
        "#,
    )
    .bind(model_id)
    .execute(state.db_manager.pool())
    .await
    .map_err(|e| format!("Failed to save diarization model setting: {}", e))?;
    log::info!("Selected speaker identification model: {}", model_id);
    Ok(())
}

#[command]
pub async fn diarization_download_model<R: Runtime>(
    app: AppHandle<R>,
    state: tauri::State<'_, AppState>,
    model_id: Option<String>,
) -> Result<(), String> {
    let selected = match model_id {
        Some(model_id) => super::models::embedding_model_by_id(&model_id)
            .map(|model| model.id.to_string())
            .ok_or_else(|| format!("Unknown diarization model: {}", model_id))?,
        None => selected_model_id(state.db_manager.pool()).await,
    };
    super::models::download_embedding_model_for_id(&app, &selected).await
}

/// Read the centroid for a speaker label from a meeting folder's speakers.json.
fn load_centroid_from_folder(folder: &str, label: &str) -> Option<Vec<f32>> {
    let path = std::path::Path::new(folder).join("speakers.json");
    let content = std::fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("speakers")?.as_array()?.iter().find_map(|s| {
        if s.get("label")?.as_str()? != label {
            return None;
        }
        let centroid: Vec<f32> = s
            .get("centroid")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        if centroid.is_empty() {
            None
        } else {
            Some(centroid)
        }
    })
}

/// Rename a speaker across all segments of a meeting. Optionally saves the
/// speaker's voice centroid (from the meeting's speakers.json) as a persistent
/// profile so future recordings label this voice by name automatically.
#[command]
pub async fn diarization_rename_speaker(
    state: tauri::State<'_, AppState>,
    meeting_id: String,
    old_label: String,
    new_name: String,
    save_profile: bool,
) -> Result<serde_json::Value, String> {
    let new_name = new_name.trim();
    if new_name.is_empty() {
        return Err("Speaker name cannot be empty".to_string());
    }
    let pool = state.db_manager.pool();

    let result =
        sqlx::query("UPDATE transcripts SET speaker = ? WHERE meeting_id = ? AND speaker = ?")
            .bind(new_name)
            .bind(&meeting_id)
            .bind(&old_label)
            .execute(pool)
            .await
            .map_err(|e| format!("Failed to rename speaker: {}", e))?;
    let updated = result.rows_affected();

    let mut profile_saved = false;
    if save_profile {
        let folder_path: Option<String> =
            sqlx::query_scalar("SELECT folder_path FROM meetings WHERE id = ?")
                .bind(&meeting_id)
                .fetch_optional(pool)
                .await
                .map_err(|e| format!("Failed to look up meeting folder: {}", e))?
                .flatten();

        if let Some(centroid) = folder_path
            .as_deref()
            .and_then(|f| load_centroid_from_folder(f, &old_label))
        {
            SpeakerProfilesRepository::create(pool, new_name, &centroid)
                .await
                .map_err(|e| format!("Failed to save voice profile: {}", e))?;
            profile_saved = true;
            log::info!(
                "Saved voice profile '{}' from meeting {}",
                new_name,
                meeting_id
            );
        } else {
            log::warn!(
                "No voice centroid found for '{}' in meeting {} - profile not saved",
                old_label,
                meeting_id
            );
        }
    }

    Ok(serde_json::json!({
        "updated_segments": updated,
        "profile_saved": profile_saved,
    }))
}

#[command]
pub async fn diarization_list_profiles(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<serde_json::Value>, String> {
    let profiles = SpeakerProfilesRepository::list(state.db_manager.pool())
        .await
        .map_err(|e| format!("Failed to list voice profiles: {}", e))?;
    Ok(profiles
        .into_iter()
        .map(|p| serde_json::json!({ "id": p.id, "name": p.name }))
        .collect())
}

#[command]
pub async fn diarization_rename_profile(
    state: tauri::State<'_, AppState>,
    id: String,
    name: String,
) -> Result<(), String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Profile name cannot be empty".to_string());
    }
    SpeakerProfilesRepository::rename(state.db_manager.pool(), &id, name)
        .await
        .map_err(|e| format!("Failed to rename voice profile: {}", e))
}

#[command]
pub async fn diarization_delete_profile(
    state: tauri::State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    SpeakerProfilesRepository::delete(state.db_manager.pool(), &id)
        .await
        .map_err(|e| format!("Failed to delete voice profile: {}", e))
}
