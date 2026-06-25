use crate::sherpa_engine::{SherpaEngine, SherpaModelInfo};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, Runtime, command};
use tauri_plugin_store::StoreExt;

pub static SHERPA_ENGINE: Mutex<Option<Arc<SherpaEngine>>> = Mutex::new(None);
static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
const SHERPA_SETTINGS_STORE: &str = "sherpa_settings.json";
const EXECUTION_PROVIDER_KEY: &str = "executionProvider";

pub fn set_models_directory<R: Runtime>(app: &AppHandle<R>) {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .expect("Failed to get app data dir");
    let models_dir = app_data_dir.join("models");
    if let Err(e) = std::fs::create_dir_all(&models_dir) {
        log::error!("Failed to create models directory: {}", e);
        return;
    }
    *MODELS_DIR.lock().unwrap() = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

#[command]
pub async fn sherpa_init() -> Result<(), String> {
    let mut guard = SHERPA_ENGINE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }

    let engine = SherpaEngine::new_with_models_dir(get_models_directory())
        .map_err(|e| format!("Failed to initialize Sherpa ONNX engine: {}", e))?;
    *guard = Some(Arc::new(engine));
    Ok(())
}

#[command]
pub async fn sherpa_get_available_models() -> Result<Vec<SherpaModelInfo>, String> {
    let engine = get_engine()?;
    engine
        .discover_models()
        .await
        .map_err(|e| format!("Failed to discover Sherpa ONNX models: {}", e))
}

#[command]
pub async fn sherpa_load_model<R: Runtime>(
    app: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    apply_saved_execution_provider(&app).await?;
    let engine = get_engine()?;
    let _ = app.emit(
        "sherpa-model-loading-started",
        serde_json::json!({ "modelName": model_name }),
    );
    let result = engine
        .load_model(&model_name)
        .await
        .map_err(|e| format!("Failed to load Sherpa ONNX model: {}", e));
    let event = if result.is_ok() {
        "sherpa-model-loading-completed"
    } else {
        "sherpa-model-loading-failed"
    };
    let _ = app.emit(event, serde_json::json!({ "modelName": model_name }));
    result
}

#[command]
pub async fn sherpa_get_current_model() -> Result<Option<String>, String> {
    Ok(get_engine()?.get_current_model().await)
}

#[command]
pub async fn sherpa_is_model_loaded() -> Result<bool, String> {
    Ok(get_engine()?.is_model_loaded().await)
}

#[command]
pub async fn sherpa_has_available_models() -> Result<bool, String> {
    let models = get_engine()?
        .discover_models()
        .await
        .map_err(|e| e.to_string())?;
    Ok(models
        .iter()
        .any(|model| matches!(model.status, crate::sherpa_engine::ModelStatus::Available)))
}

#[command]
pub async fn sherpa_validate_model_ready<R: Runtime>(app: AppHandle<R>) -> Result<String, String> {
    apply_saved_execution_provider(&app).await?;
    validate_model_ready(None).await
}

pub async fn sherpa_validate_model_ready_with_config<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<String, String> {
    apply_saved_execution_provider(app).await?;
    let configured_model =
        match crate::api::api::api_get_transcript_config(app.clone(), app.state(), None).await {
            Ok(Some(config)) if config.provider == "sherpaOnnx" && !config.model.is_empty() => {
                Some(config.model)
            }
            _ => None,
        };
    validate_model_ready(configured_model).await
}

async fn validate_model_ready(configured_model: Option<String>) -> Result<String, String> {
    let engine = get_engine()?;

    if let Some(current_model) = engine.get_current_model().await {
        if configured_model
            .as_deref()
            .map_or(true, |model| model == current_model)
            && engine.is_model_loaded().await
        {
            return Ok(current_model);
        }
        engine.unload_model().await;
    }

    let models = engine.discover_models().await.map_err(|e| e.to_string())?;
    let model_to_load = if let Some(configured_model) = configured_model {
        configured_model
    } else {
        models
            .iter()
            .find(|model| {
                matches!(model.status, crate::sherpa_engine::ModelStatus::Available)
                    && model.name.contains("int8")
            })
            .or_else(|| {
                models.iter().find(|model| {
                    matches!(model.status, crate::sherpa_engine::ModelStatus::Available)
                })
            })
            .map(|model| model.name.clone())
            .ok_or_else(|| {
                "No Sherpa ONNX models are available. Please download one first.".to_string()
            })?
    };

    let model_info = models
        .iter()
        .find(|model| model.name == model_to_load)
        .ok_or_else(|| format!("Sherpa ONNX model '{}' is not supported", model_to_load))?;

    if !matches!(
        model_info.status,
        crate::sherpa_engine::ModelStatus::Available
    ) {
        return Err(format!(
            "Sherpa ONNX model '{}' is not downloaded",
            model_to_load
        ));
    }

    engine
        .load_model(&model_to_load)
        .await
        .map_err(|e| e.to_string())?;
    Ok(model_to_load)
}

#[command]
pub async fn sherpa_transcribe_audio(audio_data: Vec<f32>) -> Result<String, String> {
    get_engine()?
        .transcribe_audio(audio_data)
        .await
        .map_err(|e| format!("Sherpa ONNX transcription failed: {}", e))
}

#[command]
pub async fn sherpa_get_models_directory() -> Result<String, String> {
    Ok(get_engine()?.models_dir().to_string_lossy().to_string())
}

#[command]
pub async fn sherpa_get_execution_provider<R: Runtime>(
    app: AppHandle<R>,
) -> Result<String, String> {
    apply_saved_execution_provider(&app).await
}

#[command]
pub async fn sherpa_set_execution_provider<R: Runtime>(
    app: AppHandle<R>,
    provider: String,
) -> Result<(), String> {
    let provider = normalize_execution_provider(&provider)?;
    get_engine()?
        .set_execution_provider(provider.clone())
        .await
        .map_err(|e| e.to_string())?;
    save_execution_provider(&app, &provider)?;
    Ok(())
}

#[command]
pub async fn sherpa_download_model<R: Runtime>(
    app: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = get_engine()?;
    let model_for_progress = model_name.clone();
    let app_for_progress = app.clone();
    let result = engine
        .download_model(&model_name, move |progress| {
            let _ = app_for_progress.emit(
                "sherpa-model-download-progress",
                serde_json::json!({ "modelName": model_for_progress, "progress": progress }),
            );
        })
        .await;

    match result {
        Ok(()) => {
            let _ = app.emit(
                "sherpa-model-download-complete",
                serde_json::json!({ "modelName": model_name }),
            );
            Ok(())
        }
        Err(e) => {
            let error = e.to_string();
            let _ = app.emit(
                "sherpa-model-download-error",
                serde_json::json!({ "modelName": model_name, "error": error }),
            );
            Err(error)
        }
    }
}

#[command]
pub async fn sherpa_delete_model(model_name: String) -> Result<String, String> {
    get_engine()?
        .delete_model(&model_name)
        .await
        .map_err(|e| e.to_string())?;
    Ok(format!("Deleted Sherpa ONNX model {}", model_name))
}

#[command]
pub async fn open_sherpa_models_folder() -> Result<(), String> {
    let path = get_engine()?.models_dir().clone();
    #[cfg(target_os = "windows")]
    std::process::Command::new("explorer")
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    #[cfg(target_os = "macos")]
    std::process::Command::new("open")
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open")
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn get_engine() -> Result<Arc<SherpaEngine>, String> {
    SHERPA_ENGINE
        .lock()
        .unwrap()
        .as_ref()
        .cloned()
        .ok_or_else(|| "Sherpa ONNX engine not initialized".to_string())
}

fn normalize_execution_provider(provider: &str) -> Result<String, String> {
    let normalized = provider.to_lowercase();
    if matches!(normalized.as_str(), "cpu" | "cuda") {
        Ok(normalized)
    } else {
        Err(format!(
            "Unsupported Sherpa execution provider '{}'. Use 'cpu' or 'cuda'.",
            provider
        ))
    }
}

fn load_saved_execution_provider<R: Runtime>(app: &AppHandle<R>) -> String {
    let store = match app.store(SHERPA_SETTINGS_STORE) {
        Ok(store) => store,
        Err(e) => {
            log::warn!("Failed to access Sherpa settings store: {}, using CPU", e);
            return "cpu".to_string();
        }
    };

    store
        .get(EXECUTION_PROVIDER_KEY)
        .and_then(|value| value.as_str().map(ToString::to_string))
        .and_then(|provider| match normalize_execution_provider(&provider) {
            Ok(provider) => Some(provider),
            Err(e) => {
                log::warn!("Invalid saved Sherpa execution provider: {}, using CPU", e);
                None
            }
        })
        .unwrap_or_else(|| "cpu".to_string())
}

fn save_execution_provider<R: Runtime>(app: &AppHandle<R>, provider: &str) -> Result<(), String> {
    let store = app
        .store(SHERPA_SETTINGS_STORE)
        .map_err(|e| format!("Failed to access Sherpa settings store: {}", e))?;
    store.set(EXECUTION_PROVIDER_KEY, serde_json::json!(provider));
    store
        .save()
        .map_err(|e| format!("Failed to save Sherpa settings: {}", e))
}

pub async fn apply_saved_execution_provider<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<String, String> {
    let provider = load_saved_execution_provider(app);
    let engine = get_engine()?;
    if engine.get_execution_provider().await != provider {
        engine
            .set_execution_provider(provider.clone())
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(provider)
}
