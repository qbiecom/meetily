use super::provider::{TranscriptResult, TranscriptionError, TranscriptionProvider};
use async_trait::async_trait;
use std::sync::Arc;

pub struct SherpaProvider {
    engine: Arc<crate::sherpa_engine::SherpaEngine>,
}

impl SherpaProvider {
    pub fn new(engine: Arc<crate::sherpa_engine::SherpaEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl TranscriptionProvider for SherpaProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        _language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        self.engine
            .transcribe_audio(audio)
            .await
            .map(|text| TranscriptResult {
                text: text.trim().to_string(),
                confidence: None,
                is_partial: false,
            })
            .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))
    }

    async fn is_model_loaded(&self) -> bool {
        self.engine.is_model_loaded().await
    }

    async fn get_current_model(&self) -> Option<String> {
        self.engine.get_current_model().await
    }

    fn provider_name(&self) -> &'static str {
        "Sherpa ONNX"
    }
}
