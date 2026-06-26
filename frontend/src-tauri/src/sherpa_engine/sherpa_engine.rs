use anyhow::Result;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationType {
    FP32,
    FP16,
    Int8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Missing,
    Downloading {
        progress: u8,
    },
    Error(String),
    Corrupted {
        file_size: u64,
        expected_min_size: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SherpaModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub quantization: QuantizationType,
    pub speed: String,
    pub status: ModelStatus,
    pub description: String,
}

#[derive(Debug)]
pub enum SherpaEngineError {
    ModelNotLoaded,
    ModelNotFound(String),
    TranscriptionFailed(String),
    DownloadFailed(String),
    IoError(std::io::Error),
    Other(String),
}

impl std::fmt::Display for SherpaEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotLoaded => write!(f, "No Sherpa ONNX model loaded"),
            Self::ModelNotFound(name) => write!(f, "Model '{}' not found", name),
            Self::TranscriptionFailed(err) => write!(f, "Transcription failed: {}", err),
            Self::DownloadFailed(err) => write!(f, "Download failed: {}", err),
            Self::IoError(err) => write!(f, "IO error: {}", err),
            Self::Other(err) => write!(f, "Error: {}", err),
        }
    }
}

impl std::error::Error for SherpaEngineError {}

impl From<std::io::Error> for SherpaEngineError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

#[derive(Clone)]
struct ModelFile {
    remote_path: &'static str,
    local_name: &'static str,
    min_bytes: u64,
}

#[derive(Clone)]
struct SherpaModelSpec {
    name: &'static str,
    repo: &'static str,
    size_mb: u32,
    quantization: QuantizationType,
    speed: &'static str,
    description: &'static str,
    encoder: &'static str,
    decoder: &'static str,
    joiner: &'static str,
    tokens: &'static str,
    model_type: Option<&'static str>,
    qwen3: bool,
    conv_frontend: Option<&'static str>,
    tokenizer: Option<&'static str>,
    files: Vec<ModelFile>,
}

fn model_specs() -> Vec<SherpaModelSpec> {
    vec![
        SherpaModelSpec {
            name: "sherpa-parakeet-tdt-0.6b-v3-int8",
            repo: "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            size_mb: 640,
            quantization: QuantizationType::Int8,
            speed: "Ultra Fast",
            description: "Sherpa ONNX NeMo Parakeet TDT 0.6B v3 int8 transducer model",
            encoder: "encoder.int8.onnx",
            decoder: "decoder.int8.onnx",
            joiner: "joiner.int8.onnx",
            tokens: "tokens.txt",
            model_type: Some("nemo_transducer"),
            qwen3: false,
            conv_frontend: None,
            tokenizer: None,
            files: vec![
                ModelFile {
                    remote_path: "encoder.int8.onnx",
                    local_name: "encoder.int8.onnx",
                    min_bytes: 500_000_000,
                },
                ModelFile {
                    remote_path: "decoder.int8.onnx",
                    local_name: "decoder.int8.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "joiner.int8.onnx",
                    local_name: "joiner.int8.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "tokens.txt",
                    local_name: "tokens.txt",
                    min_bytes: 1_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-parakeet-tdt-0.6b-v3-fp16",
            repo: "Yiivgeny/parakeet-tdt-0.6b-v3-sherpa-onnx-fp16",
            size_mb: 1300,
            quantization: QuantizationType::FP16,
            speed: "Fast",
            description: "Sherpa ONNX fp16 Parakeet TDT 0.6B v3 model with lower memory use than fp32",
            encoder: "encoder.onnx",
            decoder: "decoder.onnx",
            joiner: "joiner.onnx",
            tokens: "tokens.txt",
            model_type: Some("nemo_transducer"),
            qwen3: false,
            conv_frontend: None,
            tokenizer: None,
            files: vec![
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp16/encoder.onnx",
                    local_name: "encoder.onnx",
                    min_bytes: 500_000_000,
                },
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp16/decoder.onnx",
                    local_name: "decoder.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp16/joiner.onnx",
                    local_name: "joiner.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp16/tokens.txt",
                    local_name: "tokens.txt",
                    min_bytes: 1_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-parakeet-tdt-0.6b-v3-fp32",
            repo: "Yiivgeny/parakeet-tdt-0.6b-v3-sherpa-onnx-fp32",
            size_mb: 2550,
            quantization: QuantizationType::FP32,
            speed: "Fast",
            description: "Sherpa ONNX full precision Parakeet TDT 0.6B v3 model",
            encoder: "encoder.onnx",
            decoder: "decoder.onnx",
            joiner: "joiner.onnx",
            tokens: "tokens.txt",
            model_type: Some("nemo_transducer"),
            qwen3: false,
            conv_frontend: None,
            tokenizer: None,
            files: vec![
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/encoder.onnx",
                    local_name: "encoder.onnx",
                    min_bytes: 10_000_000,
                },
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/encoder.weights",
                    local_name: "encoder.weights",
                    min_bytes: 2_000_000_000,
                },
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/decoder.onnx",
                    local_name: "decoder.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/joiner.onnx",
                    local_name: "joiner.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/tokens.txt",
                    local_name: "tokens.txt",
                    min_bytes: 1_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-zipformer-zh-en-2026-06-03-int8",
            repo: "csukuangfj2/sherpa-onnx-x-asr-zipformer-transducer-zh-en-int8-2026-06-03",
            size_mb: 355,
            quantization: QuantizationType::Int8,
            speed: "Ultra Fast",
            description: "Sherpa ONNX Zipformer bilingual Chinese/English int8 transducer model created in 2026",
            encoder: "encoder-epoch-99-avg-1.int8.onnx",
            decoder: "decoder-epoch-99-avg-1.onnx",
            joiner: "joiner-epoch-99-avg-1.int8.onnx",
            tokens: "tokens.txt",
            model_type: None,
            qwen3: false,
            conv_frontend: None,
            tokenizer: None,
            files: vec![
                ModelFile {
                    remote_path: "encoder-epoch-99-avg-1.int8.onnx",
                    local_name: "encoder-epoch-99-avg-1.int8.onnx",
                    min_bytes: 300_000_000,
                },
                ModelFile {
                    remote_path: "decoder-epoch-99-avg-1.onnx",
                    local_name: "decoder-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "joiner-epoch-99-avg-1.int8.onnx",
                    local_name: "joiner-epoch-99-avg-1.int8.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "tokens.txt",
                    local_name: "tokens.txt",
                    min_bytes: 1_000,
                },
                ModelFile {
                    remote_path: "bpe.model",
                    local_name: "bpe.model",
                    min_bytes: 1_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-zipformer-zh-en-punct-2026-06-03-int8",
            repo: "csukuangfj2/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03",
            size_mb: 354,
            quantization: QuantizationType::Int8,
            speed: "Ultra Fast",
            description: "Sherpa ONNX Zipformer bilingual Chinese/English int8 transducer with punctuation, created in 2026",
            encoder: "encoder-epoch-99-avg-1.int8.onnx",
            decoder: "decoder-epoch-99-avg-1.onnx",
            joiner: "joiner-epoch-99-avg-1.int8.onnx",
            tokens: "tokens.txt",
            model_type: None,
            qwen3: false,
            conv_frontend: None,
            tokenizer: None,
            files: vec![
                ModelFile {
                    remote_path: "encoder-epoch-99-avg-1.int8.onnx",
                    local_name: "encoder-epoch-99-avg-1.int8.onnx",
                    min_bytes: 300_000_000,
                },
                ModelFile {
                    remote_path: "decoder-epoch-99-avg-1.onnx",
                    local_name: "decoder-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "joiner-epoch-99-avg-1.int8.onnx",
                    local_name: "joiner-epoch-99-avg-1.int8.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "tokens.txt",
                    local_name: "tokens.txt",
                    min_bytes: 1_000,
                },
                ModelFile {
                    remote_path: "bpe.model",
                    local_name: "bpe.model",
                    min_bytes: 1_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-zipformer-zh-en-2026-06-03-fp32",
            repo: "csukuangfj2/sherpa-onnx-x-asr-zipformer-transducer-zh-en-2026-06-03",
            size_mb: 1245,
            quantization: QuantizationType::FP32,
            speed: "Fast",
            description: "Sherpa ONNX Zipformer bilingual Chinese/English fp32 transducer model created in 2026",
            encoder: "encoder-epoch-99-avg-1.onnx",
            decoder: "decoder-epoch-99-avg-1.onnx",
            joiner: "joiner-epoch-99-avg-1.onnx",
            tokens: "tokens.txt",
            model_type: None,
            qwen3: false,
            conv_frontend: None,
            tokenizer: None,
            files: vec![
                ModelFile {
                    remote_path: "encoder-epoch-99-avg-1.onnx",
                    local_name: "encoder-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000_000,
                },
                ModelFile {
                    remote_path: "decoder-epoch-99-avg-1.onnx",
                    local_name: "decoder-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "joiner-epoch-99-avg-1.onnx",
                    local_name: "joiner-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "tokens.txt",
                    local_name: "tokens.txt",
                    min_bytes: 1_000,
                },
                ModelFile {
                    remote_path: "bpe.model",
                    local_name: "bpe.model",
                    min_bytes: 1_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-zipformer-zh-en-punct-2026-06-03-fp32",
            repo: "csukuangfj2/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-2026-06-03",
            size_mb: 1245,
            quantization: QuantizationType::FP32,
            speed: "Fast",
            description: "Sherpa ONNX Zipformer bilingual Chinese/English fp32 transducer with punctuation, created in 2026",
            encoder: "encoder-epoch-99-avg-1.onnx",
            decoder: "decoder-epoch-99-avg-1.onnx",
            joiner: "joiner-epoch-99-avg-1.onnx",
            tokens: "tokens.txt",
            model_type: None,
            qwen3: false,
            conv_frontend: None,
            tokenizer: None,
            files: vec![
                ModelFile {
                    remote_path: "encoder-epoch-99-avg-1.onnx",
                    local_name: "encoder-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000_000,
                },
                ModelFile {
                    remote_path: "decoder-epoch-99-avg-1.onnx",
                    local_name: "decoder-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "joiner-epoch-99-avg-1.onnx",
                    local_name: "joiner-epoch-99-avg-1.onnx",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "tokens.txt",
                    local_name: "tokens.txt",
                    min_bytes: 1_000,
                },
                ModelFile {
                    remote_path: "bpe.model",
                    local_name: "bpe.model",
                    min_bytes: 1_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-qwen3-asr-0.6b-2026-03-25-int8",
            repo: "csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
            size_mb: 1755,
            quantization: QuantizationType::Int8,
            speed: "Accurate",
            description: "Sherpa ONNX Qwen3-ASR 0.6B int8 multilingual model created in 2026; strong English support including noisy speech and rap examples",
            encoder: "encoder.int8.onnx",
            decoder: "decoder.int8.onnx",
            joiner: "",
            tokens: "",
            model_type: None,
            qwen3: true,
            conv_frontend: Some("conv_frontend.onnx"),
            tokenizer: Some("tokenizer"),
            files: vec![
                ModelFile {
                    remote_path: "conv_frontend.onnx",
                    local_name: "conv_frontend.onnx",
                    min_bytes: 40_000_000,
                },
                ModelFile {
                    remote_path: "encoder.int8.onnx",
                    local_name: "encoder.int8.onnx",
                    min_bytes: 170_000_000,
                },
                ModelFile {
                    remote_path: "decoder.int8.onnx",
                    local_name: "decoder.int8.onnx",
                    min_bytes: 700_000_000,
                },
                ModelFile {
                    remote_path: "tokenizer/merges.txt",
                    local_name: "tokenizer/merges.txt",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "tokenizer/tokenizer_config.json",
                    local_name: "tokenizer/tokenizer_config.json",
                    min_bytes: 1_000,
                },
                ModelFile {
                    remote_path: "tokenizer/vocab.json",
                    local_name: "tokenizer/vocab.json",
                    min_bytes: 1_000_000,
                },
            ],
        },
        SherpaModelSpec {
            name: "sherpa-qwen3-asr-1.7b-2026-05-17-int8",
            repo: "ilmina/qwen3-asr-1.7b-sherpa-onnx",
            size_mb: 2400,
            quantization: QuantizationType::Int8,
            speed: "High Accuracy",
            description: "Sherpa ONNX Qwen3-ASR 1.7B int8 multilingual model created in 2026 for users who prefer accuracy over disk and memory use",
            encoder: "encoder.int8.onnx",
            decoder: "decoder.int8.onnx",
            joiner: "",
            tokens: "",
            model_type: None,
            qwen3: true,
            conv_frontend: Some("conv_frontend.onnx"),
            tokenizer: Some("tokenizer"),
            files: vec![
                ModelFile {
                    remote_path: "conv_frontend.onnx",
                    local_name: "conv_frontend.onnx",
                    min_bytes: 40_000_000,
                },
                ModelFile {
                    remote_path: "encoder.int8.onnx",
                    local_name: "encoder.int8.onnx",
                    min_bytes: 350_000_000,
                },
                ModelFile {
                    remote_path: "decoder.int8.onnx",
                    local_name: "decoder.int8.onnx",
                    min_bytes: 1_800_000_000,
                },
                ModelFile {
                    remote_path: "tokenizer/merges.txt",
                    local_name: "tokenizer/merges.txt",
                    min_bytes: 1_000_000,
                },
                ModelFile {
                    remote_path: "tokenizer/tokenizer_config.json",
                    local_name: "tokenizer/tokenizer_config.json",
                    min_bytes: 1_000,
                },
                ModelFile {
                    remote_path: "tokenizer/vocab.json",
                    local_name: "tokenizer/vocab.json",
                    min_bytes: 1_000_000,
                },
            ],
        },
    ]
}

pub struct SherpaEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<Arc<OfflineRecognizer>>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    execution_provider: Arc<RwLock<String>>,
    active_downloads: Arc<RwLock<HashSet<String>>>,
    available_models: Arc<RwLock<HashMap<String, SherpaModelInfo>>>,
}

impl SherpaEngine {
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir.join("sherpa-onnx")
        } else {
            std::env::current_dir()?.join("models").join("sherpa-onnx")
        };

        if !models_dir.exists() {
            std::fs::create_dir_all(&models_dir)?;
        }

        Ok(Self {
            models_dir,
            current_model: Arc::new(RwLock::new(None)),
            current_model_name: Arc::new(RwLock::new(None)),
            execution_provider: Arc::new(RwLock::new("cpu".to_string())),
            active_downloads: Arc::new(RwLock::new(HashSet::new())),
            available_models: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn discover_models(&self) -> Result<Vec<SherpaModelInfo>> {
        let active_downloads = self.active_downloads.read().await;
        let mut models = Vec::new();

        for spec in model_specs() {
            let model_path = self.models_dir.join(spec.name);
            let status = if active_downloads.contains(spec.name) {
                ModelStatus::Downloading { progress: 0 }
            } else if model_path.exists() {
                match validate_spec_files(&model_path, &spec) {
                    Ok(()) => ModelStatus::Available,
                    Err((file_size, expected_min_size)) => ModelStatus::Corrupted {
                        file_size,
                        expected_min_size,
                    },
                }
            } else {
                ModelStatus::Missing
            };

            let info = SherpaModelInfo {
                name: spec.name.to_string(),
                path: model_path,
                size_mb: spec.size_mb,
                quantization: spec.quantization,
                speed: spec.speed.to_string(),
                status,
                description: spec.description.to_string(),
            };
            models.push(info.clone());
            self.available_models
                .write()
                .await
                .insert(spec.name.to_string(), info);
        }

        Ok(models)
    }

    pub async fn load_model(&self, model_name: &str) -> Result<(), SherpaEngineError> {
        let spec = find_spec(model_name)?;
        let model_dir = self.models_dir.join(spec.name);
        validate_spec_files(&model_dir, &spec).map_err(|_| {
            SherpaEngineError::ModelNotFound(format!("{} is missing or incomplete", model_name))
        })?;

        let provider = self.execution_provider.read().await.clone();
        let mut config = OfflineRecognizerConfig::default();
        if spec.qwen3 {
            let conv_frontend = spec.conv_frontend.ok_or_else(|| {
                SherpaEngineError::Other(format!("{} is missing Qwen3 conv frontend", model_name))
            })?;
            let tokenizer = spec.tokenizer.ok_or_else(|| {
                SherpaEngineError::Other(format!("{} is missing Qwen3 tokenizer", model_name))
            })?;
            config.model_config.qwen3_asr.conv_frontend =
                Some(model_dir.join(conv_frontend).to_string_lossy().to_string());
            config.model_config.qwen3_asr.encoder =
                Some(model_dir.join(spec.encoder).to_string_lossy().to_string());
            config.model_config.qwen3_asr.decoder =
                Some(model_dir.join(spec.decoder).to_string_lossy().to_string());
            config.model_config.qwen3_asr.tokenizer =
                Some(model_dir.join(tokenizer).to_string_lossy().to_string());
            config.model_config.qwen3_asr.max_total_len = 512;
            config.model_config.qwen3_asr.max_new_tokens = 512;
        } else {
            config.model_config.transducer = OfflineTransducerModelConfig {
                encoder: Some(model_dir.join(spec.encoder).to_string_lossy().to_string()),
                decoder: Some(model_dir.join(spec.decoder).to_string_lossy().to_string()),
                joiner: Some(model_dir.join(spec.joiner).to_string_lossy().to_string()),
            };
            config.model_config.tokens =
                Some(model_dir.join(spec.tokens).to_string_lossy().to_string());
        }
        config.model_config.provider = Some(provider.clone());
        config.model_config.model_type = spec.model_type.map(|model_type| model_type.to_string());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4) as i32)
            .unwrap_or(2);

        log::info!(
            "Loading Sherpa ONNX model '{}' with provider '{}'",
            model_name,
            provider
        );
        let recognizer = tokio::task::spawn_blocking(move || OfflineRecognizer::create(&config))
            .await
            .map_err(|e| SherpaEngineError::Other(format!("Recognizer task failed: {}", e)))?
            .ok_or_else(|| {
                SherpaEngineError::Other("Failed to create Sherpa recognizer".to_string())
            })?;

        *self.current_model.write().await = Some(Arc::new(recognizer));
        *self.current_model_name.write().await = Some(model_name.to_string());
        Ok(())
    }

    pub async fn unload_model(&self) {
        *self.current_model.write().await = None;
        *self.current_model_name.write().await = None;
    }

    pub async fn transcribe_audio(
        &self,
        audio_data: Vec<f32>,
    ) -> Result<String, SherpaEngineError> {
        let recognizer = self
            .current_model
            .read()
            .await
            .clone()
            .ok_or(SherpaEngineError::ModelNotLoaded)?;

        tokio::task::spawn_blocking(move || {
            let stream = recognizer.create_stream();
            stream.accept_waveform(16000, &audio_data);
            recognizer.decode(&stream);
            stream
                .get_result()
                .map(|result| result.text)
                .ok_or_else(|| {
                    SherpaEngineError::TranscriptionFailed("No result returned".to_string())
                })
        })
        .await
        .map_err(|e| SherpaEngineError::Other(format!("Transcription task failed: {}", e)))?
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.read().await.is_some()
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    pub async fn get_execution_provider(&self) -> String {
        self.execution_provider.read().await.clone()
    }

    pub async fn set_execution_provider(&self, provider: String) -> Result<(), SherpaEngineError> {
        let normalized = provider.to_lowercase();
        if !matches!(normalized.as_str(), "cpu" | "cuda") {
            return Err(SherpaEngineError::Other(format!(
                "Unsupported Sherpa execution provider '{}'. Use 'cpu' or 'cuda'.",
                provider
            )));
        }
        *self.execution_provider.write().await = normalized;
        self.unload_model().await;
        Ok(())
    }

    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    pub async fn download_model<F>(
        &self,
        model_name: &str,
        mut progress: F,
    ) -> Result<(), SherpaEngineError>
    where
        F: FnMut(u8) + Send,
    {
        let spec = find_spec(model_name)?;
        {
            let mut active = self.active_downloads.write().await;
            if !active.insert(model_name.to_string()) {
                return Err(SherpaEngineError::DownloadFailed(
                    "Download already in progress".to_string(),
                ));
            }
        }

        let result = self.download_model_inner(&spec, &mut progress).await;
        self.active_downloads.write().await.remove(model_name);
        result
    }

    async fn download_model_inner<F>(
        &self,
        spec: &SherpaModelSpec,
        progress: &mut F,
    ) -> Result<(), SherpaEngineError>
    where
        F: FnMut(u8) + Send,
    {
        let model_dir = self.models_dir.join(spec.name);
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir).await?;
        }
        fs::create_dir_all(&model_dir).await?;

        let client = reqwest::Client::new();
        let total_files = spec.files.len().max(1) as f32;

        for (index, file) in spec.files.iter().enumerate() {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                spec.repo, file.remote_path
            );
            let response = client
                .get(&url)
                .send()
                .await
                .map_err(|e| SherpaEngineError::DownloadFailed(e.to_string()))?;

            if !response.status().is_success() {
                return Err(SherpaEngineError::DownloadFailed(format!(
                    "Failed to download {}: HTTP {}",
                    file.remote_path,
                    response.status()
                )));
            }

            let total_size = response.content_length().unwrap_or(file.min_bytes.max(1));
            let mut downloaded = 0u64;
            let mut stream = response.bytes_stream();
            let file_path = model_dir.join(file.local_name);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).await?;
            }
            let mut writer = BufWriter::new(fs::File::create(file_path).await?);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| SherpaEngineError::DownloadFailed(e.to_string()))?;
                writer.write_all(&chunk).await?;
                downloaded += chunk.len() as u64;

                let file_progress = (downloaded as f32 / total_size as f32).clamp(0.0, 1.0);
                let overall = (((index as f32 + file_progress) / total_files) * 100.0) as u8;
                progress(overall.min(99));
            }
            writer.flush().await?;
        }

        validate_spec_files(&model_dir, spec).map_err(|_| {
            SherpaEngineError::DownloadFailed("Downloaded model failed validation".to_string())
        })?;
        progress(100);
        Ok(())
    }

    pub async fn delete_model(&self, model_name: &str) -> Result<(), SherpaEngineError> {
        let spec = find_spec(model_name)?;
        let model_dir = self.models_dir.join(spec.name);
        if model_dir.exists() {
            fs::remove_dir_all(model_dir).await?;
        }
        if self.get_current_model().await.as_deref() == Some(model_name) {
            self.unload_model().await;
        }
        Ok(())
    }
}

fn find_spec(model_name: &str) -> Result<SherpaModelSpec, SherpaEngineError> {
    model_specs()
        .into_iter()
        .find(|spec| spec.name == model_name)
        .ok_or_else(|| SherpaEngineError::ModelNotFound(model_name.to_string()))
}

fn validate_spec_files(
    model_dir: &Path,
    spec: &SherpaModelSpec,
) -> std::result::Result<(), (u64, u64)> {
    let mut file_size = 0u64;
    let mut expected_min_size = 0u64;

    for file in &spec.files {
        let min_valid_size = validation_min_bytes(file.min_bytes);
        expected_min_size += min_valid_size;
        let path = model_dir.join(file.local_name);
        match std::fs::metadata(path) {
            Ok(metadata) => {
                file_size += metadata.len();
                if metadata.len() < min_valid_size {
                    return Err((file_size, expected_min_size));
                }
            }
            Err(_) => return Err((file_size, expected_min_size)),
        }
    }

    Ok(())
}

fn validation_min_bytes(spec_min_bytes: u64) -> u64 {
    spec_min_bytes.min((spec_min_bytes / 100).max(1_024))
}
