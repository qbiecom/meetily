use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use tauri::command;

#[derive(Debug, Serialize, Deserialize)]
pub struct KiloGatewayModel {
    pub id: String,
    pub name: String,
    pub context_length: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct KiloGatewayApiModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    context_window: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum KiloGatewayModelsResponse {
    OpenAiStyle { data: Vec<KiloGatewayApiModel> },
    Direct(Vec<KiloGatewayApiModel>),
}

const KILO_GATEWAY_MODELS_URL: &str = "https://api.kilo.ai/api/gateway/models";

fn fallback_models() -> Vec<KiloGatewayModel> {
    vec![
        KiloGatewayModel {
            id: "kilo/auto".to_string(),
            name: "Kilo Auto".to_string(),
            context_length: Some(1_000_000),
        },
        KiloGatewayModel {
            id: "anthropic/claude-sonnet-4.6".to_string(),
            name: "Claude Sonnet 4.6".to_string(),
            context_length: None,
        },
        KiloGatewayModel {
            id: "openai/gpt-5.4".to_string(),
            name: "GPT-5.4".to_string(),
            context_length: None,
        },
        KiloGatewayModel {
            id: "google/gemini-3.1-pro-preview".to_string(),
            name: "Gemini 3.1 Pro Preview".to_string(),
            context_length: None,
        },
        KiloGatewayModel {
            id: "openrouter/free".to_string(),
            name: "OpenRouter Free".to_string(),
            context_length: None,
        },
    ]
}

#[command]
pub fn get_kilo_gateway_models() -> Result<Vec<KiloGatewayModel>, String> {
    let client = Client::new();
    let response = client
        .get(KILO_GATEWAY_MODELS_URL)
        .send()
        .map_err(|e| format!("Failed to make HTTP request: {}", e))?;

    if !response.status().is_success() {
        return Ok(fallback_models());
    }

    let api_response: KiloGatewayModelsResponse = response
        .json()
        .map_err(|e| format!("Failed to parse JSON response: {}", e))?;

    let data = match api_response {
        KiloGatewayModelsResponse::OpenAiStyle { data } => data,
        KiloGatewayModelsResponse::Direct(data) => data,
    };

    let models = data
        .into_iter()
        .map(|m| KiloGatewayModel {
            name: m.name.unwrap_or_else(|| m.id.clone()),
            context_length: m.context_length.or(m.context_window),
            id: m.id,
        })
        .collect::<Vec<_>>();

    if models.is_empty() {
        Ok(fallback_models())
    } else {
        Ok(models)
    }
}
