use crate::database::models::{Setting, TranscriptSetting};
use crate::summary::CustomOpenAIConfig;
use sqlx::SqlitePool;

#[derive(serde::Deserialize, Debug)]
pub struct SaveModelConfigRequest {
    pub provider: String,
    pub model: String,
    #[serde(rename = "whisperModel")]
    pub whisper_model: String,
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
    #[serde(rename = "ollamaEndpoint")]
    pub ollama_endpoint: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct SaveTranscriptConfigRequest {
    pub provider: String,
    pub model: String,
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
}

pub struct SettingsRepository;

// Transcript providers: localWhisper, deepgram, elevenLabs, groq, openai
// Summary providers: openai, claude, ollama, groq, openrouter, kilo-gateway
// NOTE: Handle data exclusion in the higher layer as this is database abstraction layer(using SELECT *)

impl SettingsRepository {
    pub async fn get_model_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<Setting>, sqlx::Error> {
        let setting = sqlx::query_as::<_, Setting>("SELECT * FROM settings LIMIT 1")
            .fetch_optional(pool)
            .await?;
        Ok(setting)
    }

    pub async fn save_model_config(
        pool: &SqlitePool,
        provider: &str,
        model: &str,
        whisper_model: &str,
        ollama_endpoint: Option<&str>,
    ) -> std::result::Result<(), sqlx::Error> {
        // Using id '1' for backward compatibility
        sqlx::query(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, ollamaEndpoint)
            VALUES ('1', $1, $2, $3, $4)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                whisperModel = excluded.whisperModel,
                ollamaEndpoint = excluded.ollamaEndpoint
            "#,
        )
        .bind(provider)
        .bind(model)
        .bind(whisper_model)
        .bind(ollama_endpoint)
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn save_api_key(
        pool: &SqlitePool,
        provider: &str,
        api_key: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        // Custom OpenAI uses JSON config (customOpenAIConfig) instead of a separate API key column
        if provider == "custom-openai" {
            return Err(sqlx::Error::Protocol(
                "custom-openai provider should use save_custom_openai_config() instead of save_api_key()".into(),
            ));
        }

        match provider {
            "openai" => {
                sqlx::query(
                    r#"
                    INSERT INTO settings (id, provider, model, whisperModel, openaiApiKey)
                    VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
                    ON CONFLICT(id) DO UPDATE SET
                        openaiApiKey = $1
                    "#,
                )
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "claude" => {
                sqlx::query(
                    r#"
                    INSERT INTO settings (id, provider, model, whisperModel, anthropicApiKey)
                    VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
                    ON CONFLICT(id) DO UPDATE SET
                        anthropicApiKey = $1
                    "#,
                )
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "ollama" => {
                sqlx::query(
                    r#"
                    INSERT INTO settings (id, provider, model, whisperModel, ollamaApiKey)
                    VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
                    ON CONFLICT(id) DO UPDATE SET
                        ollamaApiKey = $1
                    "#,
                )
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "groq" => {
                sqlx::query(
                    r#"
                    INSERT INTO settings (id, provider, model, whisperModel, groqApiKey)
                    VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
                    ON CONFLICT(id) DO UPDATE SET
                        groqApiKey = $1
                    "#,
                )
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "openrouter" => {
                sqlx::query(
                    r#"
                    INSERT INTO settings (id, provider, model, whisperModel, openRouterApiKey)
                    VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
                    ON CONFLICT(id) DO UPDATE SET
                        openRouterApiKey = $1
                    "#,
                )
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "kilo-gateway" => {
                sqlx::query(
                    r#"
                    INSERT INTO settings (id, provider, model, whisperModel, kiloGatewayApiKey)
                    VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
                    ON CONFLICT(id) DO UPDATE SET
                        kiloGatewayApiKey = $1
                    "#,
                )
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "builtin-ai" => return Ok(()), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        }

        Ok(())
    }

    pub async fn get_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        // Custom OpenAI uses JSON config - extract API key from there
        if provider == "custom-openai" {
            let config = Self::get_custom_openai_config(pool).await?;
            return Ok(config.and_then(|c| c.api_key));
        }

        let api_key = match provider {
            "openai" => {
                sqlx::query_scalar("SELECT openaiApiKey FROM settings WHERE id = '1' LIMIT 1")
                    .fetch_optional(pool)
                    .await?
            }
            "ollama" => {
                sqlx::query_scalar("SELECT ollamaApiKey FROM settings WHERE id = '1' LIMIT 1")
                    .fetch_optional(pool)
                    .await?
            }
            "groq" => {
                sqlx::query_scalar("SELECT groqApiKey FROM settings WHERE id = '1' LIMIT 1")
                    .fetch_optional(pool)
                    .await?
            }
            "claude" => {
                sqlx::query_scalar("SELECT anthropicApiKey FROM settings WHERE id = '1' LIMIT 1")
                    .fetch_optional(pool)
                    .await?
            }
            "openrouter" => {
                sqlx::query_scalar("SELECT openRouterApiKey FROM settings WHERE id = '1' LIMIT 1")
                    .fetch_optional(pool)
                    .await?
            }
            "kilo-gateway" => {
                sqlx::query_scalar("SELECT kiloGatewayApiKey FROM settings WHERE id = '1' LIMIT 1")
                    .fetch_optional(pool)
                    .await?
            }
            "builtin-ai" => return Ok(None), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };
        Ok(api_key)
    }

    pub async fn get_transcript_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<TranscriptSetting>, sqlx::Error> {
        let setting =
            sqlx::query_as::<_, TranscriptSetting>("SELECT * FROM transcript_settings LIMIT 1")
                .fetch_optional(pool)
                .await?;
        Ok(setting)
    }

    pub async fn save_transcript_config(
        pool: &SqlitePool,
        provider: &str,
        model: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model)
            VALUES ('1', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model
            "#,
        )
        .bind(provider)
        .bind(model)
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn save_transcript_api_key(
        pool: &SqlitePool,
        provider: &str,
        api_key: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        match provider {
            "localWhisper" => {
                sqlx::query(
                    r#"
                    INSERT INTO transcript_settings (id, provider, model, whisperApiKey)
                    VALUES ('1', 'parakeet', $1, $2)
                    ON CONFLICT(id) DO UPDATE SET
                        whisperApiKey = $2
                    "#,
                )
                .bind(crate::config::DEFAULT_PARAKEET_MODEL)
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "parakeet" => return Ok(()), // Parakeet doesn't need an API key, return early
            "deepgram" => {
                sqlx::query(
                    r#"
                    INSERT INTO transcript_settings (id, provider, model, deepgramApiKey)
                    VALUES ('1', 'parakeet', $1, $2)
                    ON CONFLICT(id) DO UPDATE SET
                        deepgramApiKey = $2
                    "#,
                )
                .bind(crate::config::DEFAULT_PARAKEET_MODEL)
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "elevenLabs" => {
                sqlx::query(
                    r#"
                    INSERT INTO transcript_settings (id, provider, model, elevenLabsApiKey)
                    VALUES ('1', 'parakeet', $1, $2)
                    ON CONFLICT(id) DO UPDATE SET
                        elevenLabsApiKey = $2
                    "#,
                )
                .bind(crate::config::DEFAULT_PARAKEET_MODEL)
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "groq" => {
                sqlx::query(
                    r#"
                    INSERT INTO transcript_settings (id, provider, model, groqApiKey)
                    VALUES ('1', 'parakeet', $1, $2)
                    ON CONFLICT(id) DO UPDATE SET
                        groqApiKey = $2
                    "#,
                )
                .bind(crate::config::DEFAULT_PARAKEET_MODEL)
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            "openai" => {
                sqlx::query(
                    r#"
                    INSERT INTO transcript_settings (id, provider, model, openaiApiKey)
                    VALUES ('1', 'parakeet', $1, $2)
                    ON CONFLICT(id) DO UPDATE SET
                        openaiApiKey = $2
                    "#,
                )
                .bind(crate::config::DEFAULT_PARAKEET_MODEL)
                .bind(api_key)
                .execute(pool)
                .await?;
            }
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        }

        Ok(())
    }

    pub async fn get_transcript_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        let api_key = match provider {
            "localWhisper" => {
                sqlx::query_scalar(
                    "SELECT whisperApiKey FROM transcript_settings WHERE id = '1' LIMIT 1",
                )
                .fetch_optional(pool)
                .await?
            }
            "parakeet" => return Ok(None), // Parakeet doesn't need an API key
            "deepgram" => {
                sqlx::query_scalar(
                    "SELECT deepgramApiKey FROM transcript_settings WHERE id = '1' LIMIT 1",
                )
                .fetch_optional(pool)
                .await?
            }
            "elevenLabs" => {
                sqlx::query_scalar(
                    "SELECT elevenLabsApiKey FROM transcript_settings WHERE id = '1' LIMIT 1",
                )
                .fetch_optional(pool)
                .await?
            }
            "groq" => {
                sqlx::query_scalar(
                    "SELECT groqApiKey FROM transcript_settings WHERE id = '1' LIMIT 1",
                )
                .fetch_optional(pool)
                .await?
            }
            "openai" => {
                sqlx::query_scalar(
                    "SELECT openaiApiKey FROM transcript_settings WHERE id = '1' LIMIT 1",
                )
                .fetch_optional(pool)
                .await?
            }
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };
        Ok(api_key)
    }

    pub async fn delete_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        // Custom OpenAI uses JSON config - clear the entire config
        if provider == "custom-openai" {
            sqlx::query("UPDATE settings SET customOpenAIConfig = NULL WHERE id = '1'")
                .execute(pool)
                .await?;
            return Ok(());
        }

        match provider {
            "openai" => sqlx::query("UPDATE settings SET openaiApiKey = NULL WHERE id = '1'"),
            "ollama" => sqlx::query("UPDATE settings SET ollamaApiKey = NULL WHERE id = '1'"),
            "groq" => sqlx::query("UPDATE settings SET groqApiKey = NULL WHERE id = '1'"),
            "claude" => sqlx::query("UPDATE settings SET anthropicApiKey = NULL WHERE id = '1'"),
            "openrouter" => {
                sqlx::query("UPDATE settings SET openRouterApiKey = NULL WHERE id = '1'")
            }
            "kilo-gateway" => {
                sqlx::query("UPDATE settings SET kiloGatewayApiKey = NULL WHERE id = '1'")
            }
            "builtin-ai" => return Ok(()), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        }
        .execute(pool)
        .await?;

        Ok(())
    }

    // ===== CUSTOM OPENAI CONFIG METHODS =====

    /// Gets the custom OpenAI configuration from JSON
    ///
    /// # Returns
    /// * `Ok(Some(CustomOpenAIConfig))` - Config exists and is valid JSON
    /// * `Ok(None)` - No config stored
    /// * `Err(sqlx::Error)` - Database error
    pub async fn get_custom_openai_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<CustomOpenAIConfig>, sqlx::Error> {
        use sqlx::Row;

        let row = sqlx::query(
            r#"
            SELECT customOpenAIConfig
            FROM settings
            WHERE id = '1'
            LIMIT 1
            "#,
        )
        .fetch_optional(pool)
        .await?;

        match row {
            Some(record) => {
                let config_json: Option<String> = record.get("customOpenAIConfig");

                if let Some(json) = config_json {
                    // Parse JSON into CustomOpenAIConfig
                    let config: CustomOpenAIConfig = serde_json::from_str(&json).map_err(|e| {
                        sqlx::Error::Protocol(
                            format!("Invalid JSON in customOpenAIConfig: {}", e).into(),
                        )
                    })?;

                    Ok(Some(config))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Saves the custom OpenAI configuration as JSON
    ///
    /// # Arguments
    /// * `pool` - Database connection pool
    /// * `config` - CustomOpenAIConfig to save (includes endpoint, apiKey, model, maxTokens, temperature, topP)
    ///
    /// # Returns
    /// * `Ok(())` - Config saved successfully
    /// * `Err(sqlx::Error)` - Database or JSON serialization error
    pub async fn save_custom_openai_config(
        pool: &SqlitePool,
        config: &CustomOpenAIConfig,
    ) -> std::result::Result<(), sqlx::Error> {
        // Serialize config to JSON
        let config_json = serde_json::to_string(config).map_err(|e| {
            sqlx::Error::Protocol(format!("Failed to serialize config to JSON: {}", e).into())
        })?;

        // Upsert into settings table
        sqlx::query(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, customOpenAIConfig)
            VALUES ('1', 'custom-openai', $1, 'large-v3', $2)
            ON CONFLICT(id) DO UPDATE SET
                customOpenAIConfig = excluded.customOpenAIConfig
            "#,
        )
        .bind(&config.model)
        .bind(config_json)
        .execute(pool)
        .await?;

        Ok(())
    }
}
