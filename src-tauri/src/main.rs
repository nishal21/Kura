use serde::{Deserialize, Serialize};
use serde_json::Value;
mod ai_extractor;
mod database;
mod secure_store;

use ai_extractor::{
    analyze_paper_with_target_schema, cache_transient_api_key,
    get_ollama_models as fetch_ollama_models, has_transient_api_key, AiProvider,
    EmbeddingProviderPreference,
};
use database::{
    delete_paper as delete_paper_record, export_library as export_library_records,
    find_similar_chunks, get_all_paper_summaries, init_db, save_paper_embeddings,
    save_paper_record, PaperListItem, SimilarChunkResult,
};
use secure_store::{check_api_key_secure, save_api_key_secure};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SystemInfo {
    os: String,
    architecture: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveApiKeyPayload {
    provider: String,
    #[serde(alias = "api_key")]
    api_key: String,
}

#[tauri::command]
fn ping_backend() -> &'static str {
    "Hello from Kura Backend!"
}

#[tauri::command]
fn get_system_info() -> SystemInfo {
    SystemInfo {
        os: std::env::consts::OS.to_string(),
        architecture: std::env::consts::ARCH.to_string(),
    }
}

#[tauri::command]
fn pick_pdf_file() -> Result<Option<String>, String> {
    let selected = rfd::FileDialog::new()
        .add_filter("PDF", &["pdf"])
        .pick_file()
        .map(|path| path.to_string_lossy().to_string());

    Ok(selected)
}

#[tauri::command]
fn read_local_pdf(path: String) -> Result<Vec<u8>, String> {
    let normalized_path = path.trim();
    if normalized_path.is_empty() {
        return Err("path is required".to_string());
    }

    std::fs::read(normalized_path).map_err(|error| error.to_string())
}

#[tauri::command]
fn save_api_key(
    provider: Option<String>,
    api_key: Option<String>,
    #[allow(non_snake_case)] apiKey: Option<String>,
    payload: Option<SaveApiKeyPayload>,
) -> Result<(), String> {
    let resolved_provider = payload
        .as_ref()
        .map(|value| value.provider.as_str())
        .or(provider.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "provider is required".to_string())?;

    let provider_type = resolved_provider
        .parse::<AiProvider>()
        .map_err(|error| error.to_string())?;

    let resolved_api_key = payload
        .as_ref()
        .map(|value| value.api_key.as_str())
        .or(apiKey.as_deref())
        .or(api_key.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "api_key is required".to_string())?;

    save_api_key_secure(provider_type.as_vault_id(), resolved_api_key)
        .map_err(|error| error.to_string())?;

    // Keep an in-process fallback to avoid false negatives while OS vault readback catches up.
    cache_transient_api_key(provider_type, resolved_api_key);

    const VERIFY_ATTEMPTS: usize = 12;
    const VERIFY_DELAY_MS: u64 = 200;
    let mut verified = false;

    // Credential managers can be briefly stale after a successful write.
    for attempt in 0..VERIFY_ATTEMPTS {
        match check_api_key_secure(provider_type.as_vault_id()) {
            Ok(true) => {
                verified = true;
                break;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!(
                    "Warning: API key write for '{}' succeeded, but immediate verification failed: {}",
                    provider_type.as_vault_id(),
                    error
                );
                break;
            }
        }

        if attempt + 1 < VERIFY_ATTEMPTS {
            std::thread::sleep(std::time::Duration::from_millis(VERIFY_DELAY_MS));
        }
    }

    if !verified {
        eprintln!(
            "Warning: API key write for '{}' was not yet visible during immediate verification.",
            provider_type.as_vault_id()
        );
    }

    Ok(())
}

#[tauri::command]
fn check_api_key(provider: String) -> Result<bool, String> {
    let provider_type = provider
        .trim()
        .parse::<AiProvider>()
        .map_err(|error| error.to_string())?;

    if check_api_key_secure(provider_type.as_vault_id()).map_err(|error| error.to_string())? {
        return Ok(true);
    }

    for legacy_provider in provider_type.legacy_vault_ids() {
        if check_api_key_secure(legacy_provider).map_err(|error| error.to_string())? {
            return Ok(true);
        }
    }

    if has_transient_api_key(provider_type) {
        return Ok(true);
    }

    Ok(false)
}

#[tauri::command]
fn save_paper_to_db(
    app: tauri::AppHandle,
    local_file_path: String,
    title: String,
    extracted_data: String,
) -> Result<i64, String> {
    save_paper_record(&app, &local_file_path, &title, &extracted_data)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn get_all_papers(app: tauri::AppHandle) -> Result<Vec<PaperListItem>, String> {
    get_all_paper_summaries(&app).map_err(|error| error.to_string())
}

#[tauri::command]
fn export_library(app_handle: tauri::AppHandle, export_format: String) -> Result<String, String> {
    export_library_records(&app_handle, &export_format).map_err(|error| error.to_string())
}

#[tauri::command]
fn delete_paper(app_handle: tauri::AppHandle, paper_id: i64) -> Result<(), String> {
    delete_paper_record(&app_handle, paper_id).map_err(|error| error.to_string())
}

#[tauri::command]
fn save_paper_embeddings_to_db(
    app: tauri::AppHandle,
    paper_id: i64,
    text_chunks: Vec<String>,
    embeddings: Vec<Vec<f32>>,
) -> Result<(), String> {
    save_paper_embeddings(&app, paper_id, text_chunks, embeddings).map_err(|error| error.to_string())
}

#[tauri::command]
fn find_similar_chunks_in_db(
    app: tauri::AppHandle,
    query_embedding: Vec<f32>,
    top_k: usize,
) -> Result<Vec<SimilarChunkResult>, String> {
    find_similar_chunks(&app, query_embedding, top_k).map_err(|error| error.to_string())
}

#[tauri::command]
async fn analyze_paper(
    provider: String,
    model_name: String,
    pdf_text: String,
    target_schema: Vec<String>,
    embedding_provider: Option<EmbeddingProviderPreference>,
) -> Result<Value, String> {
    let provider_type = provider.parse::<AiProvider>().map_err(|error| error.to_string())?;

    analyze_paper_with_target_schema(
        provider_type,
        &model_name,
        &pdf_text,
        &target_schema,
        embedding_provider.as_ref(),
    )
    .await
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_ollama_models() -> Result<Vec<String>, String> {
    fetch_ollama_models()
        .await
        .map_err(|error| error.to_string())
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            init_db(&app.handle()).map_err(|error| -> Box<dyn std::error::Error> {
                Box::new(error)
            })?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ping_backend,
            get_system_info,
            pick_pdf_file,
            read_local_pdf,
            save_api_key,
            check_api_key,
            save_paper_to_db,
            get_all_papers,
            export_library,
            delete_paper,
            save_paper_embeddings_to_db,
            find_similar_chunks_in_db,
            analyze_paper,
            ai_extractor::chat_with_library,
            get_ollama_models,
            ai_extractor::initialize_local_embedding,
            ai_extractor::process_paper_embeddings
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Kura Tauri application");
}
