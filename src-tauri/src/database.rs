use csv::Writer;
use rust_xlsxwriter::{Format, Workbook};
use rusqlite::{params, Connection, ErrorCode};
use serde::Serialize;
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

#[derive(Debug)]
pub enum DatabaseError {
    PathResolve { details: String },
    CreateDirectory { path: PathBuf, source: std::io::Error },
    OpenConnection { path: PathBuf, source: rusqlite::Error },
    InitializeSchema { source: rusqlite::Error },
    DeletePaper { source: rusqlite::Error },
    InsertPaper { source: rusqlite::Error },
    InsertDocumentChunks { source: rusqlite::Error },
    PaperAlreadySaved { local_file_path: String },
    QueryPapers { source: rusqlite::Error },
    QueryDocumentChunks { source: rusqlite::Error },
    InvalidExportFormat { format: String },
    ExportFailure { details: String },
    InvalidEmbeddingInput { details: String },
    InvalidPaperId { details: String },
    InvalidEmbeddingBlob { details: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperListItem {
    pub id: i64,
    pub title: String,
    pub extracted_data: String,
    pub date_added: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SimilarChunkResult {
    pub paper_title: String,
    pub chunk_text: String,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
struct ExportPaperRow {
    title: String,
    local_file_path: String,
    date_added: String,
    extracted_data: BTreeMap<String, String>,
}

impl Display for DatabaseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathResolve { details } => {
                write!(f, "Failed to resolve app local data directory: {}", details)
            }
            Self::CreateDirectory { path, source } => {
                write!(
                    f,
                    "Failed to create database directory '{}': {}",
                    path.display(),
                    source
                )
            }
            Self::OpenConnection { path, source } => {
                write!(
                    f,
                    "Failed to open SQLite database at '{}': {}",
                    path.display(),
                    source
                )
            }
            Self::InitializeSchema { source } => {
                write!(f, "Failed to initialize SQLite schema: {}", source)
            }
            Self::DeletePaper { source } => {
                write!(f, "Failed to delete paper record from database: {}", source)
            }
            Self::InsertPaper { source } => {
                write!(f, "Failed to insert paper into database: {}", source)
            }
            Self::InsertDocumentChunks { source } => {
                write!(f, "Failed to insert document chunks into database: {}", source)
            }
            Self::PaperAlreadySaved { local_file_path } => {
                write!(
                    f,
                    "A paper with this local file path is already saved: {}",
                    local_file_path
                )
            }
            Self::QueryPapers { source } => {
                write!(f, "Failed to query saved papers: {}", source)
            }
            Self::QueryDocumentChunks { source } => {
                write!(f, "Failed to query document chunks: {}", source)
            }
            Self::InvalidExportFormat { format } => {
                write!(f, "Unsupported export format '{}'. Use csv, json, xlsx, or md.", format)
            }
            Self::ExportFailure { details } => {
                write!(f, "Failed to export library: {}", details)
            }
            Self::InvalidEmbeddingInput { details } => {
                write!(f, "Invalid embedding input: {}", details)
            }
            Self::InvalidPaperId { details } => {
                write!(f, "Invalid paper identifier: {}", details)
            }
            Self::InvalidEmbeddingBlob { details } => {
                write!(f, "Invalid embedding blob in database: {}", details)
            }
        }
    }
}

impl std::error::Error for DatabaseError {}

pub fn save_paper_record(
    app_handle: &AppHandle,
    local_file_path: &str,
    title: &str,
    extracted_data: &str,
) -> Result<i64, DatabaseError> {
    let connection = open_db_connection(app_handle)?;

    let normalized_path = local_file_path.trim();
    let normalized_title = title.trim();
    let normalized_extracted_data = extracted_data.trim();

    let insert_result = connection.execute(
        "
        INSERT INTO papers (
            title,
            local_file_path,
            extracted_data,
            date_added
        ) VALUES (?1, ?2, ?3, CURRENT_TIMESTAMP)
        ",
        params![normalized_title, normalized_path, normalized_extracted_data],
    );

    match insert_result {
        Ok(_) => Ok(connection.last_insert_rowid()),
        Err(error) if is_unique_local_file_path_violation(&error) => {
            Err(DatabaseError::PaperAlreadySaved {
                local_file_path: normalized_path.to_string(),
            })
        }
        Err(source) => Err(DatabaseError::InsertPaper { source }),
    }
}

pub fn get_all_paper_summaries(app_handle: &AppHandle) -> Result<Vec<PaperListItem>, DatabaseError> {
    let connection = open_db_connection(app_handle)?;

    let mut statement = connection
        .prepare(
            "
            SELECT
                id,
                COALESCE(title, '') AS title,
                COALESCE(extracted_data, '{}') AS extracted_data,
                COALESCE(date_added, '') AS date_added
            FROM papers
            ORDER BY datetime(date_added) DESC, id DESC
            ",
        )
        .map_err(|source| DatabaseError::QueryPapers { source })?;

    let mapped_rows = statement
        .query_map([], |row| {
            Ok(PaperListItem {
                id: row.get(0)?,
                title: row.get(1)?,
                extracted_data: row.get(2)?,
                date_added: row.get(3)?,
            })
        })
        .map_err(|source| DatabaseError::QueryPapers { source })?;

    let mut papers = Vec::new();
    for mapped_row in mapped_rows {
        papers.push(mapped_row.map_err(|source| DatabaseError::QueryPapers { source })?);
    }

    Ok(papers)
}

pub fn export_library(app_handle: &AppHandle, export_format: &str) -> Result<String, DatabaseError> {
    let normalized_format = export_format.trim().to_ascii_lowercase();
    if !matches!(normalized_format.as_str(), "csv" | "json" | "xlsx" | "md") {
        return Err(DatabaseError::InvalidExportFormat {
            format: normalized_format,
        });
    }

    let save_path = match pick_export_path(&normalized_format) {
        Some(path) => path,
        None => return Ok("Cancelled".to_string()),
    };

    let rows = get_all_papers_for_export(app_handle)?;
    let dynamic_headers = collect_dynamic_headers(&rows);

    match normalized_format.as_str() {
        "csv" => write_csv_export(&save_path, &dynamic_headers, &rows)?,
        "xlsx" => write_xlsx_export(&save_path, &dynamic_headers, &rows)?,
        "json" => write_json_export(&save_path, &rows)?,
        "md" => write_markdown_export(&save_path, &dynamic_headers, &rows)?,
        _ => {
            return Err(DatabaseError::InvalidExportFormat {
                format: normalized_format,
            });
        }
    }

    Ok(save_path.to_string_lossy().to_string())
}

pub fn delete_paper(app_handle: &AppHandle, paper_id: i64) -> Result<(), DatabaseError> {
    if paper_id <= 0 {
        return Err(DatabaseError::InvalidPaperId {
            details: "paper_id must be a positive integer".to_string(),
        });
    }

    let mut connection = open_db_connection(app_handle)?;
    let transaction = connection
        .transaction()
        .map_err(|source| DatabaseError::DeletePaper { source })?;

    transaction
        .execute(
            "DELETE FROM document_chunks WHERE paper_id = ?1",
            params![paper_id],
        )
        .map_err(|source| DatabaseError::DeletePaper { source })?;

    transaction
        .execute("DELETE FROM papers WHERE id = ?1", params![paper_id])
        .map_err(|source| DatabaseError::DeletePaper { source })?;

    transaction
        .commit()
        .map_err(|source| DatabaseError::DeletePaper { source })
}

fn pick_export_path(export_format: &str) -> Option<PathBuf> {
    match export_format {
        "csv" => rfd::FileDialog::new()
            .set_file_name("kura-library.csv")
            .add_filter("CSV", &["csv"])
            .save_file(),
        "json" => rfd::FileDialog::new()
            .set_file_name("kura-library.json")
            .add_filter("JSON", &["json"])
            .save_file(),
        "xlsx" => rfd::FileDialog::new()
            .set_file_name("kura-library.xlsx")
            .add_filter("Excel Workbook", &["xlsx"])
            .save_file(),
        "md" => rfd::FileDialog::new()
            .set_file_name("kura-library.md")
            .add_filter("Markdown", &["md"])
            .save_file(),
        _ => None,
    }
}

fn get_all_papers_for_export(app_handle: &AppHandle) -> Result<Vec<ExportPaperRow>, DatabaseError> {
    let connection = open_db_connection(app_handle)?;
    let mut statement = connection
        .prepare(
            "
            SELECT
                COALESCE(title, '') AS title,
                COALESCE(local_file_path, '') AS local_file_path,
                COALESCE(extracted_data, '{}') AS extracted_data,
                COALESCE(date_added, '') AS date_added
            FROM papers
            ORDER BY datetime(date_added) DESC, id DESC
            ",
        )
        .map_err(|source| DatabaseError::QueryPapers { source })?;

    let mapped_rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|source| DatabaseError::QueryPapers { source })?;

    let mut papers = Vec::new();
    for mapped_row in mapped_rows {
        let (title, local_file_path, extracted_data_raw, date_added) =
            mapped_row.map_err(|source| DatabaseError::QueryPapers { source })?;

        papers.push(ExportPaperRow {
            title,
            local_file_path,
            date_added,
            extracted_data: parse_extracted_data_map(&extracted_data_raw),
        });
    }

    Ok(papers)
}

fn parse_extracted_data_map(raw_json: &str) -> BTreeMap<String, String> {
    let parsed = match serde_json::from_str::<Value>(raw_json) {
        Ok(value) => value,
        Err(_) => return BTreeMap::new(),
    };

    match parsed {
        Value::Object(entries) => entries
            .into_iter()
            .map(|(key, value)| (key, json_value_to_cell(value)))
            .collect(),
        _ => BTreeMap::new(),
    }
}

fn json_value_to_cell(value: Value) -> String {
    let rendered = match value {
        Value::Null => "Not mentioned".to_string(),
        Value::String(text) => text,
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(&value).unwrap_or_default(),
    };

    let normalized = rendered.trim();
    if normalized.is_empty() {
        "Not mentioned".to_string()
    } else {
        normalized.to_string()
    }
}

fn collect_dynamic_headers(rows: &[ExportPaperRow]) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for row in rows {
        for key in row.extracted_data.keys() {
            keys.insert(key.clone());
        }
    }

    keys.into_iter().collect()
}

fn build_export_headers(dynamic_headers: &[String]) -> Vec<String> {
    let mut headers = vec!["Title".to_string(), "Local Path".to_string()];
    headers.extend(dynamic_headers.iter().cloned());
    headers
}

fn build_row_values(row: &ExportPaperRow, dynamic_headers: &[String]) -> Vec<String> {
    let mut values = Vec::with_capacity(2 + dynamic_headers.len());
    values.push(row.title.clone());
    values.push(row.local_file_path.clone());

    for key in dynamic_headers {
        values.push(
            row.extracted_data
                .get(key)
                .cloned()
                .unwrap_or_else(|| "Not mentioned".to_string()),
        );
    }

    values
}

fn write_csv_export(
    output_path: &Path,
    dynamic_headers: &[String],
    rows: &[ExportPaperRow],
) -> Result<(), DatabaseError> {
    let mut writer = Writer::from_path(output_path).map_err(|error| DatabaseError::ExportFailure {
        details: error.to_string(),
    })?;

    let headers = build_export_headers(dynamic_headers);
    writer
        .write_record(&headers)
        .map_err(|error| DatabaseError::ExportFailure {
            details: error.to_string(),
        })?;

    for row in rows {
        let values = build_row_values(row, dynamic_headers);
        writer
            .write_record(values)
            .map_err(|error| DatabaseError::ExportFailure {
                details: error.to_string(),
            })?;
    }

    writer.flush().map_err(|error| DatabaseError::ExportFailure {
        details: error.to_string(),
    })
}

fn write_xlsx_export(
    output_path: &Path,
    dynamic_headers: &[String],
    rows: &[ExportPaperRow],
) -> Result<(), DatabaseError> {
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();
    let header_format = Format::new().set_bold();

    let headers = build_export_headers(dynamic_headers);
    for (column_index, header) in headers.iter().enumerate() {
        worksheet
            .write_string_with_format(0, column_index as u16, header, &header_format)
            .map_err(|error| DatabaseError::ExportFailure {
                details: error.to_string(),
            })?;
    }

    for (row_offset, row) in rows.iter().enumerate() {
        let values = build_row_values(row, dynamic_headers);
        let sheet_row = (row_offset + 1) as u32;

        for (column_index, value) in values.iter().enumerate() {
            worksheet
                .write_string(sheet_row, column_index as u16, value)
                .map_err(|error| DatabaseError::ExportFailure {
                    details: error.to_string(),
                })?;
        }
    }

    workbook
        .save(output_path)
        .map_err(|error| DatabaseError::ExportFailure {
            details: error.to_string(),
        })
}

fn write_json_export(output_path: &Path, rows: &[ExportPaperRow]) -> Result<(), DatabaseError> {
    let mut exported_rows = Vec::with_capacity(rows.len());

    for row in rows {
        let mut item = Map::new();
        item.insert("title".to_string(), Value::String(row.title.clone()));
        item.insert("date".to_string(), Value::String(row.date_added.clone()));

        for (key, value) in &row.extracted_data {
            let export_key = if item.contains_key(key) {
                format!("extracted_{}", key)
            } else {
                key.clone()
            };

            item.insert(export_key, Value::String(value.clone()));
        }

        exported_rows.push(Value::Object(item));
    }

    let file = fs::File::create(output_path).map_err(|error| DatabaseError::ExportFailure {
        details: error.to_string(),
    })?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &exported_rows).map_err(|error| DatabaseError::ExportFailure {
        details: error.to_string(),
    })
}

fn write_markdown_export(
    output_path: &Path,
    dynamic_headers: &[String],
    rows: &[ExportPaperRow],
) -> Result<(), DatabaseError> {
    let headers = build_export_headers(dynamic_headers);
    let mut markdown = String::new();

    markdown.push('|');
    for header in &headers {
        markdown.push(' ');
        markdown.push_str(&escape_markdown_cell(header));
        markdown.push(' ');
        markdown.push('|');
    }
    markdown.push('\n');

    markdown.push('|');
    for _ in &headers {
        markdown.push_str(" --- |");
    }
    markdown.push('\n');

    for row in rows {
        let values = build_row_values(row, dynamic_headers);
        markdown.push('|');
        for value in &values {
            markdown.push(' ');
            markdown.push_str(&escape_markdown_cell(value));
            markdown.push(' ');
            markdown.push('|');
        }
        markdown.push('\n');
    }

    fs::write(output_path, markdown).map_err(|error| DatabaseError::ExportFailure {
        details: error.to_string(),
    })
}

fn escape_markdown_cell(value: &str) -> String {
    value
        .replace('\r', " ")
        .replace('\n', " ")
        .replace('|', "\\|")
}

pub fn save_paper_embeddings(
    app_handle: &AppHandle,
    paper_id: i64,
    text_chunks: Vec<String>,
    embeddings: Vec<Vec<f32>>,
) -> Result<(), DatabaseError> {
    if paper_id <= 0 {
        return Err(DatabaseError::InvalidEmbeddingInput {
            details: "paper_id must be a positive integer".to_string(),
        });
    }

    if text_chunks.len() != embeddings.len() {
        return Err(DatabaseError::InvalidEmbeddingInput {
            details: format!(
                "chunks/embeddings length mismatch ({} chunks, {} embeddings)",
                text_chunks.len(),
                embeddings.len()
            ),
        });
    }

    let mut connection = open_db_connection(app_handle)?;
    let transaction = connection
        .transaction()
        .map_err(|source| DatabaseError::InsertDocumentChunks { source })?;

    transaction
        .execute(
            "DELETE FROM document_chunks WHERE paper_id = ?1",
            params![paper_id],
        )
        .map_err(|source| DatabaseError::InsertDocumentChunks { source })?;

    for (chunk, embedding) in text_chunks.into_iter().zip(embeddings.into_iter()) {
        let normalized_chunk = chunk.trim();

        if normalized_chunk.is_empty() || embedding.is_empty() {
            continue;
        }

        let embedding_blob = serialize_embedding_to_blob(&embedding);

        transaction
            .execute(
                "
                INSERT INTO document_chunks (paper_id, chunk_text, embedding)
                VALUES (?1, ?2, ?3)
                ",
                params![paper_id, normalized_chunk, embedding_blob],
            )
            .map_err(|source| DatabaseError::InsertDocumentChunks { source })?;
    }

    transaction
        .commit()
        .map_err(|source| DatabaseError::InsertDocumentChunks { source })
}

pub fn find_similar_chunks(
    app_handle: &AppHandle,
    query_embedding: Vec<f32>,
    top_k: usize,
) -> Result<Vec<SimilarChunkResult>, DatabaseError> {
    if top_k == 0 {
        return Ok(Vec::new());
    }

    if query_embedding.is_empty() {
        return Err(DatabaseError::InvalidEmbeddingInput {
            details: "query_embedding must not be empty".to_string(),
        });
    }

    let connection = open_db_connection(app_handle)?;
    let mut statement = connection
        .prepare(
            "
            SELECT
                COALESCE(p.title, '') AS paper_title,
                COALESCE(dc.chunk_text, '') AS chunk_text,
                dc.embedding
            FROM document_chunks dc
            INNER JOIN papers p ON p.id = dc.paper_id
            ",
        )
        .map_err(|source| DatabaseError::QueryDocumentChunks { source })?;

    let mapped_rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })
        .map_err(|source| DatabaseError::QueryDocumentChunks { source })?;

    let mut ranked_chunks = Vec::new();
    for mapped_row in mapped_rows {
        let (paper_title, chunk_text, embedding_blob) =
            mapped_row.map_err(|source| DatabaseError::QueryDocumentChunks { source })?;

        let stored_embedding = deserialize_embedding_blob(&embedding_blob)?;
        if let Some(similarity) = cosine_similarity(&query_embedding, &stored_embedding) {
            ranked_chunks.push(SimilarChunkResult {
                paper_title,
                chunk_text,
                similarity,
            });
        }
    }

    ranked_chunks.sort_by(|left, right| {
        right
            .similarity
            .partial_cmp(&left.similarity)
            .unwrap_or(Ordering::Equal)
    });
    ranked_chunks.truncate(top_k.min(ranked_chunks.len()));

    Ok(ranked_chunks)
}

pub fn init_db(app_handle: &AppHandle) -> Result<(), DatabaseError> {
    let connection = open_db_connection(app_handle)?;

    let needs_migration = papers_table_requires_migration(&connection)
        .map_err(|source| DatabaseError::InitializeSchema { source })?;

    if needs_migration {
        connection
            .execute_batch(
                "
                DROP TABLE IF EXISTS document_chunks;
                DROP TABLE IF EXISTS papers;
                ",
            )
            .map_err(|source| DatabaseError::InitializeSchema { source })?;
    }

    connection
        .execute_batch(
            "
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY,
                title TEXT,
                local_file_path TEXT UNIQUE,
                extracted_data TEXT NOT NULL,
                date_added DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY,
                paper_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            );

            CREATE INDEX IF NOT EXISTS idx_document_chunks_paper_id
            ON document_chunks (paper_id);
            ",
        )
        .map_err(|source| DatabaseError::InitializeSchema { source })?;

    Ok(())
}

fn open_db_connection(app_handle: &AppHandle) -> Result<Connection, DatabaseError> {
    let database_path = db_path(app_handle)?;

    Connection::open(&database_path).map_err(|source| DatabaseError::OpenConnection {
        path: database_path,
        source,
    })
}

fn db_path(app_handle: &AppHandle) -> Result<PathBuf, DatabaseError> {
    let local_app_data_dir = app_handle
        .path()
        .app_local_data_dir()
        .map_err(|error| DatabaseError::PathResolve {
            details: error.to_string(),
        })?;

    fs::create_dir_all(&local_app_data_dir).map_err(|source| DatabaseError::CreateDirectory {
        path: local_app_data_dir.clone(),
        source,
    })?;

    Ok(local_app_data_dir.join("kura.sqlite3"))
}

fn is_unique_local_file_path_violation(error: &rusqlite::Error) -> bool {
    match error {
        rusqlite::Error::SqliteFailure(inner, maybe_message) => {
            let is_constraint = inner.code == ErrorCode::ConstraintViolation;
            let message = maybe_message
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();

            is_constraint
                && (message.contains("papers.local_file_path")
                    || message.contains("unique constraint failed"))
        }
        _ => false,
    }
}

fn serialize_embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    bytes
}

fn deserialize_embedding_blob(blob: &[u8]) -> Result<Vec<f32>, DatabaseError> {
    if blob.is_empty() {
        return Err(DatabaseError::InvalidEmbeddingBlob {
            details: "embedding blob is empty".to_string(),
        });
    }

    if blob.len() % 4 != 0 {
        return Err(DatabaseError::InvalidEmbeddingBlob {
            details: format!(
                "embedding blob length {} is not divisible by 4",
                blob.len()
            ),
        });
    }

    let mut values = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    Ok(values)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.is_empty() || left.len() != right.len() {
        return None;
    }

    let mut dot = 0.0_f32;
    let mut left_norm_sq = 0.0_f32;
    let mut right_norm_sq = 0.0_f32;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm_sq += left_value * left_value;
        right_norm_sq += right_value * right_value;
    }

    if left_norm_sq <= f32::EPSILON || right_norm_sq <= f32::EPSILON {
        return None;
    }

    Some(dot / (left_norm_sq.sqrt() * right_norm_sq.sqrt()))
}

fn papers_table_requires_migration(connection: &Connection) -> Result<bool, rusqlite::Error> {
    let mut statement = connection.prepare("PRAGMA table_info(papers)")?;
    let mapped_columns = statement.query_map([], |row| row.get::<_, String>(1))?;

    let mut column_names = HashSet::new();
    for column in mapped_columns {
        column_names.insert(column?);
    }

    if column_names.is_empty() {
        return Ok(false);
    }

    let has_legacy_columns = column_names.contains("lattice_parameters")
        || column_names.contains("band_gaps")
        || column_names.contains("methodology_summary");
    let missing_extracted_data = !column_names.contains("extracted_data");

    Ok(has_legacy_columns || missing_extracted_data)
}
