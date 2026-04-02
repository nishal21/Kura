<p align="center">
  <img src="./logo.svg" alt="Kura logo" width="140" />
</p>

# Kura

Kura is a desktop research workspace for reading PDFs with AI, extracting structured evidence, and keeping every claim traceable to highlighted source text.

## What Kura Helps You Do

- Analyze scientific PDFs into a custom JSON schema.
- Click extracted values and jump to highlighted evidence in the PDF.
- Build a local "Library Vault" and chat over your saved papers with retrieval-augmented search.
- Export your full library as XLSX, CSV, JSON, or Markdown.
- Print evidence with highlights and save it as PDF.

## Quick Start

### 1. Install prerequisites

- Node.js 18+
- Rust stable toolchain
- Windows users: Visual Studio Build Tools with C++ workload
- Optional for local chat models: Ollama running on port 11434

This repo is currently pinned to:

- Rust toolchain: `stable-x86_64-pc-windows-msvc`
- Rust target: `x86_64-pc-windows-msvc`

### 2. Install dependencies

```bash
npm install
```

### 3. Run the desktop app

```bash
npm run dev
```

### 4. Complete first-run setup

Kura opens with a setup wizard on first launch.

1. Choose your chat provider.
2. Save your API key securely to the OS vault (unless using Ollama).
3. Choose embedding engine:
   - Cloud embeddings
   - Local embeddings (downloads model files on first init)

### 5. Analyze your first paper

1. Click Select PDF.
2. Wait for extraction and analysis.
3. Review extracted data in the right panel.
4. Click any value to jump to highlighted evidence in the source PDF.

## Daily Workflows

### Analyze PDFs

- Use Settings to customize extraction schema fields.
- Kura extracts only those fields and returns flat JSON output.
- Every paper is automatically saved to your local Library Vault.

### Chat with your library

- Open Library Chat from the dashboard.
- Ask a question about findings across saved papers.
- Kura retrieves relevant chunks using embeddings and sends context to your chosen model.

### Export your library

In Library Vault, use the export action buttons:

- Export as Excel (.xlsx)
- Export as CSV
- Export as JSON
- Export as Markdown

Kura opens a native Save dialog and writes the selected format to your chosen path.

### Print evidence with highlights

- Use Print Evidence in Source Document controls.
- Kura opens native print dialog with full-document print output.
- Choose Save to PDF in the print dialog to produce a sharable evidence file.

## Commands

| Command | Purpose |
|---|---|
| `npm run dev` | Start full Tauri desktop app (frontend + backend) |
| `npm run build` | Build desktop application |
| `npm run frontend:dev` | Run Vite frontend dev server only |
| `npm run frontend:build` | Build frontend assets |
| `npm run preview` | Preview built frontend assets |
| `npm run test` | TypeScript type-check (`tsc --noEmit`) |

## Configuration

### AI Provider and Model

- Configure from Settings.
- Providers include OpenAI, Anthropic, Gemini, Mistral, DeepSeek, OpenRouter routes, and Ollama.

### Extraction Schema

- Configure extraction keys from Settings.
- Kura enforces a flat JSON response aligned with your schema fields.

### Embeddings

- Cloud mode: uses configured embedding model.
- Local mode: uses fastembed models and can run fully offline after model download.

Supported local embedding choices in UI:

- `bge-small-en`
- `mxbai-embed-large`
- `nomic-embed-text`
- `all-minilm-l6-v2`

## API Reference (Tauri Commands)

Most frontend features call Tauri commands exposed from `src-tauri/src/main.rs`.

| Command | Purpose |
|---|---|
| `pick_pdf_file` | Open native PDF picker |
| `read_local_pdf` | Read local PDF bytes for viewer and extraction |
| `analyze_paper` | Run schema-driven AI extraction |
| `chat_with_library` | Ask questions over saved papers (RAG) |
| `save_paper_to_db` | Save extracted paper metadata |
| `get_all_papers` | List saved papers for Library Vault |
| `delete_paper` | Remove paper and embeddings from DB |
| `export_library` | Export library as xlsx/csv/json/md |
| `process_paper_embeddings` | Chunk text, generate embeddings, store vectors |
| `save_api_key` | Save provider key to OS vault |
| `check_api_key` | Check whether provider key exists |
| `get_ollama_models` | Detect local Ollama models |
| `initialize_local_embedding` | Download and initialize local embedding runtime |

## Security and Data

- API keys are stored in the OS credential vault via keyring integration.
- Paper metadata and embeddings are stored in a local SQLite database.
- Current database filename: `kura.sqlite3` in the app-local data directory.
- Deleting a paper from Library Vault removes DB records only, not your original PDF file.

## Print and Export Notes

- Print output is optimized for white backgrounds to reduce ink usage.
- Highlight spans retain color for print using `print-color-adjust: exact`.
- Multi-page print engine renders all pages so highlights can appear across the entire document.

## Project Structure

```text
kura/
  src/
    App.tsx
    components/
      SetupWizard.tsx
      InteractivePdfViewer.tsx
      DynamicDataTable.tsx
    lib/
      api.ts
      aiSettings.ts
      pdfExtractor.ts
  src-tauri/
    src/
      main.rs
      ai_extractor.rs
      database.rs
      secure_store.rs
```

## Troubleshooting

### "No API key found"

- Open Settings -> Manage Providers.
- Save or re-save the key for the selected backend provider.

### Ollama model detection fails

- Ensure Ollama is running locally on `http://localhost:11434`.
- Pull at least one model in Ollama.

### PDF does not highlight as expected

- Click shorter value snippets in the extraction table.
- Re-run analysis after changing extraction schema if the field shape changed.

### Print output does not include colors

- In print dialog, enable background graphics if your OS print stack requires it.
- Use Save to PDF for best fidelity.

## Documentation and Contribution

- Contribution guide: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Alternate compatibility path: [contributiion.md](./contributiion.md)

## License

This project is licensed under MIT. See [LICENSE](./LICENSE).
