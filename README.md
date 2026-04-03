<p align="center">
  <img src="./logo.svg" alt="Kura logo" width="150" />
</p>

<h1 align="center">Kura</h1>

<p align="center">
  AI-powered desktop workspace for research PDFs.
  <br />
  Extract structured evidence, verify it in-context, and ship clean reports fast.
</p>

<p align="center">
  <a href="https://github.com/nishal21/Kura/releases/tag/v1.0.0"><strong>Official Release v1.0.0</strong></a>
  &nbsp;|&nbsp;
  <a href="https://github.com/nishal21/Kura/releases"><strong>All Releases</strong></a>
  &nbsp;|&nbsp;
  <a href="./CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

## Why Kura

Kura is designed for researchers, analysts, and technical teams who need answers that are both fast and traceable.

- Read and analyze local PDFs with your preferred AI provider.
- Define your own extraction schema instead of accepting fixed templates.
- Click extracted values and jump back to highlighted evidence in the source PDF.
- Build a local paper vault and chat across your saved corpus.
- Export your library in multiple formats and print highlighted evidence.

## Install Kura (Recommended)

Use the official release assets:

- https://github.com/nishal21/Kura/releases/tag/v1.0.0

### Which file should I download?

| OS | Preferred installer type | What to pick |
|---|---|---|
| Windows | Installer package | `.msi` or `.exe` |
| macOS | Disk image | `.dmg` |
| Linux | Desktop package | `.AppImage` or `.deb` |

If the assets list is long, download only installer formats above and ignore internal runtime library files.

## 2-Minute Product Tour

1. Open Kura and complete the first-run setup wizard.
2. Choose provider and model, then save key to OS vault (not needed for local Ollama).
3. Select a PDF.
4. Kura extracts structured fields and stores the paper in Library Vault.
5. Click any extracted value to jump to highlighted source evidence.
6. Use Library Chat for cross-paper questions.
7. Export or print evidence when ready.

## Core Features

### 1. Schema-Driven Extraction

- Define exactly what Kura should extract.
- Output is flat JSON aligned to your schema fields.
- Values are constrained to verbatim source-backed text.

### 2. Interactive Evidence Trace

- Extracted values are clickable.
- PDF viewer auto-scrolls to matching highlighted evidence.
- Multi-page print mode preserves highlights in exported print/PDF output.

### 3. Retrieval Chat Over Your Library

- Papers are chunked and embedded for semantic retrieval.
- Ask questions over your saved local library.
- Works with cloud embeddings or local embedding models.

### 4. Library Management

- Persistent local Library Vault powered by SQLite.
- Safe delete removes DB records, not your source PDF file.
- Export options: Excel, CSV, JSON, Markdown.

## Build From Source

### Prerequisites

- Node.js 18+
- Rust stable toolchain
- Windows builds: Visual Studio Build Tools with C++ workload
- Optional for local LLM use: Ollama on `http://localhost:11434`

### Setup

```bash
npm install
npm run dev
```

### Production build

```bash
npm run build
```

## Developer Commands

| Command | Purpose |
|---|---|
| `npm run dev` | Run full Tauri desktop app in development |
| `npm run build` | Build desktop production binaries |
| `npm run frontend:dev` | Run Vite dev server only |
| `npm run frontend:build` | Build frontend assets |
| `npm run preview` | Preview built frontend |
| `npm run test` | TypeScript type-check |

## Configuration

### AI providers

Kura supports OpenAI, Anthropic, Gemini, Mistral, DeepSeek, OpenRouter routes, and Ollama local.

### Embedding engines

- Cloud embeddings with configurable model name.
- Local embeddings with fastembed models:
  - `bge-small-en`
  - `mxbai-embed-large`
  - `nomic-embed-text`
  - `all-minilm-l6-v2`

### Security model

- API keys are stored in the OS credential manager (keyring), not plain text app storage.
- Local metadata and embeddings are stored in app-local SQLite.

## API Surface (Tauri Commands)

| Command | Purpose |
|---|---|
| `pick_pdf_file` | Open native file picker for PDFs |
| `read_local_pdf` | Read PDF bytes for frontend viewer/extractor |
| `analyze_paper` | Run AI extraction against selected schema |
| `chat_with_library` | Ask retrieval-augmented questions |
| `save_paper_to_db` | Save analyzed paper record |
| `get_all_papers` | List Library Vault papers |
| `delete_paper` | Remove paper and embeddings from DB |
| `export_library` | Export library to xlsx/csv/json/md |
| `process_paper_embeddings` | Chunk, embed, and persist vectors |
| `save_api_key` | Save provider key securely |
| `check_api_key` | Validate provider key availability |
| `get_ollama_models` | Detect installed local Ollama models |
| `initialize_local_embedding` | Download/init local embedding runtime |

## Troubleshooting

### "No API key found"

Open Settings, then Manage Providers, then save or overwrite the provider key.

### Ollama model list is empty

Ensure Ollama is running locally and at least one model is pulled.

### Highlight did not jump exactly where expected

Click shorter extracted value snippets and rerun analysis after schema changes.

### Print output lost colors

Enable background graphics in print dialog and use Save to PDF for best fidelity.

## Repository Layout

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

## Contributing

- Main guide: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Compatibility path: [contributiion.md](./contributiion.md)

## License

MIT License. See [LICENSE](./LICENSE).
