import { invoke } from "@tauri-apps/api/core";
import { BackendAiProvider } from "./aiSettings";

export interface SystemInfo {
  readonly os: string;
  readonly architecture: string;
}

export interface LatticeParameter {
  readonly material: string;
  readonly parameter: string;
  readonly value: number;
  readonly unit: string;
}

export interface BandGap {
  readonly material: string;
  readonly value_ev: number;
  readonly transition_type: string | null;
}

export interface PaperAnalysis {
  readonly lattice_parameters: readonly LatticeParameter[];
  readonly band_gaps: readonly BandGap[];
  readonly methodology_summary: string;
}

export type ExtractionResult = Record<string, unknown>;

export interface EmbeddingProviderPreference {
  readonly engine: "cloud" | "local";
  readonly cloudModelName?: string;
  readonly localModelName?: string;
}

export interface AnalyzePaperRequest {
  readonly providerType: BackendAiProvider;
  readonly modelName: string;
  readonly targetSchema?: readonly string[];
  readonly embeddingProvider?: EmbeddingProviderPreference;
}

export interface ChatWithLibraryRequest {
  readonly providerType: BackendAiProvider;
  readonly modelName: string;
  readonly embeddingProvider?: EmbeddingProviderPreference;
}

export interface SavePaperToDbRequest {
  readonly localFilePath: string;
  readonly title: string;
  readonly extractedData: string;
}

export interface ProcessPaperEmbeddingsRequest {
  readonly paperId: number;
  readonly text: string;
  readonly embeddingProvider: EmbeddingProviderPreference;
}

export interface SavedPaperListItem {
  readonly id: number;
  readonly title: string;
  readonly extractedData: string;
  readonly dateAdded: string;
}

type Guard<T> = (value: unknown) => value is T;

const DEFAULT_LOCAL_EMBEDDING_MODEL = "mxbai-embed-large";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === "boolean";
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || isString(value);
}

function isLatticeParameter(value: unknown): value is LatticeParameter {
  return (
    isObject(value) &&
    isString(value.material) &&
    isString(value.parameter) &&
    isNumber(value.value) &&
    isString(value.unit)
  );
}

function isBandGap(value: unknown): value is BandGap {
  return (
    isObject(value) &&
    isString(value.material) &&
    isNumber(value.value_ev) &&
    isNullableString(value.transition_type)
  );
}

function isPaperAnalysis(value: unknown): value is PaperAnalysis {
  return (
    isObject(value) &&
    Array.isArray(value.lattice_parameters) &&
    value.lattice_parameters.every(isLatticeParameter) &&
    Array.isArray(value.band_gaps) &&
    value.band_gaps.every(isBandGap) &&
    isString(value.methodology_summary)
  );
}

function isExtractionResult(value: unknown): value is ExtractionResult {
  return isObject(value) && !Array.isArray(value);
}

function isSavedPaperListItem(value: unknown): value is SavedPaperListItem {
  return (
    isObject(value) &&
    isNumber(value.id) &&
    isString(value.title) &&
    isString(value.extractedData) &&
    isString(value.dateAdded)
  );
}

function isSavedPaperList(value: unknown): value is SavedPaperListItem[] {
  return Array.isArray(value) && value.every(isSavedPaperListItem);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(isString);
}

function isSystemInfo(value: unknown): value is SystemInfo {
  return (
    isObject(value) &&
    isString(value.os) &&
    isString(value.architecture)
  );
}

function normalizeEmbeddingProviderPreference(
  preference: EmbeddingProviderPreference | undefined,
): EmbeddingProviderPreference | undefined {
  if (!preference) {
    return undefined;
  }

  if (preference.engine === "local") {
    const localModelName = (preference.localModelName ?? "").trim();

    return {
      engine: "local",
      localModelName:
        localModelName.length > 0 ? localModelName : DEFAULT_LOCAL_EMBEDDING_MODEL,
    };
  }

  const cloudModelName = (preference.cloudModelName ?? "").trim();
  if (!cloudModelName) {
    throw new Error("cloudModelName is required when embedding engine is cloud");
  }

  return {
    engine: "cloud",
    cloudModelName,
  };
}

async function invokeStrict<T>(
  command: string,
  guard: Guard<T>,
  args?: Record<string, unknown>,
): Promise<T> {
  const payload = await invoke<unknown>(command, args);

  if (!guard(payload)) {
    throw new Error(`Unexpected payload returned by ${command}`);
  }

  return payload;
}

async function invokeNoResult(
  command: string,
  args?: Record<string, unknown>,
): Promise<void> {
  await invoke<unknown>(command, args);
}

export function pingBackend(): Promise<string> {
  return invokeStrict("ping_backend", isString);
}

export function getSystemInfo(): Promise<SystemInfo> {
  return invokeStrict("get_system_info", isSystemInfo);
}

export function pickPdfFile(): Promise<string | null> {
  return invokeStrict("pick_pdf_file", isNullableString);
}

export function checkApiKey(provider: BackendAiProvider): Promise<boolean> {
  return invokeStrict("check_api_key", isBoolean, { provider });
}

export async function saveApiKey(
  provider: BackendAiProvider,
  apiKey: string,
): Promise<void> {
  const normalizedApiKey = apiKey.trim();

  if (!normalizedApiKey) {
    throw new Error("API key is required");
  }

  await invokeNoResult("save_api_key", {
    provider,
    apiKey: normalizedApiKey,
    payload: {
      provider,
      apiKey: normalizedApiKey,
    },
  });
}

export async function savePaperToDb(request: SavePaperToDbRequest): Promise<number> {
  const normalizedPath = request.localFilePath.trim();
  const normalizedTitle = request.title.trim();
  const normalizedExtractedData = request.extractedData.trim();

  if (!normalizedPath) {
    throw new Error("localFilePath is required");
  }

  if (!normalizedTitle) {
    throw new Error("title is required");
  }

  if (!normalizedExtractedData) {
    throw new Error("extractedData is required");
  }

  return invokeStrict("save_paper_to_db", isNumber, {
    localFilePath: normalizedPath,
    title: normalizedTitle,
    extractedData: normalizedExtractedData,
  });
}

export function processPaperEmbeddings(
  request: ProcessPaperEmbeddingsRequest,
): Promise<boolean> {
  const normalizedText = request.text.trim();
  if (!normalizedText) {
    throw new Error("text is required to generate embeddings");
  }

  if (!Number.isInteger(request.paperId) || request.paperId <= 0) {
    throw new Error("paperId must be a positive integer");
  }

  const preference = normalizeEmbeddingProviderPreference(request.embeddingProvider);
  if (!preference) {
    throw new Error("embeddingProvider is required");
  }

  if (preference.engine === "local") {
    const localModelName = (preference.localModelName ?? "").trim();

    return invokeStrict("process_paper_embeddings", isBoolean, {
      paperId: request.paperId,
      text: normalizedText,
      embeddingProvider: {
        engine: "local",
        modelName:
          localModelName.length > 0
            ? localModelName
            : DEFAULT_LOCAL_EMBEDDING_MODEL,
      },
      localModelName:
        localModelName.length > 0 ? localModelName : DEFAULT_LOCAL_EMBEDDING_MODEL,
    });
  }

  const cloudModelName = (preference.cloudModelName ?? "").trim();

  return invokeStrict("process_paper_embeddings", isBoolean, {
    paperId: request.paperId,
    text: normalizedText,
    embeddingProvider: {
      engine: "cloud",
      modelName: cloudModelName,
    },
    localModelName: null,
  });
}

export function getAllPapers(): Promise<SavedPaperListItem[]> {
  return invokeStrict("get_all_papers", isSavedPaperList);
}

export function getOllamaModels(): Promise<string[]> {
  return invokeStrict("get_ollama_models", isStringArray);
}

export function initializeLocalEmbedding(localModelName: string): Promise<boolean> {
  const normalizedLocalModelName = localModelName.trim();

  return invokeStrict("initialize_local_embedding", isBoolean, {
    localModelName:
      normalizedLocalModelName.length > 0
        ? normalizedLocalModelName
        : DEFAULT_LOCAL_EMBEDDING_MODEL,
  });
}

export function analyzePaper(
  text: string,
  request: AnalyzePaperRequest,
): Promise<ExtractionResult> {
  const normalizedModelName = request.modelName.trim();
  const embeddingProvider = normalizeEmbeddingProviderPreference(request.embeddingProvider);
  const targetSchema = (request.targetSchema ?? [
    "Lattice Parameters",
    "Band Gaps",
    "Methodology Summary",
  ])
    .map((key) => key.trim())
    .filter((key) => key.length > 0);

  if (!normalizedModelName) {
    throw new Error("Model name is required before analysis");
  }

  if (targetSchema.length === 0) {
    throw new Error("targetSchema must include at least one key");
  }

  return invokeStrict("analyze_paper", isExtractionResult, {
    provider: request.providerType,
    modelName: normalizedModelName,
    pdfText: text,
    targetSchema,
    embeddingProvider,
  });
}

export function chatWithLibrary(
  query: string,
  request: ChatWithLibraryRequest,
): Promise<string> {
  const normalizedQuery = query.trim();
  const normalizedModelName = request.modelName.trim();
  const embeddingProvider = normalizeEmbeddingProviderPreference(request.embeddingProvider);

  if (!normalizedQuery) {
    throw new Error("Query is required before chatting with library");
  }

  if (!normalizedModelName) {
    throw new Error("Model name is required before chatting with library");
  }

  return invokeStrict("chat_with_library", isString, {
    query: normalizedQuery,
    provider: request.providerType,
    modelName: normalizedModelName,
    embeddingProvider,
  });
}
