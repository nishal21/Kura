import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Download, Loader2, Printer, Trash2 } from "lucide-react";
import DynamicDataTable from "./components/DynamicDataTable";
import InteractivePdfViewer from "./components/InteractivePdfViewer";
import SetupWizard from "./components/SetupWizard";
import {
  AiSettings,
  BackendAiProvider,
  getProviderOption,
  loadAiSettings,
  PROVIDER_OPTIONS,
  saveAiSettings,
  UiAiProvider,
} from "./lib/aiSettings";
import {
  analyzePaper,
  chatWithLibrary,
  checkApiKey,
  EmbeddingProviderPreference,
  ExtractionResult,
  getAllPapers,
  initializeLocalEmbedding,
  pickPdfFile,
  getOllamaModels,
  pingBackend,
  processPaperEmbeddings,
  saveApiKey,
  savePaperToDb,
  SavedPaperListItem,
} from "./lib/api";
import {
  extractTextFromPdf,
  isPdfExtractionErrorMessage,
} from "./lib/pdfExtractor";

type ProcessingState = "idle" | "extracting" | "analyzing" | "embedding" | "done" | "error";

type AnalysisAlertKind = "auth" | "provider" | "general";

type AnalysisAlert = {
  kind: AnalysisAlertKind;
  title: string;
  message: string;
  detail?: string;
};

type ChatMessageRole = "user" | "ai";

type ChatMessage = {
  id: number;
  role: ChatMessageRole;
  text: string;
};

type ToastKind = "success" | "error";

type LibraryExportFormat = "xlsx" | "csv" | "json" | "md";

type LibraryExportOption = {
  format: LibraryExportFormat;
  label: string;
};

type ManagedProvider = {
  id: BackendAiProvider;
  label: string;
  note?: string;
};

const MANAGED_PROVIDERS: readonly ManagedProvider[] = [
  { id: "openai", label: "OpenAI" },
  { id: "anthropic", label: "Anthropic" },
  { id: "gemini", label: "Gemini" },
  { id: "mistral", label: "Mistral" },
  { id: "deepseek", label: "DeepSeek" },
  {
    id: "openrouter",
    label: "OpenRouter",
    note: "Also used by Meta AI, Cohere, Perplexity, Groq, and Together.ai routes.",
  },
];

const OLLAMA_CONNECTION_WARNING =
  "Could not connect to Ollama. Is it running on port 11434?";

const LIBRARY_EXPORT_OPTIONS: readonly LibraryExportOption[] = [
  { format: "xlsx", label: "Export as Excel (.xlsx)" },
  { format: "csv", label: "Export as CSV" },
  { format: "json", label: "Export as JSON" },
  { format: "md", label: "Export as Markdown" },
];

type EmbeddingEngine = "cloud" | "local";

type LocalEmbeddingModelOption = {
  id: string;
  label: string;
  description: string;
};

type EmbeddingPreferences = {
  engine: EmbeddingEngine;
  cloudModelName: string;
  localModelName: string;
  localReady: boolean;
};

type EmbeddingDownloadStatus = {
  status: string;
  progress: number;
};

const EMBEDDING_PREFERENCES_STORAGE_KEY = "kura.embedding.preferences.v1";
const SETUP_COMPLETED_STORAGE_KEY = "kura.setup.completed";
const DEFAULT_CLOUD_EMBEDDING_MODEL = "text-embedding-3-small";
const DEFAULT_LOCAL_EMBEDDING_MODEL = "mxbai-embed-large";
const LOCAL_EMBEDDING_MODEL_OPTIONS: readonly LocalEmbeddingModelOption[] = [
  {
    id: "bge-small-en",
    label: "BGE-Small-EN (Recommended for speed)",
    description: "Tiny, extremely fast, great for older laptops.",
  },
  {
    id: "mxbai-embed-large",
    label: "Mxbai-Embed-Large (Recommended for accuracy)",
    description: "State-of-the-art accuracy for academic retrieval.",
  },
  {
    id: "nomic-embed-text",
    label: "Nomic-Embed-Text",
    description: "High performance with a massive 8192 token context window.",
  },
  {
    id: "all-minilm-l6-v2",
    label: "All-MiniLM-L6-v2",
    description: "The classic, ultra-lightweight standard.",
  },
];

function isSupportedLocalEmbeddingModel(value: unknown): value is string {
  if (typeof value !== "string") {
    return false;
  }

  return LOCAL_EMBEDDING_MODEL_OPTIONS.some((option) => option.id === value);
}

function getLocalEmbeddingModelOption(modelName: string): LocalEmbeddingModelOption {
  return (
    LOCAL_EMBEDDING_MODEL_OPTIONS.find((option) => option.id === modelName) ??
    LOCAL_EMBEDDING_MODEL_OPTIONS.find(
      (option) => option.id === DEFAULT_LOCAL_EMBEDDING_MODEL,
    ) ??
    LOCAL_EMBEDDING_MODEL_OPTIONS[0]
  );
}

function defaultEmbeddingPreferences(): EmbeddingPreferences {
  return {
    engine: "cloud",
    cloudModelName: DEFAULT_CLOUD_EMBEDDING_MODEL,
    localModelName: DEFAULT_LOCAL_EMBEDDING_MODEL,
    localReady: false,
  };
}

function isEmbeddingEngine(value: unknown): value is EmbeddingEngine {
  return value === "cloud" || value === "local";
}

function normalizeEmbeddingPreferences(raw: unknown): EmbeddingPreferences {
  const defaults = defaultEmbeddingPreferences();

  if (typeof raw !== "object" || raw === null) {
    return defaults;
  }

  const candidate = raw as Partial<EmbeddingPreferences>;
  const engine = isEmbeddingEngine(candidate.engine) ? candidate.engine : defaults.engine;
  const cloudModelName =
    typeof candidate.cloudModelName === "string" && candidate.cloudModelName.trim().length > 0
      ? candidate.cloudModelName.trim()
      : defaults.cloudModelName;
  const localModelName =
    isSupportedLocalEmbeddingModel(candidate.localModelName)
      ? candidate.localModelName
      : defaults.localModelName;

  return {
    engine,
    cloudModelName,
    localModelName,
    localReady: candidate.localReady === true,
  };
}

function loadEmbeddingPreferences(): EmbeddingPreferences {
  if (typeof window === "undefined") {
    return defaultEmbeddingPreferences();
  }

  const raw = window.localStorage.getItem(EMBEDDING_PREFERENCES_STORAGE_KEY);
  if (!raw) {
    return defaultEmbeddingPreferences();
  }

  try {
    return normalizeEmbeddingPreferences(JSON.parse(raw));
  } catch {
    return defaultEmbeddingPreferences();
  }
}

function saveEmbeddingPreferences(preferences: EmbeddingPreferences): void {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(
    EMBEDDING_PREFERENCES_STORAGE_KEY,
    JSON.stringify(preferences),
  );
}

function loadSetupCompleted(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return window.localStorage.getItem(SETUP_COMPLETED_STORAGE_KEY) === "true";
}

type ProviderVaultState = {
  hasKey: boolean;
  isChecking: boolean;
  isSaving: boolean;
  overwriteMode: boolean;
  draftKey: string;
  error: string | null;
};


function toEmbeddingProviderPreference(
  preferences: EmbeddingPreferences,
): EmbeddingProviderPreference {
  if (preferences.engine === "local") {
    const localModelName = preferences.localModelName.trim();
    return {
      engine: "local",
      localModelName:
        localModelName.length > 0 ? localModelName : DEFAULT_LOCAL_EMBEDDING_MODEL,
    };
  }

  const cloudModelName = preferences.cloudModelName.trim();
  return {
    engine: "cloud",
    cloudModelName: cloudModelName.length > 0 ? cloudModelName : DEFAULT_CLOUD_EMBEDDING_MODEL,
  };
}
type ProviderVaultStatus = Record<BackendAiProvider, ProviderVaultState>;

function createProviderVaultEntry(): ProviderVaultState {
  return {
    hasKey: false,
    isChecking: true,
    isSaving: false,
    overwriteMode: false,
    draftKey: "",
    error: null,
  };
}

function createInitialProviderVaultStatus(): ProviderVaultStatus {
  return {
    openai: createProviderVaultEntry(),
    anthropic: createProviderVaultEntry(),
    gemini: createProviderVaultEntry(),
    mistral: createProviderVaultEntry(),
    deepseek: createProviderVaultEntry(),
    openrouter: createProviderVaultEntry(),
    ollama: createProviderVaultEntry(),
  };
}

function deriveFileNameFromPath(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  const segments = normalized.split("/");
  const fileName = segments[segments.length - 1]?.trim();

  return fileName && fileName.length > 0 ? fileName : path;
}

function derivePaperTitle(fileName: string): string {
  const withoutPdfExtension = fileName.replace(/\.pdf$/i, "").trim();
  return withoutPdfExtension.length > 0 ? withoutPdfExtension : fileName;
}

function formatPaperDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleDateString();
}

function isPaperAlreadySavedError(message: string): boolean {
  const normalized = message.toLowerCase();

  return (
    normalized.includes("already saved") ||
    normalized.includes("unique constraint") ||
    normalized.includes("papers.local_file_path")
  );
}

function classifyAnalysisError(rawMessage: string): AnalysisAlert {
  const message = rawMessage.trim();
  const normalized = message.toLowerCase();
  const hasAuthSignal =
    normalized.includes("401") ||
    normalized.includes("403") ||
    normalized.includes("invalid api key") ||
    normalized.includes("unauthorized") ||
    normalized.includes("api key");

  const hasProviderServerSignal =
    /http\s*5\d\d/i.test(message) ||
    normalized.includes("server error") ||
    normalized.includes("endpoint returned http 5");

  if (hasAuthSignal) {
    return {
      kind: "auth",
      title: "API Key Rejected",
      message:
        "The selected provider rejected your API key. Open Settings and verify the key and provider.",
      detail: message,
    };
  }

  if (hasProviderServerSignal) {
    return {
      kind: "provider",
      title: "Provider Server Error",
      message:
        "The selected provider is currently returning a server error. Retry shortly or switch provider/model.",
      detail: message,
    };
  }

  return {
    kind: "general",
    title: "Analysis Failed",
    message,
  };
}

export default function App() {
  const [isSetupCompleted, setIsSetupCompleted] = useState<boolean>(() =>
    loadSetupCompleted(),
  );
  const [settings, setSettings] = useState<AiSettings>(() => loadAiSettings());
  const [settingsDraft, setSettingsDraft] = useState<AiSettings>(() => loadAiSettings());
  const [embeddingPreferences, setEmbeddingPreferences] = useState<EmbeddingPreferences>(() =>
    loadEmbeddingPreferences(),
  );
  const [embeddingDraft, setEmbeddingDraft] = useState<EmbeddingPreferences>(() =>
    loadEmbeddingPreferences(),
  );
  const [embeddingDownloadStatus, setEmbeddingDownloadStatus] =
    useState<EmbeddingDownloadStatus | null>(null);
  const [isInitializingLocalEmbedding, setIsInitializingLocalEmbedding] = useState(false);
  const [providerVaultStatus, setProviderVaultStatus] = useState<ProviderVaultStatus>(
    createInitialProviderVaultStatus,
  );
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isLibraryChatOpen, setIsLibraryChatOpen] = useState(false);
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);
  const [savedPapers, setSavedPapers] = useState<SavedPaperListItem[]>([]);
  const [isLoadingSavedPapers, setIsLoadingSavedPapers] = useState(true);
  const [savedPapersError, setSavedPapersError] = useState<string | null>(null);
  const [selectedPaperId, setSelectedPaperId] = useState<number | null>(null);
  const [selectedPdfName, setSelectedPdfName] = useState<string | null>(null);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [activeHighlightText, setActiveHighlightText] = useState<string | null>(null);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [isDetectingOllamaModels, setIsDetectingOllamaModels] = useState(false);
  const [ollamaModelsWarning, setOllamaModelsWarning] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<ExtractionResult | null>(null);
  const [processingState, setProcessingState] = useState<ProcessingState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analysisAlert, setAnalysisAlert] = useState<AnalysisAlert | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastKind, setToastKind] = useState<ToastKind>("success");
  const [isExportingLibrary, setIsExportingLibrary] = useState(false);
  const [activeExportFormat, setActiveExportFormat] =
    useState<LibraryExportFormat | null>(null);
  const chatMessageIdRef = useRef(1);
  const chatHistoryRef = useRef<HTMLDivElement | null>(null);
  const viewerObjectUrlRef = useRef<string | null>(null);

  const activeProvider = useMemo(
    () => getProviderOption(settings.provider),
    [settings.provider],
  );

  const draftProvider = useMemo(
    () => getProviderOption(settingsDraft.provider),
    [settingsDraft.provider],
  );

  const isOllamaSelected = draftProvider.backendProvider === "ollama";

  const persistEmbeddingPreferences = (next: EmbeddingPreferences) => {
    setEmbeddingPreferences(next);
    saveEmbeddingPreferences(next);
  };

  const showToast = (message: string, kind: ToastKind = "success") => {
    setToastKind(kind);
    setToastMessage(message);
  };

  const hydrateActiveSettingsFromStorage = () => {
    const nextSettings = loadAiSettings();
    const nextEmbeddingPreferences = loadEmbeddingPreferences();

    setSettings(nextSettings);
    setSettingsDraft({
      ...nextSettings,
      extractionSchema: [...nextSettings.extractionSchema],
    });
    setEmbeddingPreferences(nextEmbeddingPreferences);
    setEmbeddingDraft({ ...nextEmbeddingPreferences });
  };

  const handleSetupComplete = () => {
    hydrateActiveSettingsFromStorage();
    setIsSetupCompleted(true);
    setSettingsMessage("First-run setup completed. You can tweak everything in Settings anytime.");
    void refreshAllManagedProviders();
  };

  const refreshSavedPapers = async (showLoading = true) => {
    if (showLoading) {
      setIsLoadingSavedPapers(true);
    }

    setSavedPapersError(null);

    try {
      const papers = await getAllPapers();
      setSavedPapers(papers);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSavedPapersError(message);
    } finally {
      if (showLoading) {
        setIsLoadingSavedPapers(false);
      }
    }
  };

  const handleDeletePaper = async (paperId: number) => {
    const confirmed = window.confirm(
      "Are you sure you want to remove this paper from your library?",
    );

    if (!confirmed) {
      return;
    }

    try {
      await invoke<unknown>("delete_paper", { paperId });

      if (selectedPaperId === paperId) {
        setSelectedPaperId(null);
      }

      await refreshSavedPapers(false);
      showToast("Removed from Library");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSavedPapersError(`Failed to remove paper: ${message}`);
    }
  };

  const handleExport = async (format: LibraryExportFormat) => {
    if (isExportingLibrary) {
      return;
    }

    setIsExportingLibrary(true);
    setActiveExportFormat(format);
    setSavedPapersError(null);

    try {
      const savedPath = await invoke<string>("export_library", {
        exportFormat: format,
      });

      if (savedPath.trim().toLowerCase() !== "cancelled") {
        showToast(`Library exported to: ${savedPath}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showToast(`Export failed: ${message}`, "error");
    } finally {
      setIsExportingLibrary(false);
      setActiveExportFormat(null);
    }
  };

  const setManagedProviderState = (
    provider: BackendAiProvider,
    updater: (current: ProviderVaultState) => ProviderVaultState,
  ) => {
    setProviderVaultStatus((current: ProviderVaultStatus) => ({
      ...current,
      [provider]: updater(current[provider]),
    }));
  };

  const checkProviderKeyPresence = async (
    provider: BackendAiProvider,
  ): Promise<boolean> => {
    const maxAttempts = 12;
    const verifyDelayMs = 220;

    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const hasKey = await checkApiKey(provider);
      if (hasKey) {
        return true;
      }

      if (attempt + 1 < maxAttempts) {
        await new Promise<void>((resolve) => {
          window.setTimeout(resolve, verifyDelayMs);
        });
      }
    }

    return false;
  };

  const refreshManagedProvider = async (provider: BackendAiProvider): Promise<boolean> => {
    setManagedProviderState(provider, (current) => ({
      ...current,
      isChecking: true,
      error: null,
    }));

    try {
      const hasKey = await checkProviderKeyPresence(provider);

      setManagedProviderState(provider, (current) => ({
        ...current,
        hasKey,
        isChecking: false,
        isSaving: false,
        overwriteMode: hasKey ? false : current.overwriteMode,
        draftKey: "",
        error: null,
      }));

      return hasKey;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setManagedProviderState(provider, (current) => ({
        ...current,
        hasKey: current.hasKey,
        isChecking: false,
        isSaving: false,
        error: message,
      }));

      return false;
    }
  };

  const refreshAllManagedProviders = async () => {
    await Promise.all(
      MANAGED_PROVIDERS.map(async ({ id }) => {
        await refreshManagedProvider(id);
      }),
    );
  };

  const ensureProviderKeyReady = async (provider: BackendAiProvider): Promise<boolean> => {
    if (provider === "ollama") {
      return true;
    }

    return refreshManagedProvider(provider);
  };

  useEffect(() => {
    return () => {
      if (viewerObjectUrlRef.current) {
        URL.revokeObjectURL(viewerObjectUrlRef.current);
        viewerObjectUrlRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const completed = loadSetupCompleted();
    setIsSetupCompleted(completed);

    if (completed) {
      hydrateActiveSettingsFromStorage();
    }
  }, []);

  useEffect(() => {
    if (!isSetupCompleted) {
      return;
    }

    void refreshSavedPapers();
    void refreshAllManagedProviders();
  }, [isSetupCompleted]);

  useEffect(() => {
    if (!toastMessage) {
      return;
    }

    const timeout = window.setTimeout(() => {
      setToastMessage(null);
    }, 2200);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [toastMessage]);

  useEffect(() => {
    const container = chatHistoryRef.current;
    if (!container) {
      return;
    }

    container.scrollTop = container.scrollHeight;
  }, [chatMessages, isChatLoading]);

  useEffect(() => {
    let disposed = false;

    const unlistenPromise = listen<EmbeddingDownloadStatus>(
      "embedding-download-status",
      (event) => {
        if (disposed) {
          return;
        }

        const status =
          typeof event.payload.status === "string" && event.payload.status.trim().length > 0
            ? event.payload.status
            : "Preparing local embedding runtime...";
        const rawProgress =
          typeof event.payload.progress === "number" &&
          Number.isFinite(event.payload.progress)
            ? event.payload.progress
            : 0;

        setEmbeddingDownloadStatus({
          status,
          progress: Math.max(0, Math.min(1, rawProgress)),
        });
      },
    );

    return () => {
      disposed = true;
      void unlistenPromise.then((unlisten) => {
        unlisten();
      });
    };
  }, []);

  useEffect(() => {
    if (!isSettingsOpen || !isOllamaSelected) {
      return;
    }

    let canceled = false;

    setIsDetectingOllamaModels(true);
    setOllamaModelsWarning(null);

    void getOllamaModels()
      .then((models) => {
        if (canceled) {
          return;
        }

        const normalizedModels = models
          .map((model) => model.trim())
          .filter((model) => model.length > 0);

        setOllamaModels(normalizedModels);

        if (normalizedModels.length === 0) {
          setOllamaModelsWarning(OLLAMA_CONNECTION_WARNING);
          return;
        }

        setSettingsDraft((current) => {
          if (current.provider !== "ollama") {
            return current;
          }

          const currentModel = current.modelName.trim();
          if (currentModel && normalizedModels.includes(currentModel)) {
            return current;
          }

          return {
            ...current,
            modelName: normalizedModels[0],
          };
        });
      })
      .catch(() => {
        if (canceled) {
          return;
        }

        setOllamaModels([]);
        setOllamaModelsWarning(OLLAMA_CONNECTION_WARNING);
      })
      .finally(() => {
        if (!canceled) {
          setIsDetectingOllamaModels(false);
        }
      });

    return () => {
      canceled = true;
    };
  }, [isOllamaSelected, isSettingsOpen]);

  const openSettingsModal = () => {
    setSettingsDraft({
      ...settings,
      extractionSchema: [...settings.extractionSchema],
    });
    setEmbeddingDraft({ ...embeddingPreferences });
    setEmbeddingDownloadStatus(null);
    setSettingsMessage(null);
    setIsSettingsOpen(true);
    void refreshAllManagedProviders();
  };

  const closeSettingsModal = () => {
    setIsSettingsOpen(false);
  };

  const appendChatMessage = (role: ChatMessageRole, text: string) => {
    const normalized = text.trim();
    if (!normalized) {
      return;
    }

    const nextMessage: ChatMessage = {
      id: chatMessageIdRef.current,
      role,
      text: normalized,
    };

    chatMessageIdRef.current += 1;
    setChatMessages((current) => [...current, nextMessage]);
  };

  const handleSendLibraryMessage = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (isChatLoading) {
      return;
    }

    const query = chatInput.trim();
    if (!query) {
      return;
    }

    const providerType = activeProvider.backendProvider;
    const modelName = settings.modelName.trim();
    const persistedEmbeddingPreferences = loadEmbeddingPreferences();
    const embeddingProvider = toEmbeddingProviderPreference(
      persistedEmbeddingPreferences,
    );

    if (!modelName) {
      setChatError("Configure AI Provider and Model Name in Settings before using Library Chat.");
      setIsSettingsOpen(true);
      return;
    }

    const hasProviderKey = await ensureProviderKeyReady(providerType);
    if (!hasProviderKey) {
      setChatError("No API key found for this provider. Open Settings and save one in Manage Providers.");
      setIsSettingsOpen(true);
      return;
    }

    setChatInput("");
    setChatError(null);
    appendChatMessage("user", query);
    setIsChatLoading(true);

    try {
      const response = await chatWithLibrary(query, {
        providerType,
        modelName,
        embeddingProvider,
      });

      appendChatMessage("ai", response);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setChatError(message);
      appendChatMessage(
        "ai",
        `I could not search the library right now. ${message}`,
      );
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleHighlightSelection = (value: string) => {
    const normalized = value.trim();
    if (!normalized) {
      return;
    }

    setActiveHighlightText(normalized);
  };

  const handlePrintEvidence = () => {
    window.print();
  };

  const handleSchemaFieldChange = (index: number, value: string) => {
    setSettingsDraft((current: AiSettings) => ({
      ...current,
      extractionSchema: current.extractionSchema.map((field, fieldIndex) =>
        fieldIndex === index ? value : field,
      ),
    }));
  };

  const handleAddSchemaField = () => {
    setSettingsDraft((current: AiSettings) => ({
      ...current,
      extractionSchema: [...current.extractionSchema, ""],
    }));
  };

  const handleDeleteSchemaField = (index: number) => {
    setSettingsDraft((current: AiSettings) => ({
      ...current,
      extractionSchema: current.extractionSchema.filter((_, fieldIndex) => fieldIndex !== index),
    }));
  };

  const handleProviderVaultInputChange = (
    provider: BackendAiProvider,
    value: string,
  ) => {
    setManagedProviderState(provider, (current) => ({
      ...current,
      draftKey: value,
      error: null,
    }));
  };

  const handleOverwriteProviderKey = (provider: BackendAiProvider) => {
    setManagedProviderState(provider, (current) => ({
      ...current,
      overwriteMode: true,
      draftKey: "",
      error: null,
    }));
  };

  const handleSaveProviderKey = async (provider: BackendAiProvider) => {
    const providerState = providerVaultStatus[provider];
    const keyToSave = providerState.draftKey.trim();
    const providerLabel =
      MANAGED_PROVIDERS.find((entry) => entry.id === provider)?.label ?? provider;

    if (!keyToSave) {
      setManagedProviderState(provider, (current) => ({
        ...current,
        error: "Enter an API key before saving.",
      }));
      return;
    }

    setManagedProviderState(provider, (current) => ({
      ...current,
      isSaving: true,
      draftKey: "",
      error: null,
    }));

    try {
      await saveApiKey(provider, keyToSave);
      const hasKey = await refreshManagedProvider(provider);

      if (hasKey) {
        setSettingsMessage(`${providerLabel} key securely stored in OS Vault.`);
      } else {
        setSettingsMessage(
          `${providerLabel} key was saved. OS Vault status is still syncing, so click Refresh Status in a moment.`,
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSettingsMessage(`Failed to save ${providerLabel} key: ${message}`);
      setManagedProviderState(provider, (current) => ({
        ...current,
        isSaving: false,
        draftKey: "",
        error: message,
      }));
    }
  };

  const handleEmbeddingEngineChange = (engine: EmbeddingEngine) => {
    setEmbeddingDraft((current) => ({
      ...current,
      engine,
    }));

    if (engine === "cloud") {
      setEmbeddingDownloadStatus(null);
    }
  };

  const handleInitializeLocalEmbedding = async () => {
    setIsInitializingLocalEmbedding(true);
    setEmbeddingDownloadStatus({
      status: "Preparing local embedding runtime...",
      progress: 0,
    });
    setSettingsMessage("Initializing local embedding engine...");

    try {
      const isReady = await initializeLocalEmbedding(embeddingDraft.localModelName);

      if (!isReady) {
        throw new Error("Local embedding engine did not report ready state.");
      }

      const nextEmbeddingPreferences = normalizeEmbeddingPreferences({
        ...embeddingDraft,
        engine: "local",
        localReady: true,
      });

      setEmbeddingDraft(nextEmbeddingPreferences);
      persistEmbeddingPreferences(nextEmbeddingPreferences);
      setEmbeddingDownloadStatus({ status: "Ready", progress: 1 });
      setSettingsMessage("Local embedding engine is ready.");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSettingsMessage(`Failed to initialize local embedding engine: ${message}`);
      setEmbeddingDraft((current) => ({
        ...current,
        localReady: false,
      }));
    } finally {
      setIsInitializingLocalEmbedding(false);
    }
  };

  const handleDraftProviderChange = (provider: UiAiProvider) => {
    const providerOption = getProviderOption(provider);

    setSettingsDraft((current: AiSettings) => {
      const currentModel = current.modelName.trim();
      const shouldKeepCurrentModel =
        providerOption.allowCustomModel || providerOption.models.includes(currentModel);

      return {
        ...current,
        provider: providerOption.id,
        modelName: shouldKeepCurrentModel ? currentModel : providerOption.defaultModel,
      };
    });
  };

  const handleSaveSettings = () => {
    const modelName = settingsDraft.modelName.trim();
    const extractionSchema = settingsDraft.extractionSchema
      .map((field) => field.trim())
      .filter((field) => field.length > 0);
    const cloudEmbeddingModelName = embeddingDraft.cloudModelName.trim();
    const localEmbeddingModelName = embeddingDraft.localModelName.trim();

    if (!modelName) {
      setSettingsMessage("Model name is required before saving.");
      return;
    }

    if (extractionSchema.length === 0) {
      setSettingsMessage("Add at least one extraction schema field before saving.");
      return;
    }

    if (embeddingDraft.engine === "cloud" && !cloudEmbeddingModelName) {
      setSettingsMessage("Cloud embedding model name is required before saving.");
      return;
    }

    if (embeddingDraft.engine === "local" && !localEmbeddingModelName) {
      setSettingsMessage("Local embedding model name is required before saving.");
      return;
    }

    const nextSettings: AiSettings = {
      provider: settingsDraft.provider,
      modelName,
      extractionSchema,
    };

    const nextEmbeddingPreferences = normalizeEmbeddingPreferences({
      ...embeddingDraft,
      cloudModelName: cloudEmbeddingModelName,
      localModelName: localEmbeddingModelName,
    });

    saveAiSettings(nextSettings);
    persistEmbeddingPreferences(nextEmbeddingPreferences);
    setSettings(nextSettings);
    setSettingsDraft(nextSettings);
    setEmbeddingDraft(nextEmbeddingPreferences);
    setSettingsMessage("Settings, extraction schema, and embedding preferences saved locally.");
    setIsSettingsOpen(false);
  };

  const handleTestBackend = async () => {
    try {
      const result = await pingBackend();
      console.log("Kura backend response:", result);
    } catch (error) {
      console.error("Failed to ping backend:", error);
    }
  };

  const handleSelectPdf = async () => {
    const selectedPath = await pickPdfFile();
    if (!selectedPath) {
      return;
    }

    const resolvedPath = selectedPath.trim();
    const persistedSettings = loadAiSettings();
    const persistedEmbeddingPreferences = loadEmbeddingPreferences();
    const targetSchema = persistedSettings.extractionSchema
      .map((field) => field.trim())
      .filter((field) => field.length > 0);

    const analyzeRequest = {
      providerType: activeProvider.backendProvider,
      modelName: settings.modelName.trim(),
      targetSchema,
      embeddingProvider: toEmbeddingProviderPreference(persistedEmbeddingPreferences),
    };

    const selectedFileName = deriveFileNameFromPath(resolvedPath);

    setSelectedPdfName(selectedFileName);
    if (viewerObjectUrlRef.current) {
      URL.revokeObjectURL(viewerObjectUrlRef.current);
      viewerObjectUrlRef.current = null;
    }
    setViewerUrl(null);
    setActiveHighlightText(null);
    setAnalysis(null);
    setErrorMessage(null);
    setAnalysisAlert(null);

    if (!resolvedPath) {
      setProcessingState("error");
      const pathAlert: AnalysisAlert = {
        kind: "general",
        title: "Invalid File Path",
        message:
          "Unable to resolve a real local PDF path from the selected file in this runtime.",
      };
      setAnalysisAlert(pathAlert);
      setErrorMessage(pathAlert.message);
      return;
    }

    let safeUrl = "";
    try {
      const fileBytes = await invoke<Uint8Array>("read_local_pdf", {
        path: resolvedPath,
      });
      const normalizedBytes = new Uint8Array(fileBytes);
      const safeBuffer = normalizedBytes.buffer.slice(
        normalizedBytes.byteOffset,
        normalizedBytes.byteOffset + normalizedBytes.byteLength,
      );
      const blob = new Blob([safeBuffer], { type: "application/pdf" });
      safeUrl = URL.createObjectURL(blob);
      viewerObjectUrlRef.current = safeUrl;
      setViewerUrl(safeUrl);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setProcessingState("error");
      const pathAlert: AnalysisAlert = {
        kind: "general",
        title: "PDF Read Failed",
        message: `Unable to read this local PDF: ${message}`,
      };
      setAnalysisAlert(pathAlert);
      setErrorMessage(pathAlert.message);
      return;
    }

    if (!analyzeRequest.modelName) {
      setProcessingState("error");
      const configurationAlert: AnalysisAlert = {
        kind: "general",
        title: "Configuration Required",
        message: "Configure AI Provider and Model Name in Settings before analyzing a paper.",
      };
      setAnalysisAlert(configurationAlert);
      setErrorMessage(configurationAlert.message);
      setIsSettingsOpen(true);
      return;
    }

    const hasProviderKey = await ensureProviderKeyReady(analyzeRequest.providerType);
    if (!hasProviderKey) {
      setProcessingState("error");
      const missingKeyAlert: AnalysisAlert = {
        kind: "auth",
        title: "API Key Required",
        message:
          "No API key found for this provider. Open Settings and save one in Manage Providers.",
      };
      setAnalysisAlert(missingKeyAlert);
      setErrorMessage(missingKeyAlert.message);
      setIsSettingsOpen(true);
      return;
    }

    try {
      setProcessingState("extracting");
      const extractedText = await extractTextFromPdf(safeUrl);

      if (isPdfExtractionErrorMessage(extractedText)) {
        const message = extractedText.replace(/^PDF extraction failed:\s*/i, "");
        setProcessingState("error");
        setAnalysisAlert({
          kind: "general",
          title: "PDF Extraction Failed",
          message,
        });
        setErrorMessage(message);
        return;
      }

      setProcessingState("analyzing");
      const analysisResult = await analyzePaper(extractedText, analyzeRequest);

      setAnalysis(analysisResult);

      try {
        const savedPaperId = await savePaperToDb({
          localFilePath: resolvedPath,
          title: derivePaperTitle(selectedFileName),
          extractedData: JSON.stringify(analysisResult),
        });

        let embeddingReady = true;

        try {
          setProcessingState("embedding");
          const embeddingProvider = toEmbeddingProviderPreference(
            persistedEmbeddingPreferences,
          );

          embeddingReady = await processPaperEmbeddings({
            paperId: savedPaperId,
            text: extractedText,
            embeddingProvider,
          });

          if (!embeddingReady) {
            throw new Error("Embedding coordinator returned not-ready state.");
          }
        } catch (embeddingError) {
          embeddingReady = false;
          const embeddingErrorMessage =
            embeddingError instanceof Error ? embeddingError.message : String(embeddingError);

          setAnalysisAlert({
            kind: "general",
            title: "Embedding Generation Failed",
            message:
              "Analysis and library save completed, but generating retrieval embeddings failed for this paper.",
            detail: embeddingErrorMessage,
          });
        }

        await refreshSavedPapers(false);
        showToast(embeddingReady ? "Saved to Library" : "Saved; Embeddings Failed");
      } catch (saveError) {
        const saveErrorMessage =
          saveError instanceof Error ? saveError.message : String(saveError);

        if (isPaperAlreadySavedError(saveErrorMessage)) {
          await refreshSavedPapers(false);
          showToast("Already in Library");
        } else {
          setAnalysisAlert({
            kind: "general",
            title: "Library Save Failed",
            message: "Analysis completed, but saving to the local library failed.",
            detail: saveErrorMessage,
          });
        }
      }

      setProcessingState("done");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const classifiedAlert = classifyAnalysisError(message);
      setProcessingState("error");
      setAnalysisAlert(classifiedAlert);
      setErrorMessage(classifiedAlert.message);
      console.error("PDF analysis failed:", error);
    }
  };

  const statusMessage = useMemo(() => {
    if (processingState === "extracting") {
      return "Extracting text...";
    }

    if (processingState === "analyzing") {
      return "Analyzing with AI...";
    }

    if (processingState === "embedding") {
      return "Generating library embeddings...";
    }

    if (processingState === "done") {
      return "Analysis complete.";
    }

    if (processingState === "error") {
      return "Processing failed.";
    }

    return "Select a local PDF to begin extraction and analysis.";
  }, [processingState]);

  const isProcessing =
    processingState === "extracting" ||
    processingState === "analyzing" ||
    processingState === "embedding";

  if (!isSetupCompleted) {
    return <SetupWizard onComplete={handleSetupComplete} />;
  }

  return (
    <div className="kura-root text-stone-100">
      <div className="kura-shell mx-auto min-h-screen max-w-[1760px] p-4 sm:p-6 lg:p-8">
        <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="kura-sidebar kura-panel kura-cut kura-panel-animate kura-scrollbar flex max-h-[calc(100vh-3rem)] flex-col overflow-y-auto p-5 lg:sticky lg:top-6">
            <div className="flex items-start gap-4">
              <img
                src="/logo.svg"
                alt="Kura logo"
                className="h-14 w-14 border border-amber-400/40 bg-black/30 p-1.5 object-contain"
              />

              <div>
                <h1 className="kura-title text-3xl font-bold tracking-tight">Kura</h1>
              </div>
            </div>

            <div className="mt-5 space-y-2">
              <div className="kura-chip">Provider: {activeProvider.label}</div>
              <div className="kura-chip">Model: {settings.modelName}</div>
              <div className="kura-chip">
                Embeddings: {embeddingPreferences.engine === "local"
                  ? `Local (${embeddingPreferences.localModelName})`
                  : `Cloud (${embeddingPreferences.cloudModelName})`}
              </div>
            </div>

            <div className="mt-7 mb-3 space-y-2">
              <div className="flex items-center justify-between">
                <h2 className="kura-section-label">Library Vault</h2>
                <span className="kura-count-pill">{savedPapers.length}</span>
              </div>

              <div className="grid grid-cols-1 gap-1.5">
                {LIBRARY_EXPORT_OPTIONS.map((option) => {
                  const isActive = isExportingLibrary && activeExportFormat === option.format;

                  return (
                    <button
                      key={option.format}
                      type="button"
                      onClick={() => {
                        void handleExport(option.format);
                      }}
                      disabled={isExportingLibrary}
                      className="inline-flex h-8 items-center justify-start gap-2 border border-stone-400/25 bg-black/30 px-3 text-[11px] font-medium text-stone-100 transition-all hover:border-amber-400/45 hover:bg-amber-900/20 disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {isExportingLibrary ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Download className="h-3.5 w-3.5" />
                      )}
                      {isActive ? "Exporting..." : option.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {isLoadingSavedPapers ? (
              <p className="kura-subpanel px-3 py-2 text-sm text-stone-300/80">Loading saved papers...</p>
            ) : savedPapersError ? (
              <div className="kura-subpanel space-y-2 border-rose-500/40 bg-rose-900/20 px-3 py-2">
                <p className="text-xs text-rose-200">{savedPapersError}</p>
                <button
                  type="button"
                  onClick={() => {
                    void refreshSavedPapers();
                  }}
                  className="kura-btn kura-btn--danger h-8 px-3 text-[11px]"
                >
                  Retry
                </button>
              </div>
            ) : savedPapers.length === 0 ? (
              <p className="kura-subpanel px-3 py-2 text-sm text-stone-300/70">No saved papers yet.</p>
            ) : (
              <ul className="space-y-2">
                {savedPapers.map((paper: SavedPaperListItem) => (
                  <li key={paper.id} className="group relative">
                    <button
                      type="button"
                      onClick={() => setSelectedPaperId(paper.id)}
                      className={`w-full border px-3 py-2 pr-10 text-left text-sm transition-all ${
                        selectedPaperId === paper.id
                          ? "border-amber-400/70 bg-amber-500/15 text-amber-50 shadow-[0_0_0_1px_rgba(251,191,36,0.35)]"
                          : "border-stone-400/20 bg-black/25 text-stone-200 hover:border-stone-300/40 hover:bg-white/5"
                      }`}
                    >
                      <p className="truncate font-semibold tracking-[0.01em]">
                        {paper.title || "Untitled Paper"}
                      </p>
                      <p className="mt-1 text-[11px] text-stone-300/65">
                        {formatPaperDate(paper.dateAdded)}
                      </p>
                    </button>

                    <button
                      type="button"
                      aria-label={`Remove ${paper.title || "paper"} from library`}
                      title="Remove from library"
                      onClick={(event) => {
                        event.stopPropagation();
                        void handleDeletePaper(paper.id);
                      }}
                      className="absolute top-2 right-2 inline-flex h-7 w-7 items-center justify-center border border-stone-300/22 bg-black/45 text-stone-300/70 opacity-0 transition-opacity duration-150 hover:border-rose-400/55 hover:bg-rose-900/30 hover:text-rose-100 focus-visible:opacity-100 group-hover:opacity-100"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </aside>

          <main className="kura-main-dashboard space-y-6">
            <div
              className="kura-top-navigation kura-panel kura-cut kura-panel-animate p-5 sm:p-6"
              style={{ animationDelay: "90ms" }}
            >
              <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <p className="kura-section-label mb-2">Research Operations</p>
                  <h2 className="kura-title text-3xl font-bold leading-tight sm:text-4xl">
                    Paper Analysis Dashboard
                  </h2>
                  <p className="mt-2 max-w-3xl text-sm leading-6 text-stone-200/80">
                    Extract structured findings, inspect source evidence, and interrogate your
                    local paper library through retrieval-augmented chat.
                  </p>
                  <p className="mt-2 text-xs text-stone-300/65">
                    Provider Route: {activeProvider.label} | Active Model: {settings.modelName}
                  </p>
                  {settingsMessage && (
                    <p className="mt-2 text-xs text-emerald-300">{settingsMessage}</p>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  <button
                    type="button"
                    onClick={() => {
                      setChatError(null);
                      setIsLibraryChatOpen(true);
                    }}
                    className="kura-btn kura-btn--mint"
                  >
                    Library Chat
                  </button>

                  <button
                    type="button"
                    onClick={openSettingsModal}
                    className="kura-btn kura-btn--neutral"
                  >
                    Settings
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      void handleSelectPdf();
                    }}
                    disabled={isProcessing}
                    className="kura-btn kura-btn--amber disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    Select PDF
                  </button>

                  <button
                    type="button"
                    onClick={handleTestBackend}
                    className="kura-btn kura-btn--ghost"
                  >
                    Test Backend
                  </button>
                </div>
              </div>
            </div>

            <section className="kura-dashboard-grid grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
              <article
                className="kura-pdf-panel kura-panel kura-cut kura-panel-animate flex h-[65vh] min-h-[540px] flex-col p-4 sm:p-5 xl:h-[calc(100vh-16.9rem)]"
                style={{ animationDelay: "160ms" }}
              >
                <div className="kura-subpanel mb-3 p-4">
                  <p className="kura-section-label">Source Document</p>
                  <p className="mt-2 text-sm text-stone-200/85">{statusMessage}</p>
                  {selectedPdfName && (
                    <p className="mt-1 text-xs text-stone-300/65">Selected: {selectedPdfName}</p>
                  )}

                  <div className="mt-3">
                    <button
                      type="button"
                      onClick={handlePrintEvidence}
                      disabled={!viewerUrl}
                      className="kura-btn kura-btn--ghost inline-flex h-8 items-center gap-1.5 px-3 text-[11px] disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      <Printer className="h-3.5 w-3.5" />
                      Print Evidence
                    </button>
                    <p className="mt-1 text-[11px] text-stone-300/70">
                      Prints the currently visible page and highlights. Select "Save to PDF" in the dialog.
                    </p>
                  </div>

                  {analysisAlert && (
                    <div
                      className={`mt-3 border px-3 py-2 text-left ${
                        analysisAlert.kind === "auth"
                          ? "border-amber-500/50 bg-amber-900/20 text-amber-100"
                          : analysisAlert.kind === "provider"
                            ? "border-orange-500/50 bg-orange-900/20 text-orange-100"
                            : "border-rose-500/50 bg-rose-900/20 text-rose-100"
                      }`}
                    >
                      <p className="text-xs font-semibold uppercase tracking-[0.12em]">
                        {analysisAlert.title}
                      </p>
                      <p className="mt-1 text-xs">{analysisAlert.message}</p>
                      {analysisAlert.detail && (
                        <p className="mt-1 text-[11px] opacity-80">{analysisAlert.detail}</p>
                      )}
                    </div>
                  )}
                  {!analysisAlert && errorMessage && (
                    <p className="mt-3 border border-rose-500/50 bg-rose-900/20 px-3 py-2 text-xs text-rose-100">
                      {errorMessage}
                    </p>
                  )}
                </div>

                <div className="kura-pdf-viewer-host kura-subpanel min-h-0 flex flex-1 overflow-hidden p-0">
                  {viewerUrl ? (
                    <InteractivePdfViewer
                      fileUrl={viewerUrl}
                      targetText={activeHighlightText}
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center px-6 text-center">
                      <p className="max-w-md text-sm leading-6 text-stone-300/75">
                        Select a local PDF to render pages and activate intelligent text
                        highlighting.
                      </p>
                    </div>
                  )}
                </div>
              </article>

              <article
                className="kura-extraction-console kura-panel kura-cut kura-panel-animate flex h-[65vh] min-h-[540px] flex-col p-5 xl:h-[calc(100vh-16.9rem)]"
                style={{ animationDelay: "240ms" }}
              >
                <div className="mb-4">
                  <p className="kura-section-label">Extraction Console</p>
                  <h3 className="kura-title mt-2 text-2xl font-semibold">Extracted Data</h3>
                  <p className="mt-1 text-xs text-stone-300/70">
                    Click any extracted value to jump to matching evidence in the PDF viewer.
                  </p>
                </div>

                {isProcessing && (
                  <p className="mb-4 border border-teal-400/40 bg-teal-900/20 px-3 py-2 text-sm text-teal-100">
                    {statusMessage}
                  </p>
                )}

                <div className="kura-scrollbar min-h-0 flex-1 overflow-auto pr-1">
                  <DynamicDataTable
                    data={analysis}
                    activeHighlightText={activeHighlightText}
                    onValueClick={handleHighlightSelection}
                  />
                </div>
              </article>
            </section>
          </main>
        </div>
      </div>

      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6">
          <button
            type="button"
            aria-label="Close settings"
            onClick={closeSettingsModal}
            className="absolute inset-0 bg-black/75 backdrop-blur-[2px]"
          />

          <div className="kura-panel kura-cut kura-scrollbar relative max-h-[88vh] w-full max-w-3xl overflow-auto p-6 sm:p-7">
            <div className="mb-5">
              <p className="kura-section-label mb-2">Control Center</p>
              <h3 className="kura-title text-3xl font-bold text-stone-100">AI Settings</h3>
              <p className="mt-1 text-sm text-stone-300/80">
                Configure provider credentials and default model for paper analysis.
              </p>
              {settingsMessage && (
                <p
                  className={`mt-2 text-xs ${
                    settingsMessage.toLowerCase().startsWith("failed")
                      ? "text-rose-200"
                      : "text-emerald-200"
                  }`}
                >
                  {settingsMessage}
                </p>
              )}
            </div>

            <div className="space-y-6">
              <div>
                <label
                  htmlFor="provider"
                  className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/70"
                >
                  AI Provider
                </label>
                <select
                  id="provider"
                  value={settingsDraft.provider}
                  onChange={(event) =>
                    handleDraftProviderChange(event.target.value as UiAiProvider)
                  }
                  className="kura-select"
                >
                  {PROVIDER_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label
                  htmlFor="modelName"
                  className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/70"
                >
                  Model Name
                </label>

                {draftProvider.allowCustomModel ? (
                  isOllamaSelected ? (
                    <>
                      {isDetectingOllamaModels && (
                        <p className="mb-2 text-xs text-stone-300/75">Detecting local models...</p>
                      )}

                      {!isDetectingOllamaModels && ollamaModels.length > 0 ? (
                        <select
                          id="modelName"
                          value={settingsDraft.modelName}
                          onChange={(event) =>
                            setSettingsDraft((current: AiSettings) => ({
                              ...current,
                              modelName: event.target.value,
                            }))
                          }
                          className="kura-select"
                        >
                          {ollamaModels.map((model) => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          id="modelName"
                          type="text"
                          value={settingsDraft.modelName}
                          onChange={(event) =>
                            setSettingsDraft((current: AiSettings) => ({
                              ...current,
                              modelName: event.target.value,
                            }))
                          }
                          placeholder="Enter local Ollama model (e.g., llama3, mistral, phi3)"
                          className="kura-input"
                        />
                      )}

                      {ollamaModelsWarning && (
                        <p className="mt-2 text-xs text-amber-200/90">{ollamaModelsWarning}</p>
                      )}
                    </>
                  ) : (
                    <>
                      <input
                        id="modelName"
                        type="text"
                        value={settingsDraft.modelName}
                        onChange={(event) =>
                          setSettingsDraft((current: AiSettings) => ({
                            ...current,
                            modelName: event.target.value,
                          }))
                        }
                        list="openrouter-model-suggestions"
                        placeholder="Enter custom model identifier"
                        className="kura-input"
                      />
                      <datalist id="openrouter-model-suggestions">
                        {draftProvider.models.map((model) => (
                          <option key={model} value={model} />
                        ))}
                      </datalist>
                    </>
                  )
                ) : (
                  <select
                    id="modelName"
                    value={settingsDraft.modelName}
                    onChange={(event) =>
                      setSettingsDraft((current: AiSettings) => ({
                        ...current,
                        modelName: event.target.value,
                      }))
                    }
                    className="kura-select"
                  >
                    {draftProvider.models.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                )}

                {draftProvider.note && (
                  <p className="mt-2 text-xs text-amber-200/90">{draftProvider.note}</p>
                )}
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <h4 className="text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/70">
                    Extraction Schema
                  </h4>
                  <button
                    type="button"
                    onClick={handleAddSchemaField}
                    className="kura-btn kura-btn--ghost h-8 px-3 text-[11px]"
                  >
                    Add Field
                  </button>
                </div>

                <p className="mb-2 text-xs text-stone-300/70">
                  Define exactly which properties the AI should extract from the paper.
                </p>

                {settingsDraft.extractionSchema.length === 0 ? (
                  <div className="kura-subpanel px-3 py-2 text-xs text-stone-300/75">
                    No schema fields yet. Add one to continue.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {settingsDraft.extractionSchema.map((field, index) => (
                      <div key={`schema-field-${index}`} className="flex items-center gap-2">
                        <input
                          type="text"
                          value={field}
                          onChange={(event) =>
                            handleSchemaFieldChange(index, event.target.value)
                          }
                          placeholder={`Field ${index + 1} (e.g., Operating Temperature)`}
                          className="kura-input"
                        />
                        <button
                          type="button"
                          onClick={() => handleDeleteSchemaField(index)}
                          className="kura-btn kura-btn--danger h-9 px-3 text-[11px]"
                        >
                          Delete
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/70">
                  Embedding Engine
                </h4>
                <p className="mb-2 text-xs text-stone-300/70">
                  Choose how library retrieval embeddings are generated.
                </p>

                <select
                  id="embeddingEngine"
                  value={embeddingDraft.engine}
                  onChange={(event) =>
                    handleEmbeddingEngineChange(event.target.value as EmbeddingEngine)
                  }
                  className="kura-select"
                >
                  <option value="cloud">Cloud (Requires API Key)</option>
                  <option value="local">Local (FastEmbed)</option>
                </select>

                {embeddingDraft.engine === "cloud" ? (
                  <div className="mt-3">
                    <label
                      htmlFor="cloudEmbeddingModel"
                      className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/70"
                    >
                      Cloud Embedding Model
                    </label>
                    <input
                      id="cloudEmbeddingModel"
                      type="text"
                      value={embeddingDraft.cloudModelName}
                      onChange={(event) =>
                        setEmbeddingDraft((current) => ({
                          ...current,
                          cloudModelName: event.target.value,
                        }))
                      }
                      placeholder="text-embedding-3-small"
                      className="kura-input"
                    />
                  </div>
                ) : (
                  <div className="mt-3 space-y-3">
                    <div>
                      <label
                        htmlFor="localEmbeddingModel"
                        className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/70"
                      >
                        Select Local Model
                      </label>
                      <select
                        id="localEmbeddingModel"
                        value={embeddingDraft.localModelName}
                        onChange={(event) =>
                          setEmbeddingDraft((current) => ({
                            ...current,
                            localModelName: event.target.value,
                            localReady: false,
                          }))
                        }
                        className="kura-select"
                      >
                        {LOCAL_EMBEDDING_MODEL_OPTIONS.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </select>

                      <div className="mt-2 space-y-1">
                        {LOCAL_EMBEDDING_MODEL_OPTIONS.map((option) => {
                          const isSelected = option.id === embeddingDraft.localModelName;

                          return (
                            <p
                              key={`local-embedding-description-${option.id}`}
                              className={`text-xs ${
                                isSelected ? "text-amber-100" : "text-stone-300/75"
                              }`}
                            >
                              <span className="font-semibold">{option.label}</span>
                              {" - "}
                              {option.description}
                            </p>
                          );
                        })}
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => {
                        void handleInitializeLocalEmbedding();
                      }}
                      disabled={isInitializingLocalEmbedding}
                      className="kura-btn kura-btn--mint disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {isInitializingLocalEmbedding
                        ? "Initializing..."
                        : embeddingDraft.localReady
                          ? "Reinitialize Local Model"
                          : "Download & Initialize Local Model"}
                    </button>

                    {isInitializingLocalEmbedding && embeddingDownloadStatus && (
                      <div className="kura-subpanel px-3 py-3">
                        <p className="text-xs text-stone-200/90">{embeddingDownloadStatus.status}</p>
                        <div className="mt-2 h-2 overflow-hidden rounded-full bg-stone-600/65">
                          <div
                            className="h-full bg-emerald-400 transition-[width] duration-150"
                            style={{
                              width: `${Math.round(embeddingDownloadStatus.progress * 100)}%`,
                            }}
                          />
                        </div>
                        <p className="mt-1 text-[11px] text-stone-300/70">
                          {Math.round(embeddingDownloadStatus.progress * 100)}%
                        </p>
                      </div>
                    )}

                    {!isInitializingLocalEmbedding && embeddingDraft.localReady && (
                      <div className="inline-flex items-center rounded-full border border-emerald-400/45 bg-emerald-900/25 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.1em] text-emerald-100">
                        Local Engine Ready
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <h4 className="text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/70">
                    Manage Providers
                  </h4>
                  {!isOllamaSelected && (
                    <button
                      type="button"
                      onClick={() => {
                        void refreshAllManagedProviders();
                      }}
                      className="kura-btn kura-btn--ghost h-8 px-3 text-[11px]"
                    >
                      Refresh Status
                    </button>
                  )}
                </div>

                {isOllamaSelected ? (
                  <p className="kura-subpanel px-3 py-2 text-xs text-stone-300/75">
                    Runs locally. Ensure Ollama is active on port 11434.
                  </p>
                ) : (
                  <div className="space-y-3">
                    {MANAGED_PROVIDERS.map((provider) => {
                      const providerState = providerVaultStatus[provider.id];
                      const shouldShowInput =
                        !providerState.isChecking &&
                        (!providerState.hasKey || providerState.overwriteMode);

                      return (
                        <div
                          key={provider.id}
                          className="kura-subpanel p-3"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div>
                              <p className="text-sm font-medium text-stone-100">{provider.label}</p>
                              {provider.note && (
                                <p className="text-[11px] text-stone-300/60">{provider.note}</p>
                              )}
                            </div>

                            {providerState.isChecking ? (
                              <p className="text-xs text-stone-300/65">Checking OS Vault...</p>
                            ) : providerState.hasKey && !providerState.overwriteMode ? (
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-emerald-200">
                                  ✔ Key securely stored in OS Vault.
                                </span>
                                <button
                                  type="button"
                                  onClick={() => handleOverwriteProviderKey(provider.id)}
                                  className="kura-btn kura-btn--ghost h-8 px-3 text-[11px]"
                                >
                                  Overwrite
                                </button>
                              </div>
                            ) : (
                              <span className="text-xs text-stone-300/65">No key stored</span>
                            )}
                          </div>

                          {shouldShowInput && (
                            <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                              <input
                                type="password"
                                value={providerState.draftKey}
                                onChange={(event) =>
                                  handleProviderVaultInputChange(provider.id, event.target.value)
                                }
                                placeholder={`Enter ${provider.label} API key`}
                                autoComplete="off"
                                className="kura-input"
                              />
                              <button
                                type="button"
                                onClick={() => {
                                  void handleSaveProviderKey(provider.id);
                                }}
                                disabled={providerState.isSaving}
                                className="kura-btn kura-btn--mint disabled:cursor-not-allowed disabled:opacity-70"
                              >
                                {providerState.isSaving ? "Saving..." : "Save"}
                              </button>
                            </div>
                          )}

                          {providerState.error && (
                            <p className="mt-2 text-xs text-rose-200">{providerState.error}</p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="flex items-center justify-between border-t border-stone-400/20 pt-4">
                <p className="text-xs text-stone-300/65">
                  Provider and model preferences are stored locally. API keys are stored only in OS Vault.
                </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={closeSettingsModal}
                    className="kura-btn kura-btn--ghost"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleSaveSettings}
                    className="kura-btn kura-btn--mint"
                  >
                    Save Settings
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {isLibraryChatOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6">
          <button
            type="button"
            aria-label="Close library chat"
            onClick={() => setIsLibraryChatOpen(false)}
            className="absolute inset-0 bg-black/75 backdrop-blur-[2px]"
          />

          <div className="kura-panel kura-cut relative flex h-[82vh] w-full max-w-5xl flex-col overflow-hidden">
            <div className="flex items-start justify-between gap-3 border-b border-stone-400/20 px-5 py-4">
              <div>
                <p className="kura-section-label">RAG Assistant</p>
                <h3 className="kura-title mt-2 text-3xl font-bold text-stone-100">Library Chat</h3>
                <p className="mt-1 text-sm text-stone-300/80">
                  Ask questions against your saved papers using retrieval from local library chunks.
                </p>
                <p className="mt-1 text-xs text-stone-300/65">
                  Provider: {activeProvider.label} | Model: {settings.modelName}
                </p>
              </div>

              <button
                type="button"
                onClick={() => setIsLibraryChatOpen(false)}
                className="kura-btn kura-btn--ghost h-9 px-4 text-xs"
              >
                Close
              </button>
            </div>

            <div
              ref={chatHistoryRef}
              className="kura-scrollbar min-h-0 flex-1 space-y-3 overflow-y-auto bg-black/20 px-5 py-4"
            >
              {chatMessages.length === 0 && !isChatLoading ? (
                <div className="kura-subpanel px-4 py-3 text-sm text-stone-300/80">
                  Start by asking a question about your saved paper library.
                </div>
              ) : (
                chatMessages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[82%] rounded-lg border px-4 py-3 text-sm leading-6 ${
                        message.role === "user"
                          ? "border-emerald-300/45 bg-emerald-900/30 text-emerald-50"
                          : "border-stone-300/25 bg-black/35 text-stone-100"
                      }`}
                    >
                      <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-stone-300/65">
                        {message.role === "user" ? "User" : "AI"}
                      </p>
                      <p className="whitespace-pre-wrap break-words">{message.text}</p>
                    </div>
                  </div>
                ))
              )}

              {isChatLoading && (
                <div className="flex justify-start">
                  <div className="max-w-[82%] rounded-lg border border-amber-300/45 bg-amber-900/25 px-4 py-3 text-sm text-amber-100">
                    <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-amber-200/80">
                      AI
                    </p>
                    <p>Searching library...</p>
                  </div>
                </div>
              )}
            </div>

            <form
              onSubmit={(event) => {
                void handleSendLibraryMessage(event);
              }}
              className="border-t border-stone-400/20 bg-black/20 px-5 py-4"
            >
              {chatError && (
                <p className="mb-3 border border-rose-500/45 bg-rose-900/20 px-3 py-2 text-xs text-rose-100">
                  {chatError}
                </p>
              )}

              <div className="flex items-center gap-3">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="Ask about trends, methods, or findings in your library..."
                  disabled={isChatLoading}
                  className="kura-input disabled:cursor-not-allowed disabled:opacity-70"
                />

                <button
                  type="submit"
                  disabled={isChatLoading || chatInput.trim().length === 0}
                  className="kura-btn kura-btn--mint disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {isChatLoading ? "Searching..." : "Send"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {toastMessage && (
        <div
          className={`pointer-events-none fixed bottom-5 right-5 z-40 px-3 py-2 text-xs shadow-[0_12px_25px_rgba(5,20,17,0.45)] backdrop-blur-sm ${
            toastKind === "error"
              ? "border border-rose-300/45 bg-rose-900/35 text-rose-50"
              : "border border-emerald-300/45 bg-emerald-900/35 text-emerald-50"
          }`}
        >
          {toastMessage}
        </div>
      )}
    </div>
  );
}
