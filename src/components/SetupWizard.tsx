import { useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  BackendAiProvider,
  getProviderOption,
  loadAiSettings,
  saveAiSettings,
  UiAiProvider,
} from "../lib/aiSettings";
import { checkApiKey, initializeLocalEmbedding, saveApiKey } from "../lib/api";

type SetupStep = 1 | 2;
type EmbeddingEngine = "cloud" | "local";

type ChatProviderChoice = {
  id: UiAiProvider;
  label: string;
};

type LocalEmbeddingModelOption = {
  id: string;
  label: string;
  description: string;
};

type EmbeddingDownloadStatus = {
  status: string;
  progress: number;
};

type EmbeddingPreferencesDraft = {
  engine: EmbeddingEngine;
  cloudModelName: string;
  localModelName: string;
  localReady: boolean;
};

type ApiKeyState =
  | { status: "idle" | "checking" }
  | { status: "confirmed"; message: string }
  | { status: "missing"; message: string }
  | { status: "error"; message: string };

const SETUP_COMPLETED_STORAGE_KEY = "kura.setup.completed";
const EMBEDDING_PREFERENCES_STORAGE_KEY = "kura.embedding.preferences.v1";
const DEFAULT_CLOUD_EMBEDDING_MODEL = "text-embedding-3-small";
const DEFAULT_LOCAL_EMBEDDING_MODEL = "mxbai-embed-large";

const CHAT_PROVIDER_CHOICES: readonly ChatProviderChoice[] = [
  { id: "openai", label: "OpenAI" },
  { id: "mistral_ai", label: "Mistral" },
  { id: "anthropic", label: "Anthropic" },
  { id: "google_cloud_ai", label: "Gemini" },
  { id: "deepseek", label: "DeepSeek" },
  { id: "openrouter", label: "OpenRouter" },
  { id: "ollama", label: "Ollama (Local)" },
];

const LOCAL_EMBEDDING_MODEL_OPTIONS: readonly LocalEmbeddingModelOption[] = [
  {
    id: "bge-small-en",
    label: "BGE-Small-EN",
    description: "Tiny and very fast. Best for lower-end laptops.",
  },
  {
    id: "mxbai-embed-large",
    label: "Mxbai-Embed-Large",
    description: "Best retrieval quality. Recommended for most users.",
  },
  {
    id: "nomic-embed-text",
    label: "Nomic-Embed-Text",
    description: "Strong quality with long context support.",
  },
  {
    id: "all-minilm-l6-v2",
    label: "All-MiniLM-L6-v2",
    description: "Compact classic option with fast startup.",
  },
];

function isEmbeddingEngine(value: unknown): value is EmbeddingEngine {
  return value === "cloud" || value === "local";
}

function isSupportedLocalEmbeddingModel(value: unknown): value is string {
  if (typeof value !== "string") {
    return false;
  }

  return LOCAL_EMBEDDING_MODEL_OPTIONS.some((option) => option.id === value);
}

function loadEmbeddingDraft(): EmbeddingPreferencesDraft {
  if (typeof window === "undefined") {
    return {
      engine: "cloud",
      cloudModelName: DEFAULT_CLOUD_EMBEDDING_MODEL,
      localModelName: DEFAULT_LOCAL_EMBEDDING_MODEL,
      localReady: false,
    };
  }

  const raw = window.localStorage.getItem(EMBEDDING_PREFERENCES_STORAGE_KEY);
  if (!raw) {
    return {
      engine: "cloud",
      cloudModelName: DEFAULT_CLOUD_EMBEDDING_MODEL,
      localModelName: DEFAULT_LOCAL_EMBEDDING_MODEL,
      localReady: false,
    };
  }

  try {
    const parsed = JSON.parse(raw) as Partial<EmbeddingPreferencesDraft>;
    const engine = isEmbeddingEngine(parsed.engine) ? parsed.engine : "cloud";
    const cloudModelName =
      typeof parsed.cloudModelName === "string" && parsed.cloudModelName.trim().length > 0
        ? parsed.cloudModelName.trim()
        : DEFAULT_CLOUD_EMBEDDING_MODEL;
    const localModelName = isSupportedLocalEmbeddingModel(parsed.localModelName)
      ? parsed.localModelName
      : DEFAULT_LOCAL_EMBEDDING_MODEL;

    return {
      engine,
      cloudModelName,
      localModelName,
      localReady: parsed.localReady === true,
    };
  } catch {
    return {
      engine: "cloud",
      cloudModelName: DEFAULT_CLOUD_EMBEDDING_MODEL,
      localModelName: DEFAULT_LOCAL_EMBEDDING_MODEL,
      localReady: false,
    };
  }
}

function resolveInitialChatProvider(provider: UiAiProvider): UiAiProvider {
  const hasChoice = CHAT_PROVIDER_CHOICES.some((choice) => choice.id === provider);
  return hasChoice ? provider : "openai";
}

async function checkProviderKeyWithRetry(provider: BackendAiProvider): Promise<boolean> {
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
}

export type SetupWizardProps = {
  onComplete: () => void;
};

export default function SetupWizard({ onComplete }: SetupWizardProps) {
  const aiSettings = useMemo(() => loadAiSettings(), []);
  const [step, setStep] = useState<SetupStep>(1);
  const [provider, setProvider] = useState<UiAiProvider>(() =>
    resolveInitialChatProvider(aiSettings.provider),
  );
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [isSavingApiKey, setIsSavingApiKey] = useState(false);
  const [apiKeyState, setApiKeyState] = useState<ApiKeyState>({ status: "idle" });
  const [embeddingDraft, setEmbeddingDraft] = useState<EmbeddingPreferencesDraft>(() =>
    loadEmbeddingDraft(),
  );
  const [isInitializingLocalEmbedding, setIsInitializingLocalEmbedding] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<EmbeddingDownloadStatus | null>(null);
  const [finishError, setFinishError] = useState<string | null>(null);
  const [isFinishing, setIsFinishing] = useState(false);

  const providerOption = useMemo(() => getProviderOption(provider), [provider]);
  const backendProvider = providerOption.backendProvider;
  const requiresCloudKey = backendProvider !== "ollama";
  const isCloudKeyConfirmed = apiKeyState.status === "confirmed";
  const hasVerifiedCloudKey = requiresCloudKey && isCloudKeyConfirmed;

  const canAdvanceToStepTwo = !requiresCloudKey || hasVerifiedCloudKey;

  const canFinishSetup =
    embeddingDraft.engine === "local"
      ? embeddingDraft.localReady
      : hasVerifiedCloudKey;

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
        const progress =
          typeof event.payload.progress === "number" &&
          Number.isFinite(event.payload.progress)
            ? Math.max(0, Math.min(1, event.payload.progress))
            : 0;

        setDownloadStatus({ status, progress });
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
    let canceled = false;

    if (!requiresCloudKey) {
      setApiKeyState({
        status: "confirmed",
        message: "No cloud API key required for Ollama local chat.",
      });
      return;
    }

    setApiKeyInput("");
    setApiKeyState({ status: "checking" });

    void checkProviderKeyWithRetry(backendProvider)
      .then((hasKey) => {
        if (canceled) {
          return;
        }

        if (hasKey) {
          setApiKeyState({
            status: "confirmed",
            message: `${providerOption.label} key found in OS vault.`,
          });
          return;
        }

        setApiKeyState({
          status: "missing",
          message: `No ${providerOption.label} key found yet.`,
        });
      })
      .catch((error) => {
        if (canceled) {
          return;
        }

        const message = error instanceof Error ? error.message : String(error);
        setApiKeyState({
          status: "error",
          message: `Could not verify key status: ${message}`,
        });
      });

    return () => {
      canceled = true;
    };
  }, [backendProvider, providerOption.label, requiresCloudKey]);

  const handleSaveApiKey = async () => {
    const normalizedKey = apiKeyInput.trim();
    if (!normalizedKey) {
      setApiKeyState({
        status: "error",
        message: "Enter an API key before saving.",
      });
      return;
    }

    setIsSavingApiKey(true);
    setApiKeyState({ status: "checking" });

    try {
      await saveApiKey(backendProvider, normalizedKey);
      setApiKeyInput("");

      const isConfirmed = await checkProviderKeyWithRetry(backendProvider);
      if (!isConfirmed) {
        setApiKeyState({
          status: "missing",
          message:
            "Key was saved, but vault status is still syncing. Try Save again or wait a moment.",
        });
        return;
      }

      setApiKeyState({
        status: "confirmed",
        message: `${providerOption.label} key verified in OS vault.`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setApiKeyState({
        status: "error",
        message: `Failed to save key: ${message}`,
      });
    } finally {
      setIsSavingApiKey(false);
    }
  };

  const handleInitializeLocal = async () => {
    setIsInitializingLocalEmbedding(true);
    setDownloadStatus({
      status: "Preparing local embedding runtime...",
      progress: 0,
    });
    setFinishError(null);

    try {
      const ready = await initializeLocalEmbedding(embeddingDraft.localModelName);
      if (!ready) {
        throw new Error("Local embedding runtime did not report ready state.");
      }

      setEmbeddingDraft((current) => ({
        ...current,
        localReady: true,
      }));
      setDownloadStatus({
        status: "Ready",
        progress: 1,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setEmbeddingDraft((current) => ({
        ...current,
        localReady: false,
      }));
      setFinishError(`Local initialization failed: ${message}`);
    } finally {
      setIsInitializingLocalEmbedding(false);
    }
  };

  const handleFinishSetup = async () => {
    if (!canFinishSetup || isFinishing) {
      return;
    }

    setIsFinishing(true);
    setFinishError(null);

    try {
      const currentSettings = loadAiSettings();
      const selectedProvider = getProviderOption(provider);
      const nextSettings = {
        ...currentSettings,
        provider: selectedProvider.id,
        modelName: selectedProvider.defaultModel,
        extractionSchema: [...currentSettings.extractionSchema],
      };

      saveAiSettings(nextSettings);

      const nextEmbeddingPreferences: EmbeddingPreferencesDraft = {
        engine: embeddingDraft.engine,
        cloudModelName:
          embeddingDraft.cloudModelName.trim().length > 0
            ? embeddingDraft.cloudModelName.trim()
            : DEFAULT_CLOUD_EMBEDDING_MODEL,
        localModelName: embeddingDraft.localModelName,
        localReady:
          embeddingDraft.engine === "local" ? embeddingDraft.localReady : false,
      };

      window.localStorage.setItem(
        EMBEDDING_PREFERENCES_STORAGE_KEY,
        JSON.stringify(nextEmbeddingPreferences),
      );
      window.localStorage.setItem(SETUP_COMPLETED_STORAGE_KEY, "true");
      onComplete();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setFinishError(`Failed to finalize setup: ${message}`);
    } finally {
      setIsFinishing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[80] bg-black/85 backdrop-blur-sm">
      <div className="relative flex min-h-screen items-center justify-center px-4 py-6 sm:px-8">
        <div className="kura-panel kura-cut kura-panel-animate w-full max-w-4xl border border-stone-400/25 p-6 sm:p-8">
          <div className="mb-6 flex flex-col gap-4 border-b border-stone-400/20 pb-5 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="kura-section-label mb-2">First-Run Setup</p>
              <h2 className="kura-title text-3xl font-bold text-stone-100 sm:text-4xl">
                Welcome to Kura
              </h2>
              <p className="mt-2 max-w-2xl text-sm text-stone-300/85">
                Configure your chat provider and retrieval engine once, then jump straight into
                paper analysis.
              </p>
            </div>

            <div className="inline-flex items-center gap-2 rounded-full border border-stone-300/25 bg-black/35 px-3 py-2 text-xs uppercase tracking-[0.12em] text-stone-200/80">
              <span className={step === 1 ? "text-amber-200" : "text-stone-400"}>Step 1</span>
              <span className="text-stone-500">/</span>
              <span className={step === 2 ? "text-amber-200" : "text-stone-400"}>Step 2</span>
            </div>
          </div>

          {step === 1 ? (
            <section className="space-y-5">
              <div>
                <h3 className="kura-title text-2xl font-semibold text-stone-100">
                  Chat AI Setup
                </h3>
                <p className="mt-2 text-sm text-stone-300/80">
                  Choose your primary chat provider. Cloud providers need an API key saved to your
                  OS vault.
                </p>
              </div>

              <div>
                <label
                  htmlFor="setup-provider"
                  className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/75"
                >
                  Primary Chat Provider
                </label>
                <select
                  id="setup-provider"
                  value={provider}
                  onChange={(event) => {
                    setProvider(event.target.value as UiAiProvider);
                  }}
                  className="kura-select"
                >
                  {CHAT_PROVIDER_CHOICES.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              {requiresCloudKey ? (
                <div className="kura-subpanel p-4">
                  <p className="mb-2 text-xs uppercase tracking-[0.12em] text-stone-300/75">
                    Secure API Key
                  </p>
                  <div className="flex flex-col gap-2 sm:flex-row">
                    <input
                      type="password"
                      value={apiKeyInput}
                      onChange={(event) => setApiKeyInput(event.target.value)}
                      placeholder={`Enter ${providerOption.label} API key`}
                      autoComplete="off"
                      className="kura-input"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        void handleSaveApiKey();
                      }}
                      disabled={isSavingApiKey}
                      className="kura-btn kura-btn--mint whitespace-nowrap disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {isSavingApiKey ? "Saving..." : "Save API Key"}
                    </button>
                  </div>

                  <p
                    className={`mt-2 text-xs ${
                      apiKeyState.status === "confirmed"
                        ? "text-emerald-200"
                        : apiKeyState.status === "error"
                          ? "text-rose-200"
                          : "text-stone-300/75"
                    }`}
                  >
                    {apiKeyState.status === "checking"
                      ? "Checking vault status..."
                      : apiKeyState.status === "idle"
                        ? "Save your key to continue."
                        : "message" in apiKeyState
                          ? apiKeyState.message
                          : "Save your key to continue."}
                  </p>
                </div>
              ) : (
                <div className="kura-subpanel border-emerald-400/35 bg-emerald-900/20 p-4 text-xs text-emerald-100">
                  Ollama runs locally, so no cloud API key is required.
                </div>
              )}

              <div className="flex justify-end border-t border-stone-400/20 pt-4">
                <button
                  type="button"
                  onClick={() => setStep(2)}
                  disabled={!canAdvanceToStepTwo}
                  className="kura-btn kura-btn--amber disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Next
                </button>
              </div>
            </section>
          ) : (
            <section className="space-y-5">
              <div>
                <h3 className="kura-title text-2xl font-semibold text-stone-100">
                  Embedding Engine Setup
                </h3>
                <p className="mt-2 text-sm text-stone-300/80">
                  Kura needs embeddings to search your PDF library. Choose cloud for hosted
                  embeddings or local for on-device retrieval.
                </p>
              </div>

              <div>
                <label
                  htmlFor="setup-embedding-engine"
                  className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/75"
                >
                  Embedding Engine
                </label>
                <select
                  id="setup-embedding-engine"
                  value={embeddingDraft.engine}
                  onChange={(event) => {
                    const engine = event.target.value as EmbeddingEngine;
                    setEmbeddingDraft((current) => ({
                      ...current,
                      engine,
                    }));
                    setFinishError(null);
                    if (engine === "cloud") {
                      setDownloadStatus(null);
                    }
                  }}
                  className="kura-select"
                >
                  <option value="cloud">Cloud</option>
                  <option value="local">Local</option>
                </select>
              </div>

              {embeddingDraft.engine === "cloud" ? (
                <div className="space-y-3">
                  <div>
                    <label
                      htmlFor="setup-cloud-embedding-model"
                      className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/75"
                    >
                      Cloud Embedding Model
                    </label>
                    <input
                      id="setup-cloud-embedding-model"
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

                  <div
                    className={`kura-subpanel p-3 text-xs ${
                      hasVerifiedCloudKey
                        ? "border-emerald-400/35 bg-emerald-900/20 text-emerald-100"
                        : "border-amber-400/35 bg-amber-900/20 text-amber-100"
                    }`}
                  >
                    {hasVerifiedCloudKey
                      ? "Cloud key verified. You can finish setup."
                      : requiresCloudKey
                        ? "Finish is locked until a cloud provider key is confirmed in Step 1."
                        : "Cloud embeddings need a cloud provider key. Switch provider in Step 1 or choose Local embeddings."}
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div>
                    <label
                      htmlFor="setup-local-embedding-model"
                      className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-stone-300/75"
                    >
                      Local Model
                    </label>
                    <select
                      id="setup-local-embedding-model"
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
                  </div>

                  <div className="space-y-1">
                    {LOCAL_EMBEDDING_MODEL_OPTIONS.map((option) => {
                      const isSelected = option.id === embeddingDraft.localModelName;

                      return (
                        <p
                          key={`wizard-local-model-${option.id}`}
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

                  <button
                    type="button"
                    onClick={() => {
                      void handleInitializeLocal();
                    }}
                    disabled={isInitializingLocalEmbedding}
                    className="kura-btn kura-btn--mint disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isInitializingLocalEmbedding
                      ? "Downloading..."
                      : embeddingDraft.localReady
                        ? "Reinitialize Local Model"
                        : "Download & Initialize"}
                  </button>

                  {downloadStatus && (
                    <div className="kura-subpanel px-3 py-3">
                      <p className="text-xs text-stone-200/90">{downloadStatus.status}</p>
                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-stone-600/65">
                        <div
                          className="h-full bg-emerald-400 transition-[width] duration-150"
                          style={{ width: `${Math.round(downloadStatus.progress * 100)}%` }}
                        />
                      </div>
                      <p className="mt-1 text-[11px] text-stone-300/70">
                        {Math.round(downloadStatus.progress * 100)}%
                      </p>
                    </div>
                  )}

                  {embeddingDraft.localReady && !isInitializingLocalEmbedding && (
                    <div className="inline-flex items-center rounded-full border border-emerald-400/45 bg-emerald-900/25 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.1em] text-emerald-100">
                      Local Model Ready
                    </div>
                  )}
                </div>
              )}

              {finishError && (
                <p className="border border-rose-500/50 bg-rose-900/25 px-3 py-2 text-xs text-rose-100">
                  {finishError}
                </p>
              )}

              <div className="flex items-center justify-between border-t border-stone-400/20 pt-4">
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="kura-btn kura-btn--ghost"
                >
                  Back
                </button>

                <button
                  type="button"
                  onClick={() => {
                    void handleFinishSetup();
                  }}
                  disabled={!canFinishSetup || isFinishing}
                  className="kura-btn kura-btn--amber disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isFinishing ? "Finishing..." : "Finish Setup"}
                </button>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}