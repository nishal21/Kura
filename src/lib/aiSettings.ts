export type UiAiProvider =
  | "openai"
  | "anthropic"
  | "google_cloud_ai"
  | "meta_ai"
  | "mistral_ai"
  | "cohere"
  | "deepseek"
  | "perplexity"
  | "openrouter"
  | "groq"
  | "together_ai"
  | "ollama";

export type BackendAiProvider =
  | "openai"
  | "anthropic"
  | "gemini"
  | "mistral"
  | "deepseek"
  | "openrouter"
  | "ollama";

export interface ProviderOption {
  readonly id: UiAiProvider;
  readonly label: string;
  readonly backendProvider: BackendAiProvider;
  readonly models: readonly string[];
  readonly defaultModel: string;
  readonly allowCustomModel: boolean;
  readonly note?: string;
}

export interface AiSettings {
  readonly provider: UiAiProvider;
  readonly modelName: string;
  readonly extractionSchema: readonly string[];
}

const STORAGE_KEY = "kura.ai.settings.v1";

export const DEFAULT_EXTRACTION_SCHEMA: readonly string[] = [
  "Lattice Parameters",
  "Band Gaps",
  "Methodology Summary",
];

export const PROVIDER_OPTIONS: readonly ProviderOption[] = [
  {
    id: "openai",
    label: "OpenAI",
    backendProvider: "openai",
    models: ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
    defaultModel: "gpt-4.1-mini",
    allowCustomModel: false,
  },
  {
    id: "anthropic",
    label: "Anthropic",
    backendProvider: "anthropic",
    models: [
      "claude-3-5-sonnet-latest",
      "claude-3-7-sonnet-latest",
      "claude-3-5-haiku-latest",
    ],
    defaultModel: "claude-3-5-sonnet-latest",
    allowCustomModel: false,
  },
  {
    id: "google_cloud_ai",
    label: "Google Cloud AI",
    backendProvider: "gemini",
    models: ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    defaultModel: "gemini-2.0-flash",
    allowCustomModel: false,
  },
  {
    id: "meta_ai",
    label: "Meta AI",
    backendProvider: "openrouter",
    models: [
      "meta-llama/llama-3.1-70b-instruct",
      "meta-llama/llama-3.1-8b-instruct",
      "meta-llama/llama-3.3-70b-instruct",
    ],
    defaultModel: "meta-llama/llama-3.1-70b-instruct",
    allowCustomModel: false,
    note: "Current backend routes this provider through OpenRouter.",
  },
  {
    id: "mistral_ai",
    label: "Mistral AI",
    backendProvider: "mistral",
    models: ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x22b"],
    defaultModel: "mistral-large-latest",
    allowCustomModel: false,
  },
  {
    id: "cohere",
    label: "Cohere",
    backendProvider: "openrouter",
    models: ["cohere/command-r-plus", "cohere/command-r", "cohere/command-a"],
    defaultModel: "cohere/command-r-plus",
    allowCustomModel: false,
    note: "Current backend routes this provider through OpenRouter.",
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    backendProvider: "deepseek",
    models: ["deepseek-chat", "deepseek-reasoner"],
    defaultModel: "deepseek-chat",
    allowCustomModel: false,
  },
  {
    id: "perplexity",
    label: "Perplexity",
    backendProvider: "openrouter",
    models: ["perplexity/sonar-pro", "perplexity/sonar", "perplexity/sonar-reasoning"],
    defaultModel: "perplexity/sonar-pro",
    allowCustomModel: false,
    note: "Current backend routes this provider through OpenRouter.",
  },
  {
    id: "openrouter",
    label: "OpenRouter",
    backendProvider: "openrouter",
    models: [
      "openai/gpt-4.1-mini",
      "anthropic/claude-3.5-sonnet",
      "google/gemini-2.0-flash-001",
    ],
    defaultModel: "openai/gpt-4.1-mini",
    allowCustomModel: true,
  },
  {
    id: "groq",
    label: "Groq",
    backendProvider: "openrouter",
    models: [
      "meta-llama/llama-3.3-70b-versatile",
      "mixtral-8x7b-32768",
      "llama-3.1-8b-instant",
    ],
    defaultModel: "meta-llama/llama-3.3-70b-versatile",
    allowCustomModel: false,
    note: "Current backend routes this provider through OpenRouter.",
  },
  {
    id: "together_ai",
    label: "Together.ai",
    backendProvider: "openrouter",
    models: [
      "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
      "Qwen/Qwen2.5-72B-Instruct-Turbo",
      "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ],
    defaultModel: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    allowCustomModel: false,
    note: "Current backend routes this provider through OpenRouter.",
  },
  {
    id: "ollama",
    label: "Ollama (Local)",
    backendProvider: "ollama",
    models: ["llama3", "mistral", "phi3"],
    defaultModel: "llama3",
    allowCustomModel: true,
    note: "Runs locally. Ensure Ollama is active on port 11434.",
  },
] as const;

export function getProviderOption(provider: UiAiProvider): ProviderOption {
  const selected = PROVIDER_OPTIONS.find((option) => option.id === provider);
  return selected ?? PROVIDER_OPTIONS[0];
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isUiProvider(value: unknown): value is UiAiProvider {
  return PROVIDER_OPTIONS.some((option) => option.id === value);
}

export function defaultAiSettings(): AiSettings {
  const fallback = PROVIDER_OPTIONS[0];
  return {
    provider: fallback.id,
    modelName: fallback.defaultModel,
    extractionSchema: [...DEFAULT_EXTRACTION_SCHEMA],
  };
}

function normalizeExtractionSchema(schema: readonly string[] | undefined): string[] {
  if (!schema || schema.length === 0) {
    return [...DEFAULT_EXTRACTION_SCHEMA];
  }

  const deduped = new Set<string>();
  const normalized: string[] = [];

  for (const rawField of schema) {
    const field = rawField.trim();
    if (!field) {
      continue;
    }

    const normalizedKey = field.toLocaleLowerCase();
    if (deduped.has(normalizedKey)) {
      continue;
    }

    deduped.add(normalizedKey);
    normalized.push(field);
  }

  return normalized.length > 0 ? normalized : [...DEFAULT_EXTRACTION_SCHEMA];
}

export function normalizeAiSettings(settings: AiSettings): AiSettings {
  const provider = getProviderOption(settings.provider);
  const modelName = settings.modelName.trim();
  const extractionSchema = normalizeExtractionSchema(settings.extractionSchema);

  if (modelName.length > 0) {
    return {
      provider: provider.id,
      modelName,
      extractionSchema,
    };
  }

  return {
    provider: provider.id,
    modelName: provider.defaultModel,
    extractionSchema,
  };
}

export function loadAiSettings(): AiSettings {
  if (typeof window === "undefined") {
    return defaultAiSettings();
  }

  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return defaultAiSettings();
  }

  try {
    const parsed = JSON.parse(raw) as Partial<AiSettings>;

    if (!parsed || !isUiProvider(parsed.provider)) {
      return defaultAiSettings();
    }

    const provider = getProviderOption(parsed.provider);
    const modelName = typeof parsed.modelName === "string" ? parsed.modelName : provider.defaultModel;
    const extractionSchema = Array.isArray(parsed.extractionSchema)
      ? parsed.extractionSchema.filter(isString)
      : undefined;

    return normalizeAiSettings({
      provider: provider.id,
      modelName,
      extractionSchema: extractionSchema ?? [...DEFAULT_EXTRACTION_SCHEMA],
    });
  } catch {
    return defaultAiSettings();
  }
}

export function saveAiSettings(settings: AiSettings): void {
  if (typeof window === "undefined") {
    return;
  }

  const normalized = normalizeAiSettings(settings);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
}
