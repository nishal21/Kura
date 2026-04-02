type DynamicDataTableProps = {
  data: Record<string, unknown> | null;
  activeHighlightText?: string | null;
  onValueClick: (value: string) => void;
};

const NOT_MENTIONED = "Not mentioned";
const SEARCH_TARGET_LENGTH = 35;

function formatStructuredValue(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatDisplayValue(value: unknown): string {
  if (value === null || value === undefined) {
    return NOT_MENTIONED;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : NOT_MENTIONED;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return formatStructuredValue(value);
}

function stripJsonWrappingCharacters(value: string): string {
  return value
    .replace(/[\[\]{}"]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function toSearchTarget(value: string): string {
  return value.substring(0, SEARCH_TARGET_LENGTH).trim();
}

function toHighlightValue(value: unknown): string {
  if (value === null || value === undefined) {
    return NOT_MENTIONED;
  }

  if (typeof value === "string") {
    const normalized = value.trim();
    if (!normalized) {
      return NOT_MENTIONED;
    }

    return normalized;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  const structuredDisplayValue = formatStructuredValue(value);
  const searchableText = stripJsonWrappingCharacters(structuredDisplayValue);
  if (!searchableText) {
    return NOT_MENTIONED;
  }

  return searchableText;
}

function isNotMentionedValue(value: unknown): boolean {
  return typeof value === "string" && value.trim().toLowerCase() === "not mentioned";
}

export default function DynamicDataTable({
  data,
  activeHighlightText,
  onValueClick,
}: DynamicDataTableProps) {
  const entries = data ? Object.entries(data) : [];

  if (entries.length === 0) {
    return (
      <p className="kura-subpanel px-3 py-2 text-sm text-stone-300/72">
        No extracted data available yet.
      </p>
    );
  }

  return (
    <div className="overflow-hidden border border-stone-300/20 bg-black/25">
      <table className="w-full table-fixed border-collapse">
        <thead>
          <tr className="border-b border-stone-300/20 bg-black/30">
            <th className="w-1/3 px-3 py-3 text-left text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-300/72">
              Key
            </th>
            <th className="px-3 py-3 text-left text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-300/72">
              Value
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, rawValue]) => {
            const displayValue = formatDisplayValue(rawValue);
            const highlightValue = toHighlightValue(rawValue);
            const searchTarget = toSearchTarget(highlightValue);
            const isStructuredValue =
              typeof rawValue === "object" && rawValue !== null;
            const isNotMentioned = isNotMentionedValue(rawValue);
            const isActive = activeHighlightText === searchTarget;
            const canHighlight =
              searchTarget.length > 0 &&
              searchTarget.toLowerCase() !== NOT_MENTIONED.toLowerCase();

            return (
              <tr key={key} className="border-b border-stone-300/15 last:border-b-0">
                <td className="align-top px-3 py-3 text-sm font-semibold tracking-[0.01em] text-stone-100/88">
                  {key}
                </td>
                <td className="px-3 py-2">
                  <button
                    type="button"
                    onClick={() => {
                      if (!canHighlight) {
                        return;
                      }

                      onValueClick(searchTarget);
                    }}
                    disabled={!canHighlight}
                    className={`w-full border px-3 py-2 text-left transition-all ${
                      isActive
                        ? "border-amber-300/65 bg-amber-500/20 text-amber-50"
                        : isNotMentioned
                          ? "border-stone-300/16 bg-black/30 text-stone-300/55 hover:border-stone-300/28"
                          : "border-stone-300/18 bg-black/30 text-stone-100/86 hover:border-emerald-300/42 hover:bg-emerald-900/20"
                    } ${!canHighlight ? "cursor-not-allowed opacity-75" : ""}`}
                  >
                    {isStructuredValue ? (
                      <pre className="whitespace-pre-wrap break-words text-sm leading-6">
                        {displayValue}
                      </pre>
                    ) : (
                      <span className="text-sm leading-6">{displayValue}</span>
                    )}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
