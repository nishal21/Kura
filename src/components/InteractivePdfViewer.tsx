import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import "./InteractivePdfViewer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.mjs",
  import.meta.url,
).toString();

type InteractivePdfViewerProps = {
  fileUrl: string;
  targetText?: string | null;
};

type TextItemLike = {
  str?: unknown;
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function getItemText(item: unknown): string | null {
  if (typeof item !== "object" || item === null) {
    return null;
  }

  const maybeString = (item as TextItemLike).str;
  return typeof maybeString === "string" ? maybeString : null;
}

function renderHighlightedText(source: string, target: string): string {
  if (!target) {
    return escapeHtml(source);
  }

  const pattern = new RegExp(escapeRegExp(target), "gi");
  const matches = [...source.matchAll(pattern)];

  if (matches.length === 0) {
    return escapeHtml(source);
  }

  let cursor = 0;
  let html = "";

  for (const match of matches) {
    const start = match.index ?? 0;
    const chunk = match[0];

    html += escapeHtml(source.slice(cursor, start));
    html += `<span class="kura-pdf-highlight">${escapeHtml(chunk)}</span>`;
    cursor = start + chunk.length;
  }

  html += escapeHtml(source.slice(cursor));

  return html;
}

export default function InteractivePdfViewer({
  fileUrl,
  targetText,
}: InteractivePdfViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const scannedPagesRef = useRef<Set<number>>(new Set());

  const [numPages, setNumPages] = useState(0);
  const [pageWidth, setPageWidth] = useState<number | undefined>(undefined);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [firstMatchedPage, setFirstMatchedPage] = useState<number | null>(null);
  const [checkedPageCount, setCheckedPageCount] = useState(0);
  const [hasAutoScrolled, setHasAutoScrolled] = useState(false);

  const normalizedTarget = useMemo(() => targetText?.trim() ?? "", [targetText]);
  const normalizedTargetLower = useMemo(
    () => normalizedTarget.toLowerCase(),
    [normalizedTarget],
  );

  useEffect(() => {
    setNumPages(0);
    setLoadError(null);
    pageRefs.current = {};
  }, [fileUrl]);

  useEffect(() => {
    setFirstMatchedPage(null);
    setCheckedPageCount(0);
    setHasAutoScrolled(false);
    scannedPagesRef.current.clear();
  }, [fileUrl, normalizedTarget]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) {
      return;
    }

    const updateWidth = () => {
      const measured = Math.floor(node.clientWidth - 28);
      setPageWidth(measured > 320 ? measured : 320);
    };

    updateWidth();

    if (typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver(() => {
      updateWidth();
    });

    observer.observe(node);

    return () => {
      observer.disconnect();
    };
  }, []);

  const scrollToMatch = useCallback(
    (pageNumber: number) => {
      if (hasAutoScrolled || !normalizedTarget) {
        return;
      }

      const pageRoot = pageRefs.current[pageNumber];
      if (!pageRoot) {
        return;
      }

      const highlightedNode = pageRoot.querySelector(".kura-pdf-highlight");
      const nodeToScroll = highlightedNode ?? pageRoot;

      nodeToScroll.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });

      if (highlightedNode instanceof HTMLElement) {
        highlightedNode.classList.add("kura-pdf-highlight-focus");
      }

      setHasAutoScrolled(true);
    },
    [hasAutoScrolled, normalizedTarget],
  );

  useEffect(() => {
    if (firstMatchedPage === null || hasAutoScrolled || !normalizedTarget) {
      return;
    }

    const timeout = window.setTimeout(() => {
      scrollToMatch(firstMatchedPage);
    }, 0);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [firstMatchedPage, hasAutoScrolled, normalizedTarget, scrollToMatch]);

  const handlePageTextSuccess = useCallback(
    (pageNumber: number, items: readonly unknown[]) => {
      if (!normalizedTargetLower) {
        return;
      }

      if (!scannedPagesRef.current.has(pageNumber)) {
        scannedPagesRef.current.add(pageNumber);
        setCheckedPageCount((current) => current + 1);
      }

      if (firstMatchedPage !== null) {
        return;
      }

      const hasMatch = items.some((item) => {
        const pageText = getItemText(item);
        return pageText?.toLowerCase().includes(normalizedTargetLower) ?? false;
      });

      if (hasMatch) {
        setFirstMatchedPage((current) => current ?? pageNumber);
      }
    },
    [firstMatchedPage, normalizedTargetLower],
  );

  const renderTextLayerItem = useCallback(
    (textItem: { str: string }) => renderHighlightedText(textItem.str, normalizedTarget),
    [normalizedTarget],
  );

  const noMatchFound =
    normalizedTarget.length > 0 &&
    numPages > 0 &&
    checkedPageCount >= numPages &&
    firstMatchedPage === null;

  return (
    <div ref={containerRef} className="interactive-pdf-viewer flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      {noMatchFound && (
        <p className="mx-2 mt-2 rounded-md border border-amber-800/60 bg-amber-950/30 px-3 py-2 text-xs text-amber-200">
          No matching text was found in the rendered pages.
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-auto px-2 py-3">
        <Document
          key={fileUrl}
          file={fileUrl}
          className="interactive-pdf-document w-full"
          loading={<p className="px-3 py-4 text-sm text-slate-400">Loading PDF...</p>}
          noData={<p className="px-3 py-4 text-sm text-slate-400">No PDF selected.</p>}
          error={
            <p className="px-3 py-4 text-sm text-rose-300">
              {loadError ?? "Unable to load the PDF file."}
            </p>
          }
          onLoadError={(error) => {
            const message = error instanceof Error ? error.message : String(error);
            setLoadError(message);
          }}
          onLoadSuccess={({ numPages: totalPages }) => {
            setNumPages(totalPages);
            setLoadError(null);
          }}
        >
          <div className="interactive-pdf-screen-pages">
            {Array.from({ length: numPages }, (_, index) => {
              const pageNumber = index + 1;

              return (
                <div
                  key={`pdf-page-${pageNumber}`}
                  ref={(node) => {
                    pageRefs.current[pageNumber] = node;
                  }}
                  className="interactive-pdf-page-frame mb-4 rounded-lg border border-slate-800/80 bg-slate-900/40 p-3"
                >
                  <Page
                    pageNumber={pageNumber}
                    width={pageWidth}
                    renderTextLayer
                    renderAnnotationLayer
                    loading={
                      <p className="px-3 py-4 text-sm text-slate-500">
                        Rendering page {pageNumber}...
                      </p>
                    }
                    customTextRenderer={normalizedTarget ? renderTextLayerItem : undefined}
                    onRenderTextLayerSuccess={() => {
                      if (firstMatchedPage === pageNumber) {
                        scrollToMatch(pageNumber);
                      }
                    }}
                    onGetTextSuccess={({ items }) => {
                      handlePageTextSuccess(pageNumber, items);
                    }}
                  />
                </div>
              );
            })}
          </div>

          <div className="interactive-pdf-print-pages">
            {Array.from(new Array(numPages), (_el, index) => {
              const pageNumber = index + 1;

              return (
                <div
                  key={`pdf-print-page-${pageNumber}`}
                  className="interactive-pdf-page-frame mb-4 rounded-lg border border-slate-800/80 bg-slate-900/40 p-3"
                >
                  <Page
                    pageNumber={pageNumber}
                    width={pageWidth}
                    renderTextLayer
                    renderAnnotationLayer
                    loading={
                      <p className="px-3 py-4 text-sm text-slate-500">
                        Rendering page {pageNumber}...
                      </p>
                    }
                    customTextRenderer={normalizedTarget ? renderTextLayerItem : undefined}
                  />
                </div>
              );
            })}
          </div>
        </Document>
      </div>
    </div>
  );
}
