import * as pdfjsLib from "pdfjs-dist";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.mjs",
  import.meta.url,
).toString();

type TextItemLike = {
  str?: unknown;
};

const PDF_EXTRACT_ERROR_PREFIX = "PDF extraction failed: ";

function textFromItem(item: unknown): string {
  if (typeof item !== "object" || item === null) {
    return "";
  }

  const maybeText = (item as TextItemLike).str;
  return typeof maybeText === "string" ? maybeText : "";
}

export function isPdfExtractionErrorMessage(value: string): boolean {
  return value.startsWith(PDF_EXTRACT_ERROR_PREFIX);
}

export async function extractTextFromPdf(fileUrl: string): Promise<string> {
  const normalizedUrl = fileUrl.trim();

  if (!normalizedUrl) {
    return `${PDF_EXTRACT_ERROR_PREFIX}No PDF file URL was provided.`;
  }

  try {
    const pdfDocument = await pdfjsLib.getDocument(normalizedUrl).promise;
    const pageTexts: string[] = [];

    for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
      const page = await pdfDocument.getPage(pageNumber);
      const textContent = await page.getTextContent();

      const pageText = textContent.items
        .map(textFromItem)
        .map((text) => text.trim())
        .filter((text) => text.length > 0)
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();

      if (pageText.length > 0) {
        pageTexts.push(pageText);
      }
    }

    const combinedText = pageTexts.join("\n").trim();

    if (!combinedText) {
      return `${PDF_EXTRACT_ERROR_PREFIX}No extractable text found. The PDF may be scanned or image-only.`;
    }

    return combinedText;
  } catch (error) {
    console.error("PDF text extraction failed:", error);
    return `${PDF_EXTRACT_ERROR_PREFIX}Unable to read this PDF. The file may be corrupted or unsupported.`;
  }
}
