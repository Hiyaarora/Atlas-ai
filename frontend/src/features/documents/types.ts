/** Mirrors `app/schemas/documents.py`. */

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed';

/**
 * Mirrors `DocumentResponse`.
 *
 * Note the absences: no chunk count, no embedding model, no storage key. How
 * many pieces the text was cut into is an implementation detail — it tells a
 * user nothing they can act on, and showing it invites questions the product
 * should never make them ask.
 */
export interface KnowledgeDocument {
  id: string;
  filename: string;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  page_count: number;
  created_at: string;
}

export interface DocumentUploadResult {
  document: KnowledgeDocument;
  /** The conversation created alongside the document — navigate straight in. */
  conversation_id: string;
}

export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  score: number;
  excerpt: string;
}
