import { config } from '@/config/env';
import { ApiError, apiClient, authorizedFetch } from '@/lib/api/client';
import type { DocumentUploadResult, KnowledgeDocument } from './types';

export const documentsApi = {
  list: () => apiClient.get<KnowledgeDocument[]>('/documents'),

  get: (id: string) => apiClient.get<KnowledgeDocument>(`/documents/${id}`),

  remove: (id: string) => apiClient.delete<void>(`/documents/${id}`),

  reindex: (id: string) => apiClient.post<KnowledgeDocument>(`/documents/${id}/reindex`),

  upload,
};

/**
 * Uploads bypass the JSON `apiClient` but not `authorizedFetch`.
 *
 * The JSON helper sets `Content-Type: application/json` and serialises the
 * body. A multipart upload needs neither: the browser must generate the
 * `multipart/form-data` header itself, because it alone knows the boundary
 * string it will use. Setting that header by hand is the classic mistake — it
 * produces a boundary the body does not actually use, and the server sees a
 * malformed request. So we pass no headers at all and let `authorizedFetch`
 * add only `Authorization`.
 *
 * Replaying this on 401 is safe: `FormData` holding a `File` is a reference to
 * bytes on disk, not a consumed stream, so the retry re-reads it cleanly.
 *
 * No timeoutMs — a large file over a slow link would trip the 15s default.
 */
async function upload(file: File): Promise<DocumentUploadResult> {
  const body = new FormData();
  body.append('file', file);

  const response = await authorizedFetch(`${config.apiBaseUrl}/api/${config.apiVersion}/documents`, {
    method: 'POST',
    body,
  });

  const payload: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const envelope = payload as {
      error?: { code: string; message: string; request_id: string };
    } | null;
    throw new ApiError(
      response.status,
      envelope?.error?.code ?? 'upload_failed',
      envelope?.error?.message ?? 'Upload failed.',
      {},
      envelope?.error?.request_id ?? '-',
    );
  }

  return payload as DocumentUploadResult;
}
