/**
 * Chat API, including the SSE reader.
 *
 * Why not `EventSource`? The browser's built-in SSE client cannot set request
 * headers, so there is no way to attach `Authorization: Bearer …`, and it can
 * only issue GET requests — but sending a message is a POST with a body.
 *
 * So we use `fetch` and read the response stream ourselves. That also means
 * we own the framing: bytes → text → SSE frames → typed events.
 */

import { config } from '@/config/env';
import type { Citation } from '@/features/documents/types';
import { ApiError, apiClient, authorizedFetch } from '@/lib/api/client';
import type { ConversationDetail, ConversationSummary, StreamEvent } from './types';

export const chatApi = {
  listConversations: () => apiClient.get<ConversationSummary[]>('/conversations'),

  createConversation: (title?: string) =>
    apiClient.post<ConversationSummary>('/conversations', { title: title ?? null }),

  getConversation: (id: string) => apiClient.get<ConversationDetail>(`/conversations/${id}`),

  renameConversation: (id: string, title: string) =>
    apiClient.patch<ConversationSummary>(`/conversations/${id}`, { title }),

  deleteConversation: (id: string) => apiClient.delete<void>(`/conversations/${id}`),

  streamMessage,
};

/**
 * POST a message and yield reply events as they arrive.
 *
 * An async generator rather than a callback: the caller writes a plain
 * `for await` loop, and cancellation falls out of `break` for free.
 */
async function* streamMessage(
  conversationId: string,
  content: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const url = `${config.apiBaseUrl}/api/${config.apiVersion}/conversations/${conversationId}/messages`;

  // Via authorizedFetch, not bare fetch: sending a message is the single most
  // likely request to meet an expired access token, since it is what a user
  // does after leaving a tab open. It must refresh and replay like any other.
  //
  // No timeoutMs — a long reply legitimately takes longer than the 15s default,
  // and the caller's own signal already backs the Stop button.
  const response = await authorizedFetch(
    url,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ content }),
    },
    { ...(signal ? { signal } : {}) },
  );

  // Failures before the stream opens are ordinary HTTP errors (401, 404, 422)
  // and still carry the standard error envelope.
  if (!response.ok) {
    const envelope = (await response.json().catch(() => null)) as {
      error?: { code: string; message: string; request_id: string };
    } | null;
    throw new ApiError(
      response.status,
      envelope?.error?.code ?? 'unknown_error',
      envelope?.error?.message ?? response.statusText,
      {},
      envelope?.error?.request_id ?? '-',
    );
  }

  if (!response.body) {
    throw new ApiError(0, 'stream_unsupported', 'This browser cannot read streaming responses.');
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

  // Frames are delimited by a blank line, and a chunk can split one anywhere —
  // mid-word, mid-JSON, even between the \n\n. Anything not yet terminated
  // stays in the buffer until the rest arrives.
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += value;

      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const event = parseFrame(frame);
        if (event) yield event;

        boundary = buffer.indexOf('\n\n');
      }
    }
  } finally {
    // Releasing the lock lets the connection be torn down promptly when the
    // caller breaks out of the loop.
    reader.releaseLock();
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let name: string | null = null;
  let data: string | null = null;

  for (const line of frame.split('\n')) {
    if (line.startsWith('event: ')) name = line.slice(7);
    else if (line.startsWith('data: ')) data = line.slice(6);
  }

  if (!name || data === null) return null;

  // The server controls this payload, so the shapes are known; `unknown` plus
  // narrowing keeps a malformed frame from becoming a runtime crash mid-stream.
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(data) as Record<string, unknown>;
  } catch {
    return null;
  }

  const asString = (value: unknown, fallback = ''): string =>
    typeof value === 'string' ? value : fallback;

  switch (name) {
    case 'sources':
      return {
        type: 'sources',
        citations: Array.isArray(payload.citations) ? (payload.citations as Citation[]) : [],
      };
    case 'token':
      return { type: 'token', text: asString(payload.text) };
    case 'done':
      return {
        type: 'done',
        messageId: asString(payload.message_id),
        content: asString(payload.content),
        model: typeof payload.model === 'string' ? payload.model : null,
      };
    case 'error':
      return {
        type: 'error',
        code: asString(payload.code, 'unknown_error'),
        message: asString(payload.message, 'Something went wrong.'),
        requestId: asString(payload.request_id, '-'),
      };
    default:
      return null;
  }
}
