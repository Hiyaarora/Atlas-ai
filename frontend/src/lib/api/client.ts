/**
 * The single HTTP client every feature uses to reach the backend.
 *
 * Beyond issuing requests, it owns two auth concerns so no component has to:
 *
 * 1. **Attaching the access token.** Held in a module variable, not
 *    localStorage. localStorage is readable by any script on the page, so an
 *    XSS or a compromised dependency can exfiltrate a token stored there. A
 *    module variable dies with the tab, and the httpOnly refresh cookie
 *    restores the session on reload.
 *
 * 2. **Transparent refresh on 401.** When the 30-minute access token expires
 *    mid-session, the client refreshes once and replays the original request.
 *    The user never sees it.
 *
 * That second concern lives in `authorizedFetch`, deliberately separate from
 * the JSON `request` helper. Two calls — the chat SSE stream and the multipart
 * upload — cannot use `request` (one reads a stream, the other must let the
 * browser set its own multipart boundary) but must still refresh on 401.
 * Sharing the primitive is what keeps them from silently drifting into
 * "log the user out" behaviour that the rest of the app does not have.
 */

import { config } from '@/config/env';

/** Matches the backend envelope in `app/core/exceptions.py`. */
interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    details: Record<string, unknown>;
    request_id: string;
  };
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details: Record<string, unknown> = {},
    readonly requestId: string = '-',
  ) {
    super(message);
    this.name = 'ApiError';
  }

  /** True for network failures and 5xx — the class of errors worth retrying. */
  get isRetryable(): boolean {
    return this.status === 0 || this.status >= 500;
  }
}

// ---------------------------------------------------------------------------
// Auth state
// ---------------------------------------------------------------------------

let accessToken: string | null = null;

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

// There is deliberately no `getAccessToken`. Handing the token out invites
// callers to build their own `fetch` with their own `Authorization` header —
// which is exactly how the SSE stream and the upload ended up without
// refresh-on-401. Everything that needs the token goes through
// `authorizedFetch`, so the retry path can never be forgotten.

/**
 * Supplied by AuthProvider. Kept as a callback rather than a direct import so
 * this module stays free of React and of any circular dependency on the auth
 * feature.
 */
type RefreshHandler = () => Promise<boolean>;
let refreshHandler: RefreshHandler | null = null;

export function setRefreshHandler(handler: RefreshHandler | null): void {
  refreshHandler = handler;
}

/**
 * In-flight refresh, shared by every request that 401s at once.
 *
 * Without this, five parallel requests hitting an expired token would fire
 * five refreshes. Since refresh tokens rotate, four of those would present an
 * already-revoked token — logging the user out during ordinary use.
 */
let inFlightRefresh: Promise<boolean> | null = null;

function refreshOnce(): Promise<boolean> {
  if (!refreshHandler) return Promise.resolve(false);

  inFlightRefresh ??= refreshHandler().finally(() => {
    inFlightRefresh = null;
  });

  return inFlightRefresh;
}

// ---------------------------------------------------------------------------
// authorizedFetch — the one place 401s are handled
// ---------------------------------------------------------------------------

export interface AuthorizedFetchOptions {
  /**
   * Abort the request after this many ms. A hung fetch never resolves on its
   * own. Omit it for streams and uploads, which are legitimately slow.
   */
  timeoutMs?: number;
  /** Skip the refresh-on-401 dance (used by the auth endpoints themselves). */
  skipAuthRefresh?: boolean;
  /** Caller's cancellation signal — the chat Stop button, or an unmount. */
  signal?: AbortSignal;
}

/**
 * `fetch` plus the session: attaches the bearer token, and on 401 refreshes
 * once and replays the request.
 *
 * Returns the raw `Response` rather than parsed JSON, because the two callers
 * that most need it are not JSON: the SSE stream reads `response.body`, and
 * the upload posts `FormData`.
 */
export async function authorizedFetch(
  url: string,
  // `headers` is narrowed to a plain object rather than HeadersInit: we merge
  // Authorization into it by spreading, which silently produces nothing useful
  // for a Headers instance or an entry array. Narrowing makes that a compile
  // error instead of a request that mysteriously arrives unauthenticated.
  init: Omit<RequestInit, 'signal' | 'headers'> & { headers?: Record<string, string> } = {},
  { timeoutMs, skipAuthRefresh = false, signal }: AuthorizedFetchOptions = {},
): Promise<Response> {
  const attempt = async (): Promise<Response> => {
    // A fresh controller per attempt. Reusing one would mean a retry inherits
    // an already-fired timeout and aborts instantly.
    const controller = new AbortController();
    let timedOut = false;

    const forwardAbort = () => controller.abort();
    signal?.addEventListener('abort', forwardAbort, { once: true });
    if (signal?.aborted) controller.abort();

    const timer =
      timeoutMs === undefined
        ? null
        : setTimeout(() => {
            timedOut = true;
            controller.abort();
          }, timeoutMs);

    try {
      return await fetch(url, {
        ...init,
        signal: controller.signal,
        // Required for the httpOnly refresh cookie to be sent, since the API
        // is on a different origin from the dev server.
        credentials: 'include',
        // Rebuilt on every attempt: the whole point of the retry is that it
        // carries the *refreshed* token, so this cannot be hoisted out.
        headers: {
          ...init.headers,
          ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        },
      });
    } catch (cause) {
      // A caller-initiated abort must stay a DOMException. useChat identifies
      // "user pressed Stop" by that type; wrapping it would surface a spurious
      // error banner every time someone cancels a reply.
      if (signal?.aborted) throw cause;
      throw new ApiError(
        0,
        'network_error',
        timedOut ? `Request timed out after ${timeoutMs}ms` : 'Could not reach the Atlas AI API',
      );
    } finally {
      if (timer !== null) clearTimeout(timer);
      signal?.removeEventListener('abort', forwardAbort);
    }
  };

  const response = await attempt();

  if (response.status !== 401 || skipAuthRefresh) return response;

  const refreshed = await refreshOnce();
  if (!refreshed) return response;

  // Discard the body of the 401 we are about to replace, so the connection is
  // released rather than left half-read.
  void response.body?.cancel();
  return attempt();
}

// ---------------------------------------------------------------------------
// JSON request pipeline
// ---------------------------------------------------------------------------

interface RequestOptions extends Omit<RequestInit, 'body' | 'headers'> {
  body?: unknown;
  /** Plain object only — see the note on `authorizedFetch`. */
  headers?: Record<string, string>;
  /** Abort the request after this many ms. A hung fetch never resolves. */
  timeoutMs?: number;
  /** Skip the refresh-on-401 dance (used by the auth endpoints themselves). */
  skipAuthRefresh?: boolean;
}

const DEFAULT_TIMEOUT_MS = 15_000;

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, timeoutMs = DEFAULT_TIMEOUT_MS, headers, skipAuthRefresh, signal, ...rest } = options;

  const response = await authorizedFetch(
    `${config.apiBaseUrl}/api/${config.apiVersion}${path}`,
    {
      ...rest,
      headers: { 'Content-Type': 'application/json', ...headers },
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    },
    { timeoutMs, skipAuthRefresh: skipAuthRefresh ?? false, ...(signal ? { signal } : {}) },
  );

  if (response.status === 204) {
    return undefined as T;
  }

  const payload: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const envelope = payload as ErrorEnvelope | null;
    throw new ApiError(
      response.status,
      envelope?.error?.code ?? 'unknown_error',
      envelope?.error?.message ?? response.statusText,
      envelope?.error?.details ?? {},
      envelope?.error?.request_id ?? response.headers.get('X-Request-ID') ?? '-',
    );
  }

  return payload as T;
}

export const apiClient = {
  get: <T>(path: string, options?: RequestOptions) => request<T>(path, { ...options, method: 'GET' }),

  post: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(path, { ...options, method: 'POST', body }),

  patch: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(path, { ...options, method: 'PATCH', body }),

  delete: <T>(path: string, options?: RequestOptions) =>
    request<T>(path, { ...options, method: 'DELETE' }),
};
