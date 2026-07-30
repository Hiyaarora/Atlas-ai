import { apiClient } from '@/lib/api/client';
import type { LoginPayload, RegisterPayload, TokenResponse, User } from './types';

/**
 * Refresh is single-flight across the entire tab.
 *
 * Refresh tokens rotate: using one revokes it. So two concurrent refreshes
 * with the same cookie means the second presents an already-revoked token and
 * gets a 401 — logging the user out during what should be a normal session.
 *
 * That is not hypothetical. React's <StrictMode> invokes effects twice in
 * development, which fired two bootstrap refreshes in parallel and signed the
 * user out on every page reload. Deduplicating here covers every caller:
 * the session bootstrap, the 401-retry pipeline, and anything added later.
 *
 * Note the remaining limitation: this shares one promise per *tab*. Two tabs
 * reloading simultaneously still race, since they are separate JS contexts.
 * Refresh-token reuse detection would address that properly.
 */
let inFlightRefresh: Promise<TokenResponse> | null = null;

function refresh(): Promise<TokenResponse> {
  inFlightRefresh ??= apiClient
    // skipAuthRefresh is essential: a 401 from /refresh must surface as a
    // failure, not trigger another refresh attempt and recurse.
    .post<TokenResponse>('/auth/refresh', undefined, { skipAuthRefresh: true })
    .finally(() => {
      inFlightRefresh = null;
    });

  return inFlightRefresh;
}

export const authApi = {
  register: (payload: RegisterPayload) =>
    apiClient.post<TokenResponse>('/auth/register', payload, { skipAuthRefresh: true }),

  login: (payload: LoginPayload) =>
    apiClient.post<TokenResponse>('/auth/login', payload, { skipAuthRefresh: true }),

  refresh,

  logout: () => apiClient.post<void>('/auth/logout', undefined, { skipAuthRefresh: true }),

  me: () => apiClient.get<User>('/auth/me'),
};
