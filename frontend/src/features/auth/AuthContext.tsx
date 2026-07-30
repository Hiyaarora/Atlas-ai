/**
 * Authentication state for the whole app.
 *
 * The access token lives in React state and a module variable inside the API
 * client — never in localStorage. On a page reload it is gone, and the
 * provider silently re-establishes the session from the httpOnly refresh
 * cookie. That bootstrap is what `status: 'loading'` represents, and why
 * routes must wait for it rather than assuming "no token means logged out".
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { ApiError, setAccessToken, setRefreshHandler } from '@/lib/api/client';
import { authApi } from './api';
import type { LoginPayload, RegisterPayload, User } from './types';

type AuthStatus = 'loading' | 'authenticated' | 'anonymous';

interface AuthContextValue {
  user: User | null;
  status: AuthStatus;
  login: (payload: LoginPayload) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [status, setStatus] = useState<AuthStatus>('loading');

  // A ref, not state: the API client calls this outside React's render cycle,
  // and it must always see the current implementation without re-registering.
  const userRef = useRef<User | null>(null);
  userRef.current = user;

  const applySession = useCallback((accessToken: string, nextUser: User) => {
    setAccessToken(accessToken);
    setUser(nextUser);
    setStatus('authenticated');
  }, []);

  const clearSession = useCallback(() => {
    setAccessToken(null);
    setUser(null);
    setStatus('anonymous');
  }, []);

  // --- Give the API client a way to refresh expired access tokens ---------
  useEffect(() => {
    setRefreshHandler(async () => {
      try {
        const response = await authApi.refresh();
        applySession(response.access_token, response.user);
        return true;
      } catch {
        clearSession();
        return false;
      }
    });

    return () => setRefreshHandler(null);
  }, [applySession, clearSession]);

  // --- Restore the session on first load ---------------------------------
  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const response = await authApi.refresh();
        if (!cancelled) applySession(response.access_token, response.user);
      } catch (error) {
        // 401 here is the normal "not logged in" case, not a failure worth
        // surfacing. Anything else is worth knowing about in development.
        if (import.meta.env.DEV && error instanceof ApiError && error.status !== 401) {
          console.warn('Session restore failed:', error.message);
        }
        if (!cancelled) clearSession();
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [applySession, clearSession]);

  const login = useCallback(
    async (payload: LoginPayload) => {
      const response = await authApi.login(payload);
      applySession(response.access_token, response.user);
    },
    [applySession],
  );

  const register = useCallback(
    async (payload: RegisterPayload) => {
      const response = await authApi.register(payload);
      applySession(response.access_token, response.user);
    },
    [applySession],
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      // Clear locally even if the request failed. The user asked to log out;
      // a network error must not leave them looking logged in.
      clearSession();
    }
  }, [clearSession]);

  const value = useMemo(
    () => ({ user, status, login, register, logout }),
    [user, status, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used inside <AuthProvider>');
  }
  return context;
}
