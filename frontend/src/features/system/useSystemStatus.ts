/**
 * Hook that fetches backend status on mount.
 *
 * Deliberately hand-written rather than pulling in TanStack Query: at one
 * endpoint, a query library is unearned complexity. Once caching,
 * refetching, and optimistic updates are needed, that trade flips — and this
 * hook is small enough to throw away.
 */

import { useCallback, useEffect, useState } from 'react';
import { ApiError } from '@/lib/api/client';
import { systemApi } from './api';
import type { AppInfoResponse, ReadinessResponse } from './types';

interface SystemStatus {
  readiness: ReadinessResponse | null;
  info: AppInfoResponse | null;
  isLoading: boolean;
  error: ApiError | null;
  refresh: () => void;
}

export function useSystemStatus(): SystemStatus {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [info, setInfo] = useState<AppInfoResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setIsLoading(true);
    setError(null);
    try {
      // Both requests are independent — fire them together rather than
      // paying two sequential round trips.
      const [readinessResult, infoResult] = await Promise.all([
        systemApi.getReadiness(),
        systemApi.getInfo(),
      ]);
      if (signal?.aborted) return;
      setReadiness(readinessResult);
      setInfo(infoResult);
    } catch (caught) {
      if (signal?.aborted) return;
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError(0, 'unknown_error', 'Something went wrong'),
      );
    } finally {
      if (!signal?.aborted) setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    // Guard against setting state after unmount (StrictMode double-invokes
    // effects in development).
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  return { readiness, info, isLoading, error, refresh: () => void load() };
}
