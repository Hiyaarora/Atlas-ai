/**
 * System/health API calls.
 *
 * One module per feature keeps endpoint knowledge next to the feature that
 * uses it, instead of in one ever-growing `api.ts` god file.
 */

import { apiClient } from '@/lib/api/client';
import type { AppInfoResponse, ReadinessResponse } from './types';

export const systemApi = {
  getReadiness: () => apiClient.get<ReadinessResponse>('/health/ready'),
  getInfo: () => apiClient.get<AppInfoResponse>('/health/info'),
};
