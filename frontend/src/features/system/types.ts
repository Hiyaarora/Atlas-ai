/** Mirrors `app/schemas/health.py`. Keep the two in sync. */

export type ComponentStatus = 'ok' | 'degraded' | 'down';

export interface DependencyCheck {
  name: string;
  status: ComponentStatus;
  latency_ms: number | null;
  error: string | null;
}

export interface ReadinessResponse {
  status: ComponentStatus;
  dependencies: DependencyCheck[];
}

export interface AppInfoResponse {
  name: string;
  version: string;
  environment: string;
  api_version: string;
}
