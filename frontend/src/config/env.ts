/**
 * Frontend configuration.
 *
 * Mirror of the backend's `config.py`: read the environment once, validate it,
 * export a typed object. Nothing else in the app touches `import.meta.env`.
 *
 * Reminder: Vite inlines `VITE_*` variables into the bundle at build time.
 * They are visible to anyone who opens devtools — never put a secret here.
 */

interface AppConfig {
  readonly apiBaseUrl: string;
  readonly apiVersion: string;
  readonly isDev: boolean;
}

function requireEnv(key: string, value: string | undefined, fallback?: string): string {
  const resolved = value ?? fallback;
  if (!resolved) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return resolved;
}

export const config: AppConfig = {
  apiBaseUrl: requireEnv(
    'VITE_API_BASE_URL',
    import.meta.env.VITE_API_BASE_URL,
    'http://localhost:8000',
  ).replace(/\/$/, ''), // strip trailing slash so URL joins stay predictable
  apiVersion: 'v1',
  isDev: import.meta.env.DEV,
};
