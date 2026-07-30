/**
 * Live backend connectivity.
 *
 * A diagnostics panel at /status:
 * proof that browser -> API -> PostgreSQL is wired end to end, and the first
 * thing to check when something looks broken.
 */

import { useSystemStatus } from './useSystemStatus';
import type { ComponentStatus } from './types';

const STATUS_STYLES: Record<ComponentStatus, string> = {
  ok: 'text-accent bg-accent/10 ring-accent/25',
  degraded: 'text-amber-400 bg-amber-400/10 ring-amber-400/25',
  down: 'text-danger bg-danger/10 ring-danger/25',
};

function StatusBadge({ status }: { status: ComponentStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${STATUS_STYLES[status]}`}
    >
      <span className="size-1.5 rounded-full bg-current" aria-hidden="true" />
      {status}
    </span>
  );
}

export function SystemStatusPanel() {
  const { readiness, info, isLoading, error, refresh } = useSystemStatus();

  return (
    <section className="border-line bg-surface/70 w-full max-w-md rounded-2xl border p-6 backdrop-blur-xl">
      <header className="mb-5 flex items-center justify-between">
        <h2 className="text-ink-muted text-xs font-medium tracking-widest uppercase">
          System status
        </h2>
        <button
          type="button"
          onClick={refresh}
          disabled={isLoading}
          className="text-ink-faint hover:text-ink hover:bg-raised rounded-md px-2 py-1 text-xs transition disabled:opacity-40"
        >
          Refresh
        </button>
      </header>

      {isLoading && <p className="text-ink-faint text-sm">Checking backend…</p>}

      {error && (
        <div className="border-danger/30 bg-danger-soft rounded-xl border p-4">
          <p className="text-danger text-sm font-medium">{error.message}</p>
          <p className="text-danger/70 mt-1 font-mono text-xs">
            {error.code} · request {error.requestId}
          </p>
        </div>
      )}

      {!isLoading && !error && readiness && (
        <dl className="space-y-3">
          <div className="flex items-center justify-between">
            <dt className="text-ink-muted text-sm">API</dt>
            <dd>
              <StatusBadge status={readiness.status} />
            </dd>
          </div>

          {readiness.dependencies.map((dependency) => (
            <div key={dependency.name} className="flex items-center justify-between">
              <dt className="text-ink-muted text-sm capitalize">{dependency.name}</dt>
              <dd className="flex items-center gap-3">
                {dependency.latency_ms !== null && (
                  <span className="text-ink-faint font-mono text-xs">
                    {dependency.latency_ms.toFixed(1)} ms
                  </span>
                )}
                <StatusBadge status={dependency.status} />
              </dd>
            </div>
          ))}

          {info && (
            <div className="border-line text-ink-faint mt-5 border-t pt-4 text-xs">
              {info.name} v{info.version} · {info.environment} · api {info.api_version}
            </div>
          )}
        </dl>
      )}
    </section>
  );
}
