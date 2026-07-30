/**
 * Diagnostics page at /status.
 *
 * Not part of the main product flow — it exists so connectivity problems can
 * be diagnosed from the browser without opening a terminal.
 */

import { Link } from 'react-router-dom';
import { Wordmark } from '@/components/Logo';
import { SystemStatusPanel } from '@/features/system/SystemStatusPanel';

export function DashboardPage() {
  return (
    <div className="min-h-full">
      <header className="border-line border-b">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <Wordmark />
          <Link to="/" className="text-ink-muted hover:text-ink text-sm transition">
            Back to chat
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-12">
        <h1 className="text-xl font-semibold tracking-tight">Diagnostics</h1>
        <p className="text-ink-muted mt-1.5 text-sm">
          Live connectivity between the browser, the API, and PostgreSQL.
        </p>

        <div className="mt-8">
          <SystemStatusPanel />
        </div>
      </main>
    </div>
  );
}
