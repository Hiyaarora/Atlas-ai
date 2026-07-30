/**
 * The Atlas mark.
 *
 * An atlas is a book of maps, so the mark is a stylised meridian globe: a
 * ring with a latitude sweep and a plotted point. Drawn as inline SVG with
 * `currentColor` so it inherits text colour and needs no asset request.
 */

export function Logo({ className = 'size-6' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      className={className}
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="9" />
      <ellipse cx="12" cy="12" rx="4" ry="9" />
      <path d="M3.6 9h16.8M3.6 15h16.8" />
      <circle cx="16" cy="8.4" r="1.6" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Wordmark used in the sidebar and on auth screens. */
export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="flex items-center gap-2.5">
      <span className="text-accent">
        <Logo className="size-6" />
      </span>
      {!compact && (
        <span className="text-[0.95rem] font-semibold tracking-tight text-ink">
          Atlas<span className="text-ink-faint"> AI</span>
        </span>
      )}
    </span>
  );
}
