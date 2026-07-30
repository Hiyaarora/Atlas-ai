import { useEffect, useRef } from 'react';
import { SendIcon, StopIcon } from '@/components/icons';

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled: boolean;
  /** Why the composer is locked, if it is. */
  notice?: string | null;
}

const MAX_HEIGHT_PX = 200;

export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  isStreaming,
  disabled,
  notice = null,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow: reset to `auto` first so the box can also *shrink* when text is
  // deleted. Without the reset, scrollHeight only ever reports the tallest
  // size the element has held.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT_PX)}px`;

    // Only allow scrolling once the box has stopped growing. Left on `auto`,
    // the browser paints a scrollbar track inside the composer even at one
    // row, which shows through as a stray vertical bar next to the send
    // button.
    element.style.overflowY = element.scrollHeight > MAX_HEIGHT_PX ? 'auto' : 'hidden';
  }, [value]);

  const canSend = !disabled && !isStreaming && value.trim().length > 0;

  return (
    <div className="relative px-4 pb-5 sm:px-6">
      {/* Scrim: fades the transcript out as it reaches the composer instead
          of letting a line of code end in a hard horizontal cut. */}
      <div
        aria-hidden="true"
        className="from-canvas pointer-events-none absolute inset-x-0 -top-10 h-10 bg-gradient-to-t to-transparent"
      />

      <div className="mx-auto w-full max-w-3xl">
        <div className="border-line bg-raised/70 focus-within:border-accent/40 focus-within:bg-raised flex items-end gap-2 rounded-2xl border p-2 shadow-2xl backdrop-blur-xl transition">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends, Shift+Enter inserts a newline — the convention
              // every chat product uses, so it needs no explanation.
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                if (canSend) onSend();
              }
            }}
            rows={1}
            disabled={disabled}
            placeholder={notice ?? 'Ask Atlas AI…'}
            aria-label="Message"
            className="placeholder:text-ink-faint text-ink max-h-[200px] flex-1 resize-none bg-transparent px-3 py-2.5 text-[0.94rem] leading-relaxed outline-none disabled:opacity-50"
          />

          {isStreaming ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop generating"
              className="border-line text-ink-muted hover:text-ink hover:border-line-strong grid size-9 shrink-0 place-items-center rounded-xl border transition"
            >
              <StopIcon className="size-3.5" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onSend}
              disabled={!canSend}
              aria-label="Send message"
              className="bg-accent text-canvas hover:bg-accent-strong disabled:bg-overlay disabled:text-ink-faint grid size-9 shrink-0 place-items-center rounded-xl transition active:scale-95 disabled:active:scale-100"
            >
              <SendIcon className="size-4" />
            </button>
          )}
        </div>

        <p className="text-ink-faint mt-2.5 flex items-center justify-center gap-1.5 text-center text-[0.7rem]">
          {notice ? (
            <>
              <span className="bg-accent size-1.5 animate-pulse rounded-full" aria-hidden="true" />
              {notice}
            </>
          ) : (
            'Atlas AI can make mistakes. Verify anything important.'
          )}
        </p>
      </div>
    </div>
  );
}
