import { useEffect, useRef } from 'react';
import { Logo } from '@/components/Logo';
import { Markdown } from '@/components/Markdown';
import type { Citation } from '@/features/documents/types';
import { Citations } from './Citations';
import type { Message } from './types';

interface TranscriptProps {
  messages: Message[];
  streamingText: string;
  streamingCitations: Citation[];
  citationsByMessage: Record<string, Citation[]>;
  isStreaming: boolean;
  isLoading: boolean;
  userInitial: string;
}

export function Transcript({
  messages,
  streamingText,
  streamingCitations,
  citationsByMessage,
  isStreaming,
  isLoading,
  userInitial,
}: TranscriptProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const pinnedToBottom = useRef(true);

  // Only auto-scroll when the user is already at the bottom. Yanking the view
  // down while someone is reading earlier output is one of the most irritating
  // things a chat UI can do.
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;

    const onScroll = () => {
      const distanceFromBottom =
        element.scrollHeight - element.scrollTop - element.clientHeight;
      pinnedToBottom.current = distanceFromBottom < 120;
    };

    element.addEventListener('scroll', onScroll, { passive: true });
    return () => element.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (pinnedToBottom.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages.length, streamingText]);

  if (isLoading) {
    return (
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-8 px-6 py-10">
          {[0, 1].map((index) => (
            <div key={index} className="space-y-2.5">
              <div className="bg-raised h-3 w-20 animate-pulse rounded" />
              <div className="bg-raised h-3 w-full animate-pulse rounded" />
              <div className="bg-raised h-3 w-4/5 animate-pulse rounded" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
        {messages.map((message) => (
          <Turn
            key={message.id}
            message={message}
            userInitial={userInitial}
            citations={citationsByMessage[message.id] ?? []}
          />
        ))}

        {isStreaming && (
          <Turn
            message={{
              id: 'streaming',
              role: 'assistant',
              content: streamingText,
              model: null,
              created_at: '',
            }}
            userInitial={userInitial}
            citations={streamingCitations}
            isStreaming
          />
        )}

        <div ref={bottomRef} className="h-4" />
      </div>
    </div>
  );
}

/**
 * One conversational turn.
 *
 * Named turns rather than chat bubbles. Bubbles suit short back-and-forth;
 * assistant answers here run to several paragraphs with code and tables, and
 * constraining that to a rounded blob wastes horizontal space and makes code
 * blocks scroll inside a container inside a container.
 */
function Turn({
  message,
  userInitial,
  citations,
  isStreaming = false,
}: {
  message: Message;
  userInitial: string;
  citations: Citation[];
  isStreaming?: boolean;
}) {
  const isUser = message.role === 'user';

  return (
    <article className="animate-rise group py-5">
      <header className="mb-2.5 flex items-center gap-2.5">
        {isUser ? (
          <span className="bg-overlay text-ink-muted grid size-6 place-items-center rounded-full text-[0.65rem] font-semibold">
            {userInitial}
          </span>
        ) : (
          <span className="border-accent/25 bg-accent/10 text-accent grid size-6 place-items-center rounded-full border">
            <Logo className="size-3.5" />
          </span>
        )}
        <span className="text-ink text-[0.8rem] font-semibold">{isUser ? 'You' : 'Atlas'}</span>
        {message.model && (
          <span className="border-line text-ink-faint rounded-md border px-1.5 py-0.5 font-mono text-[0.62rem]">
            {message.model}
          </span>
        )}
      </header>

      <div className="pl-[2.1rem]">
        {isUser ? (
          <p className="text-ink/90 text-[0.94rem] leading-[1.75] whitespace-pre-wrap">
            {message.content}
          </p>
        ) : isStreaming && message.content === '' ? (
          <ThinkingIndicator />
        ) : (
          <div className={isStreaming ? 'stream-caret' : undefined}>
            <Markdown content={message.content} />
          </div>
        )}

        {!isUser && <Citations citations={citations} />}
      </div>
    </article>
  );
}

function ThinkingIndicator() {
  return (
    <div className="text-ink-faint flex items-center gap-1.5 py-1" aria-label="Generating reply">
      {[0, 1, 2].map((index) => (
        <span
          key={index}
          className="bg-ink-faint thinking-dot size-1.5 rounded-full"
          style={{ animationDelay: `${index * 0.16}s` }}
        />
      ))}
    </div>
  );
}
