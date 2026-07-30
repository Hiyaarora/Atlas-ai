import { useEffect, useRef, useState, type DragEvent } from 'react';
import { DocumentIcon, PlusIcon, TrashIcon } from '@/components/icons';
import { useDocuments } from './useDocuments';
import type { DocumentStatus, KnowledgeDocument } from './types';

/**
 * User-facing wording only.
 *
 * "Indexing" rather than "embedding 43 chunks": the user needs to know
 * whether they can ask a question yet, and nothing else. Implementation
 * detail in a status label is noise at best and an invitation to worry at
 * worst.
 */
const STATUS_LABEL: Record<DocumentStatus, string> = {
  pending: 'Processing',
  processing: 'Indexing',
  ready: 'Ready',
  failed: "Couldn't be read",
};

const STATUS_STYLE: Record<DocumentStatus, string> = {
  pending: 'text-amber-400',
  processing: 'text-amber-400',
  ready: 'text-accent',
  failed: 'text-danger',
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface DocumentPanelProps {
  /** Navigate into the conversation an upload created. */
  onUploaded: (conversationId: string) => void;
  /** Expose the file picker so the empty state can open it too. */
  registerTrigger?: (trigger: () => void) => void;
}

export function DocumentPanel({ onUploaded, registerTrigger }: DocumentPanelProps) {
  const { documents, isUploading, error, upload, remove, reindex } = useDocuments({ onUploaded });
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // The picker input lives here, but "Choose a document" on the empty state
  // should open it. Publishing the trigger keeps one file input in the tree
  // rather than duplicating the upload wiring in two places.
  useEffect(() => {
    registerTrigger?.(() => inputRef.current?.click());
  }, [registerTrigger]);

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    setIsDragging(false);
    if (event.dataTransfer.files.length > 0) {
      void upload(event.dataTransfer.files);
    }
  }

  return (
    <section className="flex min-h-0 flex-col">
      <header className="flex items-center justify-between px-2 pb-1.5">
        <h2 className="text-ink-faint text-[0.68rem] font-medium tracking-widest uppercase">
          Documents
        </h2>
      </header>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`mb-2 rounded-xl border border-dashed px-3 py-3 text-center transition ${
          isDragging ? 'border-accent bg-accent/5' : 'border-line hover:border-line-strong'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.pptx,.xlsx,.xlsm,.csv,.tsv,.html,.htm,.txt,.md,.markdown"
          className="hidden"
          onChange={(event) => {
            if (event.target.files) void upload(event.target.files);
            // Reset so selecting the same file twice still fires onChange.
            event.target.value = '';
          }}
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={isUploading}
          className="text-ink-muted hover:text-ink flex w-full items-center justify-center gap-1.5 text-xs transition disabled:opacity-50"
        >
          <PlusIcon className="size-3.5" />
          {isUploading ? 'Uploading…' : 'Upload a document'}
        </button>
      </div>

      {error && (
        <p role="alert" className="text-danger mb-2 px-2 text-[0.7rem]">
          {error}
        </p>
      )}

      <ul className="min-h-0 space-y-0.5 overflow-y-auto">
        {documents.map((document) => (
          <DocumentRow
            key={document.id}
            document={document}
            onRemove={() => void remove(document.id)}
            onRetry={() => void reindex(document.id)}
          />
        ))}
      </ul>
    </section>
  );
}

function DocumentRow({
  document,
  onRemove,
  onRetry,
}: {
  document: KnowledgeDocument;
  onRemove: () => void;
  onRetry: () => void;
}) {
  const isBusy = document.status === 'pending' || document.status === 'processing';

  return (
    <li className="group hover:bg-raised/60 rounded-lg px-2 py-2 transition">
      <div className="flex items-start gap-2">
        <DocumentIcon
          className={`mt-0.5 size-3.5 shrink-0 ${isBusy ? 'text-amber-400' : 'text-ink-faint'}`}
        />

        <div className="min-w-0 flex-1">
          <p className="text-ink-muted truncate text-[0.78rem]" title={document.filename}>
            {document.filename}
          </p>
          <p className="text-ink-faint mt-0.5 flex items-center gap-1.5 text-[0.66rem]">
            <span className={STATUS_STYLE[document.status]}>
              {STATUS_LABEL[document.status]}
              {isBusy && <span className="ml-1 animate-pulse">…</span>}
            </span>
            <span>·</span>
            <span>{formatSize(document.size_bytes)}</span>
            {/* Pages are meaningful to a reader — they can turn to one.
                Chunks are not, so they are not shown. */}
            {document.status === 'ready' && document.page_count > 1 && (
              <>
                <span>·</span>
                <span>{document.page_count} pages</span>
              </>
            )}
          </p>

          {document.status === 'failed' && document.error && (
            <p className="text-danger/80 mt-1 text-[0.66rem] leading-snug">
              {document.error}{' '}
              <button type="button" onClick={onRetry} className="text-accent underline">
                Try again
              </button>
            </p>
          )}
        </div>

        <button
          type="button"
          onClick={onRemove}
          aria-label={`Remove ${document.filename}`}
          className="text-ink-faint hover:text-danger rounded-md p-1 opacity-0 transition group-hover:opacity-100 focus-visible:opacity-100"
        >
          <TrashIcon className="size-3.5" />
        </button>
      </div>
    </li>
  );
}
