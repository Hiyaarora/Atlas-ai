import { Logo } from '@/components/Logo';
import { DocumentIcon, PlusIcon, SparkIcon } from '@/components/icons';

/**
 * Suggestions that work for any document.
 *
 * The previous set — "Explain a concept", "Compare approaches" — assumed a
 * technical paper. Offered against a resume, an invoice or a medical report
 * they are nonsense, and a suggestion the product cannot honour is worse than
 * no suggestion at all.
 *
 * These three are verbs that apply to every document type: summarise it,
 * pull the facts out of it, ask it something.
 */
const SUGGESTIONS = [
  {
    icon: DocumentIcon,
    title: 'Summarise this document',
    prompt: 'Summarise this document. What is it, and what are the key points?',
  },
  {
    icon: SparkIcon,
    title: 'Extract key information',
    prompt:
      'Extract the key information from this document — names, dates, figures, and any decisions or conclusions.',
  },
  {
    icon: PlusIcon,
    title: 'Ask anything',
    prompt: 'What are the most important things I should know from this document?',
  },
];

interface EmptyStateProps {
  onPick: (prompt: string) => void;
  /** The document this conversation is about, if any. */
  documentName: string | null;
  onUploadClick: () => void;
}

export function EmptyState({ onPick, documentName, onUploadClick }: EmptyStateProps) {
  return (
    <div className="animate-rise flex flex-1 flex-col items-center justify-center px-6 py-12">
      <div className="border-accent/20 bg-accent/5 text-accent mb-6 grid size-14 place-items-center rounded-2xl border">
        <Logo className="size-7" />
      </div>

      <h1 className="text-center text-[1.75rem] font-semibold tracking-tight text-balance sm:text-4xl">
        {documentName ? 'What would you like to know?' : 'Upload a document to begin'}
      </h1>

      {documentName ? (
        <p className="text-ink-muted mt-3 max-w-md text-center text-sm text-pretty">
          Answers in this conversation come from{' '}
          <span className="text-ink font-medium">{documentName}</span>, with citations back to the
          page.
        </p>
      ) : (
        <p className="text-ink-muted mt-3 max-w-md text-center text-sm text-pretty">
          Add a PDF, Word file, spreadsheet, presentation or web page. Each document gets its own
          conversation, so answers never mix between them.
        </p>
      )}

      {documentName ? (
        <div className="mt-10 grid w-full max-w-2xl gap-3 sm:grid-cols-3">
          {SUGGESTIONS.map(({ icon: Icon, title, prompt }) => (
            <button
              key={title}
              type="button"
              onClick={() => onPick(prompt)}
              className="border-line bg-raised/50 hover:border-accent/30 hover:bg-raised rounded-xl border p-4 text-left transition"
            >
              <Icon className="text-accent mb-3 size-4" />
              <p className="text-ink text-sm font-medium">{title}</p>
            </button>
          ))}
        </div>
      ) : (
        <button
          type="button"
          onClick={onUploadClick}
          className="bg-accent text-canvas hover:bg-accent-strong mt-8 rounded-xl px-5 py-2.5 text-sm font-semibold transition active:scale-[0.99]"
        >
          Choose a document
        </button>
      )}
    </div>
  );
}
