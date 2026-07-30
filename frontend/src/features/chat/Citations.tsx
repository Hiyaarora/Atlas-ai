import { useMemo, useState } from 'react';
import { DocumentIcon } from '@/components/icons';
import type { Citation } from '@/features/documents/types';

/**
 * The sources an answer was built from.
 *
 * Grouped by document, not listed per chunk. Retrieval returns chunks, and a
 * question answered from six passages of one PDF produced six identical
 * filenames stacked on top of each other — visually noisy and, worse,
 * misleading: it looks like six sources when it is one.
 *
 * The grouping is what a reader actually wants: which documents, which pages
 * within them.
 */

interface DocumentGroup {
  documentId: string;
  filename: string;
  citations: Citation[];
  bestScore: number;
}

function groupByDocument(citations: Citation[]): DocumentGroup[] {
  const groups = new Map<string, DocumentGroup>();

  for (const citation of citations) {
    const existing = groups.get(citation.document_id);
    if (existing) {
      existing.citations.push(citation);
      existing.bestScore = Math.max(existing.bestScore, citation.score);
    } else {
      groups.set(citation.document_id, {
        documentId: citation.document_id,
        filename: citation.filename,
        citations: [citation],
        bestScore: citation.score,
      });
    }
  }

  return [...groups.values()].sort((a, b) => b.bestScore - a.bestScore);
}

/** "p.2, 3, 10" — deduplicated and ordered, however retrieval ranked them. */
function formatPages(citations: Citation[]): string {
  const pages = [...new Set(citations.map((c) => c.page_number))].sort((a, b) => a - b);
  return pages.length === 1 ? `p.${pages[0]}` : `pp.${pages.join(', ')}`;
}

export function Citations({ citations }: { citations: Citation[] }) {
  const [openId, setOpenId] = useState<string | null>(null);
  const groups = useMemo(() => groupByDocument(citations), [citations]);

  if (groups.length === 0) return null;

  return (
    <div className="border-line/70 mt-4 border-t pt-3">
      <p className="text-ink-faint mb-2 text-[0.68rem] font-medium tracking-widest uppercase">
        {groups.length === 1 ? '1 source' : `${groups.length} sources`}
      </p>

      <ul className="space-y-1">
        {groups.map((group) => {
          const isOpen = openId === group.documentId;

          return (
            <li key={group.documentId}>
              <button
                type="button"
                onClick={() => setOpenId(isOpen ? null : group.documentId)}
                aria-expanded={isOpen}
                className="hover:bg-raised/60 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition"
              >
                <DocumentIcon className="text-ink-faint size-3.5 shrink-0" />
                <span className="text-ink-muted min-w-0 flex-1 truncate text-xs">
                  {group.filename}
                </span>
                <span className="text-ink-faint shrink-0 text-[0.66rem]">
                  {formatPages(group.citations)}
                </span>
                <span className="bg-raised text-ink-faint shrink-0 rounded px-1.5 py-0.5 font-mono text-[0.62rem]">
                  {group.citations.length}
                </span>
              </button>

              {isOpen && (
                <ul className="mt-1 ml-3 space-y-2">
                  {group.citations.map((citation) => (
                    <li key={citation.chunk_id} className="border-accent/30 border-l-2 pl-3">
                      <p className="text-ink-faint mb-0.5 flex items-center gap-2 text-[0.62rem]">
                        {/* The number the model was told to cite, so [3] in
                            the answer resolves to a passage here. */}
                        <span className="bg-accent/10 text-accent rounded px-1 font-mono">
                          {citation.index}
                        </span>
                        <span>page {citation.page_number}</span>
                        <span>·</span>
                        <span className="font-mono">{(citation.score * 100).toFixed(0)}%</span>
                      </p>
                      <p className="text-ink-muted text-xs leading-relaxed">{citation.excerpt}</p>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
