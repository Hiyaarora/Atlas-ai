import { useMemo, useState } from 'react';
import { Wordmark } from '@/components/Logo';
import { ChatIcon, CloseIcon, PlusIcon, SearchIcon, SignOutIcon, TrashIcon } from '@/components/icons';
import { DocumentPanel } from '@/features/documents/DocumentPanel';
import type { ConversationSummary } from './types';

interface SidebarProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  /** Navigate into the conversation an upload created. */
  onUploaded: (conversationId: string) => void;
  /** Lets the empty state open the file picker that lives down here. */
  registerUploadTrigger: (trigger: () => void) => void;
  userLabel: string;
  onSignOut: () => void;
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Groups conversations by recency.
 *
 * A flat list of forty titles is a wall. Date buckets give the eye somewhere
 * to land, and match how people actually remember their own chats — "that
 * was yesterday" rather than "that was 31 items down".
 */
function groupByRecency(conversations: ConversationSummary[]) {
  const now = Date.now();
  const day = 86_400_000;

  const buckets: { label: string; items: ConversationSummary[] }[] = [
    { label: 'Today', items: [] },
    { label: 'Yesterday', items: [] },
    { label: 'Previous 7 days', items: [] },
    { label: 'Older', items: [] },
  ];

  for (const conversation of conversations) {
    const age = now - new Date(conversation.last_message_at).getTime();
    if (age < day) buckets[0]!.items.push(conversation);
    else if (age < 2 * day) buckets[1]!.items.push(conversation);
    else if (age < 7 * day) buckets[2]!.items.push(conversation);
    else buckets[3]!.items.push(conversation);
  }

  return buckets.filter((bucket) => bucket.items.length > 0);
}

export function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onUploaded,
  registerUploadTrigger,
  userLabel,
  onSignOut,
  isOpen,
  onClose,
}: SidebarProps) {
  const [query, setQuery] = useState('');

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? conversations.filter((c) => c.title.toLowerCase().includes(needle))
      : conversations;
    return groupByRecency(filtered);
  }, [conversations, query]);

  return (
    <>
      {/* Scrim: only rendered on small screens, where the sidebar overlays
          the transcript rather than sitting beside it. */}
      {isOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          onClick={onClose}
          className="bg-canvas/70 fixed inset-0 z-30 backdrop-blur-sm lg:hidden"
        />
      )}

      <aside
        className={`bg-surface/80 border-line fixed inset-y-0 left-0 z-40 flex w-[17.5rem] flex-col border-r backdrop-blur-xl transition-transform duration-300 lg:static lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between px-4 pt-4 pb-3">
          <Wordmark />
          <button
            type="button"
            onClick={onClose}
            className="text-ink-faint hover:text-ink lg:hidden"
            aria-label="Close navigation"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="px-3 pb-3">
          <div className="group border-line bg-raised/60 focus-within:border-accent/50 flex items-center gap-2 rounded-xl border px-3 py-2 transition">
            <SearchIcon className="text-ink-faint group-focus-within:text-accent size-4 transition" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search chats"
              className="placeholder:text-ink-faint text-ink w-full bg-transparent text-sm outline-none"
            />
          </div>
        </div>

        <div className="border-line border-b px-3 pb-3">
          <DocumentPanel onUploaded={onUploaded} registerTrigger={registerUploadTrigger} />
        </div>

        <nav className="min-h-0 flex-1 space-y-5 overflow-y-auto px-3 py-3">
          {groups.length === 0 && (
            <p className="text-ink-faint px-2 py-6 text-center text-xs">
              {query ? 'No chats match that search.' : 'No conversations yet.'}
            </p>
          )}

          {groups.map((group) => (
            <div key={group.label}>
              <h2 className="text-ink-faint px-2 pb-1.5 text-[0.68rem] font-medium tracking-widest uppercase">
                {group.label}
              </h2>
              <ul className="space-y-0.5">
                {group.items.map((conversation) => {
                  const isActive = conversation.id === activeId;
                  return (
                    <li key={conversation.id}>
                      <div
                        className={`group relative flex items-center rounded-lg transition ${
                          isActive ? 'bg-overlay' : 'hover:bg-raised/70'
                        }`}
                      >
                        {/* Active marker: a short accent bar rather than a
                            filled row, so the accent stays rare. */}
                        <span
                          className={`bg-accent absolute top-1/2 left-0 h-4 w-0.5 -translate-y-1/2 rounded-r transition-opacity ${
                            isActive ? 'opacity-100' : 'opacity-0'
                          }`}
                        />
                        <button
                          type="button"
                          onClick={() => onSelect(conversation.id)}
                          className="flex min-w-0 flex-1 items-center gap-2.5 px-3 py-2.5 text-left"
                        >
                          <ChatIcon
                            className={`size-3.5 shrink-0 ${isActive ? 'text-accent' : 'text-ink-faint'}`}
                          />
                          <span
                            className={`truncate text-[0.83rem] ${isActive ? 'text-ink' : 'text-ink-muted'}`}
                            title={conversation.title}
                          >
                            {conversation.title}
                          </span>
                        </button>
                        <button
                          type="button"
                          onClick={() => onDelete(conversation.id)}
                          aria-label={`Delete ${conversation.title}`}
                          className="text-ink-faint hover:text-danger mr-2 rounded-md p-1.5 opacity-0 transition group-hover:opacity-100 focus-visible:opacity-100"
                        >
                          <TrashIcon className="size-3.5" />
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>

        <div className="border-line space-y-3 border-t p-3">
          <button
            type="button"
            onClick={onNew}
            className="bg-accent text-canvas hover:bg-accent-strong flex w-full items-center justify-between gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition active:scale-[0.99]"
          >
            New chat
            <span className="bg-canvas/15 rounded-lg p-1">
              <PlusIcon className="size-3.5" />
            </span>
          </button>

          <div className="flex items-center gap-2.5 px-1">
            <div className="bg-overlay text-ink-muted grid size-7 shrink-0 place-items-center rounded-full text-xs font-semibold">
              {userLabel.charAt(0).toUpperCase()}
            </div>
            <span className="text-ink-muted min-w-0 flex-1 truncate text-xs">{userLabel}</span>
            <button
              type="button"
              onClick={onSignOut}
              aria-label="Sign out"
              className="text-ink-faint hover:text-ink rounded-md p-1.5 transition"
            >
              <SignOutIcon className="size-4" />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
