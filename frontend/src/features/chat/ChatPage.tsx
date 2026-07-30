import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Wordmark } from '@/components/Logo';
import { MenuIcon } from '@/components/icons';
import { useAuth } from '@/features/auth/AuthContext';
import { documentsApi } from '@/features/documents/api';
import { isSettled } from '@/features/documents/useDocuments';
import type { KnowledgeDocument } from '@/features/documents/types';
import { chatApi } from './api';
import { Composer } from './Composer';
import { EmptyState } from './EmptyState';
import { Sidebar } from './Sidebar';
import { Transcript } from './Transcript';
import { useChat } from './useChat';
import type { ConversationSummary } from './types';

const DOCUMENT_POLL_MS = 1500;

export function ChatPage() {
  const { user, logout } = useAuth();

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [isNavOpen, setIsNavOpen] = useState(false);

  //: The document backing the active conversation. Drives whether the
  //: composer is usable and what the empty state says.
  const [activeDocument, setActiveDocument] = useState<KnowledgeDocument | null>(null);

  const uploadTriggerRef = useRef<(() => void) | null>(null);

  const refreshConversations = useCallback(async () => {
    const list = await chatApi.listConversations();
    setConversations(list);
    return list;
  }, []);

  const {
    messages,
    streamingText,
    streamingCitations,
    citationsByMessage,
    isStreaming,
    isLoading,
    error,
    sendMessage,
    stop,
  } = useChat(activeId, refreshConversations);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeId) ?? null,
    [conversations, activeId],
  );

  // Load the sidebar and open the most recent conversation. Nothing is created
  // here: a conversation now comes into being with a document, at upload.
  useEffect(() => {
    void (async () => {
      const list = await refreshConversations().catch(() => []);
      if (list.length > 0 && list[0]) setActiveId(list[0].id);
    })();
  }, [refreshConversations]);

  // Track the active conversation's document, polling while it is still being
  // processed so the composer unlocks the moment it is ready.
  useEffect(() => {
    const documentId = activeConversation?.document_id ?? null;
    if (!documentId) {
      setActiveDocument(null);
      return;
    }

    let cancelled = false;
    let timer: number | null = null;

    const poll = async () => {
      try {
        const document = await documentsApi.get(documentId);
        if (cancelled) return;

        setActiveDocument(document);

        if (!isSettled(document)) {
          timer = window.setTimeout(() => void poll(), DOCUMENT_POLL_MS);
        }
      } catch {
        // The document may have been deleted; the conversation survives as
        // general chat and says so.
        if (!cancelled) setActiveDocument(null);
      }
    };

    void poll();

    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [activeConversation?.document_id]);

  const openConversation = useCallback(
    async (conversationId: string) => {
      await refreshConversations();
      setActiveId(conversationId);
      setIsNavOpen(false);
    },
    [refreshConversations],
  );

  const startGeneralConversation = useCallback(async () => {
    const conversation = await chatApi.createConversation();
    await openConversation(conversation.id);
  }, [openConversation]);

  async function handleSend(content?: string) {
    const text = (content ?? draft).trim();
    if (!text || !activeId) return;

    setDraft('');
    await sendMessage(text);
  }

  const userLabel = user?.full_name ?? user?.email ?? '';

  // Locked while the document is still being ingested — asking a question
  // before the index exists would return a confident "I found nothing".
  const isDocumentProcessing =
    activeDocument !== null && !isSettled(activeDocument);

  return (
    <div className="flex h-full">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={(id) => {
          setActiveId(id);
          setIsNavOpen(false);
        }}
        onNew={() => void startGeneralConversation()}
        onDelete={(id) => {
          void (async () => {
            await chatApi.deleteConversation(id);
            const list = await refreshConversations();
            if (activeId === id) setActiveId(list[0]?.id ?? null);
          })();
        }}
        onUploaded={(conversationId) => void openConversation(conversationId)}
        registerUploadTrigger={(trigger) => {
          uploadTriggerRef.current = trigger;
        }}
        userLabel={userLabel}
        onSignOut={() => void logout()}
        isOpen={isNavOpen}
        onClose={() => setIsNavOpen(false)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="border-line flex items-center gap-3 border-b px-4 py-3 lg:hidden">
          <button
            type="button"
            onClick={() => setIsNavOpen(true)}
            aria-label="Open navigation"
            className="text-ink-muted hover:text-ink"
          >
            <MenuIcon />
          </button>
          <Wordmark compact />
        </header>

        {/* Which document this conversation is about — visible at all times, so
            it is never ambiguous where an answer came from. */}
        {activeConversation?.document_filename && messages.length > 0 && (
          <div className="border-line text-ink-faint border-b px-6 py-2 text-xs">
            Answering from{' '}
            <span className="text-ink-muted font-medium">
              {activeConversation.document_filename}
            </span>
          </div>
        )}

        {messages.length === 0 && !isStreaming && !isLoading ? (
          <EmptyState
            onPick={(prompt) => void handleSend(prompt)}
            documentName={activeConversation?.document_filename ?? null}
            onUploadClick={() => {
              setIsNavOpen(true);
              uploadTriggerRef.current?.();
            }}
          />
        ) : (
          <Transcript
            messages={messages}
            streamingText={streamingText}
            streamingCitations={streamingCitations}
            citationsByMessage={citationsByMessage}
            isStreaming={isStreaming}
            isLoading={isLoading}
            userInitial={userLabel.charAt(0).toUpperCase()}
          />
        )}

        {error && (
          <div className="mx-auto w-full max-w-3xl px-4 sm:px-6">
            <p
              role="alert"
              className="border-danger/30 bg-danger-soft text-danger rounded-xl border px-3.5 py-2.5 text-sm"
            >
              {error}
            </p>
          </div>
        )}

        <Composer
          value={draft}
          onChange={setDraft}
          onSend={() => void handleSend()}
          onStop={stop}
          isStreaming={isStreaming}
          disabled={activeId === null || isDocumentProcessing}
          notice={
            isDocumentProcessing
              ? `Preparing ${activeDocument?.filename ?? 'your document'}…`
              : activeDocument?.status === 'failed'
                ? "This document couldn't be read."
                : null
          }
        />
      </main>
    </div>
  );
}
