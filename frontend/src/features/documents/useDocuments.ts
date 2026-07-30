/**
 * Document list with status polling.
 *
 * Ingestion runs in a background task, so a freshly uploaded document is
 * `pending` and becomes `ready` seconds later with no push channel to tell
 * us. Polling is the honest solution at this scale; the interesting part is
 * knowing when to stop.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from '@/lib/api/client';
import { documentsApi } from './api';
import type { KnowledgeDocument } from './types';

const POLL_INTERVAL_MS = 1500;

/** Terminal states: nothing further will change on its own. */
export function isSettled(document: KnowledgeDocument): boolean {
  return document.status === 'ready' || document.status === 'failed';
}

interface UseDocumentsOptions {
  /** Called with the conversation the upload created, so the app can navigate. */
  onUploaded?: (conversationId: string) => void;
}

export function useDocuments({ onUploaded }: UseDocumentsOptions = {}) {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await documentsApi.list());
    } catch (caught) {
      if (caught instanceof ApiError) setError(caught.message);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll only while something is in flight, and stop the moment everything
  // settles. A timer that runs forever burns battery and fills the network
  // tab with requests that can never change anything.
  useEffect(() => {
    const pending = documents.some((document) => !isSettled(document));

    if (!pending) {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    if (timerRef.current === null) {
      timerRef.current = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    }

    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [documents, refresh]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      setError(null);
      setIsUploading(true);

      try {
        let lastConversationId: string | null = null;

        // Sequential, not Promise.all: uploads are large and the backend
        // embeds each one. Firing ten at once buys nothing and risks
        // exhausting the provider's rate limit.
        for (const file of Array.from(files)) {
          const result = await documentsApi.upload(file);
          lastConversationId = result.conversation_id;
        }

        await refresh();

        // The newest upload becomes the active document, and the user lands
        // in its conversation without touching "New chat".
        if (lastConversationId) onUploaded?.(lastConversationId);
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : 'Upload failed.');
      } finally {
        setIsUploading(false);
      }
    },
    [refresh, onUploaded],
  );

  const remove = useCallback(
    async (id: string) => {
      // Optimistic: the row disappears immediately, then we reconcile.
      setDocuments((current) => current.filter((document) => document.id !== id));
      try {
        await documentsApi.remove(id);
      } finally {
        await refresh();
      }
    },
    [refresh],
  );

  const reindex = useCallback(
    async (id: string) => {
      await documentsApi.reindex(id);
      await refresh();
    },
    [refresh],
  );

  return { documents, isUploading, error, upload, remove, reindex, refresh };
}
