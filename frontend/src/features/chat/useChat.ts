/**
 * Chat state for one conversation.
 *
 * The streaming assistant turn is kept *separate* from the persisted message
 * list. Appending tokens into the list would mean re-rendering and re-keying
 * the whole transcript on every token; instead only the streaming bubble
 * re-renders, and the finished message is committed to the list once.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Citation } from '@/features/documents/types';
import { ApiError } from '@/lib/api/client';
import { chatApi } from './api';
import type { Message } from './types';

interface ChatState {
  messages: Message[];
  streamingText: string;
  /** Sources for the reply currently being written. */
  streamingCitations: Citation[];
  /** Sources for completed replies, keyed by message id. */
  citationsByMessage: Record<string, Citation[]>;
  isStreaming: boolean;
  isLoading: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<void>;
  stop: () => void;
}

export function useChat(conversationId: string | null, onTitleChange?: () => void): ChatState {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [streamingCitations, setStreamingCitations] = useState<Citation[]>([]);
  const [citationsByMessage, setCitationsByMessage] = useState<Record<string, Citation[]>>({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  // --- Load history when the conversation changes ------------------------
  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setError(null);

    void (async () => {
      try {
        const conversation = await chatApi.getConversation(conversationId);
        if (!cancelled) {
          setMessages(conversation.messages);
          // Sources travel with the message now, so reopening a conversation
          // shows the same citations the answer was written with rather than
          // bare [1] markers pointing at nothing.
          setCitationsByMessage(
            Object.fromEntries(
              conversation.messages
                .filter((message) => message.citations.length > 0)
                .map((message) => [message.id, message.citations]),
            ),
          );
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof ApiError ? caught.message : 'Could not load conversation.');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  // Abort any in-flight stream when the component unmounts, so a closed tab
  // does not leave the request hanging.
  useEffect(() => () => abortRef.current?.abort(), []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!conversationId || isStreaming) return;

      const trimmed = content.trim();
      if (!trimmed) return;

      setError(null);

      // Optimistic user bubble. The server has the real row; this id is
      // local-only and is replaced when history is next fetched.
      const optimistic: Message = {
        id: `local-${Date.now()}`,
        role: 'user',
        content: trimmed,
        model: null,
        created_at: new Date().toISOString(),
        citations: [],
      };
      setMessages((current) => [...current, optimistic]);

      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);
      setStreamingText('');
      setStreamingCitations([]);

      let accumulated = '';
      let sources: Citation[] = [];

      try {
        for await (const event of chatApi.streamMessage(
          conversationId,
          trimmed,
          controller.signal,
        )) {
          if (event.type === 'sources') {
            sources = event.citations;
            setStreamingCitations(sources);
          } else if (event.type === 'token') {
            accumulated += event.text;
            setStreamingText(accumulated);
          } else if (event.type === 'done') {
            setMessages((current) => [
              ...current,
              {
                id: event.messageId,
                role: 'assistant',
                content: event.content,
                model: event.model,
                created_at: new Date().toISOString(),
                citations: sources,
              },
            ]);
            // Move the sources from "currently streaming" to "attached to
            // this message", so they survive the next turn.
            setCitationsByMessage((current) => ({ ...current, [event.messageId]: sources }));
            onTitleChange?.();
          } else {
            // An error event arrives *inside* a 200 response — the status code
            // was committed before generation began.
            setError(`${event.message} (request ${event.requestId})`);
          }
        }
      } catch (caught) {
        if (caught instanceof DOMException && caught.name === 'AbortError') {
          // User pressed Stop, or the component unmounted. Not an error.
        } else if (caught instanceof ApiError) {
          setError(caught.message);
        } else {
          setError('The reply could not be completed.');
        }
      } finally {
        setIsStreaming(false);
        setStreamingText('');
        setStreamingCitations([]);
        abortRef.current = null;
      }
    },
    [conversationId, isStreaming, onTitleChange],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  return {
    messages,
    streamingText,
    streamingCitations,
    citationsByMessage,
    isStreaming,
    isLoading,
    error,
    sendMessage,
    stop,
  };
}
