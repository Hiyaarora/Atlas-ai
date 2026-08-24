/** Mirrors `app/schemas/chat.py`. */

import type { Citation } from '@/features/documents/types';

export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  model: string | null;
  created_at: string;
  /** Sources this answer was grounded in, as recorded when it was written.
   *  Empty for user turns and for answers written before sources were kept. */
  citations: Citation[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  /** The document this conversation retrieves from — set once, at upload. */
  document_id: string | null;
  document_filename: string | null;
  created_at: string;
  updated_at: string;
  last_message_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: Message[];
}

/** Server-sent events emitted by POST /conversations/{id}/messages. */
export type StreamEvent =
  | { type: 'sources'; citations: Citation[] }
  | { type: 'token'; text: string }
  | { type: 'done'; messageId: string; content: string; model: string | null }
  | { type: 'error'; code: string; message: string; requestId: string };
