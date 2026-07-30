/**
 * Markdown rendering for assistant replies.
 *
 * Why a dependency here, when this project otherwise avoids them:
 *
 * Model output is markdown — fenced code, tables, lists, links. Rendering it
 * as plain text throws away most of the answer's structure. The alternative
 * to `react-markdown` is hand-rolling a parser, and the failure mode of a
 * hand-rolled one is not "a heading looks wrong", it is XSS: the obvious
 * shortcut is `dangerouslySetInnerHTML`, and the model's output is attacker-
 * influenceable the moment a user pastes in a document.
 *
 * `react-markdown` builds React elements directly and never touches innerHTML,
 * so injection is impossible by construction. That safety property — not
 * convenience — is why it earns its place.
 */

import { memo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CheckIcon, CopyIcon } from './icons';

function CodeBlock({ code, language }: { code: string; language: string | null }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access can be denied; failing silently is better than an
      // alert for something the user can still select and copy by hand.
    }
  }

  return (
    <div className="group/code border-line bg-canvas/80 my-4 overflow-hidden rounded-xl border">
      <div className="border-line/70 flex items-center justify-between border-b px-3.5 py-2">
        <span className="text-ink-faint font-mono text-[0.7rem] tracking-wide uppercase">
          {language ?? 'code'}
        </span>
        <button
          type="button"
          onClick={() => void copy()}
          className="text-ink-faint hover:text-ink flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition"
          aria-label={copied ? 'Copied' : 'Copy code'}
        >
          {copied ? (
            <>
              <CheckIcon className="text-accent size-3.5" /> Copied
            </>
          ) : (
            <>
              <CopyIcon className="size-3.5" /> Copy
            </>
          )}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 text-[0.82rem] leading-relaxed">
        <code className="font-mono text-ink/90">{code}</code>
      </pre>
    </div>
  );
}

export const Markdown = memo(function Markdown({ content }: { content: string }) {
  return (
    <div className="text-[0.94rem] leading-[1.75] text-ink/90">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const text = String(children).replace(/\n$/, '');
            // react-markdown marks fenced blocks with a `language-*` class;
            // anything without one is inline code.
            const fenced = /language-(\w+)/.exec(className ?? '');

            if (!fenced && !text.includes('\n')) {
              return (
                <code
                  className="bg-raised text-accent rounded-md px-1.5 py-0.5 font-mono text-[0.85em]"
                  {...props}
                >
                  {children}
                </code>
              );
            }

            return <CodeBlock code={text} language={fenced?.[1] ?? null} />;
          },

          // `pre` is unwrapped: CodeBlock renders its own <pre>, and nesting
          // them would double the padding and the scroll container.
          pre({ children }) {
            return <>{children}</>;
          },

          p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
          h1: ({ children }) => (
            <h1 className="text-ink mt-6 mb-3 text-lg font-semibold first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-ink mt-6 mb-3 text-base font-semibold first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-ink mt-5 mb-2 text-sm font-semibold first:mt-0">{children}</h3>
          ),
          ul: ({ children }) => (
            <ul className="mb-4 list-disc space-y-1.5 pl-5 last:mb-0">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-4 list-decimal space-y-1.5 pl-5 last:mb-0">{children}</ol>
          ),
          strong: ({ children }) => <strong className="text-ink font-semibold">{children}</strong>,
          blockquote: ({ children }) => (
            <blockquote className="border-accent/40 text-ink-muted my-4 border-l-2 pl-4 italic">
              {children}
            </blockquote>
          ),
          a: ({ children, href }) => (
            <a
              href={href}
              // Untrusted destinations: noopener stops the target page
              // reaching back through window.opener.
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent underline underline-offset-2 hover:opacity-80"
            >
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="border-line my-4 overflow-x-auto rounded-lg border">
              <table className="w-full text-left text-sm">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border-line bg-raised text-ink border-b px-3 py-2 font-semibold">
              {children}
            </th>
          ),
          td: ({ children }) => <td className="border-line/60 border-b px-3 py-2">{children}</td>,
          hr: () => <hr className="border-line my-6" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
