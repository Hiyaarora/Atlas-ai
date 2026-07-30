/**
 * Shared presentation for login and register.
 *
 * The two pages differ only in fields, copy, and which API call they make —
 * so layout, error handling, and submit state live here once.
 */

import { useState, type FormEvent, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Logo } from '@/components/Logo';
import { ApiError } from '@/lib/api/client';

interface AuthFormProps {
  title: string;
  subtitle: string;
  submitLabel: string;
  onSubmit: () => Promise<void>;
  children: ReactNode;
  footer: ReactNode;
}

export function AuthForm({
  title,
  subtitle,
  submitLabel,
  onSubmit,
  children,
  footer,
}: AuthFormProps) {
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      await onSubmit();
    } catch (caught) {
      setError(
        caught instanceof ApiError ? humanizeError(caught) : 'Something went wrong. Try again.',
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="flex min-h-full items-center justify-center px-6 py-16">
      <div className="animate-rise w-full max-w-[22rem]">
        <header className="mb-9 text-center">
          <div className="border-accent/20 bg-accent/5 text-accent mx-auto mb-6 grid size-12 place-items-center rounded-2xl border">
            <Logo className="size-6" />
          </div>
          <h1 className="text-ink text-xl font-semibold tracking-tight">{title}</h1>
          <p className="text-ink-muted mt-1.5 text-sm">{subtitle}</p>
        </header>

        <form onSubmit={handleSubmit} noValidate className="space-y-4">
          {children}

          {error && (
            <p
              role="alert"
              className="border-danger/30 bg-danger-soft text-danger rounded-xl border px-3.5 py-2.5 text-sm"
            >
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="bg-accent text-canvas hover:bg-accent-strong w-full rounded-xl px-4 py-2.5 text-sm font-semibold transition active:scale-[0.99] disabled:opacity-50 disabled:active:scale-100"
          >
            {isSubmitting ? 'Please wait…' : submitLabel}
          </button>
        </form>

        <p className="text-ink-muted mt-7 text-center text-sm">{footer}</p>
      </div>
    </main>
  );
}

/** Turn a machine-readable error code into something a person can act on. */
function humanizeError(error: ApiError): string {
  switch (error.code) {
    case 'account_not_found':
      return 'No account found for that email. Would you like to sign up?';
    case 'account_disabled':
      return 'This account has been deactivated. Contact support for help.';
    case 'authentication_error':
      return 'Incorrect email or password.';
    case 'conflict':
      return 'An account with this email already exists.';
    case 'request_validation_error':
      return 'Please check the details you entered.';
    case 'network_error':
      return 'Could not reach the server. Is the API running?';
    default:
      return error.message;
  }
}

interface FieldProps {
  label: string;
  type: string;
  value: string;
  onChange: (value: string) => void;
  autoComplete: string;
  required?: boolean;
  minLength?: number;
  hint?: string;
}

export function Field({ label, value, onChange, hint, ...rest }: FieldProps) {
  const id = `field-${label.toLowerCase().replace(/\s+/g, '-')}`;

  return (
    <div>
      <label htmlFor={id} className="text-ink-muted mb-1.5 block text-xs font-medium">
        {label}
      </label>
      <input
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="border-line bg-raised/60 text-ink placeholder:text-ink-faint focus:border-accent/50 focus:bg-raised w-full rounded-xl border px-3.5 py-2.5 text-sm transition outline-none"
        {...rest}
      />
      {hint && <p className="text-ink-faint mt-1.5 text-xs">{hint}</p>}
    </div>
  );
}

export { Link };
