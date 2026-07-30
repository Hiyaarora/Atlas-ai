/**
 * Inline icon set.
 *
 * Hand-drawn rather than pulled from an icon package: the app needs about a
 * dozen glyphs, and a dependency would ship several thousand. They share one
 * stroke weight and cap style so they read as a family.
 */

type IconProps = { className?: string };

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

export const SearchIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </svg>
);

export const PlusIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const SendIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M5 12h13M12 5l7 7-7 7" />
  </svg>
);

export const StopIcon = ({ className = 'size-4' }: IconProps) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
    <rect x="7" y="7" width="10" height="10" rx="2" />
  </svg>
);

export const ChatIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M20 15a2 2 0 0 1-2 2H8l-4 3V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z" />
  </svg>
);

export const TrashIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M4 7h16M10 11v6M14 11v6" />
    <path d="M6 7l1 13h10l1-13M9 7V4h6v3" />
  </svg>
);

export const CopyIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M5 15V5a2 2 0 0 1 2-2h8" />
  </svg>
);

export const CheckIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="m5 13 4 4L19 7" />
  </svg>
);

export const SignOutIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M15 17l5-5-5-5M20 12H9M12 20H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h6" />
  </svg>
);

export const MenuIcon = ({ className = 'size-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </svg>
);

export const CloseIcon = ({ className = 'size-5' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
);

export const SparkIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    {/* A four-point star with concave sides, plus a small companion. The
        earlier version used straight cross strokes, which at 16px read as a
        plus sign rather than a sparkle. */}
    <path d="M11 3c0 4.4 1.6 6 6 6-4.4 0-6 1.6-6 6 0-4.4-1.6-6-6-6 4.4 0 6-1.6 6-6Z" />
    <path d="M18.5 14.5c0 1.9.6 2.5 2.5 2.5-1.9 0-2.5.6-2.5 2.5 0-1.9-.6-2.5-2.5-2.5 1.9 0 2.5-.6 2.5-2.5Z" />
  </svg>
);

export const DocumentIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5M9 13h6M9 17h4" />
  </svg>
);

export const CompassIcon = ({ className = 'size-4' }: IconProps) => (
  <svg {...base} className={className}>
    <circle cx="12" cy="12" r="9" />
    <path d="m15 9-2 4-4 2 2-4z" />
  </svg>
);
