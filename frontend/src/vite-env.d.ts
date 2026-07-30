/// <reference types="vite/client" />

/** Type the app's own env vars so `import.meta.env.VITE_*` is checked. */
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
