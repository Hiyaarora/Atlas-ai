import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],

  // Load .env from the repo root rather than frontend/, so the whole project
  // has a single configuration file. Only VITE_* keys are exposed to the
  // bundle, so backend secrets in the same file are never shipped.
  envDir: fileURLToPath(new URL('..', import.meta.url)),

  resolve: {
    // `@/features/...` instead of `../../../features/...`
    // import.meta.url, not __dirname: this file is loaded as an ES module.
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },

  server: {
    port: 5173,
    strictPort: true,
  },
});
