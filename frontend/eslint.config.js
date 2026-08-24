// Flat config (ESLint 9). Deliberately built from the packages already in
// devDependencies — adding plugins here would mean a clean `npm ci` could not
// lint until someone noticed the missing install, which is the exact failure
// this file exists to fix.
import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    // Build output and dependencies are not ours to lint.
    ignores: ['dist/**', 'node_modules/**', 'eslint.config.js'],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    rules: {
      // TypeScript already proves every identifier resolves, and `no-undef`
      // cannot see ambient DOM types — leaving it on reports `document` and
      // `fetch` as undefined in a codebase where they plainly are not.
      'no-undef': 'off',

      // An unused parameter named with a leading underscore is a deliberate
      // signal that it is part of a signature and intentionally ignored.
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],

      // `any` disables the type checking this project relies on. A warning
      // rather than an error so it flags drift without blocking a build.
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
);
