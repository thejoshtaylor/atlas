import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  build: {
    // src/atlas/app.py's `app.frontend("/", directory="web/dist")`
    // (plan 03-05) reads from this exact path -- a rename here breaks
    // serving there.
    outDir: 'dist',
  },
})
