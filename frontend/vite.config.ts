import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    // During `npm run dev`, forward API calls to the backend (in Docker: nginx does this).
    proxy: { '/api': 'http://localhost:8000' },
  },
})
