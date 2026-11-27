// Fonts are bundled with the app: no request to an external font server.
import '@fontsource-variable/instrument-sans'
import '@fontsource-variable/jetbrains-mono'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router'

import App from './App.tsx'
import './App.css'
import ToastProvider from './components/ToastProvider'

const root = document.getElementById('root')
if (!root) throw new Error('Missing #root element in index.html')

createRoot(root).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <App />
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>,
)
