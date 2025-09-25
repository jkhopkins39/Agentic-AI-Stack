import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    // Force include these specific packages
    include: [
      'class-variance-authority',
      '@radix-ui/react-slot',
      '@radix-ui/react-dialog',
      '@radix-ui/react-separator', 
      '@radix-ui/react-tooltip',
      'lucide-react',
      'clsx',
      'tailwind-merge'
    ],
    // Force exclude problematic packages from pre-bundling
    exclude: []
  },
  server: {
    fs: {
      // Allow serving files from one level up to the project root
      allow: ['..']
    }
  }
})