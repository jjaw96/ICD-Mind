import { defineConfig } from 'vite'
import react from '@vitejs/react-plugin'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './',
})