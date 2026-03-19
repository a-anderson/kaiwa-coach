import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import path from 'path'

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    // wavesurfer.js 7.x ships record.js but its exports map erroneously redirects
    // the *.js pattern to *.esm.js which does not exist for the record plugin.
    // Alias the broken path directly to the real file.
    alias: {
      'wavesurfer.js/dist/plugins/record.js': path.resolve(
        __dirname,
        'node_modules/wavesurfer.js/dist/plugins/record.js',
      ),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
