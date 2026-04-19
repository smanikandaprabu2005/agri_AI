import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI backend — avoids CORS in dev
      "/chat":    { target: "http://localhost:8000", changeOrigin: true },
      "/weather": { target: "http://localhost:8000", changeOrigin: true },
      "/profile": { target: "http://localhost:8000", changeOrigin: true },
      "/health":  { target: "http://localhost:8000", changeOrigin: true },
      "/reset":   { target: "http://localhost:8000", changeOrigin: true },
      "/stats":   { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
        },
      },
    },
  },
});
