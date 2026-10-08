import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// `npm run build` writes into the Python package so `claimverifier serve` can serve the UI.
// `npm run dev` proxies API calls to a backend started with `python -m claimverifier serve`.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../claimverifier/web/dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.CLAIMVERIFIER_API ?? "http://localhost:8000", changeOrigin: true } },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
