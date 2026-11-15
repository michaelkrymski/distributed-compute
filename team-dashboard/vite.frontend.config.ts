import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  root: "frontend",
  publicDir: "../public",
  plugins: [react()],
  build: { outDir: "../frontend-dist", emptyOutDir: true },
  server: {
    host: "127.0.0.1",
    port: 5174,
    proxy: { "/api": { target: "http://127.0.0.1:5173", changeOrigin: true } },
  },
});
