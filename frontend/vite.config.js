import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api calls go to the Python backend (python app/api.py on port 8000).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
