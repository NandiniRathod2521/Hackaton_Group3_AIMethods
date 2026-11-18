import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { cloudflare } from "@cloudflare/vite-plugin";

export default defineConfig({
  plugins: [react(), cloudflare()],
  resolve: {
    alias: {
      // Only MoveNet is used; see src/pose/mediapipe-pose-stub.ts.
      "@mediapipe/pose": fileURLToPath(new URL("./src/pose/mediapipe-pose-stub.ts", import.meta.url)),
    },
  },
});
