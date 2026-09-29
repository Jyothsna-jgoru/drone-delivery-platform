import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "../../tests/e2e",
  use: { baseURL: "http://localhost:5173" },
  webServer: [
    { command: "python -m uvicorn apps.api.main:app", cwd: "../..", port: 8000, reuseExistingServer: true },
    { command: "npm run dev", cwd: ".", port: 5173, reuseExistingServer: true },
  ],
});
