import { existsSync } from "node:fs";
import { defineConfig, devices } from "@playwright/test";

const port = 3100;
const systemChromium = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ||
  (!process.env.CI && existsSync("/usr/bin/chromium") ? "/usr/bin/chromium" : undefined);

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  workers: 2,
  retries: process.env.CI ? 1 : 0,
  timeout: 45_000,
  expect: { timeout: 12_000 },
  outputDir: process.env.PLAYWRIGHT_OUTPUT_DIR || "/tmp/ai-chat-playwright-results",
  reporter: [["list"], ["html", { outputFolder: "/tmp/ai-chat-playwright-report", open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: systemChromium ? { executablePath: systemChromium } : {},
  },
  projects: [
    { name: "chromium-desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "chromium-mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: process.env.E2E_USE_PRODUCTION === "1"
      ? `npm run start -- --hostname 127.0.0.1 --port ${port}`
      : `npm run dev -- --hostname 127.0.0.1 --port ${port}`,
    url: `http://127.0.0.1:${port}/api/health`,
    timeout: 120_000,
    reuseExistingServer: false,
    env: { NEXT_PUBLIC_API_BASE_URL: "" },
  },
});
