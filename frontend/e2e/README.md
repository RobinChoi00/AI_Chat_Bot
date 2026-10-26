# Customer browser checks

Run `npm run test:e2e` from `frontend`. The suite starts Next.js, drives real
Chromium on desktop and mobile viewports, and exercises React events, forms,
storage, reloads, errors and accessibility roles. Only the backend HTTP boundary
is replaced with synthetic responses; no customer, model or email service is
called. Backend workflow and authorization integration tests run separately in
pytest.

For a machine without Chromium, run `npx playwright install --with-deps chromium`
first. An existing `/usr/bin/chromium` is used locally; set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to select a different installed Chromium.
CI installs Playwright's pinned browser and tests the production build with
`E2E_USE_PRODUCTION=1`. Failure traces and screenshots use synthetic data and are
written under `/tmp/ai-chat-playwright-results`.
