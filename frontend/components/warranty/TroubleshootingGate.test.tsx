import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import TroubleshootingGate from "./TroubleshootingGate";

const handlers = {
  onStepsCompleted: vi.fn(),
  onResolved: vi.fn(),
  onUnresolved: vi.fn(),
  onUnableToAttempt: vi.fn(),
};

describe("TroubleshootingGate", () => {
  it("hides escalation until the customer confirms the steps were tried", () => {
    const markup = renderToStaticMarkup(
      <TroubleshootingGate
        mode="troubleshooting"
        stage="review"
        stepCount={4}
        {...handlers}
      />
    );

    expect(markup).toContain("Try the troubleshooting steps above first");
    expect(markup).toContain("I’ve tried all the steps");
    expect(markup).not.toContain("the issue is still there");
    expect(markup).not.toContain("continue to team review");
  });

  it("makes self-resolution primary after troubleshooting", () => {
    const markup = renderToStaticMarkup(
      <TroubleshootingGate
        mode="troubleshooting"
        stage="outcome"
        stepCount={4}
        {...handlers}
      />
    );

    const resolved = markup.indexOf("Yes — it’s working now");
    const unresolved = markup.indexOf("No — the issue is still there");
    expect(resolved).toBeGreaterThan(-1);
    expect(unresolved).toBeGreaterThan(resolved);
  });
});
