import { renderToStaticMarkup } from "react-dom/server";
import { isValidElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import TroubleshootingGate from "./TroubleshootingGate";

const handlers = {
  onStepsCompleted: vi.fn(),
  onResolved: vi.fn(),
  onUnresolved: vi.fn(),
  onUnableToAttempt: vi.fn(),
  onDeferred: vi.fn(),
};

function buttons(node: ReactNode): { children?: ReactNode; onClick?: () => void }[] {
  if (Array.isArray(node)) return node.flatMap(buttons);
  if (!isValidElement<{ children?: ReactNode; onClick?: () => void }>(node)) return [];
  return [...(node.type === "button" ? [node.props] : []), ...buttons(node.props.children)];
}

beforeEach(() => vi.clearAllMocks());

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

  it("uses install-specific copy", () => {
    const markup = renderToStaticMarkup(
      <TroubleshootingGate
        mode="troubleshooting"
        issueType="installation"
        stage="outcome"
        stepCount={2}
        {...handlers}
      />
    );
    expect(markup).toContain("Is the chair set up correctly now?");
    expect(markup).toContain("Yes — the chair is set up now");
    expect(markup).toContain("No — I still need install help");
  });

  it("uses delivery-specific copy", () => {
    const markup = renderToStaticMarkup(
      <TroubleshootingGate
        mode="preparation"
        issueType="delivery"
        stage="outcome"
        stepCount={3}
        {...handlers}
      />
    );
    expect(markup).toContain("Send this delivery case for review?");
    expect(markup).toContain("Continue to delivery review");
    expect(markup).not.toContain("Yes — it’s working now");
  });

  it("defers a delivery case without claiming resolution or completed evidence", () => {
    const tree = TroubleshootingGate({ mode: "preparation", issueType: "delivery", stage: "review", stepCount: 3, ...handlers });
    const actions = buttons(tree);
    actions.find((button) => button.children === "Not yet — save for later")?.onClick?.();
    expect(handlers.onDeferred).toHaveBeenCalledOnce();
    expect(handlers.onResolved).not.toHaveBeenCalled();
    expect(handlers.onStepsCompleted).not.toHaveBeenCalled();
    actions.find((button) => button.children === "Continue to delivery review")?.onClick?.();
    expect(handlers.onUnresolved).toHaveBeenCalledOnce();
  });

  it("makes preparation review the primary action and requires an explicit fixed answer to resolve", () => {
    const actions = buttons(TroubleshootingGate({ mode: "preparation", stage: "outcome", stepCount: 2, ...handlers }));
    actions[0].onClick?.();
    expect(handlers.onUnresolved).toHaveBeenCalledOnce();
    expect(handlers.onResolved).not.toHaveBeenCalled();
    expect(actions[1].children).toBe("Yes — it’s working now");
    actions[1].onClick?.();
    expect(handlers.onResolved).toHaveBeenCalledOnce();
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
