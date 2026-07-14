import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import EvidenceUploader from "./EvidenceUploader";

describe("EvidenceUploader", () => {
  it("shows a back control on the final email step", () => {
    const markup = renderToStaticMarkup(
      <EvidenceUploader ticketId="ticket-1" onBack={vi.fn()} />
    );

    expect(markup).toContain("Final step — how can we reach you?");
    expect(markup).toContain("← Back");
    expect(markup).toContain("Go back to the previous troubleshooting step");
  });

  it("keeps the back control available when the email form is collapsed", () => {
    const markup = renderToStaticMarkup(
      <EvidenceUploader
        ticketId="ticket-1"
        collapsed
        onToggleCollapsed={vi.fn()}
        onBack={vi.fn()}
      />
    );

    expect(markup).toContain("Contact form is hidden");
    expect(markup).toContain("← Back");
  });
});
