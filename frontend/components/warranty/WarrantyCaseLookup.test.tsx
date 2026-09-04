import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import WarrantyCaseLookup from "./WarrantyCaseLookup";

describe("WarrantyCaseLookup", () => {
  it("asks for case reference and email", () => {
    const markup = renderToStaticMarkup(
      <WarrantyCaseLookup initialCaseReference="WR-20260904-ABCDEF" />
    );

    expect(markup).toContain("Case reference");
    expect(markup).toContain("Email on the case");
    expect(markup).toContain("WR-20260904-ABCDEF");
    expect(markup).toContain("Check case");
  });
});
