import { describe, expect, it } from "vitest";
import {
  DEFAULT_POLICY_URLS,
  resolvePolicyStoreDomain,
  resolveStorePolicyUrls,
} from "./welcomeMessage";

describe("resolvePolicyStoreDomain", () => {
  it("uses the parent Shopify store when provided", () => {
    expect(resolvePolicyStoreDomain("osakiusa.com")).toBe("osakiusa.com");
    expect(resolvePolicyStoreDomain("titanchair.com")).toBe("titanchair.com");
  });

  it("falls back for chat infrastructure hosts", () => {
    expect(resolvePolicyStoreDomain("help.osakichair.com")).toBe("osakiusa.com");
    expect(resolvePolicyStoreDomain("api.osakichair.com")).toBe("osakiusa.com");
  });
});

describe("resolveStorePolicyUrls", () => {
  it("uses /pages/ URLs for osakiusa and titanchair", () => {
    expect(resolveStorePolicyUrls("osakiusa.com")).toEqual({
      storeDomain: "osakiusa.com",
      privacy: "https://osakiusa.com/pages/privacy-policy",
      terms: "https://osakiusa.com/pages/terms-of-service",
    });
    expect(resolveStorePolicyUrls("titanchair.com")).toEqual({
      storeDomain: "titanchair.com",
      privacy: "https://titanchair.com/pages/privacy-policy",
      terms: "https://titanchair.com/pages/terms-of-service",
    });
  });

  it("uses osakimassagechair.com policy pages", () => {
    expect(resolveStorePolicyUrls("osakimassagechair.com")).toEqual({
      storeDomain: "osakimassagechair.com",
      privacy: "https://osakimassagechair.com/pages/privacy-policy",
      terms: "https://osakimassagechair.com/pages/terms-of-service",
    });
  });

  it("uses default policy links for help.osakichair.com", () => {
    expect(resolveStorePolicyUrls("help.osakichair.com")).toEqual({
      storeDomain: "osakiusa.com",
      ...DEFAULT_POLICY_URLS,
    });
  });
});
