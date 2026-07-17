import { describe, expect, it } from "vitest";
import {
  DEFAULT_POLICY_STORE_DOMAIN,
  resolvePolicyStoreDomain,
  resolveStorePolicyUrls,
} from "./welcomeMessage";

describe("resolvePolicyStoreDomain", () => {
  it("uses the parent Shopify store when provided", () => {
    expect(resolvePolicyStoreDomain("www.osakiusa.com")).toBe("www.osakiusa.com");
    expect(resolvePolicyStoreDomain("titanchair.com")).toBe("titanchair.com");
  });

  it("falls back for chat infrastructure hosts", () => {
    expect(resolvePolicyStoreDomain("help.osakichair.com")).toBe(
      DEFAULT_POLICY_STORE_DOMAIN
    );
    expect(resolvePolicyStoreDomain("api.osakichair.com")).toBe(
      DEFAULT_POLICY_STORE_DOMAIN
    );
  });
});

describe("resolveStorePolicyUrls", () => {
  it("builds Shopify policy links from the resolved store", () => {
    const urls = resolveStorePolicyUrls("www.osakiusa.com");
    expect(urls.privacy).toBe("https://www.osakiusa.com/policies/privacy-policy");
    expect(urls.terms).toBe("https://www.osakiusa.com/policies/terms-of-service");
  });

  it("uses fallback store links for help.osakichair.com", () => {
    const urls = resolveStorePolicyUrls("help.osakichair.com");
    expect(urls.privacy).toBe(
      `https://${DEFAULT_POLICY_STORE_DOMAIN}/policies/privacy-policy`
    );
    expect(urls.terms).toBe(
      `https://${DEFAULT_POLICY_STORE_DOMAIN}/policies/terms-of-service`
    );
  });
});
