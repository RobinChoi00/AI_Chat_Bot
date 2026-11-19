import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveWarrantyStoreDomain } from "./warrantyStoreDomain";

describe("resolveWarrantyStoreDomain", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("maps help portal hosts to osakiusa.com", () => {
    vi.stubGlobal("window", {
      location: {
        search: "",
        hostname: "help.osakichair.com",
      },
      document: { referrer: "" },
    });
    // document is read from global in browser; stub via location only path
    Object.defineProperty(globalThis, "document", {
      configurable: true,
      value: { referrer: "" },
    });
    expect(resolveWarrantyStoreDomain()).toBe("osakiusa.com");
  });

  it("maps store=help.osakichair.com query to osakiusa.com", () => {
    vi.stubGlobal("window", {
      location: {
        search: "?store=help.osakichair.com",
        hostname: "help.osakichair.com",
      },
    });
    Object.defineProperty(globalThis, "document", {
      configurable: true,
      value: { referrer: "" },
    });
    expect(resolveWarrantyStoreDomain()).toBe("osakiusa.com");
  });

  it("keeps titanchair.com as titanchair.com", () => {
    vi.stubGlobal("window", {
      location: {
        search: "?store=titanchair.com",
        hostname: "help.osakichair.com",
      },
    });
    Object.defineProperty(globalThis, "document", {
      configurable: true,
      value: { referrer: "" },
    });
    expect(resolveWarrantyStoreDomain()).toBe("titanchair.com");
  });
});
