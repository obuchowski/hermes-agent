import { describe, expect, it } from "vitest";

import { ptyAttachToken } from "./pty-attach-token";

class MemoryStorage {
  private readonly values = new Map<string, string>();

  clone(): MemoryStorage {
    const cloned = new MemoryStorage();
    for (const [key, value] of this.values) cloned.setItem(key, value);
    return cloned;
  }

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

function tokenFactory(...tokens: string[]): () => string {
  let index = 0;
  return () => tokens[index++] ?? `token-${index}`;
}

describe("ptyAttachToken", () => {
  it("reuses one token for the same profile and resume target", () => {
    const storage = new MemoryStorage();
    const createToken = tokenFactory("first", "unused");

    expect(
      ptyAttachToken({ scope: "programmer\0session-a", storage, createToken }),
    ).toBe("first");
    expect(
      ptyAttachToken({ scope: "programmer\0session-a", storage, createToken }),
    ).toBe("first");
  });

  it("uses different tokens for different sessions and profiles", () => {
    const storage = new MemoryStorage();
    const createToken = tokenFactory("session-a", "session-b", "default-a");

    expect(
      ptyAttachToken({ scope: "programmer\0a", storage, createToken }),
    ).toBe("session-a");
    expect(
      ptyAttachToken({ scope: "programmer\0b", storage, createToken }),
    ).toBe("session-b");
    expect(ptyAttachToken({ scope: "default\0a", storage, createToken })).toBe(
      "default-a",
    );
  });

  it("isolates the same session between browser tabs", () => {
    const firstTab = new MemoryStorage();
    const secondTab = new MemoryStorage();
    const createToken = tokenFactory("tab-one", "tab-two");

    expect(
      ptyAttachToken({
        scope: "programmer\0session-a",
        storage: firstTab,
        createToken,
      }),
    ).toBe("tab-one");
    expect(
      ptyAttachToken({
        scope: "programmer\0session-a",
        storage: secondTab,
        createToken,
      }),
    ).toBe("tab-two");
  });

  it("does not reuse the legacy localStorage token", () => {
    const storage = new MemoryStorage();
    storage.setItem("hermes.pty.token.chat", "legacy");

    expect(
      ptyAttachToken({
        scope: "programmer\0session-a",
        storage,
        createToken: () => "scoped",
      }),
    ).toBe("scoped");
  });

  it("rotates only the requested scope for a forced fresh chat", () => {
    const storage = new MemoryStorage();
    const createToken = tokenFactory("old", "other", "fresh");

    expect(
      ptyAttachToken({ scope: "programmer\0", storage, createToken }),
    ).toBe("old");
    expect(
      ptyAttachToken({ scope: "programmer\0session-b", storage, createToken }),
    ).toBe("other");
    expect(
      ptyAttachToken({
        scope: "programmer\0",
        rotate: true,
        storage,
        createToken,
      }),
    ).toBe("fresh");
    expect(
      ptyAttachToken({ scope: "programmer\0session-b", storage, createToken }),
    ).toBe("other");
  });

  it("rotates a copied token when a duplicated tab starts a new navigation", () => {
    const firstTab = new MemoryStorage();
    const createToken = tokenFactory("tab-one-token", "tab-two-token");

    expect(
      ptyAttachToken({
        scope: "programmer\0session-a",
        storage: firstTab,
        documentIdentity: {},
        navigationType: "navigate",
        createNamespace: () => "tab-one",
        createToken,
      }),
    ).toBe("tab-one-token");

    const duplicatedTab = firstTab.clone();
    expect(
      ptyAttachToken({
        scope: "programmer\0session-a",
        storage: duplicatedTab,
        documentIdentity: {},
        navigationType: "navigate",
        createNamespace: () => "tab-two",
        createToken,
      }),
    ).toBe("tab-two-token");
  });

  it("reuses the scoped token across a hard refresh", () => {
    const storage = new MemoryStorage();
    const createToken = tokenFactory("before-refresh", "must-not-rotate");

    expect(
      ptyAttachToken({
        scope: "programmer\0session-a",
        storage,
        documentIdentity: {},
        navigationType: "navigate",
        createNamespace: () => "tab-one",
        createToken,
      }),
    ).toBe("before-refresh");
    expect(
      ptyAttachToken({
        scope: "programmer\0session-a",
        storage,
        documentIdentity: {},
        navigationType: "reload",
        createNamespace: () => "must-not-rotate-namespace",
        createToken,
      }),
    ).toBe("before-refresh");
  });
});
