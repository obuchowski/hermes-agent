import { describe, expect, it } from "vitest";

import { openPtySocket } from "./pty-connect-guard";

describe("openPtySocket", () => {
  it("does not construct a socket when the effect becomes stale while resolving the URL", async () => {
    let resolveUrl!: (url: string) => void;
    const buildUrl = new Promise<string>((resolve) => {
      resolveUrl = resolve;
    });
    let current = true;
    const constructed: string[] = [];

    const pending = openPtySocket({
      buildUrl: () => buildUrl,
      isCurrent: () => current,
      createSocket: (url) => {
        constructed.push(url);
        return { url };
      },
    });
    current = false;
    resolveUrl("ws://stale.example/api/pty");

    await expect(pending).resolves.toBeNull();
    expect(constructed).toEqual([]);
  });
});
