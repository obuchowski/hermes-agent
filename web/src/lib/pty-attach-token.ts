/**
 * Identity of THIS browser tab's keep-alive PTY, sent as `?attach=`.
 *
 * The dashboard's PTY registry maps one attach token to exactly one PTY, so a
 * token two tabs present makes the second tab take the first one's terminal
 * over (the first is closed 4409 and goes dead — the reported multi-tab
 * interference). localStorage is shared per ORIGIN, so every tab read the same
 * value, and Chrome's "Duplicate tab" clones sessionStorage into the new tab,
 * so neither storage alone isolates the tabs. The token therefore lives in
 * sessionStorage *and* is claimed with a Web Lock: a tab whose token is
 * already claimed by a live document mints its own instead of sharing one.
 * See #115304.
 */

const PTY_ATTACH_TOKEN_KEY = "hermes.pty.token.chat";

/** The token this document claimed — a reconnect must not re-request the lock
 *  (Web Locks are not reentrant, so asking twice in one document deadlocks). */
const claimedTokens = new Map<string, string>();

function tabStorage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null; /* private mode / storage blocked */
  }
}

function mint(): string {
  const a = new Uint8Array(16);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Never settles, so the lock stays held for this document's lifetime. */
const HOLD: Promise<void> = new Promise<void>(() => {});

/**
 * Claim `token` for this document. True when we now hold it; the browser
 * releases it when the document unloads, so a reload of this tab finds the
 * token free again while a live second tab does not. Browsers without the Web
 * Locks API fall back to sessionStorage isolation only.
 */
async function claim(token: string): Promise<boolean> {
  const locks = typeof navigator === "undefined" ? undefined : navigator.locks;
  if (!locks) return true;
  return new Promise<boolean>((resolve) => {
    void locks.request(`hermes.pty.attach.${token}`, { ifAvailable: true }, (lock) => {
      resolve(lock !== null);
      return lock ? HOLD : Promise.resolve();
    });
  });
}

/**
 * This tab's `?attach=` token, minted when `rotate` starts a fresh session (the
 * old keep-alive PTY must NOT be reattached) or when another live tab already
 * claimed the stored one.
 */
export function ptyAttachScope(
  profile: string | null | undefined,
  resume: string | null | undefined,
): string {
  return `${profile ?? ""}\0${resume ?? ""}`;
}

export async function ptyAttachToken(rotate = false, scope = ""): Promise<string> {
  const key = scope ? `${PTY_ATTACH_TOKEN_KEY}:${encodeURIComponent(scope)}` : PTY_ATTACH_TOKEN_KEY;
  const claimedToken = claimedTokens.get(scope);
  const stored =
    claimedToken ?? tabStorage()?.getItem(key) ?? "";
  if (!rotate && stored && (stored === claimedToken || (await claim(stored)))) {
    claimedTokens.set(scope, stored);
    return stored;
  }
  const token = mint();
  try {
    tabStorage()?.setItem(key, token);
  } catch {
    /* ignore */
  }
  await claim(token);
  claimedTokens.set(scope, token);
  return token;
}
