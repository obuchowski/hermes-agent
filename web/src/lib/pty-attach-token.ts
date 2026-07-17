const PTY_ATTACH_TOKEN_KEY = "hermes.pty.token.chat";
const PTY_ATTACH_NAMESPACE_KEY = "hermes.pty.namespace.chat";

interface TokenStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export interface PtyAttachTokenOptions {
  scope: string;
  rotate?: boolean;
  storage?: TokenStorage;
  createToken?: () => string;
  createNamespace?: () => string;
  documentIdentity?: object;
  navigationType?: NavigationTimingType;
}

const fallbackTokens = new WeakMap<TokenStorage, Map<string, string>>();
const documentNamespaces = new WeakMap<object, string>();

function fallbackFor(storage: TokenStorage): Map<string, string> {
  let tokens = fallbackTokens.get(storage);
  if (!tokens) {
    tokens = new Map<string, string>();
    fallbackTokens.set(storage, tokens);
  }
  return tokens;
}

function createBrowserToken(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function browserNavigationType(): NavigationTimingType {
  const entry = performance.getEntriesByType(
    "navigation",
  )[0] as PerformanceNavigationTiming | undefined;
  return entry?.type ?? "navigate";
}

export function ptyAttachScope(
  profile: string | null | undefined,
  resume: string | null | undefined,
): string {
  return `${profile ?? ""}\0${resume ?? ""}`;
}

export function ptyAttachToken(options: PtyAttachTokenOptions): string {
  const {
    scope,
    rotate = false,
    createToken = createBrowserToken,
    createNamespace = createBrowserToken,
    navigationType = browserNavigationType(),
  } = options;
  const storage = options.storage ?? window.sessionStorage;
  const documentIdentity =
    options.documentIdentity ??
    (typeof window === "undefined" ? storage : window);
  let namespace = documentNamespaces.get(documentIdentity) ?? "";
  if (!namespace) {
    if (navigationType === "reload") {
      try {
        namespace = storage.getItem(PTY_ATTACH_NAMESPACE_KEY) ?? "";
      } catch {
        // Storage-blocked reloads fall back to a fresh document namespace.
      }
    }
    if (!namespace) namespace = createNamespace();
    documentNamespaces.set(documentIdentity, namespace);
    try {
      storage.setItem(PTY_ATTACH_NAMESPACE_KEY, namespace);
    } catch {
      // The document-scoped in-memory namespace remains usable.
    }
  }

  const key = `${PTY_ATTACH_TOKEN_KEY}:${namespace}:${encodeURIComponent(scope)}`;
  const fallback = fallbackFor(storage);
  let token = "";

  if (!rotate) {
    try {
      token = storage.getItem(key) ?? "";
    } catch {
      // Private mode can deny storage access. The in-memory fallback still
      // keeps reconnects in this page bound to one PTY.
    }
    token ||= fallback.get(key) ?? "";
  }

  if (!token) {
    token = createToken();
    fallback.set(key, token);
    try {
      storage.setItem(key, token);
    } catch {
      // The in-memory copy above remains available for this page lifetime.
    }
  }

  return token;
}
