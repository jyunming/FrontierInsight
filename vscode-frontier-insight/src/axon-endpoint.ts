/**
 * Where is Axon listening? — endpoint discovery for the extension.
 *
 * Axon's API port is not a constant. `axon-api` resolves its bind
 * address as `--port` > `AXON_PORT` > `config.yaml`'s `api.port` >
 * `8420`, so a healthy sidecar can be on a port nothing in the
 * environment mentions. The extension used to probe a hardcoded
 * `127.0.0.1:8000` — the pre-8420 default — and report "Axon not
 * detected" while Axon was up and answering.
 *
 * So we discover instead of guessing. Axon writes a per-store lock file
 * when its API server boots (`<projectsRoot>/.axon-api.lock`, JSON
 * `{host, port, pid}`) and removes it on clean shutdown, which makes it
 * authoritative: the file exists exactly when the question has an
 * answer.
 *
 * We build an ordered, de-duplicated candidate list and probe it, first
 * live one wins, rather than picking a single precedence winner. That
 * way a stale `AXON_PORT=8000` in someone's environment costs one dead
 * probe instead of masking a live sidecar. The exception is an explicit
 * base URL (`frontierInsight.axonUrl`, `AXON_API_BASE`, `RAG_API_BASE`):
 * that names one specific Axon, so it replaces the list rather than
 * heading it — see `axonCandidates`.
 *
 * This mirrors `core/axon_endpoint.py` (CLI + web). Keep them in step —
 * three-interface parity means all three surfaces agree on where Axon is.
 *
 * Deliberately imports no `vscode` API so it can be exercised from plain
 * Node; the one setting it honours is passed in by the caller.
 */
import * as fs from "fs";
import * as http from "http";
import * as net from "net";
import * as os from "os";
import * as path from "path";

/** Axon's current default port. */
export const DEFAULT_PORT = 8420;
/** The port FI assumed before Axon moved. Probed last, for back-compat. */
export const LEGACY_PORT = 8000;
export const DEFAULT_HOST = "127.0.0.1";

const LOCK_NAME = ".axon-api.lock";

/**
 * Wildcard binds are addresses to *listen* on, not addresses to connect
 * to. The lock file records the bind address, so it frequently says
 * `0.0.0.0` — observed even when uvicorn actually bound loopback.
 */
const WILDCARD_HOSTS = new Set(["0.0.0.0", "::", "[::]", "*", ""]);

export interface AxonCandidate {
    host: string;
    port: number;
    /** Where this candidate came from — shown to the user on failure. */
    source: string;
}

export interface AxonDiscovery {
    live: boolean;
    ready: boolean;
    /** The endpoint that answered, or the one we'd suggest starting. */
    baseUrl: string;
    host: string;
    port: number;
    source: string;
    /** One line per failed candidate, for the "not detected" message. */
    attempts: string[];
    error?: string;
}

export function baseUrlOf(c: { host: string; port: number }): string {
    return `http://${c.host}:${c.port}`;
}

/** Map a bind address onto something connectable. */
export function probeHost(host: string | undefined | null): string {
    if (host === undefined || host === null) return DEFAULT_HOST;
    const h = String(host).trim();
    return WILDCARD_HOSTS.has(h) ? DEFAULT_HOST : h;
}

/**
 * Parse an `AXON_API_BASE`-style value into host + port.
 *
 * Only the host and port survive: probing always speaks plain HTTP, so
 * an `https://` value resolves its default port (443) but is not
 * contacted over TLS. A TLS-fronted Axon therefore fails visibly rather
 * than appearing to work.
 */
export function splitBaseUrl(raw: string): { host: string; port: number } | undefined {
    let s = (raw || "").trim().replace(/\/+$/, "");
    if (!s) return undefined;
    if (!s.includes("//")) s = `http://${s}`;
    try {
        const u = new URL(s);
        if (!u.hostname) return undefined;
        const port = u.port
            ? Number(u.port)
            : u.protocol === "https:" ? 443 : DEFAULT_PORT;
        if (!Number.isInteger(port) || port < 1 || port > 65535) return undefined;
        // URL keeps IPv6 literals bracketed; strip so net.connect accepts it.
        return { host: probeHost(u.hostname.replace(/^\[|\]$/g, "")), port };
    } catch {
        return undefined;
    }
}

// ---------------------------------------------------------------------------
// Axon's config.yaml
// ---------------------------------------------------------------------------

/**
 * The three fields we need out of Axon's config: where the store lives
 * (so we can find the lock file) and the configured API address (a
 * fallback when no server is running to have written a lock).
 */
export interface AxonConfigFields {
    storeBase?: string;
    apiHost?: string;
    apiPort?: number;
}

export function axonConfigPath(env: NodeJS.ProcessEnv = process.env): string {
    return env.AXON_CONFIG_PATH || path.join(os.homedir(), ".config", "axon", "config.yaml");
}

/**
 * Pull `store.base`, `api.host` and `api.port` out of Axon's config.
 *
 * A targeted block read rather than a YAML parse: the extension ships
 * with zero runtime dependencies and this file is machine-written by
 * Axon's own config writer, so it is reliably flat — top-level section
 * at column 0, scalar leaves indented under it. Anything we can't make
 * sense of is simply absent, and the caller falls back to defaults.
 */
export function parseAxonConfig(text: string): AxonConfigFields {
    const out: AxonConfigFields = {};
    let section = "";
    for (const rawLine of text.split(/\r?\n/)) {
        // Strip trailing comments only when clearly not inside a value —
        // Windows paths and URLs don't contain " #".
        const line = rawLine.replace(/\s+#.*$/, "");
        if (!line.trim()) continue;

        const top = /^([A-Za-z_][\w-]*):\s*(.*)$/.exec(line);
        if (top) {
            section = top[1];
            continue;
        }
        const leaf = /^\s+([A-Za-z_][\w-]*):\s*(.*)$/.exec(line);
        if (!leaf) continue;
        const key = leaf[1];
        let value = leaf[2].trim();
        if (!value) continue;
        // Unwrap a quoted scalar.
        const quoted = /^(['"])(.*)\1$/.exec(value);
        if (quoted) value = quoted[2];
        if (value === "null" || value === "~") continue;

        if (section === "store" && key === "base") out.storeBase = value;
        else if (section === "api" && key === "host") out.apiHost = value;
        else if (section === "api" && key === "port") {
            const n = Number(value);
            if (Number.isInteger(n) && n >= 1 && n <= 65535) out.apiPort = n;
        }
    }
    return out;
}

export function readAxonConfig(env: NodeJS.ProcessEnv = process.env): AxonConfigFields {
    try {
        return parseAxonConfig(fs.readFileSync(axonConfigPath(env), "utf-8"));
    } catch {
        // No config, unreadable, or no Axon installed — all mean "use defaults".
        return {};
    }
}

// ---------------------------------------------------------------------------
// The store lock file
// ---------------------------------------------------------------------------

/**
 * `projects_root` as Axon computes it: `<store base>/AxonStore/<username>`,
 * where the base is `AXON_STORE_BASE`, else config.yaml's `store.base`,
 * else `~/.axon`.
 */
export function storeBase(cfg: AxonConfigFields, env: NodeJS.ProcessEnv = process.env): string {
    const explicit = (env.AXON_STORE_BASE || "").trim() || (cfg.storeBase || "").trim();
    if (explicit) {
        return explicit.startsWith("~")
            ? path.join(os.homedir(), explicit.slice(1))
            : explicit;
    }
    return path.join(os.homedir(), ".axon");
}

function readLockAt(lockPath: string): { host: string; port: number } | undefined {
    try {
        const info = JSON.parse(fs.readFileSync(lockPath, "utf-8"));
        const port = Number(info?.port);
        if (!Number.isInteger(port) || port < 1 || port > 65535) return undefined;
        return { host: probeHost(info?.host), port };
    } catch {
        return undefined;
    }
}

/**
 * Read the running server's lock file, if there is one.
 *
 * Tries the username-derived path first, then scans sibling user
 * directories. The scan matters because Python's `getpass.getuser()`
 * and Node's `os.userInfo().username` can disagree (domain accounts,
 * a `USERNAME` env override), and a single-user store makes the scan
 * unambiguous anyway.
 *
 * Unlike the Python side we do not validate the lock here — the caller
 * probes every candidate regardless, so a stale lock is just one more
 * dead entry rather than a wrong answer.
 */
export function lockFileEndpoints(
    cfg: AxonConfigFields,
    env: NodeJS.ProcessEnv = process.env,
): { host: string; port: number }[] {
    const root = path.join(storeBase(cfg, env), "AxonStore");
    const out: { host: string; port: number }[] = [];
    const seen = new Set<string>();

    const push = (hit: { host: string; port: number } | undefined) => {
        if (!hit) return;
        const key = `${hit.host}:${hit.port}`;
        if (seen.has(key)) return;
        seen.add(key);
        out.push(hit);
    };

    let username = "";
    try {
        username = os.userInfo().username;
    } catch {
        username = env.USERNAME || env.USER || "";
    }
    if (username) push(readLockAt(path.join(root, username, LOCK_NAME)));

    try {
        for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
            if (!entry.isDirectory() || entry.name === username) continue;
            push(readLockAt(path.join(root, entry.name, LOCK_NAME)));
        }
    } catch {
        // No store directory — nothing to scan.
    }
    return out;
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------

export interface CandidateOptions {
    /** The `frontierInsight.axonUrl` setting, when the user set one. */
    overrideUrl?: string;
    env?: NodeJS.ProcessEnv;
    /** Set false to skip the pre-8420 fallback (e.g. when suggesting a start). */
    includeLegacy?: boolean;
}

export function axonCandidates(opts: CandidateOptions = {}): AxonCandidate[] {
    const env = opts.env ?? process.env;
    const includeLegacy = opts.includeLegacy !== false;
    const cfg = readAxonConfig(env);
    const out: AxonCandidate[] = [];
    const seen = new Set<string>();

    const add = (host: string | undefined, port: number | undefined, source: string) => {
        if (port === undefined || !Number.isInteger(port) || port < 1 || port > 65535) return;
        const h = probeHost(host);
        const key = `${h}:${port}`;
        if (seen.has(key)) return;
        seen.add(key);
        out.push({ host: h, port, source });
    };

    // A full base URL names one specific Axon, and two Axon instances
    // hold different corpora — quietly using a different one because the
    // named one is down would change what a quest reads. So an explicit
    // base URL is the *only* candidate: when it's down, we say so.
    // `AXON_HOST`/`AXON_PORT` below are deliberately not treated this
    // way; FI told users to set those for years, so a stale value is
    // likely and worth searching past.
    if (opts.overrideUrl && opts.overrideUrl.trim()) {
        const parsed = splitBaseUrl(opts.overrideUrl);
        if (parsed) {
            add(parsed.host, parsed.port, "frontierInsight.axonUrl setting");
            return out;
        }
    }
    for (const varName of ["AXON_API_BASE", "RAG_API_BASE"]) {
        const raw = env[varName];
        if (!raw) continue;
        const parsed = splitBaseUrl(raw);
        if (parsed) {
            add(parsed.host, parsed.port, varName);
            return out;
        }
    }

    // Split host/port env vars — FI's historical knob.
    if (env.AXON_HOST || env.AXON_PORT) {
        const p = env.AXON_PORT ? Number(env.AXON_PORT) : DEFAULT_PORT;
        add(env.AXON_HOST || DEFAULT_HOST, p, "AXON_HOST/AXON_PORT");
    }

    // The running server's own lock file — the authoritative answer.
    for (const hit of lockFileEndpoints(cfg, env)) {
        add(hit.host, hit.port, "store lock file");
    }

    // Axon's configured API address.
    if (cfg.apiPort) add(cfg.apiHost || DEFAULT_HOST, cfg.apiPort, "axon config.yaml");

    // Static fallbacks, current default first.
    add(DEFAULT_HOST, DEFAULT_PORT, "default");
    if (includeLegacy) add(DEFAULT_HOST, LEGACY_PORT, "legacy default");

    return out;
}

/**
 * The endpoint to suggest starting a sidecar on when none is running.
 * Skips the lock file (that describes a server that already exists) and
 * the legacy port (current Axon has moved off it).
 */
export function preferredEndpoint(opts: CandidateOptions = {}): AxonCandidate {
    const list = axonCandidates({ ...opts, includeLegacy: false })
        .filter(c => c.source !== "store lock file");
    return list[0] ?? { host: DEFAULT_HOST, port: DEFAULT_PORT, source: "default" };
}

// ---------------------------------------------------------------------------
// Probing
// ---------------------------------------------------------------------------

/**
 * Raw TCP reachability, used to tell socket-level failure (nothing
 * listening) apart from above-socket failure (proxy / EDR / protocol
 * mismatch) so the error message points at the right culprit.
 */
export function tcpConnectCheck(
    host: string, port: number, timeoutMs: number,
): Promise<{ ok: boolean; error?: string }> {
    return new Promise(resolve => {
        const sock = net.connect({ host, port, family: 4 });
        const timer = setTimeout(() => {
            sock.destroy();
            resolve({ ok: false, error: `TCP timeout after ${timeoutMs}ms` });
        }, timeoutMs);
        sock.once("connect", () => {
            clearTimeout(timer);
            sock.end();
            resolve({ ok: true });
        });
        sock.once("error", err => {
            clearTimeout(timer);
            const code = (err as NodeJS.ErrnoException).code;
            resolve({ ok: false, error: code ? `TCP ${code}` : `TCP error: ${err.message}` });
        });
    });
}

/** Health payloads are tiny; cap the read so a wrong service can't stream. */
const MAX_HEALTH_BODY_BYTES = 4096;

/**
 * One HTTP GET against a health endpoint.
 *
 * Uses Node's built-in `http.request` rather than the global `fetch`
 * (undici). VSCode users repeatedly hit `TypeError: fetch failed`
 * against a localhost Axon that curl could talk to fine — the opaque
 * undici error hides the real cause (corporate proxy env vars, EDR
 * injection, stale agent connection pool, IPv6 misroute).
 * `http.request` has no global agent keepalive pool, surfaces
 * ECONNREFUSED / ETIMEDOUT / etc. directly, and accepts `family: 4` to
 * force IPv4 so a misconfigured `::1` resolution can't quietly break it.
 */
export function httpProbe(
    host: string, port: number, urlPath: string, timeoutMs: number,
): Promise<{ ok: boolean; error?: string; status?: number; body?: string }> {
    return new Promise(resolve => {
        const req = http.request(
            {
                host, port, path: urlPath, method: "GET",
                family: 4,
                // One-shot agent so we never share a poisoned global
                // keepalive pool from an earlier failure window.
                agent: new http.Agent({ keepAlive: false }),
                timeout: timeoutMs,
            },
            res => {
                const code = res.statusCode ?? 0;
                const ok = code >= 200 && code < 300;
                let body = "";
                res.setEncoding("utf-8");
                res.on("data", (chunk: string) => {
                    if (body.length < MAX_HEALTH_BODY_BYTES) body += chunk;
                });
                res.on("end", () => {
                    resolve({
                        ok, status: code, body,
                        error: ok ? undefined : `HTTP ${code}`,
                    });
                });
                res.on("error", () => resolve({ ok, status: code, error: ok ? undefined : `HTTP ${code}` }));
            },
        );
        req.on("timeout", () => {
            req.destroy();
            resolve({ ok: false, error: `HTTP timeout after ${timeoutMs}ms` });
        });
        req.on("error", err => {
            const code = (err as NodeJS.ErrnoException).code;
            resolve({ ok: false, error: code ? `HTTP ${code}` : `HTTP error: ${err.message}` });
        });
        req.end();
    });
}

/**
 * Reject a 200 that clearly isn't Axon.
 *
 * `/health/live` returns `{"status":"alive"}`. Other services live on
 * these ports — Axon's own config defaults `vllm_base_url` to
 * `localhost:8000/v1`, exactly the port FI used to probe — and a passing
 * status code alone would mislabel one of them as a warm sidecar. An
 * empty or non-JSON body is inconclusive rather than wrong, so it stays
 * accepted; only a JSON object answering with no recognisable `status`
 * is treated as "some other service".
 */
export function looksLikeAxon(body: string | undefined): boolean {
    if (!body || !body.trim()) return true;
    let parsed: unknown;
    try {
        parsed = JSON.parse(body);
    } catch {
        return true;
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return true;
    return typeof (parsed as { status?: unknown }).status === "string";
}

export interface AxonHealthResult {
    live: boolean;
    ready: boolean;
    error?: string;
}

/**
 * Probe one endpoint's `/health/live` + `/health/ready`.
 *
 * `error` is exposed so callers can show the user WHY the probe failed
 * instead of "we couldn't tell."
 */
export async function probeAxonHealth(
    host: string, port: number | string, timeoutMs: number = 5000,
): Promise<AxonHealthResult> {
    const portN = Number(port);

    const probe = async (urlPath: string): Promise<{ ok: boolean; error?: string }> => {
        const httpRes = await httpProbe(host, portN, urlPath, timeoutMs);
        if (httpRes.ok) {
            if (!looksLikeAxon(httpRes.body)) {
                return { ok: false, error: "responded, but not an Axon health endpoint" };
            }
            return { ok: true };
        }

        // HTTP failed. A sibling TCP probe separates "nothing listening"
        // from "something is listening but the request didn't get through".
        const tcpRes = await tcpConnectCheck(host, portN, timeoutMs);
        if (tcpRes.ok) {
            return {
                ok: false,
                error: `${httpRes.error} (TCP connect succeeded — HTTP request blocked; check HTTP_PROXY / antivirus / EDR)`,
            };
        }
        return { ok: false, error: `${httpRes.error}; ${tcpRes.error}` };
    };

    const live = await probe("/health/live");
    if (!live.ok) {
        return { live: false, ready: false, error: live.error };
    }
    const ready = await probe("/health/ready");
    return { live: true, ready: ready.ok, error: ready.ok ? undefined : ready.error };
}

export interface DiscoverOptions extends CandidateOptions {
    /** Per-candidate probe timeout. */
    timeoutMs?: number;
}

/**
 * Walk the candidate list and return the first live Axon.
 *
 * When nothing answers, the result describes the *preferred* endpoint
 * (what we'd tell the user to start) and carries one `attempts` line per
 * candidate, so "not detected" comes with the list of places we looked.
 */
export async function discoverAxon(opts: DiscoverOptions = {}): Promise<AxonDiscovery> {
    const timeoutMs = opts.timeoutMs ?? 5000;
    const cands = axonCandidates(opts);
    const attempts: string[] = [];

    for (const cand of cands) {
        const health = await probeAxonHealth(cand.host, cand.port, timeoutMs);
        if (health.live) {
            return {
                live: true,
                ready: health.ready,
                baseUrl: baseUrlOf(cand),
                host: cand.host,
                port: cand.port,
                source: cand.source,
                attempts,
                error: health.error,
            };
        }
        attempts.push(`${baseUrlOf(cand)} (${cand.source}): ${health.error}`);
    }

    const pref = preferredEndpoint(opts);
    return {
        live: false,
        ready: false,
        baseUrl: baseUrlOf(pref),
        host: pref.host,
        port: pref.port,
        source: "none reachable",
        attempts,
        error: attempts.join("; "),
    };
}
