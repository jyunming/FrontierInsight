/**
 * Canonical path of the per-user persistent FI bridge.
 *
 * Two OSes, two transports — but the same _idea_: a per-user IPC
 * channel managed by the OS, no port numbers or discovery files for
 * the user (or `--serve`) to worry about.
 *
 *   POSIX:  $XDG_RUNTIME_DIR/fi-bridge.sock  (fallback /tmp/fi-bridge-{uid}.sock)
 *   Windows: \\.\pipe\fi-bridge-{USERNAME}
 *
 * Node's ``net.createServer().listen(path)`` transparently handles
 * both — Unix-domain socket on POSIX, named pipe on Windows. Python's
 * side has to switch (``asyncio.open_unix_connection`` vs. a custom
 * named-pipe wrapper), so we centralise the path resolution here and
 * mirror it in ``core.vscode_bridge.persistent_bridge_path``.
 */
import * as os from "os";
import * as path from "path";

export function persistentBridgePath(): string {
    if (process.platform === "win32") {
        // Windows named pipe namespace is server-wide, but the NAME is
        // arbitrary — include the username so two users on the same
        // box (Terminal Server, Citrix, etc.) don't collide.
        const user = (process.env.USERNAME || "default").replace(/[^A-Za-z0-9_-]/g, "_");
        return `\\\\.\\pipe\\fi-bridge-${user}`;
    }
    // POSIX: prefer XDG_RUNTIME_DIR (per-user, OS-managed, auto-cleaned
    // on logout — see freedesktop.org base-dir spec). Fall back to
    // /tmp with the UID baked in so a shared host doesn't conflict
    // across users.
    const xdg = process.env.XDG_RUNTIME_DIR;
    if (xdg) return path.join(xdg, "fi-bridge.sock");
    const uid = typeof process.getuid === "function" ? process.getuid() : "default";
    return path.join(os.tmpdir(), `fi-bridge-${uid}.sock`);
}
