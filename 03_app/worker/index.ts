// Motion Atlas Worker: serves the React app and forwards keypoints to the existing Render model API.
// It never receives video. D1 analysis summaries are a later step; nothing is stored here yet.

export interface Env {
  ASSETS: Fetcher;
  /** Base URL of the Render service, e.g. https://<service>.onrender.com (no trailing slash needed). */
  RENDER_API_URL: string;
}

const MAX_KEYPOINT_BYTES = 8 * 1024 * 1024;   // ~13 min of 25 fps keypoints; video never comes here
const RENDER_TIMEOUT_MS = 150_000;            // Render free tier cold start + first model download

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

async function forward(env: Env, path: string, init: RequestInit): Promise<Response> {
  const base = (env.RENDER_API_URL ?? "").trim().replace(/\/+$/, "");
  if (!base) return json(503, { detail: "RENDER_API_URL is not configured for this Worker." });
  let upstream: Response;
  try {
    upstream = await fetch(base + path, { ...init, signal: AbortSignal.timeout(RENDER_TIMEOUT_MS) });
  } catch (err) {
    const timedOut = err instanceof DOMException && err.name === "TimeoutError";
    return json(timedOut ? 504 : 502, {
      detail: timedOut
        ? "The model server did not answer in time. It may still be waking up; try again in a minute."
        : "Could not reach the model server.",
    });
  }
  // Pass status and JSON body through (FastAPI 422 details are useful to the user).
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url);

    if (pathname === "/api/health") {
      if (request.method !== "GET") return json(405, { detail: "Use GET." });
      return forward(env, "/health", { method: "GET" });
    }

    if (pathname === "/api/analyze/keypoints") {
      if (request.method !== "POST") return json(405, { detail: "Use POST." });
      if (!(request.headers.get("content-type") ?? "").includes("application/json")) {
        return json(415, { detail: "Send keypoints as application/json." });
      }
      const length = Number(request.headers.get("content-length"));
      if (!length) return json(411, { detail: "Content-Length is required." });
      if (length > MAX_KEYPOINT_BYTES) return json(413, { detail: "Keypoint payload is too large; use a shorter clip." });
      // Stream the body straight through: {"frames": [T][17][3]}, validated by the Render API.
      return forward(env, "/predict/keypoints", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: request.body,
      });
    }

    if (pathname.startsWith("/api/")) return json(404, { detail: "Unknown API route." });
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
