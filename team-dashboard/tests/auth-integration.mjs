// Real HTTP + D1 tests. Refuse production; generate private test passwords locally.
import assert from "node:assert/strict";
import fs from "node:fs";
import crypto from "node:crypto";
const base = process.env.GLM_API_BASE ?? "http://127.0.0.1:5173";
if (!["127.0.0.1", "localhost"].includes(new URL(base).hostname))
  throw new Error("Local test database only");
const keys = JSON.parse(
  fs.readFileSync(
    process.env.GLM_TEST_KEYS_FILE ?? "work/team-access-keys.json",
    "utf8",
  ),
);
const file = "work/test-admin-password.json";
if (!fs.existsSync(file))
  fs.writeFileSync(
    file,
    JSON.stringify({ password: crypto.randomBytes(32).toString("base64url") }),
  );
const { password } = JSON.parse(fs.readFileSync(file, "utf8"));
let cookie = "";
async function call(
  path,
  { body, key, origin = base, status = 200, withCookie = true } = {},
) {
  const response = await fetch(base + path, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
      ...(origin ? { Origin: origin } : {}),
      ...(key ? { Authorization: "Bearer " + key } : {}),
      ...(withCookie && cookie ? { Cookie: cookie } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: text };
  }
  assert.ok(
    (Array.isArray(status) ? status : [status]).includes(response.status),
    `Expected ${status}, got ${response.status}: ${JSON.stringify(data)}`,
  );
  return { response, data };
}
await call("/api/auth", {
  body: { action: "setup", password },
  key: keys.michaelkrymski,
  origin: "https://evil.example",
  status: 403,
});
let { data: initial } = await call("/api/auth");
if (!initial.configured) {
  await call("/api/auth", {
    body: { action: "setup", password },
    key: keys[Object.keys(keys).find((id) => id !== "michaelkrymski")],
    status: 403,
  });
  const { response } = await call("/api/auth", {
    body: { action: "setup", password },
    key: keys.michaelkrymski,
  });
  const header = response.headers.get("set-cookie");
  assert.match(header, /HttpOnly/);
  assert.match(header, /SameSite=Strict/);
  assert.match(header, /Max-Age=43200/);
  cookie = header.split(";")[0];
} else {
  const { response } = await call("/api/auth", {
    body: { action: "login", password },
  });
  cookie = response.headers.get("set-cookie").split(";")[0];
}
await call("/api/auth", {
  body: { action: "setup", password },
  key: keys.michaelkrymski,
  status: 409,
});
assert.equal((await call("/api/auth")).data.authenticated, true);
assert.equal((await call("/api/team")).data.user.id, "michaelkrymski");
await call("/api/team", {
  body: { action: "vote", pollId: "vocab", choice: null },
  origin: "https://evil.example",
  status: [401, 403],
});
await call("/api/team", {
  body: { action: "vote", pollId: "vocab", choice: null },
  origin: "",
  status: 401,
});
await call("/api/auth", { body: { action: "logout" } });
await call("/api/team", { status: 401 }); // Revoked cookie cannot be reused.
cookie = "";
await call("/api/auth", {
  body: { action: "login", password: crypto.randomBytes(32).toString("hex") },
  status: 401,
});
const { response } = await call("/api/auth", {
  body: { action: "login", password },
});
cookie = response.headers.get("set-cookie").split(";")[0];
assert.equal((await call("/api/benchmark")).data.user.role, "coordinator");
await call("/api/auth", { body: { action: "logout" } });
cookie = "";
const statuses = await Promise.all(
  Array.from({ length: 8 }, async () => {
    const r = await fetch(base + "/api/auth", {
      method: "POST",
      headers: { Origin: base, "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "login",
        password: crypto.randomBytes(32).toString("hex"),
      }),
    });
    await r.text();
    return r.status;
  }),
);
assert.equal(statuses.filter((s) => s === 401).length, 5);
assert.equal(statuses.filter((s) => s === 429).length, 3);
await call("/api/auth", { body: { action: "login", password }, status: 429 });
console.log(
  "PASS: real API/D1 admin activation, password login, HttpOnly session, logout revocation, CSRF protection and concurrent login throttling",
);
