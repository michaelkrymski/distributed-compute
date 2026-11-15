# Team dashboard: Netlify frontend, protected API backend

> Personal repository migration: see [MIGRATION.md](../MIGRATION.md). Hosting URLs below are examples; configure your own server or enter your existing team server manually. No backend was moved by this source extraction.

The React frontend is built independently with Vite and deployed to Netlify. Shared votes, milestones, experiment records, benchmark lobbies, and benchmark results remain in Cloudflare D1 through the existing backend at `https://your-backend.example`.

Netlify serves `/api/*` through a proxy rewrite to that backend. Both the repository-root `netlify.toml` and the static `_redirects` file support this. No runtime credential is bundled into the frontend. GitHub Pages was rejected by GitHub because this private repository's current plan does not support it; Netlify is the requested replacement. Repository visibility stays private.

## Build and deploy to Netlify

From the repo root, Netlify reads `netlify.toml`:

- Base directory: `team-dashboard`
- Build command: `npm run build:frontend`
- Publish directory: `frontend-dist`
- Node: 24

For a manual deployment:

```bash
cd team-dashboard
npm ci
npm run build:frontend
```

Upload **only the contents of `frontend-dist/`** (or a ZIP containing those files at its root), not the source checkout. Keep `_redirects` in the deploy so the authenticated API proxy works. Manual uploads do not create an automatic GitHub-to-Netlify connection; repository imports use the supplied configuration. Never add secrets to a `VITE_*` variable; those are public build-time values.

## Team access keys

Each team member gets a distinct access key. `michaelkrymski` has coordinator permissions for creating, starting, and cancelling benchmarks. Members can ready their own agents and read the shared records. The API derives identity from the key, not a submitted username.

The browser requires a key before loading shared records and stores it only in sessionStorage for that tab session. All protected API reads/writes require `Authorization: Bearer ...`. The backend stores only SHA-256 key hashes in the secret runtime setting `TEAM_KEYS_JSON`. CORS permits bearer-token clients; no cookie authentication is accepted for these data endpoints. OPTIONS returns no private data.

An organizer can create fresh keys for a **new local setup** with:

```bash
node scripts/create-team-keys.mjs
```

It writes ignored `work/team-access-keys.json` (plaintext keys, owner only), `work/team-key-hashes.json`, and `.dev.vars` (local hashes). It refuses to overwrite existing keys. The existing production keys must not be regenerated as an ordinary build step. Distribute each member's own key privately; never put the key file in Git, a deployment archive, a URL, or a chat intended for the whole public. The generated file is not a source artifact.

For production, put the hash mapping into Sites' secret `TEAM_KEYS_JSON` runtime setting, then deploy the API build. The backend's platform access must permit Netlify/CLI requests; the application key check then protects every data endpoint. The backend was explicitly authorized to become reachable with these team-key checks. Do not expose a previous cookie-only API version.

## Local backend and frontend

Node 24 is recommended. From this folder:

```bash
npm ci
node scripts/create-team-keys.mjs  # new local setup only
npm run build
for migration in drizzle/*.sql; do
  node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file "$migration"
done
npm run dev -- --port 5173
```

Apply migrations only once per fresh local database; do not rerun already applied SQL. The shown loop is for an empty local DB. Windows PowerShell users can run the migration command separately for each SQL file instead. The frontend can also be developed independently with `npm run dev:frontend` on port 5174; its `/api` proxy points to the local backend on 5173. Local and hosted D1 databases are separate.

The full-stack Sites build remains `npm run build`. Its tracked `.openai/hosting.json`, D1 binding, and immutable Drizzle migrations belong to the API deployment. Publish updates through the Sites source workflow. A GitHub push does not automatically deploy that backend.

## Verification

```bash
node node_modules/typescript/bin/tsc --noEmit
node --test tests/model.test.mjs tests/benchmark.test.mjs
node tests/api-integration.mjs
```

The last command makes **actual HTTP requests to the actual local API and D1**, refusing non-loopback origins. It uses the local generated key file (override with `GLM_TEST_KEYS_FILE`) and can target another local port with `GLM_API_BASE`. It creates/cancels a local test lobby, verifies the five-agent readiness gate and coordinator permissions, and checks voting. Registration fixtures test API validation only; they are not presented as connected training workers.

The full real-network acceptance test is in `hivemind-tools/tests/e2e_real_backend.py`. It launches five separate CPU Hivemind processes against the real API/D1 for 60 seconds warmup plus 300 seconds measurement. See that folder's README. No fake backend or DHT is used in this test. Reports and logs remain ignored local files.

## What is public, and what is stored

The static interface, planning formulas, and explanatory text are public. Team votes, device details, private peer addresses, readiness, and reports require a team key. Data lives in D1 and is not part of the frontend bundle. The dashboard polls benchmark state every two seconds and ordinary team data every 15 seconds. Benchmark execution happens on the participants' locally launched processes; the website does not run GPU work or launch software on another computer.

All admitted team members may vote, finalize/reopen planning decisions, update milestones, and log ordinary experiment records. Benchmark start/cancel/create actions are coordinator-only. Custom questions do not automatically change simulator math. Simulations stay local and are labeled separately from actual results. The current benchmark UI shows the active/latest lobby; older session data remains in D1.


## Admin password and computer counts

The public Netlify dashboard now has administrator password setup/login for `michaelkrymski`. Use the coordinator key for **one-time account activation**, then choose a password (16–256 characters). The browser receives a revocable 12-hour HttpOnly/Secure/SameSite=Strict session cookie; passwords and session tokens are stored only as hashes in D1. Bootstrap activation cannot replace an existing password. Member keys and CLI coordinator credentials remain valid; no new group administrator accounts are created. Passwords, keys, test account files, results, and `.dev.vars` stay outside Git.

Migration `0002_aspiring_mephistopheles.sql` creates `admin_accounts` and `admin_sessions`; apply it along with earlier migrations for a fresh database. The site deployment applies Drizzle migrations. Keep origin validation in `lib/admin-auth.ts` aligned with the Netlify domain if it changes. Do not enable credentialed wildcard CORS. The Netlify API proxy keeps browser requests same-origin and forwards session cookies.

A benchmark accepts 1–10 independently identified computer processes. Multiple nodes can share a member key. The admin can mark ready or remove a lobby computer; members can change only their own computers. Start requires every online computer ready, freezes that list atomically, and rejects later registration. Stale/offline computers are shown but excluded from Start; remove them to free registration slots. A single-computer test reports compute only, with no fabricated communication measurement. The organizer contributes compute by launching an agent while using the admin browser session, just like anyone else.

Run `node tests/auth-integration.mjs` against an isolated local database after the other API tests. It generates an ignored local test password and tests actual D1 persistence, activation rejection, password verification, cookie flags, session revocation, browser-origin checks, and concurrent attempt limits. Its final test deliberately locks the **local test** admin account for 15 minutes; use a fresh test database for repeat acceptance runs. Never run it against production.


## Benchmark settings and model throughput

Protocol 3 adds admin-only saved warmup (0–600s), measurement (10–3600s), and
workload selection. Settings persist in the existing session JSON; no D1 schema
migration is needed. Updates and Start are conditional writes so the schedule
cannot change after Start. Workload/batch changes are disallowed once agents
register. Training mode requires identical model fingerprints across online
nodes and checks reported model/batch/token counts against the frozen session.

Synthetic connectivity testing remains model-independent. Real-training mode
loads the separately installed theoretical model workload and shows local/group
measured tokens per second plus a 500M-token estimate. Group estimates require
complete reports from every frozen participant and use the minimum measured
round count. No estimate is inferred from synthetic GFLOP/s. Update both agents
and the API together; cancel/recreate any old protocol-2 lobby.
