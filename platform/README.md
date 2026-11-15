# Distributed Compute shared platform

This service gives independent teams their own members, administrators, projects,
worker keys, benchmark coordination, training rooms, progress and histories. The
web interface lives at https://michaelkrymski.me/compute/. GPU training and the
Hivemind rendezvous run in locally launched desktop/CLI workers, not in a browser
or Cloudflare Worker. No new hosting deployment is required per team.

## Account and team flow

Create account and team together: the creator is that team's first administrator.
Other users create an account and redeem a single-use invitation. Administrators
can promote/demote/remove members. The last administrator cannot be removed or
demoted. Removing membership revokes that account's team worker keys.

Worker keys are reusable across computers, scoped to one project and account.
The full key is returned only by create/rotate responses; GET endpoints return
metadata and a short prefix. The database stores only a SHA-256 digest. Rotation
invalidates the old value; stale concurrent rotations are rejected. A lost key
cannot be recovered; regenerate it. Never put keys in URLs, logs, or Git.

Passwords are salted PBKDF2-SHA256 hashes (100,000 iterations, the Workers runtime
limit). Sessions expire after 12 hours; only their digests are stored server-side.
The frontend keeps its session token in tab-scoped sessionStorage, never passwords
or worker keys. This is a bearer-token design rather than cross-site cookies:
GitHub Pages and the backend are separate origins. Protect the frontend from XSS;
there is no claim of HttpOnly protection for JavaScript session tokens. CORS admits
only the portfolio origin, this service and localhost development. No wildcard
credentialed CORS is used. Signup/login and sensitive operations have rate limits.
Account recovery/email reset is not implemented; save account passwords securely
and appoint another trusted administrator for important teams.

## Projects and training

The included driver supports configurable decoder dimensions and token goals.
External architectures use a locally selected workload factory, explicitly passed
with `--workload-factory module:function`. The backend never chooses Python imports
or executes uploaded code. Model contracts/fingerprints must agree across peers.
The worker's default local consent limit is 25M parameters; increase it explicitly
with `--max-parameters` after reviewing device capacity. Benchmark results are
self-reported by authenticated workers, not independent proof of model quality.

Worker keys bind legacy `/api/benchmark` and `/api/rooms` requests to one project,
so existing desktop processes can connect using the service origin. Browser
controls use account-authenticated `/api/platform/teams/:team/projects/:project`
routes, checked for membership on the server. All state queries are scoped by
team/project; simultaneous updates use revision-checked writes with bounded
retries. Session settings and model parameters are frozen in session snapshots.
The current benchmark supports 1–10 computers (compiled BF16 is a single-node
profile). History is bounded to 20 sessions and 256KB per coordination record.
This initial storage design is intended for small trusted teams, not massive
clusters or high-frequency telemetry. Training progress is nominal shared token
count; benchmark time estimates exclude additional corpus/evaluation/checkpoint
costs and changing participation.

An existing self-hosted team is not automatically imported or modified. Its old
accounts/keys/data remain on its original backend. Create a team and project here
and issue new keys when intentionally moving workers to the shared platform.

## Local development and real acceptance tests

```sh
npm ci
npm run db:generate  # only after intentional schema edits
npx wrangler d1 migrations apply DB --local
npm run dev
# another terminal
npm test
npm run build
npm run validate
```

Tests use a real local Worker HTTP server and SQLite-backed D1. They create
ephemeral accounts, exercise cross-team denial, role and last-admin enforcement,
invitation claims, key rotation/revocation, and actual coordination API states.
They do not claim GPU or training-quality validation. The separate worker smoke
test uses real CPU PyTorch/Hivemind processes against this API.

## Hosting and source

The live managed deployment uses Cloudflare Workers/D1 through Sites. The public
source is mirrored in `platform/` in michaelkrymski/distributed-compute; production
records, keys, local state and source credentials are excluded. For independent
hosting, create a D1 database, configure its binding in wrangler.jsonc, apply the
generated migrations remotely, then deploy with Wrangler. Update the frontend
API origin and CORS allowlist deliberately. GitHub Pages only hosts the static
frontend and cannot run this API. Netlify supports template deploy buttons, but
deploying the old static template alone does not create this Cloudflare backend.

Migrations are schema-only, generated with Drizzle and applied before deployment.
No runtime endpoint creates tables. Keep the service project ID unchanged when
publishing updates; no runtime secrets are stored in hosting metadata.
