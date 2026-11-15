# Personal repository migration

Repository: https://github.com/michaelkrymski/distributed-compute (public). Product name: Distributed Compute. Original source remains in the organization repository; [PROVENANCE.md](PROVENANCE.md) records the snapshot.

## Existing team installation

The existing database, Netlify site, workers and organization branches remain unchanged. Existing keys work with the existing backend. In a fresh client from this repo, choose Add private server and enter the existing team's URL. That does not publish the server.

Public discovery is optional: set `COMPUTE_DIRECTORY_URL` when launching the app to use a directory deployment. No directory is contacted by default in this extracted project.

Python modules, credential storage and protocol identifiers remain stable for peer/run-bundle compatibility. Executable and installer names are now `Distributed-Compute`. The Linux launcher installs alongside the prior app. No credential is stored in source.

## New deployment, later

The personal GitHub Pages site is a separate static website, with no training backend or team data. Moving source does not move cloud accounts, databases, secrets or DNS. No automatic deployment to the old service is configured here.

The platform still uses a static frontend (optionally Netlify) and Cloudflare Workers/D1 backend:

1. Follow [PROJECT-SETUP.md](desktop/PROJECT-SETUP.md) for local development and key management. Keep `work/` and `.dev.vars` private.
2. Provision a new Workers/D1 backend when ready. `.openai/hosting.json` contains binding names only, without a live project identity. Local builds use placeholder database IDs; production needs its own bindings/hosting configuration.
3. Apply `drizzle/*.sql` to the new database. The public directory starts empty; publish listings deliberately through authenticated `/api/servers` calls.
4. Run `scripts/manage-team.mjs` to add/rotate/revoke keys. Set `TEAM_KEYS_JSON` as a backend secret. Never commit the plaintext file or secret hash map. The inherited single administrator account is `michaelkrymski`; member keys/coordinator roles work as before. Multiple administrator accounts are a separate future change.
5. Set the exact frontend origin allowlist in `lib/admin-auth.ts`. Point `/api/*` to your backend in `netlify.toml` and `public/_redirects`. These now contain `example.invalid` placeholders so a new deployment cannot accidentally target the original service.
6. Enter the new server address in the client; configure `COMPUTE_DIRECTORY_URL` only if desired. Start coordinators/agents on Tailscale using the benchmark/training guides.

Subfolder docs retain inherited team examples and compatibility/test fixture IDs. They do not represent newly created accounts or copied production credentials. Fully self-service multi-team provisioning remains future work.
