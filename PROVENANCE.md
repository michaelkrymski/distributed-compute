# Source provenance

This repository is a source-only extraction of the training/distributed compute platform from `GatorLLM/glm-0-team-2`, branch `codex/contributor-app`, commit `624d3edd3d3af49fb96c89771323c6ca4d8debee` (2026-11-15 snapshot). The original repository and deployed services were not transferred or deleted.

Only tracked files were exported. Runtime databases, team keys, passwords, datasets, checkpoints, logs, results, and the hosted project's identity were not copied. New Git history avoids carrying unreviewed historical material into another repository. Original commit authors are retained by name in [ORIGINAL_AUTHORS.txt](ORIGINAL_AUTHORS.txt); their email addresses are omitted.

The extraction retains the desktop client, coordination/benchmark tools, reference model and trainer, dashboard/backend, setup scripts and CI. The model is an included reference workload; the platform can be developed independently.

No project-wide license was present in the source checkout. Michael Krymski confirmed that the original platform code is his own, temporarily hosted in the group repository, and authorized selecting and applying a license on 2026-11-15. That original code is now licensed under Apache-2.0, copyright Michael Krymski; see LICENSE and NOTICE. Commit-author history is preserved for provenance and does not determine copyright ownership. Third-party code and assets retain their existing rights and license terms. This license addition is in the personal repository; it does not change the organization repository.

## Migration changes

Product/repository links and installer branding are now Distributed Compute. Python modules (`glmbench`, `glmtrain`, `glm_desktop`), installed package IDs, credential storage and DHT protocol namespaces remain compatible. Existing installations are not modified. Production project identity was removed; the new copy does not deploy or connect automatically to the old backend. See [MIGRATION.md](MIGRATION.md).
