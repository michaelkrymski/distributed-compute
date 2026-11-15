# GLM naming and optimized dashboard rollout

Python source packages and commands are now `glmbench`, `glmtrain`, and `glm-desktop` (`glm_desktop` module). Reinstall the editable projects after switching branches:

```sh
python -m pip install --no-deps -e ./training -e ./hivemind-tools
# In the separate desktop environment:
python -m pip install -e ./desktop
```

Use `python -m glmbench`, `python -m glmtrain`, or `python desktop/launch.py`. Dependencies and GPU drivers are unchanged. `GLM_TEAM_KEY` is preferred; the old environment variable is accepted during rollout. Installed old package commands may remain until explicitly uninstalled; they are not the current source.

The desktop bridge accepts older installed training package names. Existing `gatorllm-venv` folders are user-owned installations and are still discovered. Production hosting URLs, GitHub organization, admin cookie names, browser key storage, and the corpus DHT namespace remain unchanged to preserve access and peer interoperability. Existing credentials are valid; new generated keys use the `glm_` prefix. No database rows, datasets, checkpoints, or private results are renamed or added to Git.

The optimized dashboard supports `localSteps` (1, 2, 4, 8, 16, 32), proportional local batch sizes, and actual processed token accounting. Settings/device metadata use existing JSON columns, so no new SQL migration is required. Old sessions default to one local step. The desktop reports processed tokens per measured round rather than assuming equal single-step workloads. Benchmark throughput does not establish convergence or corpus quality.

Backend Sites version 8 and the Netlify frontend were deployed from the integration snapshot. No benchmark was started and no production settings, credentials, or results were modified by this rollout.
