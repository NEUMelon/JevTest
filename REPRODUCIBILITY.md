# Reproducibility scope

This is a curated snapshot, not a private machine backup. `verify_release.py` verifies all released scientific files without API/GPU access. CSV summaries and prediction/feature files support inspecting and, where the necessary feature/label outputs are included, recalculating reported statistics. Full source-level reruns require externally obtained canonical benchmark data, model weights and runtime dependencies. Obtain those from their original sources; use the published split/budget IDs and frozen questions, and preserve population matching.

Original paid API caches, raw requests/responses, full Qwen prompt evidence, credentials, SSH setup and trained checkpoint binaries are retained privately. This prevents a public claim that every input evidence file is downloadable here. Some archived scripts refer to private absolute paths: adapt these only after checking the accepted protocol, not by running a remote scheduler blindly. Main files are under `src/recovery`; `src/experiments` retains historical implementations and must not be treated as the corrected accepted pipeline.

The final delivery's four figures were visually reviewed. Newly drafted manuscript figures are not part of this release. Reuse numerical results only with their stated budget, seed, test population, readout and uncertainty protocol. Human annotation sheets remain incomplete and no human agreement claim is made.

No paid calls or GPU work were performed to create this release.
