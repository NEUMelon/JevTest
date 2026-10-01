# JevTest

Research code and numerical artefacts for **Probability Interfaces for Low-Label Entity Matching: A Controlled Study of Jev, Token Probabilities, and Generative LLMs** (manuscript in preparation; not an accepted publication).

Authors: Yuhao Song (corresponding author), Chang Wang, Chenyu Wang, Northeastern University.

## What is released

- Experiment/analysis code, frozen question templates and configurations.
- Label-budget pair identifiers and numerical per-example/seed outputs.
- Accepted result tables and four reviewed PDF/PNG figure pairs.
- Additional author-library string-baseline comparisons and legacy-readout diagnostics.
- File hashes and a read-only verification script.

The release excludes credentials, login configuration, raw API caches/responses/prompts, raw benchmark record text, environments and trained checkpoint binaries. Benchmarks and pretrained models must be obtained from their original sources, under their own terms. This repository does not grant rights to those external resources. Some historical result metadata refers to local paths or omitted private evidence; these are provenance records, not public links. Numerical summaries are provided separately from raw source evidence.

## Read the results in this order

1. `docs/free_completion_20261001/结论与期刊建议.md`, `新增对照表.md`, `补验结果与限制.md` and `tables/` for the latest added comparisons.
2. `jevwrangle/reports/final_delivery_20261001/` for the accepted main results, statistical scope and figures in `figures_reviewed/`.
3. Numerical files under `jevwrangle/runs/` for prediction-level inspection.

M2 is a self-built string-feature classifier. M3 uses the Magellan team's string-matching library, but is **not** a successful official Magellan replication. The WA/AG reference-deviation gate remains unresolved. Jev's low-label advantages are task-dependent; strong generative-model results and all negative findings are retained. Comparisons involving Luna use the same frozen 500 test pairs; full-test and 500-pair results are not interchangeable. Confidence intervals are descriptive, not evidence that the complete historical hypothesis family was multiplicity-adjusted. Manual E8/E12 annotations, a second fully evaluated commercial typed model and future version-drift measurements are not complete.

## Offline verification (no API or GPU)

```sh
python -X utf8 verify_release.py
```

This uses only the Python standard library and checks the released files against `PUBLIC_FILE_MANIFEST.csv`. It makes no network requests and does not rerun experiments.

## Running experiments

The research environment used Python, NumPy, pandas, SciPy, scikit-learn, PyYAML, rapidfuzz, httpx and py_stringmatching; GPU baselines additionally require compatible PyTorch, Transformers and upstream Ditto dependencies. Historical launch scripts contain project-specific paths and are not a one-command reproduction environment. See `REPRODUCIBILITY.md` for what can be reproduced with this public subset. Do not start paid API or remote GPU queues merely to inspect the results.

## Licence

Original project code is released under the MIT licence in `LICENSE`. Vendored Ditto retains its Apache-2.0 licence and upstream attribution. Dataset-derived numerical artefacts retain any applicable underlying dataset terms; the MIT code licence does not relicense benchmark data, pretrained weights, third-party material or model-provider responses. No raw provider responses are included.

## 中文说明

本仓库公开论文对应的研究代码、问题配置、预算ID、数值预测与核验表图。论文尚在准备中。请优先读取最新补验文档及最终验收目录，不将旧阶段输出直接作为论文结论。没有公开密钥、原始API返回、原始记录文本、GPU模型权重或论文草稿。现有预测可离线核验；完整API/GPU实验需要自行获取数据、模型及运行环境，不承诺下载本仓库即可自动完整重跑。
