# KineWorld / 勘境 changes

Upstream: https://github.com/dexmal/opendw

Inspected base: `e33befa8005a1585e0140dbf464566e90bc79aa1`. License: **Apache-2.0**. Original license and notices remain unchanged; code, checkpoints and datasets can have separate terms.

## 致谢 / Acknowledgements

感谢 **Dexmal / OpenDW authors** 以及所有贡献者的开源精神。你们公开研究成果、代码与复现方法，让更多研究者和小团队能够学习、验证和继续改进。勘境珍惜这些贡献，保留原始作者、提交历史、许可证与引用信息；我们的新增工作以可检查的改动和测试记录说明。

We thank Dexmal / OpenDW authors and the broader open-source community for sharing their work. Original contributions remain attributed to their authors. KineWorld's changes are documented separately, with their validation limits.

## Implemented change

Use float64 batch moments and parallel Welford merging instead of E[x^2]-E[x]^2. Reject empty, nonfinite, nonnumeric and mismatched batches. Tests compare split-batch statistics against NumPy, expose integer-square overflow and preserve state after invalid input. This does not retrain DW05.

## Validation

3 focused tests passed locally. Run `python -m unittest discover -s tests_kineworld -v` (Python 3.10+; NumPy is required for OpenDW statistics).

These are engineering/numerical checks, not a model-quality benchmark. No full checkpoint reproduction, new weights, measured leaderboard improvement or upstream endorsement is claimed. Original model descriptions and scores in the upstream README remain the authors' results.

## Project map

[KineWorld source/validation directory](https://github.com/kineworld/.github/blob/main/world-models/open-source-adoption.md) · [KineJing integration research](https://github.com/kineworld/KineJing).
