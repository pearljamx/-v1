# 最终冻结确认

本文用于课程设计提交前的最后一次确认。它不是新的功能说明，而是说明当前项目已经进入最终交付冻结状态。

## 1. 冻结结论

当前项目已经满足课程设计交付要求，具备运行、演示、测试、模型证明、文档说明、打包预检能力。

从此状态开始，不建议继续新增功能、修改检测算法、调整阈值、替换模型或改动 UI。后续只建议执行答辩前人工摄像头验证，以及最终提交或打包操作。

当交付检查脚本全部通过后，即可进入最终提交阶段。

## 2. 已完成能力摘要

项目当前已经覆盖以下能力：

- 图像检测。
- 视频检测。
- 实时摄像头检测。
- 疲劳检测。
- 分心检测。
- 生理状态展示。
- 虚拟方向盘与手部离把展示。
- 转头/转身检测展示。
- 演示模式。
- 无摄像头固定样例备用路径。
- 真实数据集 YOLO 训练入口。
- 最终验收、打包、复现、提交前检查脚本。

## 3. 最终必须保留的模型

最终课程交付包必须保留：

```text
models_data/yolo_driver_state.pt
models_data/yolo_drowsiness_cls.pt
```

这两个模型是课程 YOLO 训练证明，必须纳入最终交付包。不要为了缩小体积删除它们，也不要在提交前重新训练或替换它们。

`models_data/yolo_steering_hand.pt` 是可选增强模型，不是当前最终交付的必要证明项。

## 4. 最终建议只执行的命令

进入最终冻结状态后，只建议执行以下只读检查和测试命令：

```bash
python scripts/pre_commit_check.py
python scripts/package_check.py
python scripts/archive_dry_run.py
python scripts/submission_manifest.py
python scripts/repro_report.py
python scripts/acceptance_check.py
python -m unittest discover -s tests -p "test_*.py" -v
```

这些脚本不会训练模型、下载数据集、删除文件、执行 `git add` 或 `git commit`，其中 `archive_dry_run.py` 只生成 dry-run 报告，不真正创建压缩包。

## 5. 最后人工动作

最终提交前建议只做以下人工动作：

- 在有摄像头的设备上，按 `docs/QUICK_START_CARD.md` 或 `docs/DEMO_SCRIPT.md` 做一次 5 分钟真机演示检查。
- 提交前人工确认并纳入未跟踪交付候选：
  - `docs/`
  - `scripts/`
  - `vendor/`
  - `web/routes_demo.py`
  - `yolo/hf_drowsiness_dataset.py`
  - `yolo/train_drowsiness_cls.py`
  - `yolo/train_real_datasets.py`
- 不要加入 ignored 运行产物，例如 `.venv/`、`datasets/`、`uploads/`、`outputs/`、`runs/`、SQLite WAL/SHM、`__pycache__/` 或 `*.pyc`。

## 6. 不应继续做的事

项目冻结后不建议继续执行：

- 不要重新训练模型。
- 不要下载新数据集。
- 不要继续调检测阈值。
- 不要继续改 UI。
- 不要删除 `.pt` 模型。
- 不要强制加入 ignored 文件。

如果必须修改以上内容，应先重新运行全部验收脚本，并重新生成 `docs/REPRODUCIBILITY_REPORT.md`、`docs/SUBMISSION_MANIFEST.md` 和 `docs/ARCHIVE_DRY_RUN.md`。
