# DeepfakeBench 云端评测报告

已在 GitHub Actions 的免费公开仓库标准运行器中完成三个官方预训练模型的推理评测。项目克隆、依赖安装、模型加载、数据下载和推理均在 GitHub 托管的 Ubuntu 云端完成；用户电脑只保存返回的结果文件。

运行日期：2026-10-09  
运行记录：https://github.com/jackgreenbuskdjjdj/jack/actions/runs/37919507670  
项目：https://github.com/SCLBD/DeepfakeBench  
项目版本：`f188b1c105465e2e5377eb536a95022ae0e4522d`  
执行环境：Ubuntu 22.04 / Python 3.10.22 / PyTorch 1.12.1+cpu / CPU 4 线程，无 GPU。  
本次未进行训练。

## 数据与评测范围

使用研究团队公开发布的 [OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace)，从 wiki.zip 的真实图像与 insight.zip 的伪造图像中，按相同图像 ID 配对，固定随机种子 1024 抽取 256 对，共 512 张图像。三个模型使用同一组样本。

数据版本：`eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c`。  
图像处理：保留公开原图，按项目原始数据加载器转换 RGB、三次插值缩放为 256 × 256，再使用均值/标准差 [0.5, 0.5, 0.5] 归一化；未追加人脸裁剪。  
批次大小：16。准确率阈值：伪造概率 > 0.5。  
模型权重：DeepfakeBench 官方 v1.0.1 release；三个模型均通过严格完整参数加载。

最初尝试下载项目作者提供的 UADFV 预处理数据，Google Drive 返回“下载人数过多”限额错误。该尝试见 [失败日志](https://github.com/jackgreenbuskdjjdj/jack/actions/runs/37919114217)。因此本报告为替代数据子集的实际运行结果，不是官方 FF++ / UADFV 标准基准复现，也不是整套项目的全部模型和全部数据集评测。

## 实测结果

| 模型 | 准确率 | AUC | EER | AP | 512 张图像评测耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| meso4 | 50.20% | 0.5113 | 50.00% | 51.49% | 4.12 秒 |
| meso4Inception | 47.46% | 0.4704 | 52.34% | 48.34% | 12.03 秒 |
| xception | 50.39% | 0.4455 | 55.08% | 47.44% | 64.75 秒 |

耗时包含推理与评测中的数据加载，不包含环境安装、数据包下载和权重下载。AUC/AP 越高越好，EER 越低越好。

在此固定子集和当前图像处理方式下，三个模型区分能力整体接近随机。数据分布和图像处理方式与标准 benchmark 不同；这些结果不能用于否定或复现论文报告的标准数据集性能。没有反转预测分数、调整阈值或依据测试结果重新挑选样本。

## 运行适配

为使用免费 CPU 运行器，仅导入本次评测所需的 detector、backbone、loss 和 dataset，去掉 test.py 中未使用的训练模块导入。Xception 在严格加载完整 detector 权重之前，跳过会被完整权重覆盖的 ImageNet 初始化。模型网络结构和 forward 方法保持项目原实现；数据加载、test_epoch 和指标计算使用项目原函数。

公开图像被组织为每张图像一个样本目录，避免样本 ID 聚合冲突。JSON 中的 video_auc 在这次单图样本下等于图像 AUC，不应解释为视频评测指标。

## 返回文件

结果压缩包含：

- evaluation_report.json：完整指标、版本、数据与权重校验值、混淆矩阵。
- 三个 *_predictions.csv：逐图真实标签与伪造概率，各 512 行。
- deepfakeface_roc.png：三个模型的 ROC 曲线。
- data_audit.json：所选图像的来源、原图尺寸和 SHA256。
- cloud_run.log：云端数据下载、模型加载、推理与指标日志。
- run_manifest.json：运行配置。

逐图 CSV 已复核：每个文件均含真实/伪造各 256 张，正确分类数分别为 Meso4 257、MesoInception 243、Xception 258，与报告准确率一致。

[GitHub 结果包](https://github.com/jackgreenbuskdjjdj/jack/actions/runs/37919507670/artifacts/11611456437)在云端保留 7 天；随本报告返回的本地 ZIP 副本可长期保存。
