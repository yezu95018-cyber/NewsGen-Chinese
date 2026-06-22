# 基于预训练生成模型的中文新闻标题与摘要生成系统

## 项目简介

本项目为自然语言处理技术课程 Project，面向中文新闻正文，基于 Randeng-T5 预训练生成模型实现新闻标题生成、新闻摘要生成、关键词提取、结果分析和历史记录管理。系统采用标题生成与摘要生成统一建模的 Text-to-Text 多任务学习方式，通过不同任务提示词控制模型生成目标，并使用 Gradio 构建可交互的课程展示系统。

本项目不是简单调用大模型 API，而是在开源预训练模型基础上进行数据处理、模型微调、自动评测和系统集成，体现从数据准备、模型训练、推理封装到前端展示的完整自然语言生成流程。

## 项目功能

- 新闻标题生成
- 新闻摘要生成
- 关键词提取
- 摘要长度控制
- 关键词覆盖率分析
- 生成耗时统计
- 用户登录 / 切换用户
- 我的生成记录
- 历史记录导出

## 技术路线

- 预训练生成模型 Randeng-T5
- Text-to-Text 任务建模
- 多任务学习：`生成标题：` 与 `生成摘要：`
- jieba TF-IDF 关键词提取
- Gradio 前端系统
- JSON 本地历史记录存储

## 模型与训练信息

| 项目 | 内容 |
| --- | --- |
| 基础模型 | `IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese` |
| 训练数据 | 标题生成 5000 条，摘要生成 5000 条 |
| 验证数据 | 标题生成 1000 条，摘要生成 1000 条 |
| 训练轮数 | 1 epoch |
| 模型大小 | 296M |
| BLEU | 0.1602 |
| ROUGE | 0.2837 |
| ROUGE-1 | 0.3458 |
| ROUGE-2 | 0.2290 |
| ROUGE-L | 0.3074 |

## 环境要求

建议环境：

- Python 3.9
- torch
- transformers==4.26.1
- pytorch-lightning==1.4.9
- torchmetrics==0.5.0
- sentencepiece
- jieba
- rouge
- nltk
- scikit-learn
- gradio

完整依赖见 [requirements.txt](requirements.txt)。

## 安装与运行

```bash
conda create -n newsgen python=3.9 -y
conda activate newsgen
pip install -r requirements.txt
python app.py
```

启动成功后，在浏览器打开：

```text
http://127.0.0.1:7860
```

## 数据说明

GitHub 仓库仅提供 `samples/` 中的小样例数据，用于展示数据格式和任务输入输出结构。完整训练数据体积较大，不直接上传 GitHub。

样例文件包括：

- `samples/multitask_sample.json`：多任务混合样例，包含标题生成和摘要生成样本。
- `samples/title_sample.json`：标题生成样例。
- `samples/summary_sample.json`：摘要生成样例。

完整数据本地目录约定如下：

```text
data/title_raw/
data/summary_raw/
data/title_processed/
data/summary_processed/
data/multitask_processed/
```

## 模型文件说明

训练好的 `.ckpt` checkpoint 文件体积较大，不上传 GitHub。本地运行前需要将训练好的 checkpoint 放到：

```text
outputs/multitask_v2_10000/
```

系统启动时会自动在该目录下查找 `.ckpt` 文件并加载模型。

## 项目结构

```text
NewsGen-Chinese/
├── app.py                         # Gradio 前端系统入口
├── baseline/                      # 训练框架与模型封装
├── scripts/                       # 数据处理、预测和推理工具
│   ├── inference_utils.py          # 前端推理封装
│   ├── prepare_title_data.py       # 标题数据预处理
│   ├── prepare_summary_data.py     # 摘要数据预处理
│   ├── make_multitask_data.py      # 多任务数据合并
│   └── predict_multitask.py        # 命令行预测脚本
├── data/                          # 本地数据目录，GitHub 不上传大数据
├── docs/                          # 使用说明、报告、演示材料
│   └── screenshots/                # 运行截图说明
├── samples/                       # GitHub 展示用小样例数据
├── outputs/                       # 模型输出目录，GitHub 不上传 checkpoint
├── exports/                       # 系统导出记录目录，GitHub 不提交真实记录
├── logs/                          # 训练日志目录，GitHub 不上传
├── requirements.txt
├── README.md
└── .gitignore
```

## 演示说明

建议演示时长为 3 到 5 分钟：

1. 介绍项目背景、任务目标和技术路线。
2. 启动 `python app.py`，展示系统首页与三个 Tab。
3. 输入或选择示例新闻，演示标题生成和摘要生成。
4. 展示关键词、生成耗时、标题字数、摘要字数和关键词覆盖率。
5. 切换到“我的记录”页面，演示刷新历史记录和导出 JSON。
6. 切换到“系统说明”页面，介绍模型、数据规模和验证指标。

## 小组协作说明

为了方便两人小组协作，`docs/` 目录已经整理了报告、PPT 和演示视频相关材料：

- 报告素材看 [docs/报告写作素材.md](docs/报告写作素材.md)。
- 实验结果看 [docs/实验结果汇总.md](docs/实验结果汇总.md)。
- 演示流程看 [docs/演示视频脚本.md](docs/演示视频脚本.md)。
- PPT 大纲看 [docs/演示PPT大纲.md](docs/演示PPT大纲.md)。
- 截图要求看 [docs/截图与材料说明.md](docs/截图与材料说明.md)。
- 组员快速阅读项目可先看 [docs/组员阅读指南.md](docs/组员阅读指南.md)。

建议分工为：成员 A 负责模型训练、数据处理、前端系统实现和实验记录整理；成员 B 负责项目报告撰写、PPT 制作、演示视频录制和截图整理。最终分工可以根据实际贡献调整。

## 课程提交材料

课程提交建议包含：

- GitHub 仓库代码链接
- `samples/` 小样例数据
- 完整数据来源与准备说明
- 训练好的 checkpoint 文件或网盘链接
- 项目报告
- 使用说明文档
- 演示 PPT
- 演示视频
- 运行界面截图和测试结果截图
