# 基于预训练生成模型的中文新闻标题与摘要生成系统

本项目为自然语言技术课程大作业，面向中文新闻文本生成任务，实现中文新闻标题生成、新闻摘要生成以及多任务统一生成。项目基于中文预训练生成模型 Randeng-T5-77M-MultiTask-Chinese，将标题生成和摘要生成统一建模为 Text-to-Text 任务，并通过任务提示词区分不同生成目标。

本项目不是直接调用大语言模型 API，而是基于公开数据集进行数据预处理、模型微调、自动评测和预测展示，重点体现完整的自然语言生成任务流程。

## 一、项目功能

目前已实现：

1. 新闻标题生成：输入新闻正文，自动生成简洁标题。
2. 新闻摘要生成：输入新闻正文，自动生成新闻摘要。
3. 多任务统一生成：使用同一个模型同时支持标题生成和摘要生成。
4. 数据预处理：支持标题数据和摘要数据转换为统一 JSONL 格式。
5. 自动评测：训练阶段计算 BLEU、ROUGE-1、ROUGE-2、ROUGE-L 等指标。
6. 模型预测：支持通过命令行加载 checkpoint 并生成标题或摘要。

后续计划实现：

1. 关键词提取；
2. 关键词引导生成；
3. 摘要长度控制；
4. 前端可视化演示界面；
5. 生成耗时、字数、关键词覆盖率等结果分析。

最终系统目标是：用户输入一篇中文新闻正文，系统自动提取关键词，并生成对应的新闻标题和新闻摘要。

## 二、模型方法

本项目使用 Randeng-T5-77M-MultiTask-Chinese 作为基础预训练生成模型。该模型属于 Text-to-Text 生成模型，可以将不同自然语言处理任务统一表示为“输入文本到输出文本”的形式。

本项目构造两类任务提示词：

```text
生成标题：新闻正文
生成摘要：新闻正文
```

模型根据输入前缀判断当前任务是生成标题还是生成摘要。例如：

```json
{"src": "生成标题：新闻正文", "tgt": "新闻标题"}
{"src": "生成摘要：新闻正文", "tgt": "新闻摘要"}
```

在多任务训练阶段，标题生成样本和摘要生成样本被混合到同一个训练集中，使同一个模型能够根据不同任务提示完成不同生成目标。

## 三、数据说明

### 1. 标题生成数据

标题生成数据采用新闻正文—标题对，每条样本包含：

```json
{"content": "新闻正文", "title": "新闻标题"}
```

预处理后转换为统一的 JSONL 格式：

```json
{"src": "生成标题：新闻正文", "tgt": "新闻标题"}
```

其中，`src` 为模型输入，`tgt` 为模型需要学习生成的目标标题。

### 2. 摘要生成数据

摘要生成数据采用 CLTS 中文长文本摘要数据集。原始数据为平行文本格式：

```text
train.src    训练集新闻正文
train.tgt    训练集人工摘要
valid.src    验证集新闻正文
valid.tgt    验证集人工摘要
test.src     测试集新闻正文
test.tgt     测试集人工摘要
```

本地数据统计结果如下：

```text
train.src    148317 行
train.tgt    148317 行
valid.src     20393 行
valid.tgt     20393 行
```

原始文本采用分字格式，例如：

```text
山 西 省 晋 中 市
```

预处理时去除字与字之间的空格，恢复为正常中文文本：

```text
山西省晋中市
```

随后转换为统一的 JSONL 格式：

```json
{"src": "生成摘要：新闻正文", "tgt": "人工摘要"}
```

### 3. 多任务数据构建

多任务训练时，将标题生成数据和摘要生成数据合并，并随机打乱，得到统一的多任务训练集。

示例：

```json
{"src": "生成标题：新闻正文A", "tgt": "新闻标题A"}
{"src": "生成摘要：新闻正文B", "tgt": "新闻摘要B"}
{"src": "生成标题：新闻正文C", "tgt": "新闻标题C"}
{"src": "生成摘要：新闻正文D", "tgt": "新闻摘要D"}
```

这里的“合并”指的是数据层面的合并，不是把两个已经训练好的模型进行权重融合。最终模型从同一个 Randeng-T5 初始化，并在混合后的多任务数据集上继续微调，得到一个同时支持标题生成和摘要生成的 checkpoint。

## 四、项目结构

```text
NewsGen-Chinese/
├── baseline/                    # 基础训练代码
├── scripts/                     # 数据处理与预测脚本
│   ├── prepare_title_data.py
│   ├── prepare_summary_data.py
│   ├── make_multitask_data.py
│   ├── predict_title.py
│   └── predict_multitask.py
├── data/                        # 数据目录，本仓库不上传大规模原始数据
├── outputs/                     # 模型输出目录，本仓库不上传 checkpoint
├── requirements.txt
├── README.md
└── .gitignore
```

说明：

* `baseline/`：基础训练框架；
* `scripts/prepare_title_data.py`：标题数据预处理脚本；
* `scripts/prepare_summary_data.py`：摘要数据预处理脚本；
* `scripts/make_multitask_data.py`：多任务数据合并脚本；
* `scripts/predict_multitask.py`：多任务模型预测脚本；
* `data/`：本地数据目录；
* `outputs/`：训练输出目录。

由于原始数据和模型 checkpoint 文件较大，GitHub 仓库默认不上传 `data/title_raw/`、`data/summary_raw/`、`data/*_processed/` 和 `outputs/`。

## 五、环境安装

建议使用 conda 创建环境：

```bash
conda create -n newsgen python=3.9 -y
conda activate newsgen
pip install -r requirements.txt
```

主要依赖包括：

```text
torch
transformers==4.26.1
pytorch-lightning==1.4.9
torchmetrics==0.5.0
sentencepiece
jieba
rouge
nltk
scikit-learn
numpy
pandas
tqdm
protobuf==3.20.3
```

如果使用 NumPy 2.x 时出现 `np.Inf` 相关报错，可以在 `baseline/train.py` 开头加入兼容补丁：

```python
import numpy as np
if not hasattr(np, "Inf"):
    np.Inf = np.inf
```

## 六、数据处理方法

### 1. 标题数据处理

生成标题训练集：

```bash
python scripts/prepare_title_data.py \
  --input data/title_raw/train.json \
  --output data/title_processed/title_train_5000.json \
  --max_samples 5000
```

生成标题验证集：

```bash
python scripts/prepare_title_data.py \
  --input data/title_raw/dev.json \
  --output data/title_processed/title_dev_1000.json \
  --max_samples 1000
```

### 2. 摘要数据处理

生成摘要训练集：

```bash
python scripts/prepare_summary_data.py \
  --src data/summary_raw/train.src \
  --tgt data/summary_raw/train.tgt \
  --output data/summary_processed/summary_train_5000.json \
  --max_samples 5000
```

生成摘要验证集：

```bash
python scripts/prepare_summary_data.py \
  --src data/summary_raw/valid.src \
  --tgt data/summary_raw/valid.tgt \
  --output data/summary_processed/summary_dev_1000.json \
  --max_samples 1000
```

### 3. 构建多任务训练集

构建训练集：

```bash
python scripts/make_multitask_data.py \
  --title_file data/title_processed/title_train_5000.json \
  --summary_file data/summary_processed/summary_train_5000.json \
  --output data/multitask_processed/multitask_train_10000.json
```

构建验证集：

```bash
python scripts/make_multitask_data.py \
  --title_file data/title_processed/title_dev_1000.json \
  --summary_file data/summary_processed/summary_dev_1000.json \
  --output data/multitask_processed/multitask_dev_2000.json
```

## 七、模型训练

### 1. 本地 CPU 调试训练

```bash
python baseline/train.py \
  --train_file data/multitask_processed/multitask_train_2000.json \
  --dev_file data/multitask_processed/multitask_dev_400.json \
  --batch_size 1 \
  --max_epochs 1 \
  --max_source_length 512 \
  --max_target_length 128 \
  --model_path IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese \
  --gpus 0 \
  --lr 5e-5 \
  --num_workers 0 \
  --plugins "" \
  --output_dir outputs/multitask_v1_2000
```

### 2. 正式多任务训练

```bash
python baseline/train.py \
  --train_file data/multitask_processed/multitask_train_10000.json \
  --dev_file data/multitask_processed/multitask_dev_2000.json \
  --batch_size 1 \
  --max_epochs 1 \
  --max_source_length 512 \
  --max_target_length 128 \
  --model_path IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese \
  --gpus 0 \
  --lr 5e-5 \
  --num_workers 0 \
  --plugins "" \
  --output_dir outputs/multitask_v2_10000
```

如果在 GPU 服务器上训练，可以将 `--gpus 0` 修改为 `--gpus 1`，并根据显存情况适当增大 `batch_size`。

## 八、模型预测

### 1. 生成标题

```bash
CKPT=$(find outputs/multitask_v1_2000 -name "*.ckpt" | head -n 1)

python scripts/predict_multitask.py \
  --ckpt "$CKPT" \
  --task title \
  --text "新闻正文..."
```

### 2. 生成摘要

```bash
CKPT=$(find outputs/multitask_v1_2000 -name "*.ckpt" | head -n 1)

python scripts/predict_multitask.py \
  --ckpt "$CKPT" \
  --task summary \
  --text "新闻正文..."
```

## 九、当前实验进度

### 1. 标题生成 smoke test

使用 100 条训练样本和 20 条验证样本完成标题生成流程验证，确认数据读取、模型训练、验证评测和 checkpoint 保存均可正常运行。

### 2. 标题生成初步训练

使用 1000 条训练样本和 200 条验证样本进行标题生成初步训练。示例输入为高校人工智能专业建设相关新闻，模型生成结果为：

```text
进一步加强人工智能相关专业建设
```

### 3. 摘要生成 smoke test

使用 100 条训练样本和 20 条验证样本完成摘要生成流程验证，得到验证结果：

```text
bleu=0.0847
rouge=0.2099
rouge-1=0.2852
rouge-2=0.1387
rouge-l=0.2436
```

### 4. 多任务模型训练

使用标题 1000 条、摘要 1000 条构建多任务训练集，使用标题 200 条、摘要 200 条构建多任务验证集，完成多任务 v1 训练，得到验证结果：

```text
bleu=0.1313
rouge=0.2579
rouge-1=0.3198
rouge-2=0.2039
rouge-l=0.2809
```

多任务模型可以通过 `生成标题：` 和 `生成摘要：` 两类任务提示词，在同一个 checkpoint 下完成标题生成和摘要生成。

## 十、后续计划

后续将继续完善以下内容：

1. 扩大多任务训练数据规模，构建 10000 条训练样本和 2000 条验证样本；
2. 在服务器 GPU 环境下完成正式训练；
3. 增加关键词提取模块；
4. 增加关键词引导生成模块；
5. 增加摘要长度控制功能；
6. 设计前端可视化演示页面；
7. 增加生成耗时、字数统计、关键词覆盖率等结果分析；
8. 整理实验截图、结果表格和课程报告。
