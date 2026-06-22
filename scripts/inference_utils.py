import time
from argparse import Namespace
import sys
from pathlib import Path

import jieba.analyse
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import inspect
from baseline import models as models_module

_MODEL_CLASSES = [
    obj for name, obj in inspect.getmembers(models_module, inspect.isclass)
    if obj.__module__ == models_module.__name__
    and hasattr(obj, "load_from_checkpoint")
]

if not _MODEL_CLASSES:
    raise ImportError("没有在 baseline.models 中找到可用的 PyTorch Lightning 模型类")

T5FineTuner = _MODEL_CLASSES[0]
print(f"使用模型类：{T5FineTuner.__name__}")


def clean_title_output(text, max_chars=18):
    text = text.strip()

    for sep in ["。", "，", ",", "；", ";", "！", "？", "\n"]:
        if sep in text:
            text = text.split(sep)[0].strip()
            break

    remove_tails = ["的相关通知", "相关通知", "的相关方案", "相关方案", "的相关政策", "相关政策"]
    for tail in remove_tails:
        if text.endswith(tail):
            text = text[:-len(tail)].strip()

    return text[:max_chars].strip()


def extract_keywords(text, top_k=8):
    keywords = jieba.analyse.extract_tags(text, topK=top_k)
    return [kw.strip() for kw in keywords if kw.strip()]


def keyword_coverage(keywords, generated_text):
    if not keywords:
        return 0.0
    hit = sum(1 for kw in keywords if kw in generated_text)
    return hit / len(keywords)


class NewsGenerator:
    def __init__(self, ckpt_path=None, device=None):
        if ckpt_path is None:
            ckpts = list(Path("outputs/multitask_v2_10000").glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError("没有找到 outputs/multitask_v2_10000 下的 .ckpt 模型文件")
            ckpt_path = str(ckpts[0])

        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_args = Namespace(
            model_path="IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese",
            model_type="t5",
            recompute=False,
            learning_rate=5e-5,
            lr=5e-5,
            weight_decay=0.0,
            adam_epsilon=1e-8,
            warmup_steps=0,
            max_source_length=512,
            max_target_length=128,
            train_batch_size=1,
            eval_batch_size=1,
            batch_size=1,
            n_gpu=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
        )
        self.light_model = T5FineTuner.load_from_checkpoint(ckpt_path, args=model_args)
        self.light_model.to(self.device)
        self.light_model.eval()

        self.model = self.light_model.model
        self.tokenizer = self.light_model.tokenizer

    def generate(self, task, text, summary_length="中", beams=4):
        if task == "title":
            prompt = "生成标题：" + text
            max_target_length = 18
        else:
            prompt = "生成摘要：" + text
            length_map = {
                "短": 64,
                "中": 128,
                "长": 160,
            }
            max_target_length = length_map.get(summary_length, 128)

        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_target_length,
                num_beams=beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )

        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if task == "title":
            pred = clean_title_output(pred, max_chars=18)

        return pred

    def run(self, text, summary_length="中", top_k=8):
        start = time.time()

        title = self.generate("title", text, summary_length=summary_length)
        summary = self.generate("summary", text, summary_length=summary_length)

        cost = time.time() - start

        keywords = extract_keywords(text, top_k=top_k)
        title_cov = keyword_coverage(keywords, title)
        summary_cov = keyword_coverage(keywords, summary)

        return {
            "title": title,
            "summary": summary,
            "keywords": keywords,
            "time_cost": round(cost, 2),
            "title_length": len(title),
            "summary_length": len(summary),
            "title_keyword_coverage": round(title_cov * 100, 2),
            "summary_keyword_coverage": round(summary_cov * 100, 2),
        }
