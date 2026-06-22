import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline"
sys.path.insert(0, str(BASELINE))

from models import LightModel


def build_args(model_path):
    return SimpleNamespace(
        model_path=model_path,
        model_type=None,
        lr=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        warmup_proportion=0.1,
        max_source_length=512,
        max_target_length=128,
        beams=4,
        rdrop=False,
        rdrop_coef=5,
        ls_eps=0.0,
        recompute=False,
    )


def load_model(ckpt_path, model_path):
    args = build_args(model_path)
    light_model = LightModel(args)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    light_model.load_state_dict(ckpt["state_dict"], strict=False)

    light_model.eval()
    return light_model


def clean_title_output(text, max_chars=18):
    text = text.strip()

    # 先按新闻标题常见断点截断，避免输出导语长句
    for sep in ["。", "，", ",", "；", ";", "！", "？", "\n"]:
        if sep in text:
            text = text.split(sep)[0].strip()
            break

    # 去掉一些容易让标题变长的套话
    remove_tails = ["的相关通知", "相关通知", "的相关方案", "相关方案", "的相关政策", "相关政策"]
    for tail in remove_tails:
        if text.endswith(tail):
            text = text[:-len(tail)].strip()

    return text[:max_chars].strip()

def generate(light_model, task, text, beams=4):
    tokenizer = light_model.tokenizer
    model = light_model.model

    if task == "title":
        prompt = "生成标题："
        max_target_length = 18
    elif task == "summary":
        prompt = "生成摘要："
        max_target_length = 128
    else:
        raise ValueError("task must be title or summary")

    src = prompt + text.strip()

    inputs = tokenizer(
        [src],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_attention_mask=True,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_target_length,
            num_beams=beams,
            early_stopping=True,
        )

    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if task == "title":
        pred = clean_title_output(pred, max_chars=18)
    return pred.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task", required=True, choices=["title", "summary"])
    parser.add_argument("--text", required=True)
    parser.add_argument("--model_path", default="IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese")
    parser.add_argument("--beams", type=int, default=4)
    args = parser.parse_args()

    light_model = load_model(args.ckpt, args.model_path)
    result = generate(light_model, args.task, args.text, beams=args.beams)

    if args.task == "title":
        print("\n生成标题：")
    else:
        print("\n生成摘要：")
    print(result)


if __name__ == "__main__":
    main()
