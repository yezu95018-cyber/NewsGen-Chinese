import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# 让脚本能找到 baseline 里的 models.py / utils.py / tokenizer.py
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
        max_target_length=64,
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
    state_dict = ckpt["state_dict"]
    light_model.load_state_dict(state_dict, strict=False)

    light_model.eval()
    return light_model


def generate_title(light_model, text, max_source_length=512, max_target_length=64, beams=4):
    tokenizer = light_model.tokenizer
    model = light_model.model

    src = "生成标题：" + text.strip()

    inputs = tokenizer(
        [src],
        padding=True,
        truncation=True,
        max_length=max_source_length,
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

    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return pred.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model_path", default="IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    light_model = load_model(args.ckpt, args.model_path)
    title = generate_title(light_model, args.text)

    print("\n生成标题：")
    print(title)


if __name__ == "__main__":
    main()
