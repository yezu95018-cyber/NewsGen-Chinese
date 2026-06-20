import argparse
import json
import re
from pathlib import Path


def clean_text(text):
    text = text.strip()
    # 这个数据集是分字格式：山 西 省 晋 中 市
    # 这里直接去掉所有空白，恢复成正常中文：山西省晋中市
    text = re.sub(r"\s+", "", text)
    return text


def convert(src_path, tgt_path, output_path, max_samples=None):
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    with src_path.open("r", encoding="utf-8") as fs, \
         tgt_path.open("r", encoding="utf-8") as ft, \
         output_path.open("w", encoding="utf-8") as fo:

        for src_line, tgt_line in zip(fs, ft):
            src = clean_text(src_line)
            tgt = clean_text(tgt_line)

            if not src or not tgt:
                skipped += 1
                continue
            if len(src) < 50 or len(tgt) < 5:
                skipped += 1
                continue

            item = {
                "src": "生成摘要：" + src,
                "tgt": tgt
            }

            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

            if max_samples is not None and count >= max_samples:
                break

    print(f"Saved {count} samples to {output_path}")
    print(f"Skipped {skipped} samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    convert(args.src, args.tgt, args.output, args.max_samples)


if __name__ == "__main__":
    main()
