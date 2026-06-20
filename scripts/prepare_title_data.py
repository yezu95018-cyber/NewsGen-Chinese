import argparse
import json
from pathlib import Path


def load_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data


def clean_text(text):
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def convert(input_path, output_path, max_samples=None):
    data = load_json_or_jsonl(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            title = clean_text(item.get("title", ""))
            content = clean_text(item.get("content", ""))

            if not title or not content:
                continue
            if len(content) < 30 or len(title) < 2:
                continue

            src = f"生成标题：{content}"
            tgt = title

            f.write(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")
            count += 1

            if max_samples is not None and count >= max_samples:
                break

    print(f"Saved {count} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    convert(args.input, args.output, args.max_samples)


if __name__ == "__main__":
    main()
