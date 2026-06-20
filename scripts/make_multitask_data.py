import argparse
import json
import random
from pathlib import Path


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title_file", required=True)
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    title_data = read_jsonl(args.title_file)
    summary_data = read_jsonl(args.summary_file)

    data = title_data + summary_data
    random.seed(args.seed)
    random.shuffle(data)

    write_jsonl(data, args.output)

    print(f"title samples: {len(title_data)}")
    print(f"summary samples: {len(summary_data)}")
    print(f"total samples: {len(data)}")
    print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
