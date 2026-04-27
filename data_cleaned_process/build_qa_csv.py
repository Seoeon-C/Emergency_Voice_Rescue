"""
헬스케어 질의응답 데이터셋 → CSV 변환 스크립트

구조: 라벨링데이터/1.질문/<카테고리>/<질환명>/<의도>/HC-Q-*.json
      라벨링데이터/2.답변/<카테고리>/<질환명>/<의도>/HC-A-*.json

같은 폴더 경로(카테고리/질환명/의도)끼리 Q-A 매핑
"""

import json
import csv
import os
import random
from pathlib import Path
from collections import defaultdict

BASE = Path("D:/20260424/120.초거대AI 사전학습용 헬스케어 질의응답 데이터/3.개방데이터/1.데이터")

SPLITS = ["Training", "Validation"]

OUTPUT_CSV = Path("c:/study/project/mountain/healthcare_qa.csv")


def get_answer_text(answer_obj):
    parts = [
        answer_obj.get("intro", ""),
        answer_obj.get("body", ""),
        answer_obj.get("conclusion", ""),
    ]
    return " ".join(p.strip() for p in parts if p.strip())


def collect_answers_by_folder(labeling_root: Path) -> dict[str, list[str]]:
    """답변 폴더를 순회하며 {상대경로: [answer_text, ...]} 딕셔너리 반환"""
    answer_dir = labeling_root / "2.답변"
    folder_answers: dict[str, list[str]] = defaultdict(list)

    if not answer_dir.exists():
        return folder_answers

    for json_file in answer_dir.rglob("HC-A-*.json"):
        rel = json_file.parent.relative_to(answer_dir)
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            text = get_answer_text(data.get("answer", {}))
            if text:
                folder_answers[str(rel)].append(text)
        except Exception:
            pass

    return folder_answers


def process_split(split: str, folder_answers: dict[str, list[str]], writer):
    question_dir = BASE / split / "02.라벨링데이터" / "1.질문"
    if not question_dir.exists():
        print(f"[{split}] 질문 폴더 없음, 건너뜀")
        return 0

    all_files = list(question_dir.rglob("HC-Q-*.json"))
    total = len(all_files)
    print(f"[{split}] 총 질문 파일: {total}개")

    count = 0
    skipped = 0

    for i, json_file in enumerate(all_files):
        if i % 50000 == 0 and i > 0:
            print(f"  진행: {i}/{total} ({i*100//total}%)")

        rel = str(json_file.parent.relative_to(question_dir))
        answers = folder_answers.get(rel)
        if not answers:
            skipped += 1
            continue

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            question = data.get("question", "").strip()
            if not question:
                continue

            answer = random.choice(answers)
            writer.writerow({"질문": question, "답변": answer})
            count += 1
        except Exception:
            pass

    print(f"[{split}] 완료 — 작성: {count}행, 스킵: {skipped}행")
    return count


def main():
    random.seed(42)
    total = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["질문", "답변"])
        writer.writeheader()

        for split in SPLITS:
            print(f"\n=== {split} 처리 중 ===")
            labeling_root = BASE / split / "02.라벨링데이터"
            print(f"  답변 파일 수집 중...")
            folder_answers = collect_answers_by_folder(labeling_root)
            print(f"  답변 폴더 수: {len(folder_answers)}")
            total += process_split(split, folder_answers, writer)

    print(f"\n완료! 총 {total}행 → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
