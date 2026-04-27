"""
헬스케어 질의응답 데이터셋 → CSV 변환 (대용량 최적화 버전)

주요 개선사항:
- 메모리 최적화: 답변 텍스트 대신 파일 경로만 인덱싱
- 체크포인트: 중단 후 이어서 처리 가능
- 배치 저장: 5,000개마다 CSV append
- tqdm 진행바
- ThreadPoolExecutor: JSON 파싱 병렬화 (HDD I/O는 순차 유지)
"""

import json
import csv
import random
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE = Path("D:/20260424/120.초거대AI 사전학습용 헬스케어 질의응답 데이터/3.개방데이터/1.데이터")
SPLITS = ["Training"]
OUTPUT_CSV = Path("c:/study/project/mountain/healthcare_qa_training.csv")
CHECKPOINT_FILE = Path("c:/study/project/mountain/processed_files.txt")
BATCH_SIZE = 1000
MAX_WORKERS = 4  # HDD라서 너무 많으면 오히려 느려짐


def get_answer_text(json_path: Path) -> str:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        a = data.get("answer", {})
        parts = [a.get("intro", ""), a.get("body", ""), a.get("conclusion", "")]
        return " ".join(p.strip() for p in parts if p.strip())
    except Exception:
        return ""


def collect_answer_paths(labeling_root: Path) -> dict[str, list[Path]]:
    """답변 파일 경로만 인덱싱 (텍스트는 나중에 읽음 → 메모리 절약)"""
    answer_dir = labeling_root / "2.답변"
    folder_answers: dict[str, list[Path]] = defaultdict(list)

    if not answer_dir.exists():
        return folder_answers

    print(f"  답변 경로 인덱싱 중...")
    for json_file in tqdm(answer_dir.rglob("HC-A-*.json"), desc="  답변 인덱싱", unit="파일"):
        rel = str(json_file.parent.relative_to(answer_dir))
        folder_answers[rel].append(json_file)

    return folder_answers


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_checkpoint(paths: list[str]):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


def parse_question(json_path: Path) -> tuple[str, str] | None:
    """질문 파일 파싱 — (file_str, question_text) 반환"""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        q = data.get("question", "").strip()
        return (str(json_path), q) if q else None
    except Exception:
        return None


def write_batch(rows: list[dict], first_write: bool):
    mode = "w" if first_write else "a"
    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if first_write:
            writer.writeheader()
        writer.writerows(rows)


def process_split(split: str, folder_answers: dict[str, list[Path]],
                  processed: set[str], is_first_split: bool) -> int:
    question_dir = BASE / split / "02.라벨링데이터" / "1.질문"
    if not question_dir.exists():
        print(f"[{split}] 질문 폴더 없음, 건너뜀")
        return 0

    all_q_files = [
        f for f in question_dir.rglob("HC-Q-*.json")
        if str(f) not in processed
    ]
    print(f"[{split}] 미처리 질문 파일: {len(all_q_files):,}개")

    batch: list[dict] = []
    checkpoint_batch: list[str] = []
    total_written = 0
    skipped = 0
    first_write = is_first_split and not OUTPUT_CSV.exists()

    with tqdm(total=len(all_q_files), desc=f"[{split}]", unit="파일") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(parse_question, f): f for f in all_q_files}

            for future in as_completed(futures):
                q_file = futures[future]
                result = future.result()
                pbar.update(1)

                if result is None:
                    skipped += 1
                    continue

                file_str, question = result
                rel = str(q_file.parent.relative_to(question_dir))
                answer_paths = folder_answers.get(rel)

                if not answer_paths:
                    skipped += 1
                    checkpoint_batch.append(file_str)
                    continue

                answer_path = random.choice(answer_paths)
                answer = get_answer_text(answer_path)
                if not answer:
                    skipped += 1
                    checkpoint_batch.append(file_str)
                    continue

                batch.append({"question": question, "answer": answer})
                checkpoint_batch.append(file_str)

                if len(batch) >= BATCH_SIZE:
                    write_batch(batch, first_write)
                    save_checkpoint(checkpoint_batch)
                    total_written += len(batch)
                    first_write = False
                    batch.clear()
                    checkpoint_batch.clear()

    # 남은 배치 저장
    if batch:
        write_batch(batch, first_write)
        save_checkpoint(checkpoint_batch)
        total_written += len(batch)

    print(f"[{split}] 완료 — 작성: {total_written:,}행, 스킵: {skipped:,}행")
    return total_written


def main():
    random.seed(42)

    processed = load_checkpoint()
    if processed:
        print(f"체크포인트 로드: {len(processed):,}개 파일 이미 처리됨, 이어서 시작")

    total = 0
    for i, split in enumerate(SPLITS):
        print(f"\n=== {split} 처리 중 ===")
        labeling_root = BASE / split / "02.라벨링데이터"
        folder_answers = collect_answer_paths(labeling_root)
        print(f"  답변 폴더 수: {len(folder_answers):,}")
        is_first = (i == 0 and not processed)
        total += process_split(split, folder_answers, processed, is_first)

    print(f"\n전체 완료! 총 {total:,}행 → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
