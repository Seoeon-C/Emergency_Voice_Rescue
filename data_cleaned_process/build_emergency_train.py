"""
산악 응급 특화 헬스케어 QA 추출 — Training 셋
입력 : D:/20260424/.../Training
출력 : emergency_train.csv  (question, answer)
"""

import json
import csv
import random
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────────────
BASE          = Path("D:/20260424/120.초거대AI 사전학습용 헬스케어 질의응답 데이터/3.개방데이터/1.데이터")
SPLIT         = "Training"
OUTPUT_CSV    = Path("c:/study/project/mountain/emergency_train.csv")
CHECKPOINT    = Path("c:/study/project/mountain/emergency_train_checkpoint.txt")
BATCH_SIZE    = 1000
MAX_WORKERS   = 4

# ── 추출 대상 질환 목록 ────────────────────────────────────────────
# "ALL" → 해당 카테고리 전체 포함
TARGET: dict[str, list[str] | str] = {
    "응급질환": "ALL",
    "근골격질환": [
        "염좌", "십자 인대 손상", "아킬레스 건 파열", "아킬레스 건염",
        "요추 추간판 탈출증", "좌골신경통", "골관절염",
    ],
    "순환기질환": [
        "급성 심근경색증", "협심증", "불안정형 협심증",
        "저혈압", "심계항진", "폐 혈전색전증",
    ],
    "호흡기질환": [
        "공기가슴증(기흉)", "천식", "성인 호흡곤란 증후군",
    ],
    "뇌신경정신질환": [
        "뇌졸중", "뇌출혈", "뇌경색", "발작",
    ],
    "감염성질환": [
        "파상풍", "패혈증", "라임병",
    ],
    "기타": [
        "음식 알레르기",
    ],
    "피부질환": [
        "동상", "봉와직염", "두드러기", "옻 중독",
    ],
    "귀코목질환": [
        "비출혈",
    ],
    "소화기질환": [
        "급성 위장염", "노로바이러스 장염", "복부 통증", "복막염",
    ],
    "성형미용 및 재건": [
        "안면 골절",
    ],
    "신장비뇨기질환": [
        "요로결석", "신장 결석",
    ],
    "유방내분비질환": [
        "당뇨병성 케톤산증", "골다공증",
    ],
    "치과질환": [
        "턱관절 탈구",
    ],
    "눈질환": [
        "결막염", "알레르기성 결막염",
    ],
}


def is_target(category: str, disease: str) -> bool:
    rule = TARGET.get(category)
    if rule is None:
        return False
    if rule == "ALL":
        return True
    return disease in rule


# ── 답변 경로 인덱싱 ───────────────────────────────────────────────
def collect_answer_paths(labeling_root: Path) -> dict[str, list[Path]]:
    answer_dir = labeling_root / "2.답변"
    folder_answers: dict[str, list[Path]] = defaultdict(list)
    if not answer_dir.exists():
        return folder_answers

    print("  답변 경로 인덱싱 중...")
    for json_file in tqdm(answer_dir.rglob("HC-A-*.json"), desc="  답변 인덱싱", unit="파일"):
        parts = json_file.parent.relative_to(answer_dir).parts
        if len(parts) >= 2 and is_target(parts[0], parts[1]):
            rel = str(json_file.parent.relative_to(answer_dir))
            folder_answers[rel].append(json_file)

    return folder_answers


def get_answer_text(json_path: Path) -> str:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        a = data.get("answer", {})
        parts = [a.get("intro", ""), a.get("body", ""), a.get("conclusion", "")]
        return " ".join(p.strip() for p in parts if p.strip())
    except Exception:
        return ""


# ── 체크포인트 ─────────────────────────────────────────────────────
def load_checkpoint() -> set[str]:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_checkpoint(paths: list[str]):
    with open(CHECKPOINT, "a", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")


# ── 질문 파싱 ──────────────────────────────────────────────────────
def parse_question(json_path: Path) -> tuple[str, str] | None:
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        q = data.get("question", "").strip()
        return (str(json_path), q) if q else None
    except Exception:
        return None


# ── 배치 CSV 저장 ──────────────────────────────────────────────────
def write_batch(rows: list[dict], first_write: bool):
    mode = "w" if first_write else "a"
    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if first_write:
            writer.writeheader()
        writer.writerows(rows)


# ── 메인 ───────────────────────────────────────────────────────────
def main():
    random.seed(42)

    processed = load_checkpoint()
    if processed:
        print(f"체크포인트 로드: {len(processed):,}개 파일 이미 처리됨, 이어서 시작")

    labeling_root = BASE / SPLIT / "02.라벨링데이터"
    folder_answers = collect_answer_paths(labeling_root)
    print(f"  대상 답변 폴더 수: {len(folder_answers):,}")

    question_dir = labeling_root / "1.질문"
    print("  대상 질문 파일 수집 중... (시간 소요)")
    all_q_files = []
    for json_file in tqdm(question_dir.rglob("HC-Q-*.json"), desc="  질문 스캔", unit="파일"):
        if str(json_file) in processed:
            continue
        parts = json_file.parent.relative_to(question_dir).parts
        if len(parts) >= 2 and is_target(parts[0], parts[1]):
            all_q_files.append(json_file)

    print(f"[{SPLIT}] 미처리 대상 질문 파일: {len(all_q_files):,}개")

    batch: list[dict] = []
    ckpt_batch: list[str] = []
    total_written = 0
    skipped = 0
    first_write = not OUTPUT_CSV.exists()

    with tqdm(total=len(all_q_files), desc=f"[{SPLIT}]", unit="파일") as pbar:
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
                    ckpt_batch.append(file_str)
                    continue

                answer = get_answer_text(random.choice(answer_paths))
                if not answer:
                    skipped += 1
                    ckpt_batch.append(file_str)
                    continue

                batch.append({"question": question, "answer": answer})
                ckpt_batch.append(file_str)

                if len(batch) >= BATCH_SIZE:
                    write_batch(batch, first_write)
                    save_checkpoint(ckpt_batch)
                    total_written += len(batch)
                    first_write = False
                    batch.clear()
                    ckpt_batch.clear()

    if batch:
        write_batch(batch, first_write)
        save_checkpoint(ckpt_batch)
        total_written += len(batch)

    print(f"\n완료! 작성: {total_written:,}행 | 스킵: {skipped:,}행 → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
