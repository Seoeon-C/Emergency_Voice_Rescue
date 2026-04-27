"""
데이터셋 병합 스크립트

입력:
  - '팀원이 줄 firstaid train 파일명'  : firstaid 학습용 (팀원 제공, 90%)
  - '팀원이 줄 firstaid val 파일명'    : firstaid 검증용 (팀원 제공, 10%)
  - cleaned_emergency_train.csv        : 응급 특화 헬스케어 QA 학습용 (전처리 완료)
  - cleaned_emergency_val.csv          : 응급 특화 헬스케어 QA 검증용 (전처리 완료)

출력:
  - merged_train.csv  : 학습 데이터 (firstaid_train 전체 + emergency_train 전체)
  - merged_val.csv    : 검증 데이터 (firstaid_val 전체 + emergency_val 전체)
"""

import pandas as pd

# ── 입력 파일 설정 ─────────────────────────────────────────
INPUT_FIRSTAID_TRAIN  = "dataset/Cleaned_FirstAidQA_google_ko_train.csv"
INPUT_FIRSTAID_VAL    = "dataset/Cleaned_FirstAidQA_google_ko_val.csv"
INPUT_EMERGENCY_TRAIN = "dataset/cleaned_emergency_train.csv"
INPUT_EMERGENCY_VAL   = "dataset/cleaned_emergency_val.csv"

OUTPUT_TRAIN = "dataset/merged_train.csv"
OUTPUT_VAL   = "dataset/merged_val.csv"

RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[["question", "answer"]]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].str.strip()
    df["answer"]   = df["answer"].str.strip()
    df = df[df["question"].str.len() > 5]
    df = df[df["answer"].str.len() > 10]
    df = df.drop_duplicates(subset=["question"])
    return df.reset_index(drop=True)


def main():
    print("=== 데이터 로드 ===")

    firstaid_train = load_csv(INPUT_FIRSTAID_TRAIN)
    print(f"  firstaid train    : {len(firstaid_train):,}행")

    firstaid_val = load_csv(INPUT_FIRSTAID_VAL)
    print(f"  firstaid val      : {len(firstaid_val):,}행")

    emergency_train = load_csv(INPUT_EMERGENCY_TRAIN)
    print(f"  emergency train   : {len(emergency_train):,}행")

    emergency_val = load_csv(INPUT_EMERGENCY_VAL)
    print(f"  emergency val     : {len(emergency_val):,}행")

    print("\n=== 병합 및 정제 ===")
    train = clean(pd.concat([firstaid_train, emergency_train], ignore_index=True))
    val   = clean(pd.concat([firstaid_val,   emergency_val],   ignore_index=True))

    train = train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val   = val.sample(frac=1,   random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  최종 train : {len(train):,}행")
    print(f"  최종 val   : {len(val):,}행")

    train.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    val.to_csv(OUTPUT_VAL,     index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {OUTPUT_TRAIN}, {OUTPUT_VAL}")


if __name__ == "__main__":
    main()