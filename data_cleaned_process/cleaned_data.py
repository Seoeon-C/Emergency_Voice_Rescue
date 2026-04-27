import pandas as pd
import re
import numpy as np # 빈 값 처리를 위해 추가

# 1. 파일 불러오기 (실제 변환하신 CSV 파일명으로 변경하세요)
file_path = 'emergency_val.csv' 
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ '{file_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    exit()

print(f"▶ 1. 초기 데이터 로드 완료: 총 {len(df)}행")

# 2. 필수 컬럼 무결성 체크
# 우리가 원하는 'question', 'answer' 영문 컬럼이 정확히 있는지 확인합니다.
if 'question' not in df.columns or 'answer' not in df.columns:
    print("❌ 오류: 'question' 또는 'answer' 컬럼이 없습니다. 앞서 만든 CSV 파일을 확인해주세요.")
    exit()

# 3. 결측치(NaN) 1차 제거
# 질문이나 답변 중 하나라도 아예 없는(비어있는) 불량 데이터 줄을 삭제합니다.
before_null = len(df)
df = df.dropna(subset=['question', 'answer'])
print(f"▶ 2. 초기 결측치(빈 칸) 제거: {before_null - len(df)}개 삭제됨")

# 4. 텍스트 정제 함수 정의
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 정제 규칙: 한글, 영문, 숫자, 기본 띄어쓰기, 필수 문장부호만 남깁니다.
    # 영문 포함: CPR, AED, COVID-19 등 의료 약어 보존
    clean = re.sub(r'[^가-힣a-zA-Z0-9\s.,?()/-]', '', text)
    
    # 여러 칸 띄워진 공백을 한 칸으로 줄이고, 문장 앞뒤의 쓸데없는 공백을 자릅니다.
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

print("✨ 3. 텍스트 전처리(기호 제거 및 공백 정리) 진행 중...")
df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

# 5. 정제 후 빈 값(Empty String) 2차 제거
# "!!!" 같은 기호만 있던 데이터는 위 함수를 거치면 ""(빈 문자열)이 됩니다.
# 이를 찾아내서 완전히 삭제합니다.
df.replace("", np.nan, inplace=True)
before_empty = len(df)
df = df.dropna(subset=['question', 'answer'])
print(f"▶ 4. 정제 후 내용이 비어버린 데이터 제거: {before_empty - len(df)}개 삭제됨")

# 6. 중복 데이터 제거
# 질문과 답변이 완전히 똑같은 쌍이 있다면 하나만 남기고 지웁니다. (과적합 방지)
before_dup = len(df)
df = df.drop_duplicates(subset=['question', 'answer'], keep='first')
print(f"▶ 5. 완전히 중복된 문답 세트 제거: {before_dup - len(df)}개 삭제됨")

# 7. 최종 저장
final_len = len(df)
print(f"\n✅ [최종 결과] 전처리가 완벽하게 끝났습니다! 남은 데이터: {final_len}행")

if final_len > 0:
    output_path = 'cleaned_dataset_final.csv' # 저장될 파일 이름
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 깨끗해진 파일이 저장되었습니다: {output_path}")
else:
    print("❌ 경고: 전처리 후 남은 데이터가 없습니다. 원본 데이터의 상태를 확인해주세요.")