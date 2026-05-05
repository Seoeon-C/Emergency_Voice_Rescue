"""
임시 테스트 서버 (real 서버 완성 전 로컬 테스트용)

실제 서버 흐름:
  1. 핸드폰 앱이 /upload 로 WAV 파일 전송  →  received/ 에 저장
  2. 서버가 음성 분석 (AI 판단)
  3. 위험 상황이면 announcements/ 에 있는 안내방송 .mp3 URL 반환
  4. 핸드폰 앱이 .mp3 를 임시 다운로드 → 스피커 재생 → 즉시 삭제

폴더 구조:
  received/       ← 핸드폰에서 올라온 녹음 WAV 파일
  announcements/  ← 안내방송용 .mp3 파일 (미리 준비해둬야 함)

실행 방법:
  pip install fastapi uvicorn python-multipart
  python simple_server.py
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# ─────────────────────────────────────────────
#  설정값 (실제 서버 배포 시 환경변수 또는 설정파일로 교체)
# ─────────────────────────────────────────────
SERVER_HOST = "172.30.1.73"   # 예: "192.168.0.10"
SERVER_PORT = 8000

RECEIVED_DIR     = "received"                    # 핸드폰에서 올라온 녹음 파일
ANNOUNCEMENT_DIR = "backend/tts_to_mp3"         # 안내방송 .mp3 파일

app = FastAPI()

os.makedirs(RECEIVED_DIR, exist_ok=True)
os.makedirs(ANNOUNCEMENT_DIR, exist_ok=True)

# announcements/ 폴더만 외부에 노출 (received/ 는 노출 불필요)
app.mount("/announcements", StaticFiles(directory=ANNOUNCEMENT_DIR), name="announcements")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...), device_id: str = Form(...)):
    """
    핸드폰 앱에서 녹음 파일을 받는 엔드포인트.

    반환값:
      - announcement_url: 재생할 안내방송 .mp3 의 URL
                          위험 없음으로 판단하면 빈 문자열 "" 반환
    """
    save_path = os.path.join(RECEIVED_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    print(f"[수신] 장치={device_id}  파일={file.filename}  →  {save_path}")

    # ── 여기서 실제 서버는 AI 분석을 수행합니다 ──────────────────────────
    # 예시: result = analyze_audio(save_path)
    # 분석 결과에 따라 announcements/ 안의 적절한 .mp3 파일명을 선택해서 반환
    # ──────────────────────────────────────────────────────────────────────

    # 테스트: 항상 status1 반환
    # 실제 서버: 분석 결과에 따라 파일명 선택 (예: "male_warning_status1.mp3", "male_warning_status2.mp3")
    announcement_file = "male_warning_status1.mp3"
    announcement_path = os.path.join(ANNOUNCEMENT_DIR, announcement_file)

    if os.path.exists(announcement_path):
        announcement_url = f"http://{SERVER_HOST}:{SERVER_PORT}/announcements/{announcement_file}"
    else:
        announcement_url = ""
        print(f"[경고] {announcement_path} 파일이 없습니다.")

    # 위험 없음으로 판단했을 때는 아래처럼 반환
    # announcement_url = ""

    print(f"[반환] announcement_url={announcement_url or '(없음)'}")

    return {
        "status": "success",
        "announcement_url": announcement_url,  # .mp3 URL or ""
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
