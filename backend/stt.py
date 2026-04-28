\
"""
stt.py

Whisper API STT 모듈

역할:
- 사람 말소리 또는 unknown 오디오를 텍스트로 변환
- "도와주세요", "살려주세요", "119" 등 응급구조신호 판단에 사용
"""

from __future__ import annotations

from pathlib import Path
from openai import OpenAI

from config import settings


class WhisperAPI:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 .env에 필요합니다.")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_stt_model

    def transcribe(self, audio_path: str | Path, language: str = "ko") -> str:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"STT 입력 파일이 없습니다: {audio_path}")

        with audio_path.open("rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language,
            )

        return (result.text or "").strip()
