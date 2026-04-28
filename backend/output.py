\
"""
output.py

출력 모듈

역할:
1. 상황별 고정 TTS 메시지 출력
2. 상황 데이터 로컬 기록
3. 상황실 webhook 전송
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pygame
import requests

from config import settings
from decision import DecisionResult
from environmental_sound import SoundEvent


FIXED_TTS_MESSAGES: Dict[str, str] = {
    "NONE": "",
    "INTRUSION_WARN_1": "출입이 허가되지 않은 위험 구역입니다. 즉시 안전한 곳으로 이동해 주세요.",
    "INTRUSION_WARN_2": "위험 구역에 계속 머무르고 있습니다. 위치 정보가 상황실로 전송되었습니다. 즉시 퇴장해 주세요.",
    "EMERGENCY_GUIDE": "응급 상황이 감지되었습니다. 가능한 경우 안전한 위치로 이동하고 구조 안내를 기다려 주세요.",
    "EVACUATION_GUIDE": "위험 상황이 감지되었습니다. 즉시 현재 위치에서 벗어나 안전한 곳으로 대피해 주세요.",
}


class FixedMessageSpeaker:
    """
    고정 메시지 출력.
    간단하고 안정적으로 하기 위해 pygame으로 미리 준비된 음성 파일을 재생합니다.

    기본 파일 위치:
    assets/tts/INTRUSION_WARN_1.mp3
    assets/tts/INTRUSION_WARN_2.mp3
    assets/tts/EMERGENCY_GUIDE.mp3
    assets/tts/EVACUATION_GUIDE.mp3

    파일이 없으면 콘솔에 문구만 출력합니다.
    """

    def __init__(self, tts_dir: str = "assets/tts") -> None:
        self.tts_dir = Path(tts_dir)
        self.tts_dir.mkdir(parents=True, exist_ok=True)
        pygame.mixer.init()

    def speak(self, tts_key: str) -> None:
        message = FIXED_TTS_MESSAGES.get(tts_key, "")

        if not message:
            return

        print(f"[TTS] {message}")

        audio_path = self.tts_dir / f"{tts_key}.mp3"

        if not audio_path.exists():
            print(f"[TTS] 음성 파일 없음: {audio_path}")
            print("[TTS] assets/tts 폴더에 같은 이름의 mp3 파일을 넣으면 실제 방송처럼 재생됩니다.")
            return

        pygame.mixer.music.load(str(audio_path))
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


class EventLoggerAndMessenger:
    def __init__(self, log_dir: str = "outputs/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record_and_send(
        self,
        decision: DecisionResult,
        sound_event: SoundEvent,
        stt_text: str,
        dwell_seconds: float,
    ) -> Dict:
        event = {
            "event_time": datetime.now().isoformat(timespec="seconds"),
            "zone_name": settings.zone_name,
            "location_text": settings.location_text,
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "situation": decision.situation,
            "situation_name": decision.situation_name,
            "risk_level": decision.risk_level,
            "reason": decision.reason,
            "action": decision.action,
            "tts_key": decision.tts_key,
            "emergency_candidate": decision.emergency_candidate,
            "sound": {
                "label": sound_event.label,
                "confidence": round(sound_event.confidence, 4),
                "raw_label": sound_event.raw_label,
                "person_detected": sound_event.person_detected,
                "danger_sound_detected": sound_event.danger_sound_detected,
            },
            "stt_text": stt_text,
            "dwell_seconds": round(dwell_seconds, 2),
            "note": "본 메시지는 상황실 모니터링 및 출동 판단용입니다.",
        }

        self._write_local(event)

        if decision.send_to_control_room:
            self._send_webhook(event)

        return event

    def _write_local(self, event: Dict) -> None:
        date = datetime.now().strftime("%Y%m%d")
        log_path = self.log_dir / f"events_{date}.jsonl"

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        print(f"[LOG] 이벤트 기록 완료: {log_path}")

    def _send_webhook(self, event: Dict) -> None:
        if not settings.control_room_webhook:
            print("[CONTROL_ROOM] Webhook 미설정. 로컬 로그만 저장했습니다.")
            return

        try:
            response = requests.post(
                settings.control_room_webhook,
                json=event,
                timeout=5,
            )
            print(f"[CONTROL_ROOM] 전송 완료: HTTP {response.status_code}")
        except Exception as exc:
            print(f"[WARN] 상황실 전송 실패: {exc}")
